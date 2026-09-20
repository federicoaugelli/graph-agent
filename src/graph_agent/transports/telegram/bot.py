from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from graph_agent.config import TelegramChannelConfig
from graph_agent.core.service import AgentService
from graph_agent.core.sessions import SessionManager
from graph_agent.core.state import ApprovalMode
from graph_agent.events import ApprovalRequestEvent, ErrorEvent, Event, FileEvent, TokenEvent
from graph_agent.media import image_content_part, is_image
from graph_agent.transports.telegram.formatting import (
    escape_html,
    html_to_plain,
    markdown_to_telegram_html,
    split_telegram_html,
)

HTML = "HTML"

APPROVE_PREFIX = "approve:"
DENY_PREFIX = "deny:"

AUTO_COMMAND = "/auto"
MANUAL_COMMAND = "/manual"
NEW_COMMAND = "/new"

COMMANDS: list[BotCommand] = [
    BotCommand(command="new", description="Start a fresh session"),
    BotCommand(command="auto", description="Approve sensitive tools automatically"),
    BotCommand(command="manual", description="Ask before sensitive tools"),
]

CHANNEL = "telegram"
INCOMING_DIR = "incoming"
TYPING_ACTION = "typing"
TYPING_REFRESH_SECONDS = 4.0
STREAM_EDIT_INTERVAL = 1.0

MODE_REPLIES: dict[ApprovalMode, str] = {
    "auto": "Auto mode on: sensitive tools now run directly, without asking.",
    "manual": "Manual mode on: sensitive tools will ask for your approval.",
}


async def send_telegram_html(bot: Bot, chat_id: int, text: str) -> None:
    """Send a markdown reply rendered as Telegram HTML, chunked and with fallback.

    Shared by the interactive transport and the scheduler's Telegram sink so both
    channels render agent output the same way.
    """
    for chunk in split_telegram_html(markdown_to_telegram_html(text)):
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=HTML)
        except TelegramBadRequest:
            await bot.send_message(chat_id=chat_id, text=html_to_plain(chunk))


class TelegramBot:
    """Telegram transport (aiogram 3, long polling).

    Contract (Phase 3 tests pin this down):
    - session identifier = str(chat_id), thread id via SessionManager
    - handle_message: allowed_user_ids gate (empty = deny all), /new resets the thread, /auto and
      /manual toggle the approval mode; anything else runs the agent and streams
      the reply via a send_message placeholder then edit_message_text
    - streaming edits stay plain text; the final reply is converted from markdown
      to the Telegram HTML subset and sent with parse_mode=HTML (fallback: plain)
    - handle_file: downloads document/photo into workspace/incoming/{chat_id}/
      and runs the agent with the caption plus the saved path
    - ApprovalRequestEvent -> send_message with InlineKeyboardMarkup whose buttons
      have callback_data f"{APPROVE_PREFIX}{approval_id}" / f"{DENY_PREFIX}{approval_id}"
    - FileEvent -> send_document
    - handle_approval_callback: parse decision, service.resume(), answer_callback_query
    """

    def __init__(
        self,
        service: AgentService,
        config: TelegramChannelConfig,
        sessions: SessionManager,
    ) -> None:
        self.service = service
        self.config = config
        self.sessions = sessions
        token = os.environ.get(config.token_env, "")
        self.bot = Bot(token=token)
        self.dp = Dispatcher()

    def is_allowed(self, user_id: int) -> bool:
        """Whether the user may talk to the agent (empty allowlist = deny all)."""
        return user_id in self.config.allowed_user_ids

    async def set_commands(self) -> None:
        """Publish the slash-command menu so clients can list it under ``/``."""
        await self.bot.set_my_commands(COMMANDS)

    async def start(self) -> None:
        """Register handlers on self.dp and start long polling."""
        await self.set_commands()
        self.dp.message.register(self.handle_message, F.text)
        self.dp.message.register(self.handle_file, F.document | F.photo)
        self.dp.callback_query.register(
            self.handle_approval_callback,
            F.data.startswith(APPROVE_PREFIX) | F.data.startswith(DENY_PREFIX),
        )
        await self.dp.start_polling(self.bot, handle_signals=False)

    async def stop(self) -> None:
        await self.bot.session.close()

    async def handle_message(self, message: Message) -> None:
        """Text handler: /new resets, /auto and /manual toggle mode, else run the agent."""
        if message.text is None or message.from_user is None:
            return
        if not self.is_allowed(message.from_user.id):
            return

        chat_id = message.chat.id
        identifier = str(chat_id)
        command = message.text.strip().lower().split(maxsplit=1)[0]

        if command == NEW_COMMAND:
            thread_id = await self.sessions.reset(CHANNEL, identifier)
            await self.bot.send_message(chat_id=chat_id, text=f"[new session] {thread_id}")
            return

        thread_id = await self.sessions.thread_id(CHANNEL, identifier)

        if command in (AUTO_COMMAND, MANUAL_COMMAND):
            mode: ApprovalMode = "auto" if command == AUTO_COMMAND else "manual"
            await self.service.set_approval_mode(thread_id, mode)
            await self.bot.send_message(chat_id=chat_id, text=MODE_REPLIES[mode])
            return

        await self._stream_reply(chat_id, self.service.run(thread_id, message.text))

    async def handle_file(self, message: Message) -> None:
        """File handler: download document/photo into the workspace, then run the agent."""
        if message.from_user is None or not self.is_allowed(message.from_user.id):
            return

        attachment = _attachment(message)
        if attachment is None:
            return

        file, filename = attachment
        chat_id = message.chat.id
        identifier = str(chat_id)
        relative = Path(INCOMING_DIR) / identifier / filename
        destination = Path(self.service.config.agent.workspace) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        await self.bot.download(file=file, destination=destination)

        caption = (message.caption or "").strip()
        kind = "image" if is_image(destination) else "file"
        note = f"[user sent a {kind}, saved in the workspace at {relative}]"
        prompt = f"{caption}\n{note}".strip() if caption else note

        thread_id = await self.sessions.thread_id(CHANNEL, identifier)
        if is_image(destination):
            content: list[dict[str, Any]] = [
                {"type": "text", "text": prompt},
                image_content_part(destination),
            ]
            await self._stream_reply(chat_id, self.service.run(thread_id, content))
        else:
            await self._stream_reply(chat_id, self.service.run(thread_id, prompt))

    async def handle_approval_callback(self, callback: CallbackQuery) -> None:
        """Callback handler: approve/deny -> service.resume(), then stream result."""
        data = callback.data or ""

        if callback.from_user is None or not self.is_allowed(callback.from_user.id):
            await self.bot.answer_callback_query(
                callback_query_id=callback.id, text="Not authorized"
            )
            return

        if data.startswith(APPROVE_PREFIX):
            approved = True
            approval_id = data[len(APPROVE_PREFIX) :]
        elif data.startswith(DENY_PREFIX):
            approved = False
            approval_id = data[len(DENY_PREFIX) :]
        else:
            await self.bot.answer_callback_query(
                callback_query_id=callback.id, text="Unknown approval"
            )
            return

        await self.bot.answer_callback_query(
            callback_query_id=callback.id, text="Approved" if approved else "Denied"
        )

        if callback.message is None:
            return

        chat_id = callback.message.chat.id
        thread_id = await self.sessions.thread_id(CHANNEL, str(chat_id))
        await self._stream_reply(chat_id, self.service.resume(thread_id, approval_id, approved))

    async def _stream_reply(self, chat_id: int, stream: AsyncIterator[Event]) -> None:
        typing_done = asyncio.Event()
        typing_task = asyncio.create_task(self._keep_typing(chat_id, typing_done))
        await self._send_typing(chat_id)

        placeholder: Message | None = None
        text_parts: list[str] = []
        approval: ApprovalRequestEvent | None = None
        streaming = True
        last_edit = 0.0

        try:
            async for event in stream:
                if isinstance(event, TokenEvent):
                    text_parts.append(event.delta)
                    current = "".join(text_parts)
                    if placeholder is None:
                        if current:
                            typing_done.set()
                            placeholder = await self.bot.send_message(chat_id=chat_id, text=current)
                            last_edit = time.monotonic()
                    elif streaming and time.monotonic() - last_edit >= STREAM_EDIT_INTERVAL:
                        try:
                            await self.bot.edit_message_text(
                                chat_id=chat_id,
                                message_id=placeholder.message_id,
                                text=current,
                            )
                            last_edit = time.monotonic()
                        except TelegramBadRequest:
                            streaming = False
                        except TelegramRetryAfter as exc:
                            streaming = False
                            await asyncio.sleep(exc.retry_after)
                elif isinstance(event, ErrorEvent):
                    text_parts.append(f"\n[error] {event.message}")
                elif isinstance(event, FileEvent):
                    await self.bot.send_document(
                        chat_id=chat_id,
                        document=FSInputFile(event.path),
                        caption=event.caption,
                    )
                elif isinstance(event, ApprovalRequestEvent):
                    approval = event
        finally:
            typing_done.set()
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task

        final_text = "".join(text_parts).strip()
        if final_text:
            if placeholder is not None:
                await self._edit_html(chat_id, placeholder.message_id, final_text)
            else:
                await send_telegram_html(self.bot, chat_id, final_text)

        if approval is not None:
            await self._send_approval_request(chat_id, approval)

    async def _send_typing(self, chat_id: int) -> None:
        with contextlib.suppress(TelegramBadRequest):
            await self.bot.send_chat_action(chat_id=chat_id, action=TYPING_ACTION)

    async def _keep_typing(self, chat_id: int, done: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(done.wait(), timeout=TYPING_REFRESH_SECONDS)
                return
            except TimeoutError:
                pass
            await self._send_typing(chat_id)

    async def _edit_html(self, chat_id: int, message_id: int, text: str) -> None:
        """Replace a message with the markdown rendered as Telegram HTML.

        Long replies are split (Telegram caps messages at 4096 chars), the first
        chunk replaces the streaming placeholder, the rest go as new messages.
        If Telegram rejects the HTML it falls back to plain text.
        """
        chunks = split_telegram_html(markdown_to_telegram_html(text))
        for index, chunk in enumerate(chunks):
            if index == 0:
                await self._replace_message(chat_id, message_id, chunk)
            else:
                try:
                    await self.bot.send_message(chat_id=chat_id, text=chunk, parse_mode=HTML)
                except TelegramBadRequest:
                    await self.bot.send_message(chat_id=chat_id, text=html_to_plain(chunk))

    async def _replace_message(self, chat_id: int, message_id: int, chunk: str) -> None:
        """Rewrite the streaming placeholder, degrading on parse errors or flood control.

        Tries HTML first, then plain text; each attempt tolerates one RetryAfter by
        waiting, and if editing stays impossible the chunk is sent as a new message.
        """
        for parse_mode, body in ((HTML, chunk), (None, html_to_plain(chunk))):
            for attempt in range(2):
                try:
                    await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=body,
                        parse_mode=parse_mode,
                    )
                    return
                except TelegramBadRequest:
                    break
                except TelegramRetryAfter as exc:
                    if attempt == 0:
                        await asyncio.sleep(exc.retry_after)
                        continue
                    break
        await self.bot.send_message(chat_id=chat_id, text=html_to_plain(chunk))

    async def _send_approval_request(self, chat_id: int, approval: ApprovalRequestEvent) -> None:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Approve",
                        callback_data=f"{APPROVE_PREFIX}{approval.approval_id}",
                    ),
                    InlineKeyboardButton(
                        text="Deny",
                        callback_data=f"{DENY_PREFIX}{approval.approval_id}",
                    ),
                ]
            ]
        )
        payload = json.dumps(approval.args, ensure_ascii=False, indent=2)
        caption = (
            f"<b>Approval required:</b> {escape_html(approval.name)}\n"
            f"<pre>{escape_html(payload)}</pre>"
        )
        try:
            await self.bot.send_message(
                chat_id=chat_id, text=caption, reply_markup=keyboard, parse_mode=HTML
            )
        except TelegramBadRequest:
            await self.bot.send_message(chat_id=chat_id, text=caption, reply_markup=keyboard)


def _attachment(message: Message) -> tuple[Any, str] | None:
    if message.document is not None:
        name = message.document.file_name or f"document_{message.document.file_unique_id}"
        return message.document, name
    if message.photo:
        photo = message.photo[-1]
        return photo, f"photo_{photo.file_unique_id}.jpg"
    return None
