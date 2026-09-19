from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
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

APPROVE_PREFIX = "approve:"
DENY_PREFIX = "deny:"

AUTO_COMMAND = "/auto"
MANUAL_COMMAND = "/manual"
NEW_COMMAND = "/new"

CHANNEL = "telegram"
INCOMING_DIR = "incoming"

MODE_REPLIES: dict[ApprovalMode, str] = {
    "auto": "Auto mode on: sensitive tools now run directly, without asking.",
    "manual": "Manual mode on: sensitive tools will ask for your approval.",
}


class TelegramBot:
    """Telegram transport (aiogram 3, long polling).

    Contract (Phase 3 tests pin this down):
    - session identifier = str(chat_id), thread id via SessionManager
    - handle_message: allowed_user_ids gate (empty = deny all), /new resets the thread, /auto and
      /manual toggle the approval mode; anything else runs the agent and streams
      the reply via a send_message placeholder then edit_message_text
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

    async def start(self) -> None:
        """Register handlers on self.dp and start long polling."""
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
        note = f"[user sent a file, saved in the workspace at {relative}]"
        prompt = f"{caption}\n{note}".strip() if caption else note

        thread_id = await self.sessions.thread_id(CHANNEL, identifier)
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
        placeholder = await self.bot.send_message(chat_id=chat_id, text="...")
        text_parts: list[str] = []
        approval: ApprovalRequestEvent | None = None

        async for event in stream:
            if isinstance(event, TokenEvent):
                text_parts.append(event.delta)
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=placeholder.message_id,
                    text="".join(text_parts),
                )
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

        final_text = "".join(text_parts).strip()
        if final_text:
            await self.bot.edit_message_text(
                chat_id=chat_id, message_id=placeholder.message_id, text=final_text
            )

        if approval is not None:
            await self._send_approval_request(chat_id, approval)

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
        caption = (
            f"Approval required: {approval.name}\n"
            f"{json.dumps(approval.args, ensure_ascii=False, indent=2)}"
        )
        await self.bot.send_message(chat_id=chat_id, text=caption, reply_markup=keyboard)


def _attachment(message: Message) -> tuple[Any, str] | None:
    if message.document is not None:
        name = message.document.file_name or f"document_{message.document.file_unique_id}"
        return message.document, name
    if message.photo:
        photo = message.photo[-1]
        return photo, f"photo_{photo.file_unique_id}.jpg"
    return None