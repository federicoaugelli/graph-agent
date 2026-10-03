from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from langchain_core.messages import BaseMessage, HumanMessage

from graph_agent.config import RealtimeChannelConfig
from graph_agent.core.service import AgentService
from graph_agent.events import DoneEvent, ErrorEvent, TokenEvent
from graph_agent.logging import get_logger
from graph_agent.models.realtime import (
    RealtimeBackend,
    RealtimeConnection,
    build_realtime_backend,
)
from graph_agent.transports.realtime.text import normalize_for_speech

logger = get_logger(__name__)

DELEGATE_TOOL_NAME = "delegate_to_brain"

DELEGATE_INSTRUCTION = (
    "You are the realtime voice front-end of an autonomous agent called the 'brain'. "
    "The brain owns persistent memory, files, tools and multi-step reasoning; you do "
    "not. RULE: whenever the user asks about personal facts, previous context, files, "
    "the web, or asks you to do, remember, find, send or check anything, you MUST call "
    "the `delegate_to_brain` tool with a self-contained prompt instead of answering. "
    "The tool output is the brain's reply: relay it to the user in natural speech. "
    "NEVER say that you cannot remember, access or do something: delegate instead. "
    "Chat directly only for pure small talk that needs neither memory nor tools."
)

DELEGATE_REQUEST_SUFFIX = (
    "\n\n[Reply in plain spoken language for a voice assistant: short conversational "
    "prose. No markdown, no lists, no code blocks, no emojis, no asterisks.]"
)

FILLER_INSTRUCTION = (
    "The main agent is now working on the request. Say one very short filler phrase "
    "in the user's language to acknowledge it (for example 'un attimo', 'ci sto "
    "pensando', 'sto lavorando'), then stop. Do not call any tool and do not answer "
    "the request yet."
)

DELEGATE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": DELEGATE_TOOL_NAME,
    "description": (
        "Delegate a task, question or long-running request to the main agent. "
        "Use it whenever the user needs tools, memory, files, or multi-step reasoning. "
        "Pass a self-contained prompt with all the context the main agent needs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Self-contained instruction for the main agent.",
            }
        },
        "required": ["prompt"],
    },
}


def _filler_response() -> dict[str, Any]:
    return {
        "type": "response.create",
        "response": {"instructions": FILLER_INSTRUCTION, "tool_choice": "none"},
    }


def _is_delegate_tool(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    if tool.get("name") == DELEGATE_TOOL_NAME:
        return True
    nested = tool.get("function")
    return isinstance(nested, dict) and nested.get("name") == DELEGATE_TOOL_NAME


def _ensure_delegate_tool(session: dict[str, Any]) -> None:
    tools = session.setdefault("tools", [])
    if isinstance(tools, list) and not any(_is_delegate_tool(tool) for tool in tools):
        tools.append(DELEGATE_TOOL)


def _ensure_delegate_instructions(session: dict[str, Any]) -> None:
    existing = session.get("instructions")
    if not isinstance(existing, str) or not existing:
        session["instructions"] = DELEGATE_INSTRUCTION
    elif DELEGATE_INSTRUCTION not in existing:
        session["instructions"] = f"{existing}\n\n{DELEGATE_INSTRUCTION}"


def _ensure_voice(session: dict[str, Any], voice: str | None) -> None:
    if voice:
        session["voice"] = voice


def _delegate_session_update(instructions: str, voice: str | None = None) -> dict[str, Any]:
    session: dict[str, Any] = {"tools": [DELEGATE_TOOL], "instructions": instructions}
    _ensure_voice(session, voice)
    return {"type": "session.update", "session": session}


def _log_session_update(update: dict[str, Any]) -> None:
    session = update.get("session")
    if not isinstance(session, dict):
        return
    tools = [
        tool.get("name")
        for tool in session.get("tools", [])
        if isinstance(tool, dict) and tool.get("name")
    ]
    logger.info(
        "realtime session configured",
        tools=tools,
        instructions_chars=len(session.get("instructions") or ""),
    )


@dataclass(slots=True)
class _DelegateCall:
    call_id: str
    arguments: str


def _delegate_call(event: dict[str, Any]) -> _DelegateCall | None:
    """Extract a delegate invocation across OpenAI/Qwen function-call dialects."""
    etype = event.get("type")
    if etype == "response.function_call_arguments.done":
        if event.get("name") == DELEGATE_TOOL_NAME:
            return _DelegateCall(
                str(event.get("call_id") or ""),
                str(event.get("arguments") or "{}"),
            )
    elif etype == "response.output_item.done":
        item = event.get("item")
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") == DELEGATE_TOOL_NAME
        ):
            arguments = str(item.get("arguments") or "")
            if arguments:
                return _DelegateCall(str(item.get("call_id") or ""), arguments)
    return None


def _text_from_item(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text") or part.get("input_text") or part.get("transcript")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return " ".join(parts)


@dataclass
class _SessionState:
    """Per-connection state: the realtime transcript fed to the brain on delegate."""

    transcript: list[BaseMessage] = field(default_factory=list)
    handled_calls: set[str] = field(default_factory=set)

    def record_user_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "conversation.item.input_audio_transcription.completed":
            self._record(str(event.get("transcript") or ""))
        elif etype == "conversation.item.created":
            item = event.get("item")
            if isinstance(item, dict) and item.get("role") == "user":
                self._record(_text_from_item(item))

    def _record(self, text: str) -> None:
        text = text.strip()
        if text:
            self.transcript.append(HumanMessage(content=text))

    def take_transcript(self) -> list[BaseMessage]:
        messages = self.transcript
        self.transcript = []
        return messages


class StarletteWebSocketConnection:
    """JSON-framed adapter over a FastAPI/Starlette server-side WebSocket."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    async def send(self, event: dict[str, Any]) -> None:
        await self._ws.send_json(event)

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            try:
                message = await self._ws.receive_json()
            except (WebSocketDisconnect, json.JSONDecodeError):
                return
            if isinstance(message, dict):
                yield message

    async def close(self) -> None:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await self._ws.close()


class RealtimeBridge:
    """Bidirectional bridge: client (OpenAI Realtime) <-> upstream realtime backend.

    The bridge is mounted on the agent HTTP server (same host, port and bearer
    token) and forwards events between the client and the configured backend,
    intercepting the ``delegate_to_brain`` function call to run the LangGraph agent.
    """

    def __init__(
        self,
        service: AgentService,
        config: RealtimeChannelConfig,
        backend_factory: Callable[[RealtimeChannelConfig], RealtimeBackend] | None = None,
    ) -> None:
        self.service = service
        self.config = config
        self._backend_factory = backend_factory or build_realtime_backend

    async def handle(self, client: RealtimeConnection, session_id: str) -> None:
        backend = self._backend_factory(self.config)
        await backend.connect()
        try:
            await self.run(client, backend, session_id)
        finally:
            await backend.close()

    async def run(
        self, client: RealtimeConnection, backend: RealtimeBackend, session_id: str
    ) -> None:
        state = _SessionState()
        tasks = [
            asyncio.create_task(self._client_to_backend(client, backend)),
            asyncio.create_task(self._backend_to_client(client, backend, session_id, state)),
        ]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for task in done:
                task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _client_to_backend(
        self, client: RealtimeConnection, backend: RealtimeBackend
    ) -> None:
        async for event in client.events():
            if event.get("type") == "session.update":
                session = event.get("session")
                if isinstance(session, dict):
                    _ensure_delegate_tool(session)
                    _ensure_delegate_instructions(session)
                    _ensure_voice(session, self.config.voice)
            await backend.send(event)

    async def _backend_to_client(
        self,
        client: RealtimeConnection,
        backend: RealtimeBackend,
        session_id: str,
        state: _SessionState | None = None,
    ) -> None:
        state = state or _SessionState()
        pending: set[asyncio.Task[None]] = set()
        finished = False
        try:
            async for event in backend.events():
                etype = event.get("type")
                if etype == "session.created":
                    update = _delegate_session_update(
                        self._session_instructions(), self.config.voice
                    )
                    _log_session_update(update)
                    await backend.send(update)
                elif etype == "error":
                    logger.warning("realtime backend error", error=event.get("error"))
                else:
                    call = _delegate_call(event)
                    if call is not None and call.call_id not in state.handled_calls:
                        state.handled_calls.add(call.call_id)
                        await self._handle_delegate(call, backend, session_id, state, pending)
                        continue

                state.record_user_event(event)
                await client.send(event)
            finished = True
        finally:
            if not finished:
                for task in pending:
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    def _session_instructions(self) -> str:
        base = self.config.instructions or self.service.base_system_prompt
        if base:
            return f"{base}\n\n{DELEGATE_INSTRUCTION}"
        return DELEGATE_INSTRUCTION

    async def _handle_delegate(
        self,
        call: _DelegateCall,
        backend: RealtimeBackend,
        session_id: str,
        state: _SessionState,
        pending: set[asyncio.Task[None]],
    ) -> None:
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        prompt = str(args.get("prompt") or args.get("task") or "").strip()

        if not prompt:
            await self._send_delegate_output(
                backend,
                call.call_id,
                "delegate_to_brain requires a non-empty 'prompt' argument "
                "describing the task for the main agent.",
            )
            return

        transcript = state.take_transcript()
        logger.info(
            "delegating to brain",
            call_id=call.call_id,
            transcript_messages=len(transcript),
            prompt_chars=len(prompt),
        )

        await backend.send(_filler_response())
        task = asyncio.create_task(
            self._finish_delegate(backend, call.call_id, session_id, prompt, transcript)
        )
        pending.add(task)
        task.add_done_callback(pending.discard)

    async def _finish_delegate(
        self,
        backend: RealtimeBackend,
        call_id: str,
        session_id: str,
        prompt: str,
        transcript: list[BaseMessage],
    ) -> None:
        try:
            output = await self._run_agent(session_id, prompt, transcript)
            await self._send_delegate_output(backend, call_id, output)
        except Exception:
            logger.exception("delegate_to_brain failed", call_id=call_id)

    @staticmethod
    async def _send_delegate_output(backend: RealtimeBackend, call_id: str, output: str) -> None:
        await backend.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            }
        )
        await backend.send({"type": "response.create"})

    async def _run_agent(
        self,
        session_id: str,
        prompt: str,
        transcript: list[BaseMessage] | None = None,
    ) -> str:
        messages = [*(transcript or []), HumanMessage(content=f"{prompt}{DELEGATE_REQUEST_SUFFIX}")]
        parts: list[str] = []
        final_text: str | None = None
        async for event in self.service.run(session_id, messages, approval_mode="auto"):
            if isinstance(event, TokenEvent):
                parts.append(event.delta)
            elif isinstance(event, ErrorEvent):
                parts.append(f"[error] {event.message}")
            elif isinstance(event, DoneEvent):
                final_text = event.final_text
        text = "".join(parts)
        if not text and final_text:
            text = final_text
        return normalize_for_speech(text) or "(no response)"
