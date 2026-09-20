from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from graph_agent.config import RealtimeChannelConfig
from graph_agent.core.service import AgentService
from graph_agent.events import DoneEvent, ErrorEvent, TokenEvent
from graph_agent.models.realtime import (
    RealtimeBackend,
    RealtimeConnection,
    build_realtime_backend,
)

DELEGATE_TOOL_NAME = "delegate_to_brain"

DELEGATE_INSTRUCTION = (
    "You are the realtime voice interface of an autonomous agent that owns persistent "
    "memory, files, tools and multi-step reasoning. For anything that needs them "
    "(personal facts, prior context, files, research, actions) call the "
    "`delegate_to_brain` tool with a self-contained prompt. Never claim you lack memory "
    "or capabilities: delegate instead."
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


def _delegate_session_update(instructions: str) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {"tools": [DELEGATE_TOOL], "instructions": instructions},
    }


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
        with contextlib.suppress(RuntimeError):
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
        tasks = [
            asyncio.create_task(self._client_to_backend(client, backend)),
            asyncio.create_task(self._backend_to_client(client, backend, session_id)),
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
            await backend.send(event)

    async def _backend_to_client(
        self, client: RealtimeConnection, backend: RealtimeBackend, session_id: str
    ) -> None:
        async for event in backend.events():
            if event.get("type") == "session.created":
                await backend.send(_delegate_session_update(self._session_instructions()))
            elif self._is_delegate_call(event):
                await self._handle_delegate(event, backend, session_id)
                continue
            await client.send(event)

    def _session_instructions(self) -> str:
        base = self.service.default_system_prompt
        if base:
            return f"{base}\n\n{DELEGATE_INSTRUCTION}"
        return DELEGATE_INSTRUCTION

    def _is_delegate_call(self, event: dict[str, Any]) -> bool:
        return (
            event.get("type") == "response.function_call_arguments.done"
            and event.get("name") == DELEGATE_TOOL_NAME
        )

    async def _handle_delegate(
        self, event: dict[str, Any], backend: RealtimeBackend, session_id: str
    ) -> None:
        call_id = str(event.get("call_id") or "")
        try:
            args = json.loads(event.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        prompt = str(args.get("prompt") or args.get("task") or "").strip()
        if prompt:
            output = await self._run_agent(session_id, prompt)
        else:
            output = (
                "delegate_to_brain requires a non-empty 'prompt' argument "
                "describing the task for the main agent."
            )
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

    async def _run_agent(self, session_id: str, prompt: str) -> str:
        parts: list[str] = []
        final_text: str | None = None
        async for event in self.service.run(session_id, prompt, approval_mode="auto"):
            if isinstance(event, TokenEvent):
                parts.append(event.delta)
            elif isinstance(event, ErrorEvent):
                parts.append(f"[error] {event.message}")
            elif isinstance(event, DoneEvent):
                final_text = event.final_text
        text = "".join(parts)
        if not text and final_text:
            text = final_text
        return text or "(no response)"
