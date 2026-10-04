from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from graph_agent.config import RealtimeChannelConfig
from graph_agent.core.service import AgentService
from graph_agent.logging import get_logger
from graph_agent.models.realtime import (
    RealtimeBackend,
    RealtimeCapabilities,
    RealtimeConnection,
    build_realtime_backend,
)
from graph_agent.transports.realtime.delegate import (
    DELEGATE_INSTRUCTION,
    DelegationSupervisor,
    _delegate_call,
    _ensure_delegate_instructions,
    _ensure_delegate_tool,
    _ensure_voice,
    _log_session_update,
    _normalize_turn_detection,
)

logger = get_logger(__name__)


def _capabilities(backend: RealtimeBackend) -> RealtimeCapabilities:
    return getattr(backend, "capabilities", RealtimeCapabilities())


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
        supervisor = DelegationSupervisor(
            self.service,
            backend,
            session_id,
            capabilities=_capabilities(backend),
            instructions=self._session_instructions(),
            voice=self.config.voice,
        )
        tasks = [
            asyncio.create_task(self._client_to_backend(client, backend, supervisor)),
            asyncio.create_task(self._backend_to_client(client, backend, supervisor)),
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
            await supervisor.wait()
            await supervisor.aclose()

    async def _client_to_backend(
        self,
        client: RealtimeConnection,
        backend: RealtimeBackend,
        supervisor: DelegationSupervisor,
    ) -> None:
        async for event in client.events():
            etype = event.get("type")
            if etype == "session.update":
                session = event.get("session")
                if isinstance(session, dict):
                    _ensure_delegate_tool(session)
                    _ensure_delegate_instructions(session)
                    _ensure_voice(session, self.config.voice)
                    _normalize_turn_detection(session, supervisor.capabilities)
            elif etype == "input_audio_buffer.speech_started":
                await supervisor.on_speech_started()
            await backend.send(event)

    async def _backend_to_client(
        self,
        client: RealtimeConnection,
        backend: RealtimeBackend,
        supervisor: DelegationSupervisor,
    ) -> None:
        async for event in backend.events():
            etype = event.get("type")
            if etype == "session.created":
                update = supervisor.session_update()
                _log_session_update(update)
                await backend.send(update)
            elif etype == "error":
                logger.warning("realtime backend error", error=event.get("error"))
            else:
                call = _delegate_call(event)
                if call is not None:
                    await supervisor.handle_call(call)
                    continue

            supervisor.record_event(event)
            await client.send(event)

    def _session_instructions(self) -> str:
        base = self.config.instructions or self.service.base_system_prompt
        if base:
            return f"{base}\n\n{DELEGATE_INSTRUCTION}"
        return DELEGATE_INSTRUCTION
