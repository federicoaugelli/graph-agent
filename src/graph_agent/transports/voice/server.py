from __future__ import annotations

import asyncio
from typing import cast

from graph_agent.config import VoiceChannelConfig
from graph_agent.core.sessions import SessionManager
from graph_agent.logging import get_logger
from graph_agent.transports.realtime.bridge import RealtimeBridge
from graph_agent.transports.voice.audiosocket import (
    MESSAGE_TYPE_UUID,
    AudioSocketConnection,
    format_call_id,
    read_frame,
)

logger = get_logger(__name__)


class AudioSocketServer:
    """TCP server accepting Asterisk AudioSocket connections.

    Asterisk's dialplan ``AudioSocket(uuid,host:port)`` opens one TCP connection
    per call, starting with the UUID frame. Each call is bridged to the shared
    realtime backend session, reusing ``delegate_to_brain``.

    Outbound calls are pre-registered by :class:`OutboundCaller` so the brain can
    speak first through a per-call greeting.
    """

    def __init__(
        self,
        bridge: RealtimeBridge,
        config: VoiceChannelConfig,
        sessions: SessionManager,
    ) -> None:
        self._bridge = bridge
        self._config = config
        self._sessions = sessions
        self._server: asyncio.AbstractServer | None = None
        self._pending_greetings: dict[str, str | None] = {}

    def register_outbound(self, call_id: str, greeting: str | None) -> None:
        self._pending_greetings[call_id] = greeting

    def cancel_outbound(self, call_id: str) -> None:
        self._pending_greetings.pop(call_id, None)

    @property
    def bound_port(self) -> int | None:
        if self._server is None:
            return None
        sockets = cast("asyncio.Server", self._server).sockets
        if not sockets:
            return None
        return int(sockets[0].getsockname()[1])

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, self._config.host, self._config.port
        )
        logger.info(
            "audiosocket listening",
            host=self._config.host,
            port=self.bound_port,
        )

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        server = self._server
        if server is None:
            return
        async with server:
            await server.serve_forever()

    async def stop(self) -> None:
        server = self._server
        if server is not None:
            server.close()
            await server.wait_closed()
            self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        frame = await read_frame(reader)
        if frame is None or frame[0] != MESSAGE_TYPE_UUID:
            writer.close()
            return

        call_id = format_call_id(frame[1])
        outbound = call_id in self._pending_greetings
        greeting = self._pending_greetings.pop(call_id, self._config.greeting)
        connection = AudioSocketConnection(reader, writer, self._config, greeting=greeting)
        connection.start()
        logger.info("audiosocket call started", call_id=call_id, outbound=outbound)
        try:
            await self._bridge.handle(connection, await self._sessions.thread_id())
        except Exception:
            logger.exception("audiosocket call failed", call_id=call_id)
        finally:
            await connection.close()
            logger.info("audiosocket call ended", call_id=call_id)
