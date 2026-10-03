from __future__ import annotations

import asyncio
from array import array
from collections.abc import AsyncIterator
from typing import Any, cast

from conftest import fake_text_response
from graph_agent.config import AppConfig, RealtimeChannelConfig, VoiceChannelConfig
from graph_agent.core.service import AgentService
from graph_agent.core.sessions import SessionManager
from graph_agent.models.realtime import RealtimeBackend
from graph_agent.transports.realtime.bridge import RealtimeBridge
from graph_agent.transports.voice import AudioSocketServer
from graph_agent.transports.voice.audiosocket import (
    MESSAGE_TYPE_AUDIO,
    MESSAGE_TYPE_UUID,
    encode_frame,
    read_frame,
)


class EchoAudioBackend:
    """Realtime backend double: echoes every appended audio buffer back."""

    def __init__(self) -> None:
        self.connected = False
        self.sent: list[dict[str, Any]] = []
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def connect(self) -> None:
        self.connected = True

    async def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)
        if event.get("type") == "input_audio_buffer.append":
            await self._queue.put({"type": "response.audio.delta", "delta": event["audio"]})

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            yield await self._queue.get()

    async def close(self) -> None:
        self.connected = False


def _bridge(service: AgentService, backend: EchoAudioBackend) -> RealtimeBridge:
    config = RealtimeChannelConfig(enabled=True, backend="openai", model="fake-realtime")
    typed = cast(RealtimeBackend, backend)
    return RealtimeBridge(service, config, backend_factory=lambda _: typed)


async def _start_server(app_config: AppConfig) -> tuple[AudioSocketServer, EchoAudioBackend, Any]:
    service = AgentService(app_config)
    await service.setup(fake_text_response("ok"))
    backend = EchoAudioBackend()
    server = AudioSocketServer(
        _bridge(service, backend),
        VoiceChannelConfig(enabled=True, host="127.0.0.1", port=0, sample_rate=16000),
        SessionManager(service),
    )
    await server.start()
    return server, backend, service


async def test_server_roundtrip(app_config: AppConfig) -> None:
    server, backend, service = await _start_server(app_config)
    assert server.bound_port is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", server.bound_port)
    try:
        writer.write(encode_frame(MESSAGE_TYPE_UUID, bytes(range(16))))
        writer.write(encode_frame(MESSAGE_TYPE_AUDIO, array("h", list(range(160))).tobytes()))
        await writer.drain()

        frame = await asyncio.wait_for(read_frame(reader), timeout=5)
        assert frame is not None
        assert frame[0] == MESSAGE_TYPE_AUDIO
        echoed = array("h")
        echoed.frombytes(frame[1])
        assert len(echoed) == 160

        sent_types = [event["type"] for event in backend.sent]
        assert "session.update" in sent_types
        assert "input_audio_buffer.append" in sent_types
    finally:
        writer.close()
        await writer.wait_closed()
        await server.stop()
        await service.shutdown()


async def test_server_outbound_call_speaks_registered_greeting(app_config: AppConfig) -> None:
    server, backend, service = await _start_server(app_config)
    call_id = "00010203-0405-0607-0809-0a0b0c0d0e0f"
    server.register_outbound(call_id, "saluta il chiamante")
    assert server.bound_port is not None
    _reader, writer = await asyncio.open_connection("127.0.0.1", server.bound_port)
    try:
        writer.write(encode_frame(MESSAGE_TYPE_UUID, bytes(range(16))))
        await writer.drain()
        for _ in range(100):
            if any(event["type"] == "response.create" for event in backend.sent):
                break
            await asyncio.sleep(0.01)

        item = next(event for event in backend.sent if event["type"] == "conversation.item.create")
        assert item["item"]["content"][0]["text"] == "saluta il chiamante"
        assert any(event["type"] == "response.create" for event in backend.sent)
    finally:
        writer.close()
        await writer.wait_closed()
        await server.stop()
        await service.shutdown()


async def test_server_drops_connection_without_uuid(app_config: AppConfig) -> None:
    server, _, service = await _start_server(app_config)
    assert server.bound_port is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", server.bound_port)
    try:
        writer.write(encode_frame(MESSAGE_TYPE_AUDIO, b"\x00\x00"))
        await writer.drain()
        assert await asyncio.wait_for(reader.read(1), timeout=5) == b""
    finally:
        writer.close()
        await writer.wait_closed()
        await server.stop()
        await service.shutdown()
