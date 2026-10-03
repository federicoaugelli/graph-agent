from __future__ import annotations

import asyncio
import base64
from array import array
from typing import Any

from graph_agent.config import VoiceChannelConfig
from graph_agent.transports.voice.audiosocket import (
    MESSAGE_TYPE_AUDIO,
    MESSAGE_TYPE_HANGUP,
    MESSAGE_TYPE_UUID,
    AudioSocketConnection,
    build_greeting_events,
    encode_frame,
    format_call_id,
    read_frame,
)


class FakeWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closing = False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self.closing

    def close(self) -> None:
        self.closing = True

    async def wait_closed(self) -> None:
        return None


def _connection(
    incoming: bytes, config: VoiceChannelConfig | None = None
) -> tuple[AudioSocketConnection, asyncio.StreamReader, FakeWriter]:
    reader = asyncio.StreamReader()
    reader.feed_data(incoming)
    reader.feed_eof()
    writer = FakeWriter()
    connection = AudioSocketConnection(reader, writer, config or VoiceChannelConfig())
    return connection, reader, writer


def test_encode_frame_header() -> None:
    frame = encode_frame(MESSAGE_TYPE_AUDIO, b"\x01\x02")
    assert frame == bytes((MESSAGE_TYPE_AUDIO, 0, 2, 1, 2))


async def test_read_frame_roundtrip() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(encode_frame(MESSAGE_TYPE_UUID, bytes(range(16))))
    reader.feed_eof()

    assert await read_frame(reader) == (MESSAGE_TYPE_UUID, bytes(range(16)))
    assert await read_frame(reader) is None


async def test_read_frame_truncated_returns_none() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"\x10\x00\x05\x01")
    reader.feed_eof()

    assert await read_frame(reader) is None


def test_format_call_id_groups_uuid() -> None:
    assert format_call_id(bytes(range(16))) == "00010203-0405-0607-0809-0a0b0c0d0e0f"


def test_format_call_id_falls_back_on_bad_length() -> None:
    assert format_call_id(b"\x01\x02") == "0102"


async def test_connection_emits_session_update_and_resampled_audio() -> None:
    source = array("h", list(range(160))).tobytes()
    incoming = encode_frame(MESSAGE_TYPE_AUDIO, source) + encode_frame(MESSAGE_TYPE_HANGUP)
    connection, _, _ = _connection(incoming, VoiceChannelConfig(sample_rate=16000))

    events: list[dict[str, Any]] = []
    async for event in connection.events():
        events.append(event)

    assert [event["type"] for event in events] == [
        "session.update",
        "input_audio_buffer.append",
    ]
    session = events[0]["session"]
    assert session["input_audio_format"] == "pcm16"
    assert session["turn_detection"] == {"type": "server_vad"}

    audio = base64.b64decode(events[1]["audio"])
    assert len(audio) == 320 * 2


async def test_connection_greeting_yields_user_turn_then_response() -> None:
    connection, _, _ = _connection(b"", VoiceChannelConfig(greeting="saluta"))

    events = [event async for event in connection.events()]

    assert events[1:] == build_greeting_events("saluta")
    assert events[1]["item"]["content"][0]["text"] == "saluta"
    assert events[2] == {"type": "response.create", "response": {"tool_choice": "none"}}


async def test_connection_writes_resampled_backend_audio() -> None:
    connection, _, writer = _connection(b"", VoiceChannelConfig(sample_rate=16000))
    connection.start()
    backend_audio = array("h", list(range(320))).tobytes()

    await connection.send(
        {"type": "response.audio.delta", "delta": base64.b64encode(backend_audio).decode()}
    )
    for _ in range(50):
        if len(writer.data) >= 3 + 160 * 2:
            break
        await asyncio.sleep(0.01)

    assert writer.data[0] == MESSAGE_TYPE_AUDIO
    length = int.from_bytes(writer.data[1:3], "big")
    assert length == 160 * 2
    assert bytes(writer.data[3 : 3 + length]) == array("h", list(range(0, 320, 2))).tobytes()
    await connection.close()


async def test_connection_ignores_non_audio_events() -> None:
    connection, _, writer = _connection(b"", VoiceChannelConfig())

    await connection.send({"type": "response.done"})
    await connection.send({"type": "response.audio.delta", "delta": ""})

    assert writer.data == b""


async def test_connection_drops_output_when_user_interrupts() -> None:
    connection, _, _ = _connection(b"", VoiceChannelConfig(sample_rate=16000))
    delta = base64.b64encode(array("h", list(range(320))).tobytes()).decode()

    await connection.send({"type": "response.audio.delta", "delta": delta})
    assert not connection._outgoing.empty()

    await connection.send({"type": "input_audio_buffer.speech_started"})
    assert connection._send_buffer == b""
    assert connection._outgoing.empty()

    await connection.send({"type": "response.audio.delta", "delta": delta})
    assert connection._outgoing.empty()

    await connection.send({"type": "response.created"})
    await connection.send({"type": "response.audio.delta", "delta": delta})
    assert not connection._outgoing.empty()
