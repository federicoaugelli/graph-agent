from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from graph_agent.config import VoiceChannelConfig
from graph_agent.logging import get_logger
from graph_agent.transports.voice.audio import resample_pcm16

logger = get_logger(__name__)

MESSAGE_TYPE_HANGUP = 0x00
MESSAGE_TYPE_UUID = 0x01
MESSAGE_TYPE_DTMF = 0x03
MESSAGE_TYPE_AUDIO = 0x10
MESSAGE_TYPE_ERROR = 0xFF

HEADER_SIZE = 3
TELEPHONY_RATE = 8000
UUID_BYTES = 16
FRAME_SAMPLES = 160
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_INTERVAL_SECONDS = 0.02


def encode_frame(message_type: int, payload: bytes = b"") -> bytes:
    """Encode one AudioSocket message: ``type(1B) + length(2B BE) + payload``."""
    return bytes((message_type,)) + len(payload).to_bytes(2, "big") + payload


async def read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes] | None:
    """Read one AudioSocket message, or ``None`` when the stream ends early."""
    try:
        header = await reader.readexactly(HEADER_SIZE)
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    message_type = header[0]
    length = int.from_bytes(header[1:HEADER_SIZE], "big")
    if length == 0:
        return message_type, b""
    try:
        payload = await reader.readexactly(length)
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    return message_type, payload


def format_call_id(payload: bytes) -> str:
    """Format the 16-byte AudioSocket UUID as the canonical ``8-4-4-4-12`` string."""
    raw = payload.hex()
    if len(raw) != UUID_BYTES * 2:
        return raw
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"


def build_session_update(config: VoiceChannelConfig) -> dict[str, Any]:
    """The session configuration the phone leg sends to the realtime backend."""
    return {
        "type": "session.update",
        "session": {
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad"},
        },
    }


def build_greeting_events(greeting: str) -> list[dict[str, Any]]:
    """Make the brain speak first: a synthetic user turn plus a response request.

    Qwen-Omni-Realtime rejects a bare ``response.create`` ("conversation has no
    messages"), so the greeting is carried by a user message; ``tool_choice: none``
    keeps the opening turn from delegating.
    """
    return [
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": greeting}],
            },
        },
        {"type": "response.create", "response": {"tool_choice": "none"}},
    ]


def _audio_delta(event: dict[str, Any]) -> str | None:
    if event.get("type") not in ("response.audio.delta", "response.output_audio.delta"):
        return None
    delta = event.get("delta")
    return delta if isinstance(delta, str) and delta else None


class AudioSocketConnection:
    """Asterisk AudioSocket <-> OpenAI Realtime ``RealtimeConnection`` adapter.

    The phone leg is the "client": inbound slin16 audio becomes
    ``input_audio_buffer.append`` events and backend audio deltas are written back
    as 8 kHz slin16 frames.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        config: VoiceChannelConfig,
        greeting: str | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._config = config
        self._output_rate = config.output_sample_rate or config.sample_rate
        self._closed = False
        self._greeting = greeting if greeting is not None else config.greeting
        self._event_types: dict[str, int] = {}
        self._audio_bytes = 0
        self._send_buffer = bytearray()
        self._outgoing: asyncio.Queue[bytes] = asyncio.Queue()
        self._pump: asyncio.Task[None] | None = None
        self._interrupted = False

    def start(self) -> None:
        """Start the real-time audio writer (20 ms frames, paced)."""
        if self._pump is None:
            self._pump = asyncio.create_task(self._pump_audio())

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        yield build_session_update(self._config)
        if self._greeting:
            for event in build_greeting_events(self._greeting):
                yield event

        while not self._closed:
            frame = await read_frame(self._reader)
            if frame is None:
                return
            message_type, payload = frame
            if message_type == MESSAGE_TYPE_AUDIO:
                audio = resample_pcm16(payload, TELEPHONY_RATE, self._config.sample_rate)
                if audio:
                    yield {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio).decode("ascii"),
                    }
            elif message_type == MESSAGE_TYPE_HANGUP:
                logger.info("audiosocket hangup frame")
                return
            elif message_type == MESSAGE_TYPE_DTMF:
                logger.info("audiosocket dtmf", key=payload.decode("ascii", errors="replace"))
            elif message_type == MESSAGE_TYPE_ERROR:
                logger.warning(
                    "audiosocket error frame", detail=payload.decode("utf-8", errors="replace")
                )

    async def send(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type"))
        self._event_types[event_type] = self._event_types.get(event_type, 0) + 1

        if event_type == "input_audio_buffer.speech_started":
            self._interrupted = True
            self._clear_pending_audio()
            return
        if event_type == "response.created":
            self._interrupted = False

        delta = _audio_delta(event)
        if delta is not None:
            if self._interrupted:
                return
            audio = resample_pcm16(
                base64.b64decode(delta), self._output_rate, TELEPHONY_RATE
            )
            self._audio_bytes += len(audio)
            self._queue_audio(audio)
        elif event_type == "error":
            logger.warning("realtime error", error=event.get("error"))

    def _clear_pending_audio(self) -> None:
        """Drop buffered playback so barge-in stops the model immediately."""
        self._send_buffer.clear()
        while not self._outgoing.empty():
            self._outgoing.get_nowait()

    def _queue_audio(self, audio: bytes) -> None:
        self._send_buffer.extend(audio)
        while len(self._send_buffer) >= FRAME_BYTES:
            frame = bytes(self._send_buffer[:FRAME_BYTES])
            del self._send_buffer[:FRAME_BYTES]
            self._outgoing.put_nowait(frame)

    async def _pump_audio(self) -> None:
        while True:
            frame = await self._outgoing.get()
            await self._write(MESSAGE_TYPE_AUDIO, frame)
            await asyncio.sleep(FRAME_INTERVAL_SECONDS)

    async def close(self) -> None:
        if self._event_types:
            logger.info(
                "audiosocket backend events",
                events=self._event_types,
                audio_bytes=self._audio_bytes,
            )
        self._closed = True
        pump = self._pump
        self._pump = None
        if pump is not None:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
        with contextlib.suppress(ConnectionError, RuntimeError):
            self._writer.close()
            await self._writer.wait_closed()

    async def _write(self, message_type: int, payload: bytes) -> None:
        if self._closed or self._writer.is_closing():
            return
        try:
            self._writer.write(encode_frame(message_type, payload))
            await self._writer.drain()
        except (ConnectionError, RuntimeError):
            self._closed = True
