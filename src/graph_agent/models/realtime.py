from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.server import ServerConnection

from graph_agent.config import RealtimeChannelConfig

WebSocketLike = ClientConnection | ServerConnection


class RealtimeConnection(Protocol):
    """JSON event stream shared by the client transport and the upstream backend."""

    async def send(self, event: dict[str, Any]) -> None: ...

    def events(self) -> AsyncIterator[dict[str, Any]]: ...

    async def close(self) -> None: ...


class RealtimeBackend(RealtimeConnection, Protocol):
    """Upstream realtime provider. Speaks OpenAI Realtime events at its boundary."""

    @property
    def capabilities(self) -> RealtimeCapabilities: ...

    async def connect(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RealtimeCapabilities:
    """Optional realtime features a backend may not implement.

    The bridge degrades gracefully when a backend lacks a feature, so the same
    agent-agnostic orchestration works across OpenAI, Qwen and local models.
    """

    supports_cancel: bool = True
    supports_progress: bool = True
    supports_semantic_vad: bool = True


class WebSocketConnection:
    """JSON-framed adapter over a websockets connection (client or server side)."""

    def __init__(self, ws: WebSocketLike) -> None:
        self._ws = ws

    async def send(self, event: dict[str, Any]) -> None:
        await self._ws.send(json.dumps(event))

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        async for raw in self._ws:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data

    async def close(self) -> None:
        await self._ws.close()


def to_ws_scheme(url: str) -> str:
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


def _api_key(config: RealtimeChannelConfig) -> str:
    if config.api_key_env is None:
        return "unused"
    return os.environ.get(config.api_key_env, "") or "unused"


def build_openai_url(config: RealtimeChannelConfig) -> str:
    base = to_ws_scheme(config.api_base.rstrip("/"))
    if not base.endswith("/v1/realtime"):
        base = f"{base}/v1/realtime"
    return f"{base}?model={quote(config.model)}" if config.model else base


def auth_headers(config: RealtimeChannelConfig, *, beta: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {_api_key(config)}"}
    if beta:
        headers["OpenAI-Beta"] = "realtime=v1"
    return headers


class ConnectionBackend:
    """Shared lifecycle for WebSocket-backed realtime providers."""

    def __init__(self, config: RealtimeChannelConfig) -> None:
        self.config = config
        self._conn: WebSocketConnection | None = None

    @property
    def capabilities(self) -> RealtimeCapabilities:
        return RealtimeCapabilities()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require(self) -> WebSocketConnection:
        if self._conn is None:
            raise RuntimeError("realtime backend is not connected")
        return self._conn


class OpenAIRealtimeBackend(ConnectionBackend):
    """Backend for any OpenAI-compatible realtime endpoint (LiteLLM proxy, OpenAI, ...)."""

    async def connect(self) -> None:
        ws = await websockets.connect(
            build_openai_url(self.config),
            additional_headers=auth_headers(self.config, beta=True),
        )
        self._conn = WebSocketConnection(ws)

    async def send(self, event: dict[str, Any]) -> None:
        await self._require().send(event)

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        async for event in self._require().events():
            yield event


def build_realtime_backend(config: RealtimeChannelConfig) -> RealtimeBackend:
    """Select the backend implementation from configuration."""
    if config.backend == "qwen":
        from graph_agent.models.realtime_qwen import QwenRealtimeBackend

        return QwenRealtimeBackend(config)
    return OpenAIRealtimeBackend(config)
