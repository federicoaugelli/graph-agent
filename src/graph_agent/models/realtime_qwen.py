from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote

import websockets
from websockets.asyncio.client import ClientConnection

from graph_agent.config import RealtimeChannelConfig
from graph_agent.models.realtime import WebSocketConnection, to_ws_scheme


def build_qwen_url(config: RealtimeChannelConfig) -> str:
    base = to_ws_scheme(config.api_base.rstrip("/"))
    return f"{base}?model={quote(config.model)}" if config.model else base


def qwen_headers(config: RealtimeChannelConfig) -> dict[str, str]:
    key = ""
    if config.api_key_env is not None:
        key = os.environ.get(config.api_key_env, "")
    return {"Authorization": f"Bearer {key or 'unused'}"}


def _to_qwen_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if isinstance(tool.get("function"), dict):
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        },
    }


def _to_qwen_session(session: dict[str, Any]) -> dict[str, Any]:
    out = dict(session)
    out["input_audio_format"] = "pcm"
    out["output_audio_format"] = "pcm"
    tools = out.get("tools")
    if isinstance(tools, list):
        out["tools"] = [_to_qwen_tool(tool) for tool in tools if isinstance(tool, dict)]
        if out["tools"]:
            out["enable_search"] = False
    return out


def to_qwen(event: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI Realtime event into the Qwen-Omni-Realtime dialect.

    Most client events share the OpenAI shape, so only ``session.update`` needs
    adaptation: Qwen uses flat PCM formats and nested function tool definitions.
    """
    if event.get("type") == "session.update":
        session = event.get("session")
        return {
            "type": "session.update",
            "session": _to_qwen_session(session if isinstance(session, dict) else {}),
        }
    return event


def from_qwen(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate a Qwen server event into OpenAI Realtime events.

    Qwen server events already follow the OpenAI Realtime beta schema, so this is
    a pass-through kept as a seam for dialect drift.
    """
    return [message]


class QwenRealtimeBackend:
    """Backend for the Qwen-Omni-Realtime WebSocket API (Alibaba Model Studio)."""

    def __init__(self, config: RealtimeChannelConfig) -> None:
        self.config = config
        self._conn: WebSocketConnection | None = None

    async def connect(self) -> None:
        ws: ClientConnection = await websockets.connect(
            build_qwen_url(self.config),
            additional_headers=qwen_headers(self.config),
        )
        self._conn = WebSocketConnection(ws)

    async def send(self, event: dict[str, Any]) -> None:
        await self._require().send(to_qwen(event))

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        async for message in self._require().events():
            for event in from_qwen(message):
                yield event

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require(self) -> WebSocketConnection:
        if self._conn is None:
            raise RuntimeError("realtime backend is not connected")
        return self._conn
