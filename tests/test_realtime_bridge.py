from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from conftest import ScriptedLLMBackend, fake_text_response
from graph_agent.config import AppConfig, RealtimeChannelConfig
from graph_agent.core.service import AgentService
from graph_agent.core.sessions import SessionManager
from graph_agent.events import DoneEvent
from graph_agent.models.llm import StreamChunk, ToolCallRequest
from graph_agent.tools.base import ToolContext, ToolSpec
from graph_agent.transports.http import create_app
from graph_agent.transports.realtime.bridge import (
    DELEGATE_INSTRUCTION,
    DELEGATE_TOOL_NAME,
    RealtimeBridge,
)


class FakeConnection:
    """Scripted in-memory realtime connection. ``block`` keeps the stream open."""

    def __init__(
        self, incoming: list[dict[str, Any]] | None = None, *, block: bool = False
    ) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._incoming = incoming or []
        self._block = block

    async def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        for event in self._incoming:
            yield event
        if self._block:
            await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True


class FakeBackend(FakeConnection):
    connected = False

    async def connect(self) -> None:
        self.connected = True


def realtime_config() -> RealtimeChannelConfig:
    return RealtimeChannelConfig(enabled=True, backend="openai", model="fake-realtime")


async def test_client_to_backend_injects_delegate_tool(app_config: AppConfig) -> None:
    service = AgentService(app_config)
    await service.setup(fake_text_response("ok"))
    bridge = RealtimeBridge(service, realtime_config())
    backend = FakeConnection()
    client = FakeConnection([{"type": "session.update", "session": {"instructions": "hi"}}])

    await bridge._client_to_backend(client, backend)

    tool_names = [tool["name"] for tool in backend.sent[0]["session"]["tools"]]
    assert DELEGATE_TOOL_NAME in tool_names
    instructions = backend.sent[0]["session"]["instructions"]
    assert instructions.startswith("hi")
    assert DELEGATE_INSTRUCTION in instructions
    await service.shutdown()


async def test_client_to_backend_passthrough(app_config: AppConfig) -> None:
    service = AgentService(app_config)
    await service.setup(fake_text_response("ok"))
    bridge = RealtimeBridge(service, realtime_config())
    backend = FakeConnection()
    event = {"type": "input_audio_buffer.append", "audio": "AAAA"}
    client = FakeConnection([event])

    await bridge._client_to_backend(client, backend)

    assert backend.sent == [event]
    await service.shutdown()


async def test_backend_to_client_declares_tool_and_delegates(app_config: AppConfig) -> None:
    llm = fake_text_response("risposta dell'agente")
    service = AgentService(app_config)
    await service.setup(llm)
    bridge = RealtimeBridge(service, realtime_config())

    call = {
        "type": "response.function_call_arguments.done",
        "name": DELEGATE_TOOL_NAME,
        "call_id": "call_1",
        "arguments": json.dumps({"prompt": "ciao agente"}),
    }
    backend = FakeBackend([{"type": "session.created", "session": {}}, call])
    client = FakeConnection(block=True)

    await bridge.run(client, backend, "realtime:test")

    sent_types = [event["type"] for event in backend.sent]
    assert "session.update" in sent_types
    assert "conversation.item.create" in sent_types
    assert backend.sent[-1] == {"type": "response.create"}

    session_update = next(event for event in backend.sent if event["type"] == "session.update")
    assert DELEGATE_INSTRUCTION in session_update["session"]["instructions"]

    tool_output = next(
        event for event in backend.sent if event["type"] == "conversation.item.create"
    )
    assert tool_output["item"]["type"] == "function_call_output"
    assert tool_output["item"]["call_id"] == "call_1"
    assert tool_output["item"]["output"] == "risposta dell'agente"

    assert llm.calls, "the agent backend was never called"
    assert llm.calls[0][-1].content == "ciao agente"
    assert call not in client.sent
    await service.shutdown()


class _NeedsApprovalTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="needs_approval",
            description="tool that always requires explicit approval",
            parameters={"type": "object", "properties": {}},
            requires_approval=True,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        return "approved and executed"


async def test_delegate_uses_auto_approval(app_config: AppConfig) -> None:
    llm = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[
                        ToolCallRequest(id="tc1", name="needs_approval", args={}),
                    ]
                )
            ],
            [StreamChunk(delta_text="fatto", finish_reason="stop")],
        ]
    )
    service = AgentService(app_config)
    await service.setup(llm)
    service.registry.register(_NeedsApprovalTool())
    bridge = RealtimeBridge(service, realtime_config())

    backend = FakeBackend(
        [
            {"type": "session.created", "session": {}},
            {
                "type": "response.function_call_arguments.done",
                "name": DELEGATE_TOOL_NAME,
                "call_id": "call_1",
                "arguments": json.dumps({"prompt": "esegui il tool"}),
            },
        ]
    )
    client = FakeConnection(block=True)

    await bridge.run(client, backend, "realtime:test")

    output = next(event for event in backend.sent if event["type"] == "conversation.item.create")[
        "item"
    ]["output"]
    assert output == "fatto"
    await service.shutdown()


async def test_delegate_handles_empty_prompt(app_config: AppConfig) -> None:
    llm = fake_text_response("non dovrei essere chiamato")
    service = AgentService(app_config)
    await service.setup(llm)
    bridge = RealtimeBridge(service, realtime_config())

    backend = FakeBackend(
        [
            {
                "type": "response.function_call_arguments.done",
                "name": DELEGATE_TOOL_NAME,
                "call_id": "call_1",
                "arguments": "not-json",
            }
        ]
    )
    client = FakeConnection(block=True)

    await bridge.run(client, backend, "realtime:test")

    output = next(event for event in backend.sent if event["type"] == "conversation.item.create")[
        "item"
    ]["output"]
    assert "non-empty" in output
    assert not llm.calls
    await service.shutdown()


class _DoneOnlyService:
    def run(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            yield DoneEvent(session_id="realtime:test", final_text="solo final text")

        return _gen()

    @property
    def default_system_prompt(self) -> str | None:
        return None


async def test_run_agent_falls_back_to_done_event() -> None:
    bridge = RealtimeBridge(cast(Any, _DoneOnlyService()), realtime_config())

    assert await bridge._run_agent("realtime:test", "ciao") == "solo final text"


async def test_run_returns_when_backend_stream_ends(app_config: AppConfig) -> None:
    service = AgentService(app_config)
    await service.setup(fake_text_response("ok"))
    bridge = RealtimeBridge(service, realtime_config())
    backend = FakeBackend([])
    client = FakeConnection(block=True)

    await bridge.run(client, backend, "realtime:test")

    assert client.sent == []
    await service.shutdown()


async def test_run_returns_when_client_stream_ends(app_config: AppConfig) -> None:
    service = AgentService(app_config)
    await service.setup(fake_text_response("ok"))
    bridge = RealtimeBridge(service, realtime_config())
    backend = FakeBackend(block=True)
    client = FakeConnection([])

    await asyncio.wait_for(bridge.run(client, backend, "realtime:test"), timeout=1)

    assert backend.sent == []
    await service.shutdown()


class RecordingBridge:
    def __init__(self) -> None:
        self.sessions: list[str] = []

    async def handle(self, client: Any, session_id: str) -> None:
        self.sessions.append(session_id)
        async for _ in client.events():
            pass
        await client.close()


def _ws_app(app_config: AppConfig, bridge: RecordingBridge) -> TestClient:
    service = AgentService(app_config)
    typed = cast(RealtimeBridge, bridge)
    app = create_app(service, app_config, SessionManager(service), typed)
    return TestClient(app)


def test_realtime_ws_rejects_missing_token(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_WS_TOKEN", "secret")
    app_config.channels.http.api_key_env = "TEST_WS_TOKEN"
    client = _ws_app(app_config, RecordingBridge())

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/v1/realtime"):
        pass


def test_realtime_ws_accepts_query_token(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_WS_TOKEN", "secret")
    app_config.channels.http.api_key_env = "TEST_WS_TOKEN"
    bridge = RecordingBridge()
    client = _ws_app(app_config, bridge)

    with client.websocket_connect("/v1/realtime?api_key=secret"):
        pass

    assert len(bridge.sessions) == 1
    assert bridge.sessions[0].startswith("realtime:")
