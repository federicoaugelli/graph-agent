from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from conftest import ScriptedLLMBackend
from graph_agent.config import AppConfig
from graph_agent.core.graph import build_agent_graph
from graph_agent.core.service import AgentService
from graph_agent.events import (
    ApprovalRequestEvent,
    DoneEvent,
    ErrorEvent,
    FileEvent,
    MetricsEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from graph_agent.models.llm import StreamChunk, ToolCallRequest
from graph_agent.tools.base import ToolContext, ToolRegistry, ToolSpec

_EVENT_TYPES = (
    ApprovalRequestEvent,
    DoneEvent,
    ErrorEvent,
    FileEvent,
    MetricsEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class ConcurrentProbeTool:
    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="probe",
            description="Probe concurrency.",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        return args["n"]


class RecordingEchoTool:
    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echo input.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        self.executed.append(args)
        return args["text"]


async def collect_events(graph: Any, state: Any) -> list[Any]:
    events: list[Any] = []
    async for chunk in graph.astream(state, stream_mode="custom"):
        if isinstance(chunk, dict):
            event = chunk.get("event")
            if isinstance(event, _EVENT_TYPES):
                events.append(event)
    return events


async def test_tool_calls_run_in_parallel(app_config: AppConfig) -> None:
    registry = ToolRegistry()
    probe = ConcurrentProbeTool()
    registry.register(probe)

    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[
                        ToolCallRequest(id="a", name="probe", args={"n": 1}),
                        ToolCallRequest(id="b", name="probe", args={"n": 2}),
                        ToolCallRequest(id="c", name="probe", args={"n": 3}),
                    ],
                    finish_reason="tool_calls",
                )
            ],
            [StreamChunk(delta_text="done", finish_reason="stop")],
        ]
    )

    graph = build_agent_graph(backend, registry, app_config)
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="probe")],
        "session_id": "parallel",
        "iterations": 0,
    }

    await collect_events(graph, state)

    assert probe.max_active == 3


async def test_repeated_tool_calls_are_short_circuited(app_config: AppConfig) -> None:
    app_config.agent.repeat_tool_call_limit = 2
    tool = RecordingEchoTool()
    responses = [
        [
            StreamChunk(
                tool_calls=[ToolCallRequest(id=f"c{i}", name="echo", args={"text": "same"})],
                finish_reason="tool_calls",
            )
        ]
        for i in range(4)
    ]
    responses.append([StreamChunk(delta_text="done", finish_reason="stop")])
    backend = ScriptedLLMBackend(responses)

    service = AgentService(app_config)
    await service.setup(backend)
    service.registry.register(tool)

    events = [event async for event in service.run("loop", "go")]

    assert tool.executed == [{"text": "same"}, {"text": "same"}]
    repeated = [event for event in events if isinstance(event, ToolResultEvent) and event.is_error]
    assert any("repeated" in str(event.result) for event in repeated)

    await service.shutdown()


async def test_tool_output_is_truncated(app_config: AppConfig) -> None:
    app_config.agent.max_tool_output_chars = 20
    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[ToolCallRequest(id="c1", name="echo", args={"text": "x" * 100})],
                    finish_reason="tool_calls",
                )
            ],
            [StreamChunk(delta_text="done", finish_reason="stop")],
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    service.registry.register(RecordingEchoTool())

    _ = [event async for event in service.run("trim", "go")]

    tool_messages = [m for m in backend.calls[-1] if isinstance(m, ToolMessage)]
    assert tool_messages
    assert "truncated" in str(tool_messages[0].content)

    await service.shutdown()
