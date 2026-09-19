from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from conftest import ScriptedLLMBackend
from graph_agent.config import AppConfig
from graph_agent.core.graph import build_agent_graph
from graph_agent.core.state import AgentState
from graph_agent.events import (
    ApprovalRequestEvent,
    DoneEvent,
    ErrorEvent,
    Event,
    FileEvent,
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
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class EchoTool:
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
        return args["text"]


class SendFileTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="send_file",
            description="Send a file.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        return FileEvent(path="/ws/a.txt", caption="ecco")


def make_state(input_text: str) -> AgentState:
    return {
        "messages": [HumanMessage(content=input_text)],
        "session_id": "test",
        "iterations": 0,
        "pending_approval": None,
    }


async def collect_custom_events(graph: Any, state: AgentState) -> list[Event]:
    events: list[Event] = []
    async for chunk in graph.astream(state, stream_mode="custom"):
        if isinstance(chunk, dict):
            event = chunk.get("event")
            if isinstance(event, _EVENT_TYPES):
                events.append(event)
    return events


async def test_graph_emits_text_event(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend([[StreamChunk(delta_text="hello", finish_reason="stop")]])
    graph = build_agent_graph(backend, ToolRegistry(), app_config)

    events = await collect_custom_events(graph, make_state("hi"))

    assert len(events) == 1
    assert isinstance(events[0], TokenEvent)
    assert events[0].delta == "hello"


async def test_graph_runs_tool_loop(app_config: AppConfig) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[ToolCallRequest(id="call_1", name="echo", args={"text": "hi"})],
                    finish_reason="tool_calls",
                )
            ],
            [StreamChunk(delta_text="done", finish_reason="stop")],
        ]
    )

    graph = build_agent_graph(backend, registry, app_config)
    events = await collect_custom_events(graph, make_state("echo hi"))

    assert [type(event) for event in events] == [ToolCallEvent, ToolResultEvent, TokenEvent]
    assert isinstance(events[1], ToolResultEvent)
    assert events[1].result == "hi"
    assert events[1].is_error is False
    assert len(backend.calls) == 2


async def test_graph_emits_file_event_instead_of_tool_result(app_config: AppConfig) -> None:
    registry = ToolRegistry()
    registry.register(SendFileTool())

    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[ToolCallRequest(id="call_1", name="send_file", args={})],
                    finish_reason="tool_calls",
                )
            ],
            [StreamChunk(delta_text="done", finish_reason="stop")],
        ]
    )

    graph = build_agent_graph(backend, registry, app_config)
    events = await collect_custom_events(graph, make_state("manda"))

    assert [type(event) for event in events] == [ToolCallEvent, FileEvent, TokenEvent]
    assert isinstance(events[1], FileEvent)
    assert events[1].path == "/ws/a.txt"
    assert events[1].caption == "ecco"


async def test_graph_respects_max_iterations(app_config: AppConfig) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[
                        ToolCallRequest(id=f"call_{index}", name="echo", args={"text": "x"})
                    ],
                    finish_reason="tool_calls",
                )
            ]
            for index in range(10)
        ]
    )

    graph = build_agent_graph(backend, registry, app_config, max_iterations=2)
    await collect_custom_events(graph, make_state("loop"))

    assert len(backend.calls) == 2
