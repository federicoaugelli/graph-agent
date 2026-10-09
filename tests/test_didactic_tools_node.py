from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from graph_agent.config import AppConfig
from graph_agent.core.state import AgentState
from graph_agent.didactic.tools_node import run_tools_node
from graph_agent.events import FileEvent, ToolCallEvent, ToolResultEvent
from graph_agent.tools.base import ToolContext, ToolRegistry, ToolSpec


class OrderRecordingTool:
    def __init__(self) -> None:
        self.order: list[str] = []

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
        self.order.append(str(args["text"]))
        return args["text"]


class ArchiveFileTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="send_file",
            description="Send a file.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        return FileEvent(path="/ws/a.txt", caption="ecco")


def make_state(calls: list[dict[str, Any]]) -> AgentState:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": str(call["id"]),
                        "name": str(call["name"]),
                        "args": call["args"],
                        "type": "tool_call",
                    }
                    for call in calls
                ],
            )
        ],
        "session_id": "didactic",
        "iterations": 0,
    }


async def test_runs_tools_sequentially(app_config: AppConfig, tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = OrderRecordingTool()
    registry.register(tool)
    events: list[dict[str, Any]] = []

    outputs = await run_tools_node(
        make_state(
            [
                {"id": "c1", "name": "echo", "args": {"text": "first"}},
                {"id": "c2", "name": "echo", "args": {"text": "second"}},
            ]
        ),
        registry,
        workspace=tmp_path,
        config=app_config,
        emit=events.append,
    )

    assert tool.order == ["first", "second"]
    assert [message.content for message in outputs if isinstance(message, ToolMessage)] == [
        "first",
        "second",
    ]
    assert [type(story["event"]) for story in events] == [
        ToolCallEvent,
        ToolResultEvent,
        ToolCallEvent,
        ToolResultEvent,
    ]


async def test_file_event_replaces_tool_result(app_config: AppConfig, tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ArchiveFileTool())
    events: list[dict[str, Any]] = []

    outputs = await run_tools_node(
        make_state([{"id": "c1", "name": "send_file", "args": {}}]),
        registry,
        workspace=tmp_path,
        config=app_config,
        emit=events.append,
    )

    assert isinstance(outputs[0], ToolMessage)
    assert str(outputs[0].content) == "file sent: /ws/a.txt"
    assert ToolCallEvent in [type(story["event"]) for story in events]
    assert FileEvent in [type(story["event"]) for story in events]
    assert ToolResultEvent not in [type(story["event"]) for story in events]


async def test_unknown_tool_becomes_error_result(app_config: AppConfig, tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []

    outputs = await run_tools_node(
        make_state([{"id": "c1", "name": "nope", "args": {}}]),
        ToolRegistry(),
        workspace=tmp_path,
        config=app_config,
        emit=events.append,
    )

    assert str(outputs[0].content) == "unknown tool: nope"
    result_events = [
        story["event"] for story in events if isinstance(story["event"], ToolResultEvent)
    ]
    assert result_events and result_events[0].is_error is True
