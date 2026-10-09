from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from conftest import ScriptedLLMBackend
from graph_agent.config import AppConfig
from graph_agent.core.service import AgentService
from graph_agent.events import CancelledEvent, DoneEvent
from graph_agent.models.llm import StreamChunk, ToolCallRequest
from graph_agent.tools.base import ToolContext, ToolSpec


def _last_text(messages: list[BaseMessage]) -> str:
    content = messages[-1].content
    return content if isinstance(content, str) else ""


class GatedBackend:
    """Backend whose slow replies block until released, so a test can barge in."""

    def __init__(self) -> None:
        self.calls: list[list[BaseMessage]] = []
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def _reply(self, messages: list[BaseMessage]) -> str:
        text = _last_text(messages)
        if text.startswith("slow"):
            self.entered.set()
            await self.gate.wait()
            return "slow-done"
        return f"fast:{text}"

    async def astream(
        self, messages: list[BaseMessage], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        yield StreamChunk(delta_text=await self._reply(messages), finish_reason="stop")

    async def acomplete(
        self, messages: list[BaseMessage], tools: list[dict[str, Any]] | None = None
    ) -> StreamChunk:
        self.calls.append(messages)
        return StreamChunk(delta_text=await self._reply(messages), finish_reason="stop")


class BlockingTool:
    """Tool that blocks mid-execution so a test can cancel the run inside tools_node."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="block",
            description="Block until released.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        self.entered.set()
        await self.release.wait()
        return "done"


def _pending_tool_call_ids(messages: list[BaseMessage]) -> list[str]:
    pending: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            pending = [str(call["id"]) for call in message.tool_calls]
        elif isinstance(message, ToolMessage):
            if str(message.tool_call_id) in pending:
                pending.remove(str(message.tool_call_id))
        elif pending:
            return pending
    return pending


async def test_new_run_cancels_the_in_flight_one(app_config: AppConfig) -> None:
    backend = GatedBackend()
    service = AgentService(app_config)
    await service.setup(backend)

    first_events: list[Any] = []

    async def consume_first() -> None:
        async for event in service.run("s1", "slow task"):
            first_events.append(event)

    first = asyncio.create_task(consume_first())
    await backend.entered.wait()

    second_events = [event async for event in service.run("s1", "fast follow-up")]

    assert isinstance(second_events[-1], DoneEvent)
    assert second_events[-1].final_text == "fast:fast follow-up"

    await asyncio.wait_for(first, timeout=1.0)
    assert not any(isinstance(event, DoneEvent) for event in first_events)
    assert any(isinstance(event, CancelledEvent) for event in first_events)

    await service.shutdown()


async def test_runs_in_other_sessions_are_not_cancelled(app_config: AppConfig) -> None:
    backend = GatedBackend()
    service = AgentService(app_config)
    await service.setup(backend)

    first_events: list[Any] = []

    async def consume_first() -> None:
        async for event in service.run("session-a", "slow task"):
            first_events.append(event)

    first = asyncio.create_task(consume_first())
    await backend.entered.wait()

    second_events = [event async for event in service.run("session-b", "fast task")]
    assert isinstance(second_events[-1], DoneEvent)
    assert second_events[-1].final_text == "fast:fast task"
    assert not first.done(), "a run in another session must not be cancelled"

    backend.gate.set()
    await asyncio.wait_for(first, timeout=1.0)
    assert isinstance(first_events[-1], DoneEvent)
    assert first_events[-1].final_text == "slow-done"

    await service.shutdown()


async def test_barge_in_mid_tool_leaves_a_repairable_history(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[ToolCallRequest(id="call_1", name="block", args={})],
                    finish_reason="tool_calls",
                )
            ],
            [StreamChunk(delta_text="after barge-in", finish_reason="stop")],
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    tool = BlockingTool()
    service.registry.register(tool)

    first_events: list[Any] = []

    async def consume_first() -> None:
        async for event in service.run("s1", "first"):
            first_events.append(event)

    first = asyncio.create_task(consume_first())
    await asyncio.wait_for(tool.entered.wait(), timeout=1.0)

    second_events = [event async for event in service.run("s1", "second")]
    await asyncio.wait_for(first, timeout=1.0)

    assert isinstance(second_events[-1], DoneEvent)
    assert second_events[-1].final_text == "after barge-in"
    assert _pending_tool_call_ids(backend.calls[-1]) == []

    await service.shutdown()


async def test_shutdown_cancels_active_runs(app_config: AppConfig) -> None:
    backend = GatedBackend()
    service = AgentService(app_config)
    await service.setup(backend)

    first_events: list[Any] = []

    async def consume_first() -> None:
        async for event in service.run("s1", "slow task"):
            first_events.append(event)

    first = asyncio.create_task(consume_first())
    await backend.entered.wait()

    await service.shutdown()
    await asyncio.wait_for(first, timeout=1.0)
    assert any(isinstance(event, CancelledEvent) for event in first_events)
