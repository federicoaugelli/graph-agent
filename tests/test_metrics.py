from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import BaseMessage

from conftest import ScriptedLLMBackend
from graph_agent.config import AppConfig
from graph_agent.core.service import AgentService
from graph_agent.events import DoneEvent, Event, MetricsEvent, ToolResultEvent
from graph_agent.models.llm import StreamChunk, ToolCallRequest
from graph_agent.tools.base import ToolContext, ToolSpec


class SlowEchoTool:
    def __init__(self, delay: float = 0.03) -> None:
        self.delay = delay

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
        await asyncio.sleep(self.delay)
        return args["text"]


class SlowBackend:
    def __init__(self, responses: list[list[StreamChunk]], delay: float = 0.03) -> None:
        self.responses = responses
        self.delay = delay
        self.calls: list[list[BaseMessage]] = []

    async def acomplete(
        self, messages: list[BaseMessage], tools: list[dict[str, Any]] | None = None
    ) -> StreamChunk:
        self.calls.append(messages)
        await asyncio.sleep(self.delay)
        merged = StreamChunk(finish_reason="stop")
        for chunk in self.responses.pop(0):
            merged.delta_text += chunk.delta_text
            merged.tool_calls.extend(chunk.tool_calls)
        return merged

    async def astream(
        self, messages: list[BaseMessage], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        await asyncio.sleep(self.delay)
        for chunk in self.responses.pop(0):
            yield chunk


def tool_call_chunks(call_id: str, text: str) -> list[StreamChunk]:
    return [
        StreamChunk(
            tool_calls=[ToolCallRequest(id=call_id, name="echo", args={"text": text})],
            finish_reason="tool_calls",
        )
    ]


def text_chunks(text: str) -> list[StreamChunk]:
    return [StreamChunk(delta_text=text, finish_reason="stop")]


def metrics_of(events: list[Event]) -> MetricsEvent:
    metrics = [event for event in events if isinstance(event, MetricsEvent)]
    assert len(metrics) == 1
    return metrics[0]


async def test_metrics_reports_counts_and_durations(app_config: AppConfig) -> None:
    app_config.agent.latency_metrics = True
    backend = SlowBackend([tool_call_chunks("call_1", "hi"), text_chunks("done")])
    service = AgentService(app_config)
    await service.setup(backend)
    service.registry.register(SlowEchoTool())

    events = [event async for event in service.run("m1", "echo hi")]

    metrics = metrics_of(events)
    assert metrics.iterations == 2
    assert metrics.llm_calls == 2
    assert metrics.tool_calls == 1
    assert any(isinstance(event, ToolResultEvent) for event in events)

    await service.shutdown()


async def test_metrics_reports_llm_and_tool_time(app_config: AppConfig) -> None:
    app_config.agent.latency_metrics = True
    backend = SlowBackend([tool_call_chunks("call_1", "hi"), text_chunks("done")])
    service = AgentService(app_config)
    await service.setup(backend)
    service.registry.register(SlowEchoTool())

    events = [event async for event in service.run("m2", "echo hi")]

    metrics = metrics_of(events)
    assert metrics.llm_duration_ms >= 20
    assert metrics.tool_duration_ms >= 20
    assert metrics.total_duration_ms >= metrics.llm_duration_ms
    assert metrics.total_duration_ms >= metrics.tool_duration_ms

    await service.shutdown()


async def test_metrics_emitted_before_done(app_config: AppConfig) -> None:
    app_config.agent.latency_metrics = True
    backend = ScriptedLLMBackend([text_chunks("hello")])
    service = AgentService(app_config)
    await service.setup(backend)

    events = [event async for event in service.run("m3", "hi")]

    assert isinstance(events[-1], DoneEvent)
    assert isinstance(events[-2], MetricsEvent)
    assert metrics_of(events).tool_calls == 0
    assert metrics_of(events).llm_calls == 1

    await service.shutdown()
