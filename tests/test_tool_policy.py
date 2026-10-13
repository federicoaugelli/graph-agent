from __future__ import annotations

import asyncio
from typing import Any

from conftest import ScriptedLLMBackend
from graph_agent.config import AppConfig
from graph_agent.core.service import AgentService
from graph_agent.events import DoneEvent, ToolResultEvent
from graph_agent.models.llm import StreamChunk, ToolCallRequest
from graph_agent.tools.base import ToolContext, ToolSpec
from graph_agent.tools.policy import (
    RunState,
    SearchBudgetPolicy,
    SearchDedupPolicy,
    ToolCallInfo,
)


def _search(query: str, call_id: str) -> StreamChunk:
    return StreamChunk(
        tool_calls=[ToolCallRequest(id=call_id, name="web_search", args={"query": query})],
        finish_reason="tool_calls",
    )


class FakeSearchTool:
    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description="Fake search.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        self.executed.append(args)
        return [
            {
                "title": "t",
                "url": f"https://example.com/{len(self.executed)}",
                "snippet": "s",
            }
        ]


async def test_budget_allows_up_to_the_limit_then_blocks() -> None:
    policy = SearchBudgetPolicy(2)
    state = RunState()
    call = ToolCallInfo("web_search", {"query": "a"})

    decisions = [await policy.apply(call, state) for _ in range(3)]

    assert [decision.blocks for decision in decisions] == [False, False, True]
    assert "budget" in str(decisions[-1].result)


def test_search_policies_only_apply_to_web_search() -> None:
    assert SearchBudgetPolicy(1).applies_to("web_search")
    assert not SearchBudgetPolicy(1).applies_to("echo")
    assert SearchDedupPolicy(0.8).applies_to("web_search")
    assert not SearchDedupPolicy(0.8).applies_to("echo")


async def test_budget_is_enforced_under_concurrent_calls() -> None:
    policy = SearchBudgetPolicy(3)
    state = RunState()
    call = ToolCallInfo("web_search", {"query": "x"})

    decisions = await asyncio.gather(*(policy.apply(call, state) for _ in range(10)))

    assert sum(1 for decision in decisions if decision.blocks) == 7
    assert state.counts["web_search"] == 3


async def test_dedup_drops_near_duplicate_queries() -> None:
    policy = SearchDedupPolicy(0.8)
    state = RunState()

    first = await policy.apply(
        ToolCallInfo("web_search", {"query": "dottorato AFAM musica"}), state
    )
    duplicate = await policy.apply(
        ToolCallInfo("web_search", {"query": "dottorato AFAM musica 2026"}), state
    )
    fresh = await policy.apply(
        ToolCallInfo("web_search", {"query": "concerto pianoforte roma"}), state
    )

    assert not first.blocks
    assert duplicate.blocks
    assert "Already searched" in str(duplicate.result)
    assert not fresh.blocks


async def test_dedup_batch_keeps_only_new_queries() -> None:
    policy = SearchDedupPolicy(0.8)
    state = RunState()
    await policy.apply(ToolCallInfo("web_search", {"query": "langgraph checkpoint"}), state)

    decision = await policy.apply(
        ToolCallInfo(
            "web_search",
            {
                "query": "langgraph checkpoint sqlite",
                "queries": ["langgraph checkpoint sqlite", "python asyncio queue"],
            },
        ),
        state,
    )

    assert not decision.blocks
    assert decision.args["query"] == "python asyncio queue"
    assert decision.args["queries"] == ["python asyncio queue"]


async def test_search_budget_short_circuits_within_a_run(app_config: AppConfig) -> None:
    app_config.tools.web_search.max_calls_per_run = 1
    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[
                        ToolCallRequest(id="s1", name="web_search", args={"query": "alpha beta"}),
                        ToolCallRequest(id="s2", name="web_search", args={"query": "gamma delta"}),
                    ],
                    finish_reason="tool_calls",
                )
            ],
            [StreamChunk(delta_text="done", finish_reason="stop")],
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    tool = FakeSearchTool()
    service.registry.register(tool)

    events = [event async for event in service.run("run", "go")]

    assert len(tool.executed) == 1
    assert isinstance(events[-1], DoneEvent)
    assert any(
        isinstance(event, ToolResultEvent) and "budget" in str(event.result) for event in events
    )

    await service.shutdown()


async def test_search_budget_blocks_across_rounds(app_config: AppConfig) -> None:
    app_config.tools.web_search.max_calls_per_run = 1
    backend = ScriptedLLMBackend(
        [
            [_search("alpha beta", "s1")],
            [_search("gamma delta", "s2")],
            [StreamChunk(delta_text="done", finish_reason="stop")],
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    tool = FakeSearchTool()
    service.registry.register(tool)

    events = [event async for event in service.run("run", "go")]

    assert len(tool.executed) == 1
    assert any(
        isinstance(event, ToolResultEvent) and "budget" in str(event.result) for event in events
    )

    await service.shutdown()


async def test_search_dedup_short_circuits_repeated_query(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend(
        [
            [_search("dottorato AFAM musica", "s1")],
            [_search("dottorato AFAM musica 2026", "s2")],
            [StreamChunk(delta_text="done", finish_reason="stop")],
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    tool = FakeSearchTool()
    service.registry.register(tool)

    events = [event async for event in service.run("run", "go")]

    assert len(tool.executed) == 1
    assert any(
        isinstance(event, ToolResultEvent) and "Already searched" in str(event.result)
        for event in events
    )

    await service.shutdown()


async def test_search_budget_resets_between_runs(app_config: AppConfig) -> None:
    app_config.tools.web_search.max_calls_per_run = 1
    backend = ScriptedLLMBackend(
        [
            [_search("alpha beta", "s1")],
            [StreamChunk(delta_text="done1", finish_reason="stop")],
            [_search("gamma delta", "s2")],
            [StreamChunk(delta_text="done2", finish_reason="stop")],
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    tool = FakeSearchTool()
    service.registry.register(tool)

    _ = [event async for event in service.run("run", "one")]
    _ = [event async for event in service.run("run", "two")]

    assert len(tool.executed) == 2

    await service.shutdown()
