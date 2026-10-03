from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from conftest import ScriptedLLMBackend
from graph_agent.config import AppConfig
from graph_agent.core.compaction import SUMMARY_INSTRUCTIONS, SUMMARY_PREFIX, compact
from graph_agent.core.service import AgentService
from graph_agent.events import Event
from graph_agent.models.llm import StreamChunk


def humans(*contents: str) -> list[BaseMessage]:
    return [HumanMessage(content=content) for content in contents]


async def test_no_compaction_under_limit() -> None:
    backend = ScriptedLLMBackend([[StreamChunk(delta_text="unused")]])

    result = await compact(backend, humans("a", "b", "c", "d"), token_limit=10_000, keep=2)

    assert result is None
    assert backend.calls == []


async def test_disabled_when_limit_is_zero() -> None:
    backend = ScriptedLLMBackend([[StreamChunk(delta_text="unused")]])

    result = await compact(backend, humans("a", "b", "c", "d"), token_limit=0, keep=1)

    assert result is None
    assert backend.calls == []


async def test_compaction_summarizes_oldest_messages() -> None:
    backend = ScriptedLLMBackend([[StreamChunk(delta_text="riassunto", finish_reason="stop")]])

    result = await compact(backend, humans("a", "b", "c", "d"), token_limit=1, keep=2)

    assert result is not None
    assert isinstance(result[0], RemoveMessage)
    assert isinstance(result[1], SystemMessage)
    assert result[1].content == f"{SUMMARY_PREFIX}riassunto"
    assert [message.content for message in result[2:]] == ["c", "d"]

    summarization_call = backend.calls[0]
    assert summarization_call[0].content == SUMMARY_INSTRUCTIONS
    assert [message.content for message in summarization_call[1:]] == ["a", "b"]


async def test_split_never_lands_on_a_tool_message() -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"id": "1", "name": "x", "args": {}, "type": "tool_call"}],
        ),
        ToolMessage(content="result", tool_call_id="1"),
        HumanMessage(content="next"),
    ]
    backend = ScriptedLLMBackend([[StreamChunk(delta_text="riassunto", finish_reason="stop")]])

    result = await compact(backend, messages, token_limit=1, keep=2)

    assert result is not None
    kept = result[2:]
    assert len(kept) == 3
    assert isinstance(kept[0], AIMessage)


async def _drain(stream: AsyncIterator[Event]) -> None:
    async for _ in stream:
        pass


async def test_service_compacts_across_turns(app_config: AppConfig) -> None:
    app_config.agent.context_token_limit = 1
    app_config.agent.compaction_keep_messages = 1
    backend = ScriptedLLMBackend(
        [
            [StreamChunk(delta_text="risposta1", finish_reason="stop")],
            [StreamChunk(delta_text="riassunto", finish_reason="stop")],
            [StreamChunk(delta_text="risposta2", finish_reason="stop")],
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)

    await _drain(service.run("turni", "uno"))
    await _drain(service.run("turni", "due"))

    final_agent_call = [str(message.content) for message in backend.calls[-1]]
    assert any(SUMMARY_PREFIX in content for content in final_agent_call)

    await service.shutdown()


async def test_context_stats_counts_messages(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend([[StreamChunk(delta_text="risposta", finish_reason="stop")]])
    service = AgentService(app_config)
    await service.setup(backend)

    await _drain(service.run("stat", "ciao"))
    count, tokens = await service.context_stats("stat")

    assert count >= 2
    assert tokens > 0

    await service.shutdown()


async def test_compact_session_forces_compaction(app_config: AppConfig) -> None:
    app_config.agent.compaction_keep_messages = 1
    backend = ScriptedLLMBackend(
        [
            [StreamChunk(delta_text="r1", finish_reason="stop")],
            [StreamChunk(delta_text="r2", finish_reason="stop")],
            [StreamChunk(delta_text="riassunto", finish_reason="stop")],
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)

    await _drain(service.run("force", "uno"))
    await _drain(service.run("force", "due"))
    count_before, _ = await service.context_stats("force")

    compacted = await service.compact_session("force")

    assert compacted is True
    count_after, _ = await service.context_stats("force")
    assert count_after < count_before

    await service.shutdown()
