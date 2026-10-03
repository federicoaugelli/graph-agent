from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage

from graph_agent.config import ModelConfig
from graph_agent.models.llm import OpenAICompatBackend


class FakeChatModel:
    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self.chunks = chunks
        self.bound_tools: list[Any] | None = None
        self.calls: list[list[BaseMessage]] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> FakeChatModel:
        self.bound_tools = list(tools)
        return self

    async def astream(
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[AIMessageChunk]:
        self.calls.append(messages)
        for chunk in self.chunks:
            yield chunk


def fake_backend(
    chunks: list[AIMessageChunk], **config_kwargs: Any
) -> tuple[OpenAICompatBackend, FakeChatModel]:
    model = FakeChatModel(chunks)
    backend = OpenAICompatBackend(
        ModelConfig(model="fake-brain", api_base="http://localhost:4000", **config_kwargs),
        model=cast(Any, model),
    )
    return backend, model


def text_chunk(text: str, finish_reason: str | None = None) -> AIMessageChunk:
    return AIMessageChunk(content=text, response_metadata={"finish_reason": finish_reason})


def tool_chunk(
    index: int,
    id: str | None = None,
    name: str | None = None,
    args: str | None = None,
    finish_reason: str | None = None,
) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"index": index, "id": id, "name": name, "args": args}],
        response_metadata={"finish_reason": finish_reason},
    )


async def test_astream_text() -> None:
    backend, model = fake_backend([text_chunk("Hello "), text_chunk("world", "stop")])

    chunks = [chunk async for chunk in backend.astream([HumanMessage(content="hi")])]

    assert "".join(chunk.delta_text for chunk in chunks) == "Hello world"
    assert chunks[-1].finish_reason == "stop"
    assert all(not chunk.tool_calls for chunk in chunks)
    assert model.calls[0] == [HumanMessage(content="hi")]


async def test_astream_tool_call_assembled() -> None:
    backend, _ = fake_backend(
        [
            tool_chunk(0, id="call_1", name="filesystem"),
            tool_chunk(0, args='{"act'),
            tool_chunk(0, args='ion": "read_file",'),
            tool_chunk(0, args=' "path": "a.txt"}'),
            tool_chunk(0, finish_reason="tool_calls"),
        ]
    )

    chunks = [chunk async for chunk in backend.astream([HumanMessage(content="leggi")])]

    tool_calls = [tc for chunk in chunks for tc in chunk.tool_calls]
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].name == "filesystem"
    assert tool_calls[0].args == {"action": "read_file", "path": "a.txt"}
    assert chunks[-1].finish_reason == "tool_calls"


async def test_astream_two_parallel_tool_calls() -> None:
    backend, _ = fake_backend(
        [
            tool_chunk(0, id="a", name="filesystem"),
            tool_chunk(1, id="b", name="web_search"),
            tool_chunk(0, args='{"path": "."}'),
            tool_chunk(1, args='{"query": "x"}'),
            tool_chunk(0, finish_reason="tool_calls"),
        ]
    )

    chunks = [chunk async for chunk in backend.astream([])]

    tool_calls = [tc for chunk in chunks for tc in chunk.tool_calls]
    assert {(tc.id, tc.name) for tc in tool_calls} == {
        ("a", "filesystem"),
        ("b", "web_search"),
    }
    by_id = {tc.id: tc.args for tc in tool_calls}
    assert by_id["a"] == {"path": "."}
    assert by_id["b"] == {"query": "x"}


async def test_astream_binds_tools() -> None:
    backend, model = fake_backend([text_chunk("ok", "stop")])
    tools = [{"type": "function", "function": {"name": "echo"}}]

    _ = [chunk async for chunk in backend.astream([HumanMessage(content="hi")], tools=tools)]

    assert model.bound_tools == tools


async def test_acomplete_aggregates() -> None:
    backend, _ = fake_backend([text_chunk("ab"), text_chunk("cd", "stop")])

    result = await backend.acomplete([HumanMessage(content="hi")])

    assert result.delta_text == "abcd"
    assert result.finish_reason == "stop"


def test_default_model_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_KEY_ENV", "sk-test")
    backend = OpenAICompatBackend(
        ModelConfig(
            model="fake-brain",
            api_base="http://localhost:4000",
            api_key_env="FAKE_KEY_ENV",
        )
    )

    assert backend.model.model_name == "fake-brain"
    assert backend.model.openai_api_base == "http://localhost:4000"
    assert backend.model.openai_api_key.get_secret_value() == "sk-test"
