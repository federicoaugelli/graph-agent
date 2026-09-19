from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from graph_agent.config import ModelConfig
from graph_agent.models.llm import OpenAICompatBackend, to_openai_messages


def _delta(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _chunk(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> Any:
    choice = SimpleNamespace(delta=_delta(content, tool_calls), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _tool_call_delta(
    index: int = 0,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        async def gen() -> Any:
            for chunk in self.chunks:
                yield chunk

        return gen()


def fake_backend(chunks: list[Any], **config_kwargs: Any) -> tuple[OpenAICompatBackend, Any]:
    completions = FakeCompletions(chunks)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAICompatBackend(
        ModelConfig(model="fake-brain", api_base="http://localhost:4000", **config_kwargs),
        client=client,  # type: ignore[arg-type]
    )
    return backend, completions


def test_to_openai_messages_basic() -> None:
    messages: list[BaseMessage] = [
        SystemMessage(content="sei utile"),
        HumanMessage(content="hello"),
    ]
    out = to_openai_messages(messages)
    assert out == [
        {"role": "system", "content": "sei utile"},
        {"role": "user", "content": "hello"},
    ]


def test_to_openai_messages_tool_roundtrip() -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="leggi a.txt"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "filesystem",
                    "args": {"action": "read_file", "path": "a.txt"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="contenuto", tool_call_id="call_1"),
    ]
    out = to_openai_messages(messages)
    assert out[1]["role"] == "assistant"
    tool_calls = out[1]["tool_calls"]
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "filesystem"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "action": "read_file",
        "path": "a.txt",
    }
    assert out[2] == {"role": "tool", "content": "contenuto", "tool_call_id": "call_1"}


async def test_astream_text() -> None:
    backend, completions = fake_backend(
        [
            _chunk(content="Hello "),
            _chunk(content="world"),
            _chunk(finish_reason="stop"),
        ]
    )

    chunks = [chunk async for chunk in backend.astream([HumanMessage(content="hi")])]

    assert "".join(chunk.delta_text for chunk in chunks) == "Hello world"
    assert chunks[-1].finish_reason == "stop"
    assert all(not chunk.tool_calls for chunk in chunks)

    call = completions.calls[0]
    assert call["model"] == "fake-brain"
    assert call["stream"] is True
    assert call["messages"] == [{"role": "user", "content": "hi"}]


async def test_astream_tool_call_assembled() -> None:
    backend, _ = fake_backend(
        [
            _chunk(tool_calls=[_tool_call_delta(id="call_1", name="filesystem")]),
            _chunk(tool_calls=[_tool_call_delta(arguments='{"act')]),
            _chunk(tool_calls=[_tool_call_delta(arguments='ion": "read_file",')]),
            _chunk(tool_calls=[_tool_call_delta(arguments=' "path": "a.txt"}')]),
            _chunk(finish_reason="tool_calls"),
        ]
    )

    chunks = [chunk async for chunk in backend.astream([HumanMessage(content="leggi")])]

    tool_calls = [tc for chunk in chunks for tc in chunk.tool_calls]
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].name == "filesystem"
    assert tool_calls[0].args == {"action": "read_file", "path": "a.txt"}


async def test_astream_two_parallel_tool_calls() -> None:
    backend, _ = fake_backend(
        [
            _chunk(tool_calls=[_tool_call_delta(index=0, id="a", name="filesystem")]),
            _chunk(tool_calls=[_tool_call_delta(index=1, id="b", name="web_search")]),
            _chunk(tool_calls=[_tool_call_delta(index=0, arguments='{"path": "."}')]),
            _chunk(tool_calls=[_tool_call_delta(index=1, arguments='{"query": "x"}')]),
            _chunk(finish_reason="tool_calls"),
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


async def test_astream_passes_tools_and_params() -> None:
    backend, completions = fake_backend(
        [_chunk(finish_reason="stop")], temperature=0.5, max_tokens=123
    )

    tools = [{"type": "function", "function": {"name": "echo"}}]
    _ = [chunk async for chunk in backend.astream([HumanMessage(content="hi")], tools=tools)]

    call = completions.calls[0]
    assert call["tools"] == tools
    assert call["temperature"] == 0.5
    assert call["max_tokens"] == 123


async def test_acomplete_aggregates() -> None:
    backend, _ = fake_backend(
        [
            _chunk(content="ab"),
            _chunk(content="cd"),
            _chunk(finish_reason="stop"),
        ]
    )

    result = await backend.acomplete([HumanMessage(content="hi")])

    assert result.delta_text == "abcd"
    assert result.finish_reason == "stop"


def test_default_client_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_KEY_ENV", "sk-test")
    backend = OpenAICompatBackend(
        ModelConfig(
            model="fake-brain",
            api_base="http://localhost:4000",
            api_key_env="FAKE_KEY_ENV",
        )
    )
    assert str(backend.client.base_url).startswith("http://localhost:4000")
    assert backend.client.api_key == "sk-test"
