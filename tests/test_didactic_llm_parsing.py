from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from graph_agent.didactic.llm_parsing import (
    ToolCallFragment,
    assemble_tool_calls,
    to_openai_messages,
)


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


def test_to_openai_messages_keeps_multimodal_parts() -> None:
    content = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
    ]
    messages: list[BaseMessage] = [HumanMessage(content=content)]

    out = to_openai_messages(messages)

    assert out == [{"role": "user", "content": content}]


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


def fragment(
    index: int,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> ToolCallFragment:
    return ToolCallFragment(index=index, id=id, name=name, arguments=arguments)


def test_assemble_tool_calls_single() -> None:
    fragments = [
        fragment(0, id="call_1", name="filesystem"),
        fragment(0, arguments='{"act'),
        fragment(0, arguments='ion": "read_file",'),
        fragment(0, arguments=' "path": "a.txt"}'),
    ]

    calls = assemble_tool_calls(fragments)

    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "filesystem"
    assert calls[0].args == {"action": "read_file", "path": "a.txt"}


def test_assemble_tool_calls_parallel() -> None:
    fragments = [
        fragment(0, id="a", name="filesystem"),
        fragment(1, id="b", name="web_search"),
        fragment(0, arguments='{"path": "."}'),
        fragment(1, arguments='{"query": "x"}'),
    ]

    calls = assemble_tool_calls(fragments)

    by_id: dict[str, Any] = {call.id: call.args for call in calls}
    assert by_id == {"a": {"path": "."}, "b": {"query": "x"}}
