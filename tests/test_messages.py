from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from graph_agent.core.messages import INTERRUPTED_TOOL_RESULT, repair_tool_call_pairs


def ai_call(*call_ids: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"id": call_id, "name": "echo", "args": {}, "type": "tool_call"} for call_id in call_ids
        ],
    )


def pending_ids(messages: list[BaseMessage]) -> list[str]:
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


def test_valid_pair_is_unchanged() -> None:
    messages = [
        HumanMessage(content="hi"),
        ai_call("c1"),
        ToolMessage(content="ok", tool_call_id="c1"),
    ]
    assert repair_tool_call_pairs(messages) == messages


def test_dangling_call_at_end_gets_synthetic_result() -> None:
    messages = [HumanMessage(content="hi"), ai_call("c1")]

    repaired = repair_tool_call_pairs(messages)

    assert repaired[:2] == messages
    assert isinstance(repaired[2], ToolMessage)
    assert repaired[2].tool_call_id == "c1"
    assert repaired[2].content == INTERRUPTED_TOOL_RESULT
    assert pending_ids(repaired) == []


def test_dangling_call_before_next_turn_is_filled() -> None:
    messages = [ai_call("c1"), HumanMessage(content="next")]

    repaired = repair_tool_call_pairs(messages)

    assert isinstance(repaired[1], ToolMessage)
    assert repaired[1].tool_call_id == "c1"
    assert isinstance(repaired[2], HumanMessage)
    assert pending_ids(repaired) == []


def test_orphan_tool_message_is_dropped() -> None:
    messages = [ToolMessage(content="stray", tool_call_id="ghost")]

    assert repair_tool_call_pairs(messages) == []


def test_parallel_calls_are_all_answered() -> None:
    messages = [ai_call("a", "b"), ToolMessage(content="ra", tool_call_id="a")]

    repaired = repair_tool_call_pairs(messages)

    ids = [m.tool_call_id for m in repaired if isinstance(m, ToolMessage)]
    assert ids == ["a", "b"]
