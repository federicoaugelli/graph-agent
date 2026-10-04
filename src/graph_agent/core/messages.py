from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

INTERRUPTED_TOOL_RESULT = (
    "Tool call was not executed: the previous run was interrupted before producing a "
    "result. Treat this as a failure and do not assume the action happened."
)


def repair_tool_call_pairs(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Make a message history valid for OpenAI-compatible backends.

    An assistant message carrying ``tool_calls`` must be followed by one ``tool``
    message per ``tool_call_id``. A run can leave the checkpoint with a dangling
    assistant tool-call message (iteration limit, cancelled run, abandoned
    approval, crash); the API then rejects every later turn with HTTP 400. Missing
    results are filled with a synthetic ``ToolMessage`` and orphan ``ToolMessage``
    (no matching call) are dropped.
    """
    repaired: list[BaseMessage] = []
    pending: list[str] = []

    def flush() -> None:
        for call_id in pending:
            repaired.append(ToolMessage(content=INTERRUPTED_TOOL_RESULT, tool_call_id=call_id))
        pending.clear()

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            flush()
            repaired.append(message)
            pending.extend(str(call["id"]) for call in message.tool_calls)
        elif isinstance(message, ToolMessage):
            call_id = str(message.tool_call_id)
            if call_id in pending:
                pending.remove(call_id)
                repaired.append(message)
        else:
            flush()
            repaired.append(message)

    flush()
    return repaired
