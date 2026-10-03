"""Didactic archive: hand-rolled OpenAI chat-request parsing.

What this module is
    The original hand-written code that translated LangChain ``BaseMessage``
    objects into the OpenAI Chat Completions request format, and reassembled
    streamed tool-call fragments emitted by the API.

What it implements
    ``to_openai_messages``  -> ``list[BaseMessage]`` to OpenAI message dicts.
    ``_as_text``            -> normalise a message content payload to a string.
    ``assemble_tool_calls`` -> merge streamed tool-call fragments (by ``index``)
                               into complete calls with parsed JSON arguments.

What built-in replaces it
    ``langchain_openai.ChatOpenAI`` performs both steps internally: it accepts
    ``list[BaseMessage]`` directly and yields ``AIMessageChunk`` objects whose
    ``+`` operator merges streamed tool-call fragments, parsing the arguments
    into a dict. If you only need the raw conversion (without a chat model),
    ``langchain_core.messages.convert_to_openai_messages`` is the built-in.

Why the built-in is preferred
    The manual version handles only the simplest shapes. The built-in also
    covers multimodal content blocks, refusals/reasoning fields, provider
    quirks, retries and streaming edge cases, and it is maintained upstream.

What to look at while reading
    - How ``ToolMessage`` carries ``tool_call_id`` and how an assistant turn
      encodes ``tool_calls`` (id/type/function.name/function.arguments).
    - Why streamed tool calls must be accumulated by ``index`` and why
      ``arguments`` is concatenated as a *string* before ``json.loads``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


@dataclass(slots=True)
class ToolCallRequest:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ToolCallFragment:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


def to_openai_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages -> OpenAI chat format (role/content/tool_calls dicts)."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        raw = msg.content
        content: Any = raw if isinstance(raw, (str, list)) else str(raw or "")
        if isinstance(msg, ToolMessage):
            out.append(
                {"role": "tool", "content": _as_text(content), "tool_call_id": msg.tool_call_id}
            )
        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": _as_text(content)}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                    }
                    for tc in msg.tool_calls
                ]
            out.append(entry)
        else:
            role = {"system": "system", "human": "user"}.get(msg.type, msg.type)
            out.append({"role": role, "content": content})
    return out


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str) if content else ""


@dataclass(slots=True)
class _PendingCall:
    id: str = ""
    name: str = ""
    arguments: str = field(default="")


def assemble_tool_calls(fragments: list[ToolCallFragment]) -> list[ToolCallRequest]:
    """Merge streamed tool-call fragments, grouped by ``index``, into full calls."""
    pending: dict[int, _PendingCall] = {}
    for fragment in fragments:
        slot = pending.setdefault(fragment.index, _PendingCall())
        if fragment.id:
            slot.id = fragment.id
        if fragment.name:
            slot.name = fragment.name
        if fragment.arguments:
            slot.arguments += fragment.arguments
    return [
        ToolCallRequest(
            id=slot.id,
            name=slot.name,
            args=json.loads(slot.arguments or "{}"),
        )
        for slot in pending.values()
    ]
