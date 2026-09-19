from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from graph_agent.config import ModelConfig


@dataclass(slots=True)
class ToolCallRequest:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class StreamChunk:
    delta_text: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str | None = None
    raw: Any = None


def to_openai_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages -> OpenAI chat format (role/content/tool_calls dicts)."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if isinstance(msg, ToolMessage):
            out.append({"role": "tool", "content": content, "tool_call_id": msg.tool_call_id})
        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": content}
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


class LLMBackend(Protocol):
    async def acomplete(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> StreamChunk: ...

    def astream(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...


class OpenAICompatBackend:
    """Backend for any OpenAI-compatible endpoint (the LiteLLM proxy, vLLM, ...).

    Agnostic by design: no provider SDK here, just base_url + model name.
    The client is injectable for tests.
    """

    def __init__(self, config: ModelConfig, client: AsyncOpenAI | None = None) -> None:
        self.config = config
        self.model = config.model
        api_key = os.environ.get(config.api_key_env, "unused") if config.api_key_env else "unused"
        self.client = client or AsyncOpenAI(base_url=config.api_base, api_key=api_key)

    async def acomplete(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> StreamChunk:
        merged = StreamChunk()
        async for chunk in self.astream(messages, tools=tools):
            merged.delta_text += chunk.delta_text
            merged.tool_calls.extend(chunk.tool_calls)
            if chunk.finish_reason is not None:
                merged.finish_reason = chunk.finish_reason
        return merged

    async def astream(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        response = cast(
            AsyncStream[ChatCompletionChunk],
            await self.client.chat.completions.create(
                model=self.model,
                messages=cast(list[ChatCompletionMessageParam], to_openai_messages(messages)),
                tools=cast(Any, tools),
                stream=True,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ),
        )
        pending: dict[int, dict[str, str]] = {}
        async for raw_chunk in response:
            if not raw_chunk.choices:
                continue
            choice = raw_chunk.choices[0]
            delta = choice.delta
            for fragment in delta.tool_calls or []:
                slot = pending.setdefault(fragment.index, {"id": "", "name": "", "arguments": ""})
                if fragment.id:
                    slot["id"] = fragment.id
                if fragment.function is not None and fragment.function.name:
                    slot["name"] = fragment.function.name
                if fragment.function is not None and fragment.function.arguments:
                    slot["arguments"] += fragment.function.arguments
            if delta.content:
                yield StreamChunk(delta_text=delta.content)
            if choice.finish_reason is not None:
                yield StreamChunk(
                    finish_reason=choice.finish_reason,
                    tool_calls=[
                        ToolCallRequest(
                            id=slot["id"],
                            name=slot["name"],
                            args=json.loads(slot["arguments"] or "{}"),
                        )
                        for slot in pending.values()
                    ],
                )
                pending.clear()
