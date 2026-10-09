from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

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
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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


def _default_model(config: ModelConfig) -> ChatOpenAI:
    api_key = os.environ.get(config.api_key_env, "unused") if config.api_key_env else "unused"
    return ChatOpenAI(
        model=config.model,
        base_url=config.api_base,
        api_key=SecretStr(api_key),
        temperature=config.temperature,
        max_completion_tokens=config.max_tokens,
        stream_usage=True,
    )


class OpenAICompatBackend:
    """Backend for any OpenAI-compatible endpoint (the LiteLLM proxy, vLLM, ...).

    Delegates request building, message conversion and streaming tool-call
    assembly to ``ChatOpenAI``; this class only adapts the result to
    ``StreamChunk`` so the graph stays provider-agnostic.
    """

    def __init__(self, config: ModelConfig, model: ChatOpenAI | None = None) -> None:
        self.config = config
        self.model = model or _default_model(config)

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
        runnable = self.model.bind_tools(tools) if tools else self.model
        stream = cast(AsyncIterator[AIMessageChunk], runnable.astream(messages))

        assembled: AIMessageChunk | None = None
        async for chunk in stream:
            assembled = chunk if assembled is None else assembled + chunk
            if chunk.text:
                yield StreamChunk(delta_text=chunk.text)

        if assembled is None:
            return

        finish_reason = assembled.response_metadata.get("finish_reason")
        tool_calls = assembled.tool_calls
        usage = assembled.usage_metadata
        prompt_tokens = int(usage["input_tokens"]) if usage else None
        completion_tokens = int(usage["output_tokens"]) if usage else None
        if finish_reason is not None or tool_calls or usage:
            yield StreamChunk(
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tool_calls=[
                    ToolCallRequest(
                        id=str(tool_call["id"] or ""),
                        name=str(tool_call["name"]),
                        args=dict(tool_call["args"]),
                    )
                    for tool_call in tool_calls
                ],
            )
