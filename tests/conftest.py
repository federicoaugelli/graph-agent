from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import BaseMessage

from graph_agent.config import AppConfig, ModelConfig, ModelsConfig, load_config
from graph_agent.models.llm import StreamChunk, ToolCallRequest

MINIMAL_CONFIG_YAML = """
models:
  brain:
    model: fake-brain
    api_base: http://localhost:4000
"""


@pytest.fixture
def app_config(tmp_path: Any) -> AppConfig:
    from pathlib import Path

    path: Path = tmp_path
    config_file = path / "config.yaml"
    config_file.write_text(MINIMAL_CONFIG_YAML)
    config = load_config(config_file)
    config.agent.workspace = path / "workspace"
    config.agent.checkpointer_db = path / "checkpoints.sqlite"
    config.tools.filesystem.root = path / "fs"
    return config


@pytest.fixture
def minimal_config() -> AppConfig:
    return AppConfig(models=ModelsConfig(brain=ModelConfig(model="fake-brain")))


class FakeLLMBackend:
    """Fake backend for deterministic tests: replies with scripted chunks."""

    def __init__(self, script: list[StreamChunk]) -> None:
        self.script = script
        self.calls: list[list[BaseMessage]] = []

    async def acomplete(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> StreamChunk:
        self.calls.append(messages)
        merged = StreamChunk(finish_reason="stop")
        for chunk in self.script:
            merged.delta_text += chunk.delta_text
            merged.tool_calls.extend(chunk.tool_calls)
        return merged

    async def astream(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        for chunk in self.script:
            yield chunk


class ScriptedLLMBackend:
    """Fake backend for multi-turn tests: each model call consumes one scripted reply."""

    def __init__(self, responses: list[list[StreamChunk]]) -> None:
        self.responses = responses
        self.calls: list[list[BaseMessage]] = []

    async def acomplete(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> StreamChunk:
        self.calls.append(messages)
        merged = StreamChunk(finish_reason="stop")
        for chunk in self.responses.pop(0):
            merged.delta_text += chunk.delta_text
            merged.tool_calls.extend(chunk.tool_calls)
        return merged

    async def astream(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        for chunk in self.responses.pop(0):
            yield chunk


def fake_text_response(text: str) -> FakeLLMBackend:
    return FakeLLMBackend([StreamChunk(delta_text=text, finish_reason="stop")])


def fake_tool_call_response(tool_call_id: str, name: str, args: dict[str, Any]) -> FakeLLMBackend:
    return FakeLLMBackend(
        [StreamChunk(tool_calls=[ToolCallRequest(id=tool_call_id, name=name, args=args)])]
    )
