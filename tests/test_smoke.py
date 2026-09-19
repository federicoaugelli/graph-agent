from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import FakeLLMBackend, fake_text_response
from graph_agent.bus import EventBus
from graph_agent.config import AppConfig, SandboxConfig, WebSearchToolConfig, load_config
from graph_agent.events import DoneEvent, TokenEvent
from graph_agent.tools.base import ToolContext, ToolRegistry, ToolSpec
from graph_agent.tools.filesystem import FilesystemTool
from graph_agent.tools.shell import ShellTool
from graph_agent.tools.websearch import WebSearchTool


class EchoTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echo back the input.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        return args["text"]


def test_registry_roundtrip() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    assert registry.names() == ["echo"]
    schema = registry.to_openai_schema()
    assert schema[0]["function"]["name"] == "echo"
    assert schema[0]["type"] == "function"


def test_registry_rejects_duplicates() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(ValueError):
        registry.register(EchoTool())


def test_default_tools_have_specs(tmp_path: Path) -> None:
    tools = [
        FilesystemTool(tmp_path),
        ShellTool(SandboxConfig()),
        WebSearchTool(WebSearchToolConfig()),
    ]
    names = {tool.spec.name for tool in tools}
    assert names == {"filesystem", "shell", "web_search"}
    assert ShellTool(SandboxConfig()).spec.requires_approval


def test_load_config_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("models:\n  brain:\n    model: openai/fake\n")
    config = load_config(config_file)
    assert config.agent.max_iterations == 25
    assert config.channels.http.enabled is False
    assert config.logging.json_output is False


def test_load_config_requires_models(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("agent:\n  max_iterations: 5\n")
    with pytest.raises(ValueError):
        load_config(config_file)


async def test_event_bus_pubsub() -> None:
    bus = EventBus()
    queue = bus.subscribe("session-1")
    await bus.publish("session-1", TokenEvent(delta="hello"))
    await bus.publish("session-2", TokenEvent(delta="ignored"))
    event = queue.get_nowait()
    assert isinstance(event, TokenEvent)
    assert event.delta == "hello"
    assert queue.empty()
    bus.unsubscribe("session-1", queue)
    await bus.publish("session-1", DoneEvent(session_id="session-1"))
    assert queue.empty()


async def test_fake_llm_backend() -> None:
    backend: FakeLLMBackend = fake_text_response("hello")
    chunk = await backend.acomplete([])
    assert chunk.delta_text == "hello"
    assert chunk.finish_reason == "stop"
    assert backend.calls == [[]]


async def test_echo_tool_execute(tmp_path: Path, minimal_config: AppConfig) -> None:
    ctx = ToolContext(session_id="test", workspace=tmp_path, config=minimal_config)
    result = await EchoTool().execute({"text": "hi"}, ctx)
    assert result == "hi"
