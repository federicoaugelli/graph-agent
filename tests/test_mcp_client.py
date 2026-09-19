from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import types

from conftest import fake_text_response
from graph_agent.config import AppConfig, MCPServerConfig
from graph_agent.core.service import AgentService
from graph_agent.mcp.client import MCPClientManager, MCPTool
from graph_agent.tools.base import ToolContext, ToolRegistry

STUB_SERVER = Path(__file__).parent / "mcp_stub_server.py"


class FakeSession:
    def __init__(self, result: types.CallToolResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.CallToolResult:
        self.calls.append((name, arguments))
        return self.result


def text_result(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)])


def make_tool(session: Any) -> MCPTool:
    return MCPTool(
        server="kb",
        tool_name="echo",
        description="Echo back text.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        session=session,
    )


def make_ctx(minimal_config: AppConfig) -> ToolContext:
    return ToolContext(session_id="t", workspace=Path("/tmp"), config=minimal_config)


def stub_server_config() -> MCPServerConfig:
    return MCPServerConfig(name="stub", command=sys.executable, args=[str(STUB_SERVER)])


def test_spec_uses_prefixed_name_and_schema() -> None:
    tool = make_tool(FakeSession(text_result("x")))

    spec = tool.spec

    assert spec.name == "mcp_kb_echo"
    assert spec.description == "Echo back text."
    assert spec.parameters["properties"]["text"] == {"type": "string"}
    assert spec.requires_approval is False


async def test_execute_forwards_name_and_args(minimal_config: AppConfig) -> None:
    session = FakeSession(text_result("echo: hi"))
    tool = make_tool(session)

    result = await tool.execute({"text": "hi"}, make_ctx(minimal_config))

    assert result == "echo: hi"
    assert session.calls == [("echo", {"text": "hi"})]


async def test_execute_joins_multiple_text_blocks(minimal_config: AppConfig) -> None:
    result = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="line 1"),
            types.TextContent(type="text", text="line 2"),
        ]
    )
    tool = make_tool(FakeSession(result))

    assert await tool.execute({}, make_ctx(minimal_config)) == "line 1\nline 2"


async def test_execute_marks_non_text_blocks(minimal_config: AppConfig) -> None:
    result = types.CallToolResult(
        content=[types.ImageContent(type="image", data="AAAA", mimeType="image/png")]
    )
    tool = make_tool(FakeSession(result))

    assert await tool.execute({}, make_ctx(minimal_config)) == "[image]"


async def test_execute_uses_structured_content_when_no_text(minimal_config: AppConfig) -> None:
    result = types.CallToolResult(content=[], structuredContent={"answer": 42})
    tool = make_tool(FakeSession(result))

    raw = await tool.execute({}, make_ctx(minimal_config))

    assert json.loads(raw) == {"answer": 42}


async def test_execute_raises_on_error_result(minimal_config: AppConfig) -> None:
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text="boom")],
        isError=True,
    )
    tool = make_tool(FakeSession(result))

    with pytest.raises(RuntimeError, match="boom"):
        await tool.execute({}, make_ctx(minimal_config))


def test_register_tools_is_idempotent_within_registry() -> None:
    manager = MCPClientManager([])
    tool = make_tool(FakeSession(text_result("x")))
    manager._tools = [tool]
    registry = ToolRegistry()

    registered = manager.register_tools(registry)

    assert registered == [tool]
    assert registry.names() == ["mcp_kb_echo"]
    assert registry.get("mcp_kb_echo") is tool


async def test_connect_all_lists_and_registers_real_server(minimal_config: AppConfig) -> None:
    manager = MCPClientManager([stub_server_config()])
    await manager.connect_all()
    try:
        registry = ToolRegistry()
        registered = manager.register_tools(registry)

        assert {tool.spec.name for tool in registered} == {"mcp_stub_echo", "mcp_stub_add"}

        echo = registry.get("mcp_stub_echo")
        assert "text" in echo.spec.parameters["properties"]
        result = await echo.execute({"text": "ciao"}, make_ctx(minimal_config))
        assert result == "echo: ciao"
    finally:
        await manager.shutdown()


async def test_service_setup_registers_mcp_tools(app_config: AppConfig) -> None:
    app_config.mcp.servers = [stub_server_config()]
    service = AgentService(app_config)
    await service.setup(fake_text_response("ok"))

    assert "mcp_stub_echo" in service.registry.names()

    await service.shutdown()