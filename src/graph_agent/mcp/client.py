from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import get_default_environment, stdio_client

from graph_agent.config import MCPServerConfig
from graph_agent.tools.base import Tool, ToolContext, ToolRegistry, ToolSpec


@dataclass(slots=True)
class MCPTool:
    """An external MCP tool exposed to the agent as a regular registry Tool."""

    server: str
    tool_name: str
    description: str
    parameters: dict[str, Any]
    session: ClientSession

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=f"mcp_{self.server}_{self.tool_name}",
            description=self.description,
            parameters=self.parameters,
            requires_approval=False,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        result = await self.session.call_tool(self.tool_name, args)
        text = _format_result(result)
        if result.is_error:
            raise RuntimeError(text or f"mcp tool error: {self.tool_name}")
        return text


class MCPClientManager:
    """Connect to external MCP servers (stdio) and bridge their tools into the ToolRegistry.

    Sessions are kept open for the lifetime of the manager via a single
    AsyncExitStack; connect_all() and shutdown() must run in the same asyncio
    task because the underlying stdio transport uses anyio cancel scopes.
    """

    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self.servers = servers
        self._sessions: dict[str, ClientSession] = {}
        self._tools: list[MCPTool] = []
        self._stack: AsyncExitStack | None = None

    async def connect_all(self) -> None:
        self._stack = AsyncExitStack()
        try:
            for server in self.servers:
                await self._connect(server)
        except BaseException:
            await self.shutdown()
            raise

    async def _connect(self, server: MCPServerConfig) -> None:
        stack = self._stack
        if stack is None:
            raise RuntimeError("MCPClientManager is not connected; call connect_all() first")

        params = StdioServerParameters(
            command=server.command,
            args=list(server.args),
            env=_server_env(server),
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[server.name] = session

        listed = await session.list_tools()
        for tool in listed.tools:
            self._tools.append(
                MCPTool(
                    server=server.name,
                    tool_name=tool.name,
                    description=tool.description or "",
                    parameters=dict(tool.input_schema or {}),
                    session=session,
                )
            )

    async def shutdown(self) -> None:
        stack = self._stack
        self._stack = None
        self._sessions.clear()
        self._tools.clear()
        if stack is not None:
            await stack.aclose()

    def register_tools(self, registry: ToolRegistry) -> list[Tool]:
        registered: list[Tool] = []
        for tool in self._tools:
            registry.register(tool)
            registered.append(tool)
        return registered

    @property
    def tools(self) -> list[MCPTool]:
        return list(self._tools)


def _server_env(server: MCPServerConfig) -> dict[str, str] | None:
    if not server.env:
        return None
    return {**get_default_environment(), **server.env}


def _format_result(result: types.CallToolResult) -> str:
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
        else:
            parts.append(f"[{block.type}]")

    text = "\n".join(parts)
    if not text and result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False, default=str)
    return text