from __future__ import annotations

from mcp.server.mcpserver import MCPServer

server = MCPServer("stub")


@server.tool()
def echo(text: str) -> str:
    """Echo back the given text."""
    return f"echo: {text}"


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    server.run("stdio")