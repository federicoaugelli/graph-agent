from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from langchain_core.utils.function_calling import convert_to_openai_tool

from graph_agent.config import AppConfig


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False


@dataclass(slots=True)
class ToolContext:
    session_id: str
    workspace: Path
    config: AppConfig


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any: ...


def resolve_within(root: Path, raw: str) -> Path:
    """Resolve ``raw`` under ``root``, rejecting paths that escape it."""
    base = root.resolve()
    resolved = (base / raw).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"path outside workspace: {raw}")
    return resolved


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"tool gia' registrato: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def to_openai_schema(self) -> list[dict[str, Any]]:
        return [
            convert_to_openai_tool(
                {
                    "name": tool.spec.name,
                    "description": tool.spec.description,
                    "parameters": tool.spec.parameters,
                }
            )
            for tool in self._tools.values()
        ]
