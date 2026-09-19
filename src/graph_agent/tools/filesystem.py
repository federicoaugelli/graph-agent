from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from graph_agent.tools.base import ToolContext, ToolSpec


class FilesystemTool:
    """Filesystem operations confined to an allowed root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="filesystem",
            description="Read, write and manage files and directories inside the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read_file", "write_file", "list_dir", "mkdir", "move", "delete"],
                    },
                    "path": {"type": "string"},
                    "destination": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["action", "path"],
            },
            requires_approval=False,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        action = args.get("action")
        path = self._safe_path(args.get("path", "."))
        match action:
            case "read_file":
                return path.read_text()
            case "write_file":
                content = args.get("content")
                if content is None:
                    raise ValueError("write_file requires 'content'")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                return str(path.relative_to(self.root))
            case "list_dir":
                if not path.is_dir():
                    raise ValueError(f"not a directory: {path}")
                return sorted(entry.name for entry in path.iterdir())
            case "mkdir":
                path.mkdir(parents=True, exist_ok=True)
            case "move":
                destination = self._safe_path(args["destination"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(destination))
            case "delete":
                if path.is_dir():
                    raise ValueError(f"cannot delete directory: {path}")
                path.unlink()
            case _:
                raise ValueError(f"unknown action: {action}")

    def _safe_path(self, raw: str) -> Path:
        candidate = (self.root / raw).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"path outside workspace: {raw}")
        return candidate
