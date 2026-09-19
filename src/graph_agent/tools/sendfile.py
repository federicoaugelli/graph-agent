from __future__ import annotations

from typing import Any

from graph_agent.events import FileEvent
from graph_agent.tools.base import ToolContext, ToolSpec


class SendFileTool:
    """Emit a FileEvent asking the active transport to deliver a workspace file.

    The transport decides how to render it (Telegram send_document, CLI prints
    the path, HTTP returns a textual note). The path is confined to the
    workspace, like the filesystem tool.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="send_file",
            description=(
                "Send a file from the workspace to the user on the current channel. "
                "Use a path relative to the workspace."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "caption": {"type": "string"},
                },
                "required": ["path"],
            },
            requires_approval=False,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        raw = args.get("path")
        if not raw:
            raise ValueError("send_file requires 'path'")

        root = ctx.workspace.resolve()
        candidate = (root / raw).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"path outside workspace: {raw}")
        if not candidate.is_file():
            raise ValueError(f"not a file: {raw}")

        caption = args.get("caption")
        return FileEvent(path=str(candidate), caption=caption if caption else None)