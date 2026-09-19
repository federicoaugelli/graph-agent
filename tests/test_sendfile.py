from __future__ import annotations

from pathlib import Path

import pytest

from graph_agent.config import AppConfig
from graph_agent.events import FileEvent
from graph_agent.tools.base import ToolContext
from graph_agent.tools.sendfile import SendFileTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def ctx(workspace: Path, minimal_config: AppConfig) -> ToolContext:
    return ToolContext(session_id="test", workspace=workspace, config=minimal_config)


@pytest.fixture
def tool() -> SendFileTool:
    return SendFileTool()


async def test_returns_file_event(tool: SendFileTool, ctx: ToolContext, workspace: Path) -> None:
    target = workspace / "a.txt"
    target.write_text("hi")

    result = await tool.execute({"path": "a.txt", "caption": "ecco"}, ctx)

    assert isinstance(result, FileEvent)
    assert result.path == str(target.resolve())
    assert result.caption == "ecco"


async def test_caption_is_optional(tool: SendFileTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.txt").write_text("hi")

    result = await tool.execute({"path": "a.txt"}, ctx)

    assert isinstance(result, FileEvent)
    assert result.caption is None


async def test_traversal_blocked(tool: SendFileTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace.parent / "secret.txt").write_text("x")

    with pytest.raises(ValueError):
        await tool.execute({"path": "../secret.txt"}, ctx)


async def test_missing_file_raises(tool: SendFileTool, ctx: ToolContext) -> None:
    with pytest.raises(ValueError):
        await tool.execute({"path": "nope.txt"}, ctx)


async def test_directory_is_rejected(tool: SendFileTool, ctx: ToolContext, workspace: Path) -> None:
    (workspace / "sub").mkdir()

    with pytest.raises(ValueError):
        await tool.execute({"path": "sub"}, ctx)


async def test_missing_path_raises(tool: SendFileTool, ctx: ToolContext) -> None:
    with pytest.raises(ValueError):
        await tool.execute({}, ctx)