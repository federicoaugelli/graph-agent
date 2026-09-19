from __future__ import annotations

from pathlib import Path

import pytest

from graph_agent.config import AppConfig
from graph_agent.tools.base import ToolContext
from graph_agent.tools.filesystem import FilesystemTool


@pytest.fixture
def root(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def tool(root: Path) -> FilesystemTool:
    return FilesystemTool(root)


@pytest.fixture
def ctx(tmp_path: Path, minimal_config: AppConfig) -> ToolContext:
    return ToolContext(session_id="test", workspace=tmp_path, config=minimal_config)


def test_spec_has_destination(tool: FilesystemTool) -> None:
    assert "destination" in tool.spec.parameters["properties"]


async def test_read_file(tool: FilesystemTool, ctx: ToolContext, root: Path) -> None:
    (root / "a.txt").write_text("hello")
    assert await tool.execute({"action": "read_file", "path": "a.txt"}, ctx) == "hello"


async def test_read_absolute_path_inside_root(
    tool: FilesystemTool, ctx: ToolContext, root: Path
) -> None:
    (root / "a.txt").write_text("x")
    assert await tool.execute({"action": "read_file", "path": str(root / "a.txt")}, ctx) == "x"


async def test_read_missing_file_raises(tool: FilesystemTool, ctx: ToolContext) -> None:
    with pytest.raises(FileNotFoundError):
        await tool.execute({"action": "read_file", "path": "nope.txt"}, ctx)


async def test_read_traversal_blocked(
    tool: FilesystemTool, ctx: ToolContext, tmp_path: Path
) -> None:
    (tmp_path / "outside.txt").write_text("segreto")
    with pytest.raises(ValueError):
        await tool.execute({"action": "read_file", "path": "../outside.txt"}, ctx)


async def test_absolute_path_outside_root_blocked(
    tool: FilesystemTool, ctx: ToolContext, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("segreto")
    with pytest.raises(ValueError):
        await tool.execute({"action": "read_file", "path": str(outside)}, ctx)


async def test_write_file_creates_parents(
    tool: FilesystemTool, ctx: ToolContext, root: Path
) -> None:
    result = await tool.execute(
        {"action": "write_file", "path": "notes/a.md", "content": "hello"}, ctx
    )
    assert (root / "notes" / "a.md").read_text() == "hello"
    assert result == "notes/a.md"


async def test_write_file_overwrites(tool: FilesystemTool, ctx: ToolContext, root: Path) -> None:
    for text in ("v1", "v2"):
        await tool.execute({"action": "write_file", "path": "a.txt", "content": text}, ctx)
    assert (root / "a.txt").read_text() == "v2"


async def test_write_file_without_content_raises(
    tool: FilesystemTool, ctx: ToolContext, root: Path
) -> None:
    with pytest.raises(ValueError):
        await tool.execute({"action": "write_file", "path": "a.txt"}, ctx)


async def test_list_dir_sorted(tool: FilesystemTool, ctx: ToolContext, root: Path) -> None:
    (root / "b.txt").write_text("b")
    (root / "a.txt").write_text("a")
    (root / "sub").mkdir()
    result = await tool.execute({"action": "list_dir", "path": "."}, ctx)
    assert result == ["a.txt", "b.txt", "sub"]


async def test_list_dir_on_file_raises(tool: FilesystemTool, ctx: ToolContext, root: Path) -> None:
    (root / "a.txt").write_text("a")
    with pytest.raises(ValueError):
        await tool.execute({"action": "list_dir", "path": "a.txt"}, ctx)


async def test_mkdir_nested(tool: FilesystemTool, ctx: ToolContext, root: Path) -> None:
    await tool.execute({"action": "mkdir", "path": "x/y/z"}, ctx)
    assert (root / "x/y/z").is_dir()


async def test_move_creates_parents(tool: FilesystemTool, ctx: ToolContext, root: Path) -> None:
    (root / "a.txt").write_text("hello")
    await tool.execute({"action": "move", "path": "a.txt", "destination": "b/a.txt"}, ctx)
    assert not (root / "a.txt").exists()
    assert (root / "b" / "a.txt").read_text() == "hello"


async def test_move_traversal_blocked(tool: FilesystemTool, ctx: ToolContext, root: Path) -> None:
    (root / "a.txt").write_text("hello")
    with pytest.raises(ValueError):
        await tool.execute({"action": "move", "path": "a.txt", "destination": "../out.txt"}, ctx)


async def test_delete_file(tool: FilesystemTool, ctx: ToolContext, root: Path) -> None:
    (root / "a.txt").write_text("a")
    await tool.execute({"action": "delete", "path": "a.txt"}, ctx)
    assert not (root / "a.txt").exists()


async def test_delete_dir_forbidden(tool: FilesystemTool, ctx: ToolContext, root: Path) -> None:
    (root / "sub").mkdir()
    with pytest.raises(ValueError):
        await tool.execute({"action": "delete", "path": "sub"}, ctx)


async def test_delete_missing_raises(tool: FilesystemTool, ctx: ToolContext) -> None:
    with pytest.raises(FileNotFoundError):
        await tool.execute({"action": "delete", "path": "nope.txt"}, ctx)


async def test_unknown_action_raises(tool: FilesystemTool, ctx: ToolContext, root: Path) -> None:
    with pytest.raises(ValueError):
        await tool.execute({"action": "teleport", "path": "a.txt"}, ctx)
