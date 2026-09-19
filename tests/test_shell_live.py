from __future__ import annotations

import os
from pathlib import Path

import pytest

from graph_agent.config import SandboxConfig
from graph_agent.tools.shell import BwrapRunner

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_TESTS") != "1",
        reason="live tests need bubblewrap: RUN_LIVE_TESTS=1",
    ),
]


@pytest.fixture
def bwrap_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "hello.txt").write_text("hello from jail\n")
    return ws


async def test_bwrap_runs_real_command(bwrap_workspace: Path) -> None:
    runner = BwrapRunner(SandboxConfig(backend="bwrap"))
    result = await runner.run(
        "cat hello.txt && echo eseguito > out.txt",
        timeout_seconds=30,
        workspace=bwrap_workspace,
    )
    assert result.exit_code == 0
    assert "hello from jail" in result.stdout
    assert (bwrap_workspace / "out.txt").read_text() == "eseguito\n"


async def test_bwrap_root_is_readonly(bwrap_workspace: Path) -> None:
    runner = BwrapRunner(SandboxConfig(backend="bwrap"))
    result = await runner.run(
        "touch /etc/pwned",
        timeout_seconds=30,
        workspace=bwrap_workspace,
    )
    assert result.exit_code != 0
    assert "Read-only" in result.stderr


async def test_bwrap_no_network(bwrap_workspace: Path) -> None:
    runner = BwrapRunner(SandboxConfig(backend="bwrap"))
    result = await runner.run(
        "timeout 3 bash -c '</dev/tcp/1.1.1.1/443' && echo UP || echo DOWN",
        timeout_seconds=30,
        workspace=bwrap_workspace,
    )
    assert result.exit_code == 0
    assert "DOWN" in result.stdout


async def test_bwrap_timeout_kills(bwrap_workspace: Path) -> None:
    runner = BwrapRunner(SandboxConfig(backend="bwrap"))
    result = await runner.run("sleep 30", timeout_seconds=2, workspace=bwrap_workspace)
    assert result.exit_code == -1
    assert "timed out" in result.stderr
