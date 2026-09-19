from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from graph_agent.config import AppConfig, SandboxConfig
from graph_agent.tools.base import ToolContext
from graph_agent.tools.shell import ExecResult, ShellTool


class FakeRunner:
    def __init__(self, result: ExecResult | None = None, error: Exception | None = None) -> None:
        self.result = result or ExecResult(stdout="out", stderr="", exit_code=0)
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        command: str,
        *,
        timeout_seconds: int,
        workspace: Path,
    ) -> ExecResult:
        self.calls.append(
            {"command": command, "timeout_seconds": timeout_seconds, "workspace": workspace}
        )
        if self.error is not None:
            raise self.error
        return self.result


def make_ctx(app_config: AppConfig) -> ToolContext:
    return ToolContext(
        session_id="t",
        workspace=Path("/tmp/ws"),
        config=app_config,
    )


def make_tool(app_config: AppConfig, runner: FakeRunner) -> ShellTool:
    return ShellTool(app_config.sandbox, runner=runner)


def test_spec_requires_approval(app_config: AppConfig) -> None:
    tool = ShellTool(SandboxConfig())
    assert tool.spec.name == "shell"
    assert tool.spec.requires_approval is True
    assert "command" in tool.spec.parameters["required"]


async def test_execute_delegates_to_runner_with_defaults(
    app_config: AppConfig,
) -> None:
    runner = FakeRunner()
    tool = make_tool(app_config, runner)
    ctx = make_ctx(app_config)

    result = await tool.execute({"command": "echo hi"}, ctx)

    assert runner.calls == [
        {"command": "echo hi", "timeout_seconds": 120, "workspace": Path("/tmp/ws")}
    ]
    assert result == {"stdout": "out", "stderr": "", "exit_code": 0}


async def test_execute_clamps_timeout_to_config_max(app_config: AppConfig) -> None:
    runner = FakeRunner()
    tool = make_tool(app_config, runner)
    ctx = make_ctx(app_config)

    await tool.execute({"command": "x", "timeout_seconds": 9999}, ctx)
    assert runner.calls[0]["timeout_seconds"] == app_config.sandbox.timeout_seconds

    await tool.execute({"command": "x", "timeout_seconds": 5}, ctx)
    assert runner.calls[1]["timeout_seconds"] == 5


async def test_execute_rejects_blank_command(app_config: AppConfig) -> None:
    runner = FakeRunner()
    tool = make_tool(app_config, runner)
    ctx = make_ctx(app_config)

    with pytest.raises(ValueError):
        await tool.execute({"command": "   "}, ctx)

    assert runner.calls == []


async def test_execute_truncates_long_stdout(app_config: AppConfig) -> None:
    app_config.sandbox.max_output_bytes = 100
    runner = FakeRunner(ExecResult(stdout="A" * 500, stderr="B" * 500, exit_code=0))
    tool = make_tool(app_config, runner)
    ctx = make_ctx(app_config)

    result = await tool.execute({"command": "x"}, ctx)

    assert result["stdout"].startswith("A")
    assert "[truncated" in result["stdout"]
    assert len(result["stdout"].encode()) <= 100 + 64
    assert len(result["stderr"].encode()) <= 100 + 64


async def test_execute_leaves_short_output_untouched(app_config: AppConfig) -> None:
    runner = FakeRunner(ExecResult(stdout="ok\n", stderr="warn\n", exit_code=3))
    tool = make_tool(app_config, runner)
    ctx = make_ctx(app_config)

    result = await tool.execute({"command": "x"}, ctx)

    assert result == {"stdout": "ok\n", "stderr": "warn\n", "exit_code": 3}


async def test_execute_propagates_runner_errors(app_config: AppConfig) -> None:
    runner = FakeRunner(error=RuntimeError("sandbox is on fire"))
    tool = make_tool(app_config, runner)
    ctx = make_ctx(app_config)

    with pytest.raises(RuntimeError, match="sandbox"):
        await tool.execute({"command": "x"}, ctx)


def make_deny_config(patterns: list[str]) -> SandboxConfig:
    return SandboxConfig(deny_patterns=patterns)


async def test_deny_pattern_blocks_before_runner(app_config: AppConfig) -> None:
    runner = FakeRunner()
    config = make_deny_config([r"\brm\s+-rf"])
    tool = ShellTool(config, runner=runner)
    ctx = make_ctx(app_config)

    with pytest.raises(ValueError, match="denied"):
        await tool.execute({"command": "rm -rf /"}, ctx)

    assert runner.calls == []


async def test_deny_pattern_is_case_insensitive(app_config: AppConfig) -> None:
    runner = FakeRunner()
    tool = ShellTool(make_deny_config([r"shutdown"]), runner=runner)
    ctx = make_ctx(app_config)

    with pytest.raises(ValueError):
        await tool.execute({"command": "SHUTDOWN now"}, ctx)


async def test_allow_when_no_pattern_matches(app_config: AppConfig) -> None:
    runner = FakeRunner()
    tool = ShellTool(make_deny_config([r"\brm\s+-rf"]), runner=runner)
    ctx = make_ctx(app_config)

    await tool.execute({"command": "ls -la && rm file.txt"}, ctx)

    assert runner.calls[0]["command"] == "ls -la && rm file.txt"


def test_invalid_deny_regex_raises_at_init() -> None:
    with pytest.raises(ValueError, match="deny_patterns"):
        ShellTool(make_deny_config(["[unbalanced"]))


async def test_default_config_blocks_sudo(app_config: AppConfig) -> None:
    runner = FakeRunner()
    config = app_config.sandbox.model_copy(update={"deny_patterns": ["\\bsudo\\b"]})
    tool = ShellTool(config, runner=runner)
    ctx = make_ctx(app_config)
    with pytest.raises(ValueError, match="denied"):
        await tool.execute({"command": "sudo reboot"}, ctx)
