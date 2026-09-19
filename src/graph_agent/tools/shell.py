from __future__ import annotations

import asyncio
import os
import re
import signal
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from graph_agent.config import SandboxConfig
from graph_agent.tools.base import ToolContext, ToolSpec


@dataclass(slots=True)
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


class SandboxRunner(Protocol):
    async def run(
        self,
        command: str,
        *,
        timeout_seconds: int,
        workspace: Path,
    ) -> ExecResult: ...


class BwrapRunner:
    """Run commands jailed by bubblewrap: no daemon, no images, user namespaces."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config

    def _build_argv(self, command: str, workspace: Path) -> list[str]:
        ws = str(Path(workspace).resolve())
        argv = [
            "bwrap",
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "PATH",
            "/usr/local/bin:/usr/bin:/bin",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            ws,
            ws,
        ]
        for path in self.config.mask_paths:
            resolved = Path(path).resolve()
            if resolved.is_dir():
                argv += ["--tmpfs", str(resolved)]
            else:
                argv += ["--bind", "/dev/null", str(resolved)]
        argv += [
            "--chdir",
            ws,
            "--",
            "/bin/sh",
            "-c",
            command,
        ]
        return argv

    async def run(
        self,
        command: str,
        *,
        timeout_seconds: int,
        workspace: Path,
    ) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            *self._build_argv(command, workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            timed_out = True
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = await proc.communicate()

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        if timed_out:
            err = f"command timed out after {timeout_seconds}s\n{err}"
            return ExecResult(stdout=out, stderr=err, exit_code=-1)
        return ExecResult(stdout=out, stderr=err, exit_code=proc.returncode or 0)


class ShellTool:
    """Execute shell commands jailed by bubblewrap."""

    def __init__(self, config: SandboxConfig, runner: SandboxRunner | None = None) -> None:
        self.config = config
        self._runner = runner
        self._denied: list[re.Pattern[str]] = []
        for pattern in config.deny_patterns:
            try:
                self._denied.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                raise ValueError(f"shell: invalid deny_patterns regex {pattern!r}: {exc}") from exc

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="shell",
            description="Execute a shell command inside the bubblewrap sandbox.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                },
                "required": ["command"],
            },
            requires_approval=True,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("shell: command must not be empty")

        for rx in self._denied:
            if rx.search(command):
                raise ValueError(
                    f"shell: command denied by sandbox policy (pattern: {rx.pattern!r})"
                )

        requested = args.get("timeout_seconds")
        timeout = (
            self.config.timeout_seconds
            if requested is None
            else max(1, min(int(requested), self.config.timeout_seconds))
        )

        if self._runner is None:
            self._runner = BwrapRunner(self.config)
        result = await self._runner.run(
            command,
            timeout_seconds=timeout,
            workspace=ctx.workspace,
        )

        return {
            "stdout": self._truncate(result.stdout),
            "stderr": self._truncate(result.stderr),
            "exit_code": result.exit_code,
        }

    def _truncate(self, text: str) -> str:
        limit = self.config.max_output_bytes
        encoded = text.encode()
        if len(encoded) <= limit:
            return text
        clipped = encoded[:limit].decode("utf-8", errors="ignore")
        return f"{clipped}\n...[truncated {len(encoded) - limit} bytes]"
