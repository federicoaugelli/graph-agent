from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any

import pytest

from graph_agent.config import SandboxConfig
from graph_agent.tools import shell as shell_module
from graph_agent.tools.shell import BwrapRunner


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int, delay: float = 0.0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.pid = 4242
        self._delay = delay
        self._first = True

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._first and self._delay:
            self._first = False
            await asyncio.sleep(self._delay)
        return self._stdout, self._stderr


class SpawnRecorder:
    def __init__(self, proc: FakeProcess) -> None:
        self.proc = proc
        self.argv: list[str] = []
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, *argv: str, **kwargs: Any) -> FakeProcess:
        self.argv = list(argv)
        self.kwargs = kwargs
        return self.proc


@pytest.fixture
def killpg_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    calls: list[tuple[int, int]] = []

    def fake_getpgid(pid: int) -> int:
        return 9999

    def fake_killpg(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))

    monkeypatch.setattr(shell_module.os, "getpgid", fake_getpgid)
    monkeypatch.setattr(shell_module.os, "killpg", fake_killpg)
    return calls


def patch_spawn(monkeypatch: pytest.MonkeyPatch, proc: FakeProcess) -> SpawnRecorder:
    recorder = SpawnRecorder(proc)
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", recorder)
    return recorder


async def test_argv_pins_security_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = patch_spawn(monkeypatch, FakeProcess(b"hi\n", b"", 0))
    runner = BwrapRunner(SandboxConfig())

    result = await runner.run("echo hi", timeout_seconds=5, workspace=Path("/tmp/ws"))

    argv = recorder.argv
    assert argv[0] == "bwrap"
    assert "--unshare-all" in argv
    assert "--unshare-user" in argv
    assert "--disable-userns" in argv
    assert "--die-with-parent" in argv
    assert argv[argv.index("--ro-bind") + 1 :][:2] == ["/", "/"]
    i = argv.index("--bind")
    assert argv[i + 1 : i + 3] == ["/tmp/ws", "/tmp/ws"]
    assert argv[argv.index("--chdir") + 1] == "/tmp/ws"
    assert argv[-4:] == ["--", "/bin/sh", "-c", "echo hi"]
    assert result.exit_code == 0
    assert result.stdout == "hi\n"


async def test_masks_are_applied_after_system_bind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    secret_file = tmp_path / ".env"
    secret_file.write_text("K=1")
    config = SandboxConfig(mask_paths=[secret_dir, secret_file])
    recorder = patch_spawn(monkeypatch, FakeProcess(b"", b"", 0))
    runner = BwrapRunner(config)

    await runner.run("true", timeout_seconds=5, workspace=Path("/tmp/ws"))

    argv = recorder.argv
    idx_ro = argv.index("--ro-bind")
    tmpfs_idx = argv.index("--tmpfs", argv.index("--tmpfs") + 1)
    assert argv[tmpfs_idx : tmpfs_idx + 2] == ["--tmpfs", str(secret_dir.resolve())]
    i = argv.index("--bind", argv.index("--bind") + 1)
    assert argv[i : i + 3] == ["--bind", "/dev/null", str(secret_file.resolve())]
    assert idx_ro < argv.index(str(secret_dir.resolve()))
    assert idx_ro < argv.index(str(secret_file.resolve()))


async def test_system_root_is_bound_before_proc_and_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = patch_spawn(monkeypatch, FakeProcess(b"", b"", 0))
    runner = BwrapRunner(SandboxConfig())

    await runner.run("true", timeout_seconds=5, workspace=Path("/tmp/ws"))

    argv = recorder.argv
    assert argv.index("--ro-bind") < argv.index("--proc")
    assert argv.index("--ro-bind") < argv.index("--dev")


async def test_network_shares_host_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = patch_spawn(monkeypatch, FakeProcess(b"", b"", 0))
    runner = BwrapRunner(SandboxConfig(network=True))

    await runner.run("true", timeout_seconds=5, workspace=Path("/tmp/ws"))

    argv = recorder.argv
    assert "--share-net" in argv
    assert argv.index("--unshare-all") < argv.index("--share-net")


async def test_network_is_isolated_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = patch_spawn(monkeypatch, FakeProcess(b"", b"", 0))
    runner = BwrapRunner(SandboxConfig())

    await runner.run("true", timeout_seconds=5, workspace=Path("/tmp/ws"))

    assert "--share-net" not in recorder.argv


async def test_writable_binds_are_added_after_system_bind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host = tmp_path / "usr-local"
    config = SandboxConfig(writable_binds={"/usr/local": str(host)})
    recorder = patch_spawn(monkeypatch, FakeProcess(b"", b"", 0))
    runner = BwrapRunner(config)

    await runner.run("true", timeout_seconds=5, workspace=Path("/tmp/ws"))

    argv = recorder.argv
    idx = argv.index("--bind", argv.index("--bind") + 1)
    assert argv[idx : idx + 3] == ["--bind", str(host.resolve()), "/usr/local"]
    assert argv.index("--ro-bind") < idx
    assert host.is_dir()


async def test_stdout_stderr_and_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_spawn(monkeypatch, FakeProcess(b"out\n", b"err\n", 3))
    runner = BwrapRunner(SandboxConfig())

    result = await runner.run("x", timeout_seconds=5, workspace=Path("/tmp/ws"))

    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert result.exit_code == 3


async def test_process_is_own_session(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = patch_spawn(monkeypatch, FakeProcess(b"", b"", 0))
    runner = BwrapRunner(SandboxConfig())
    await runner.run("x", timeout_seconds=5, workspace=Path("/tmp/ws"))
    assert recorder.kwargs.get("start_new_session") is True


async def test_timeout_kills_process_group(
    monkeypatch: pytest.MonkeyPatch, killpg_calls: Any
) -> None:
    patch_spawn(monkeypatch, FakeProcess(b"partial\n", b"", -9, delay=1.5))
    runner = BwrapRunner(SandboxConfig())

    result = await runner.run("sleep 100", timeout_seconds=1, workspace=Path("/tmp/ws"))

    assert killpg_calls == [(9999, int(signal.SIGKILL))]
    assert result.exit_code == -1
    assert "timed out" in result.stderr
    assert result.stdout == "partial\n"
