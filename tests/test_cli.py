from __future__ import annotations

import builtins
from typing import Any

import pytest

from conftest import ScriptedLLMBackend, fake_text_response
from graph_agent.config import AppConfig
from graph_agent.core.service import AgentService
from graph_agent.core.sessions import SessionManager
from graph_agent.events import FileEvent
from graph_agent.models.llm import StreamChunk, ToolCallRequest
from graph_agent.tools.base import ToolContext, ToolSpec
from graph_agent.transports.cli import _render, run_cli


class ConfirmTool:
    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="confirm",
            description="Do something dangerous.",
            parameters={"type": "object", "properties": {}, "required": []},
            requires_approval=True,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        self.executed.append(args)
        return "done"


def scripted_input(monkeypatch: pytest.MonkeyPatch, values: list[str]) -> None:
    iterator = iter(values)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)


async def make_service(app_config: AppConfig, final_texts: str) -> tuple[AgentService, ConfirmTool]:
    tool = ConfirmTool()
    backend = ScriptedLLMBackend(
        [
            [
                StreamChunk(
                    tool_calls=[ToolCallRequest(id="c1", name="confirm", args={})],
                    finish_reason="tool_calls",
                )
            ],
            [StreamChunk(delta_text=final_texts, finish_reason="stop")],
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    service.registry.register(tool)
    return service, tool


async def test_cli_approves_and_resumes(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, tool = await make_service(app_config, "done")
    scripted_input(monkeypatch, ["burn", "y"])

    try:
        await run_cli(service, SessionManager(service), identifier="test")
    finally:
        await service.shutdown()

    assert tool.executed == [{}]


async def test_cli_denies_on_n(app_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    service, tool = await make_service(app_config, "ok never mind")
    scripted_input(monkeypatch, ["burn", "n"])

    try:
        await run_cli(service, SessionManager(service), identifier="test")
    finally:
        await service.shutdown()

    assert tool.executed == []


async def test_cli_new_starts_a_new_thread(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = AgentService(app_config)
    await service.setup(fake_text_response("ok"))
    sessions = SessionManager(service)
    seen: list[str] = []
    original = service.run

    async def spy_run(session_id: str, user_input: str, **kwargs: Any) -> Any:
        seen.append(session_id)
        async for event in original(session_id, user_input, **kwargs):
            yield event

    monkeypatch.setattr(service, "run", spy_run)
    scripted_input(monkeypatch, ["/new", "hello"])

    try:
        await run_cli(service, sessions, identifier="test")
    finally:
        await service.shutdown()

    assert seen == ["cli:test:2"]


def test_render_file_event(capsys: pytest.CaptureFixture[str]) -> None:
    _render(FileEvent(path="/ws/a.txt", caption="ecco"))

    out = capsys.readouterr().out
    assert "/ws/a.txt" in out
    assert "ecco" in out


async def test_cli_eof_at_approval_denies(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, tool = await make_service(app_config, "hello")
    scripted_input(monkeypatch, ["burn"])

    try:
        await run_cli(service, SessionManager(service), identifier="test")
    finally:
        await service.shutdown()

    assert tool.executed == []
