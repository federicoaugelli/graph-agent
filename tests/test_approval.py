from __future__ import annotations

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
from graph_agent.tools.base import ToolContext, ToolSpec


class ConfirmTool:
    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="confirm",
            description="Do something dangerous.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            requires_approval=True,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        self.executed.append(args)
        return args["text"]


def tool_call_response(tool_call_id: str, args: dict[str, Any]) -> list[StreamChunk]:
    return [
        StreamChunk(
            tool_calls=[ToolCallRequest(id=tool_call_id, name="confirm", args=args)],
            finish_reason="tool_calls",
        )
    ]


def text_response(text: str) -> list[StreamChunk]:
    return [StreamChunk(delta_text=text, finish_reason="stop")]


async def make_service(
    app_config: AppConfig,
) -> tuple[AgentService, ScriptedLLMBackend, ConfirmTool]:
    tool = ConfirmTool()
    backend = ScriptedLLMBackend(
        [
            tool_call_response("call_1", {"text": "boom"}),
            text_response("end"),
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    service.registry.register(tool)
    return service, backend, tool


async def test_run_pauses_on_approval_request(app_config: AppConfig) -> None:
    service, backend, tool = await make_service(app_config)

    events = [event async for event in service.run("s1", "do it")]

    assert len(events) == 1
    assert isinstance(events[0], ApprovalRequestEvent)
    assert events[0].name == "confirm"
    assert events[0].args == {"text": "boom"}
    assert events[0].approval_id == "call_1"
    assert tool.executed == []
    assert len(backend.calls) == 1

    await service.shutdown()


async def test_resume_approved_executes_tool(app_config: AppConfig) -> None:
    service, backend, tool = await make_service(app_config)
    _ = [event async for event in service.run("s2", "do it")]

    events = [event async for event in service.resume("s2", "call_1", approved=True)]

    assert [type(event) for event in events] == [
        ToolCallEvent,
        ToolResultEvent,
        TokenEvent,
        DoneEvent,
    ]
    assert isinstance(events[1], ToolResultEvent)
    assert events[1].result == "boom"
    assert events[1].is_error is False
    assert isinstance(events[3], DoneEvent)
    assert events[3].final_text == "end"
    assert tool.executed == [{"text": "boom"}]
    assert len(backend.calls) == 2

    await service.shutdown()


async def test_resume_denied_feeds_refusal_to_model(app_config: AppConfig) -> None:
    service, backend, tool = await make_service(app_config)
    _ = [event async for event in service.run("s3", "do it")]

    events = [event async for event in service.resume("s3", "call_1", approved=False)]

    assert [type(event) for event in events] == [ToolResultEvent, TokenEvent, DoneEvent]
    assert isinstance(events[0], ToolResultEvent)
    assert events[0].is_error is True
    assert "denied" in str(events[0].result)
    assert tool.executed == []

    last_call = backend.calls[-1]
    tool_message = last_call[-1]
    assert "denied" in str(tool_message.content)

    await service.shutdown()


async def test_auto_mode_override_skips_interrupt(app_config: AppConfig) -> None:
    service, backend, tool = await make_service(app_config)

    events = [event async for event in service.run("s5", "do it", approval_mode="auto")]

    assert [type(event) for event in events] == [
        ToolCallEvent,
        ToolResultEvent,
        TokenEvent,
        DoneEvent,
    ]
    assert tool.executed == [{"text": "boom"}]
    assert len(backend.calls) == 2

    await service.shutdown()


async def test_set_approval_mode_persists_in_session(app_config: AppConfig) -> None:
    tool = ConfirmTool()
    backend = ScriptedLLMBackend(
        [
            tool_call_response("call_x", {"text": "boom"}),
            text_response("end"),
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    service.registry.register(tool)

    await service.set_approval_mode("s6", "auto")
    events = [event async for event in service.run("s6", "do it")]

    assert not any(isinstance(event, ApprovalRequestEvent) for event in events)
    assert tool.executed == [{"text": "boom"}]

    await service.shutdown()


async def test_session_mode_survives_later_turns(app_config: AppConfig) -> None:
    tool = ConfirmTool()
    backend = ScriptedLLMBackend(
        [
            tool_call_response("call_a", {"text": "first"}),
            text_response("ok a"),
            tool_call_response("call_b", {"text": "second"}),
            text_response("ok b"),
        ]
    )
    service = AgentService(app_config)
    await service.setup(backend)
    service.registry.register(tool)

    await service.set_approval_mode("s7", "auto")

    events1 = [event async for event in service.run("s7", "go a")]
    events2 = [event async for event in service.run("s7", "go b")]

    assert not any(isinstance(event, ApprovalRequestEvent) for event in events1 + events2)
    assert tool.executed == [{"text": "first"}, {"text": "second"}]

    await service.shutdown()


async def test_approval_survives_service_restart(app_config: AppConfig) -> None:
    service1, _, _ = await make_service(app_config)
    events = [event async for event in service1.run("s4", "do it")]
    assert isinstance(events[0], ApprovalRequestEvent)
    await service1.shutdown()

    tool2 = ConfirmTool()
    backend2 = ScriptedLLMBackend([text_response("recovered")])
    service2 = AgentService(app_config)
    await service2.setup(backend2)
    service2.registry.register(tool2)

    events2 = [event async for event in service2.resume("s4", "call_1", approved=True)]

    assert [type(event) for event in events2] == [
        ToolCallEvent,
        ToolResultEvent,
        TokenEvent,
        DoneEvent,
    ]
    assert tool2.executed == [{"text": "boom"}]
    assert isinstance(events2[-1], DoneEvent)
    assert events2[-1].final_text == "recovered"

    await service2.shutdown()
