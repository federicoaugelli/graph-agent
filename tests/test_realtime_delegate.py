from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, cast

from graph_agent.events import DoneEvent, TokenEvent, ToolResultEvent
from graph_agent.models.realtime import RealtimeCapabilities
from graph_agent.transports.realtime.delegate import (
    DELEGATE_TOOL_NAME,
    DelegationSupervisor,
    SessionContext,
    _delegate_call,
    _DelegateCall,
    _normalize_turn_detection,
    _parse_prompt,
)


class FakeBackend:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)


class StubService:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.calls: list[Any] = []

    def run(
        self, session_id: str, messages: Any, *, approval_mode: str | None = None
    ) -> AsyncIterator[Any]:
        self.calls.append((session_id, messages, approval_mode))

        async def gen() -> AsyncIterator[Any]:
            for event in self.events:
                yield event

        return gen()


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def _supervisor(
    service: Any,
    backend: FakeBackend,
    *,
    clock: FakeClock | None = None,
    capabilities: RealtimeCapabilities | None = None,
) -> DelegationSupervisor:
    return DelegationSupervisor(
        cast(Any, service),
        cast(Any, backend),
        "s",
        clock=clock or FakeClock(),
        capabilities=capabilities,
    )


def _is_progress(event: dict[str, Any]) -> bool:
    return event.get("type") == "response.create" and "still working" in (
        event.get("response", {}).get("instructions") or ""
    )


def test_session_context_records_both_roles_and_drains() -> None:
    state = SessionContext()
    state.record_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "ciao",
        }
    )
    state.record_event({"type": "response.output_audio_transcript.done", "transcript": "ehi"})

    messages = state.take_context()

    assert [message.type for message in messages] == ["human", "ai"]
    assert [str(message.content) for message in messages] == ["ciao", "ehi"]
    assert state.take_context() == []


def test_session_context_bounds_buffer() -> None:
    state = SessionContext()
    for index in range(100):
        state.record_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": f"m{index}",
            }
        )
    assert len(state.buffer) == 40
    assert str(state.buffer[-1].content) == "m99"


def test_session_context_ignores_empty_and_other_events() -> None:
    state = SessionContext()
    state.record_event({"type": "response.output_audio.delta", "delta": "AAAA"})
    state.record_event({"type": "response.output_text.done", "text": "  "})
    assert state.buffer == []


def test_session_context_revision_helpers() -> None:
    state = SessionContext()
    assert state.is_current(0)
    revision = state.next_revision()
    assert revision == 1
    assert state.is_current(1)
    assert not state.is_current(0)


def test_delegate_call_from_output_item() -> None:
    call = _delegate_call(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": DELEGATE_TOOL_NAME,
                "call_id": "c9",
                "arguments": json.dumps({"prompt": "x"}),
            },
        }
    )
    assert call is not None
    assert call.call_id == "c9"

    empty = _delegate_call(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": DELEGATE_TOOL_NAME,
                "call_id": "c9",
                "arguments": "",
            },
        }
    )
    assert empty is None


def test_delegate_call_from_arguments_done() -> None:
    call = _delegate_call(
        {
            "type": "response.function_call_arguments.done",
            "name": DELEGATE_TOOL_NAME,
            "call_id": "c1",
            "arguments": json.dumps({"prompt": "x"}),
        }
    )
    assert call == _DelegateCall("c1", json.dumps({"prompt": "x"}))


def test_parse_prompt_variants() -> None:
    assert _parse_prompt(json.dumps({"prompt": " a "})) == "a"
    assert _parse_prompt(json.dumps({"task": "b"})) == "b"
    assert _parse_prompt("not-json") == ""
    assert _parse_prompt(json.dumps({"prompt": "  "})) == ""


def test_normalize_turn_detection_downgrades_for_limited_backend() -> None:
    limited = RealtimeCapabilities(supports_semantic_vad=False)
    beta = {"turn_detection": {"type": "semantic_vad"}}
    ga = {"audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}}

    _normalize_turn_detection(beta, limited)
    _normalize_turn_detection(ga, limited)

    assert beta["turn_detection"]["type"] == "server_vad"
    assert ga["audio"]["input"]["turn_detection"]["type"] == "server_vad"


def test_normalize_turn_detection_keeps_semantic_when_supported() -> None:
    session = {"turn_detection": {"type": "semantic_vad"}}
    _normalize_turn_detection(session, RealtimeCapabilities())
    assert session["turn_detection"]["type"] == "semantic_vad"


def test_handle_call_with_empty_prompt_reports_error() -> None:
    backend = FakeBackend()
    supervisor = _supervisor(StubService([]), backend)

    asyncio.run(supervisor.handle_call(_DelegateCall("c1", "not-json")))

    output = backend.sent[0]["item"]["output"]
    assert "non-empty" in output


def test_finish_drops_stale_revision() -> None:
    backend = FakeBackend()
    supervisor = _supervisor(StubService([DoneEvent(session_id="s", final_text="ok")]), backend)
    supervisor.context.revision = 2

    asyncio.run(supervisor._finish("c1", revision=1, prompt="p", context=[]))

    assert backend.sent == []


def test_handle_call_supersedes_previous_delegate() -> None:
    backend = FakeBackend()
    supervisor = _supervisor(StubService([DoneEvent(session_id="s", final_text="ok")]), backend)
    slow_task: asyncio.Task[None] | None = None

    async def slow() -> None:
        await asyncio.Event().wait()

    async def scenario() -> None:
        nonlocal slow_task
        slow_task = asyncio.create_task(slow())
        supervisor.context.active = slow_task
        await supervisor.handle_call(_DelegateCall("c2", json.dumps({"prompt": "nuovo"})))
        active = supervisor.context.active
        assert active is not None
        await active
        await supervisor.aclose()

    asyncio.run(scenario())

    assert slow_task is not None and slow_task.cancelled()
    output = next(event for event in backend.sent if event["type"] == "conversation.item.create")
    assert output["item"]["output"] == "ok"


def test_run_brain_emits_progress_on_tool_result() -> None:
    backend = FakeBackend()
    clock = FakeClock(start=1000.0)
    service = StubService(
        [
            ToolResultEvent(tool_call_id="t", name="web_search", result="x"),
            TokenEvent(delta="fatto"),
            DoneEvent(session_id="s", final_text="fatto"),
        ]
    )
    supervisor = _supervisor(service, backend, clock=clock)
    supervisor._last_progress = 0.0

    result = asyncio.run(supervisor._run_brain("p", []))

    assert result == "fatto"
    assert any(_is_progress(event) for event in backend.sent)


def test_progress_is_throttled() -> None:
    backend = FakeBackend()
    clock = FakeClock(start=100.0)
    supervisor = _supervisor(StubService([]), backend, clock=clock)
    supervisor._last_progress = 100.0

    asyncio.run(supervisor._maybe_progress())
    assert backend.sent == []

    clock.t = 106.0
    asyncio.run(supervisor._maybe_progress())
    assert len(backend.sent) == 1

    asyncio.run(supervisor._maybe_progress())
    assert len(backend.sent) == 1


def test_progress_suppressed_without_capability() -> None:
    backend = FakeBackend()
    clock = FakeClock(start=100.0)
    supervisor = _supervisor(
        StubService([]),
        backend,
        clock=clock,
        capabilities=RealtimeCapabilities(supports_progress=False),
    )
    supervisor._last_progress = 0.0

    asyncio.run(supervisor._maybe_progress())

    assert backend.sent == []


def test_speech_started_sends_cancel_when_supported() -> None:
    backend = FakeBackend()
    supervisor = _supervisor(StubService([]), backend)

    async def scenario() -> None:
        supervisor.context.active = asyncio.create_task(asyncio.Event().wait())
        await supervisor.on_speech_started()
        supervisor.context.active.cancel()

    asyncio.run(scenario())

    assert backend.sent == [{"type": "response.cancel"}]


def test_speech_started_skipped_without_capability() -> None:
    backend = FakeBackend()
    supervisor = _supervisor(
        StubService([]),
        backend,
        capabilities=RealtimeCapabilities(supports_cancel=False),
    )

    async def scenario() -> None:
        supervisor.context.active = asyncio.create_task(asyncio.Event().wait())
        await supervisor.on_speech_started()
        supervisor.context.active.cancel()

    asyncio.run(scenario())

    assert backend.sent == []


def test_run_brain_falls_back_to_done_event() -> None:
    service = StubService([DoneEvent(session_id="s", final_text="solo final text")])
    supervisor = _supervisor(service, FakeBackend())

    assert asyncio.run(supervisor._run_brain("ciao", [])) == "solo final text"
