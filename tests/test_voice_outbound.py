from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from graph_agent.config import AppConfig, VoiceChannelConfig
from graph_agent.tools.base import ToolContext
from graph_agent.transports.voice.outbound import CallPhoneTool, OutboundCaller, VoiceSink


class FakeServer:
    def __init__(self) -> None:
        self.pending: dict[str, str | None] = {}

    def register_outbound(self, call_id: str, greeting: str | None) -> None:
        self.pending[call_id] = greeting

    def cancel_outbound(self, call_id: str) -> None:
        self.pending.pop(call_id, None)


class FakeAmi:
    def __init__(self, message: str = "Originate successfully queued") -> None:
        self.calls: list[tuple[str, str]] = []
        self._message = message

    async def originate(self, number: str, call_uuid: str) -> str:
        self.calls.append((number, call_uuid))
        return self._message


class FailingAmi:
    async def originate(self, number: str, call_uuid: str) -> str:
        raise RuntimeError("boom")


def _caller(
    config: VoiceChannelConfig, ami: Any, server: FakeServer | None = None
) -> tuple[OutboundCaller, FakeServer]:
    fake = server or FakeServer()
    return OutboundCaller(cast(Any, fake), config, ami=cast(Any, ami)), fake


async def test_caller_registers_greeting_and_dials_default_target() -> None:
    ami = FakeAmi()
    caller, server = _caller(VoiceChannelConfig(default_target="333", greeting="ciao"), ami)

    result = await caller.call()

    assert "333" in result
    number, call_uuid = ami.calls[0]
    assert number == "333"
    assert server.pending == {call_uuid: "ciao"}


async def test_caller_uses_explicit_number_and_greeting() -> None:
    ami = FakeAmi()
    caller, server = _caller(VoiceChannelConfig(default_target="333"), ami)

    await caller.call("999", "dimmi")

    number, call_uuid = ami.calls[0]
    assert number == "999"
    assert server.pending == {call_uuid: "dimmi"}


async def test_caller_cleans_up_pending_on_failure() -> None:
    caller, server = _caller(VoiceChannelConfig(default_target="333"), FailingAmi())

    with pytest.raises(RuntimeError):
        await caller.call()

    assert server.pending == {}


async def test_caller_requires_a_target() -> None:
    caller, _ = _caller(VoiceChannelConfig(), FakeAmi())

    with pytest.raises(ValueError):
        await caller.call()


async def test_call_phone_tool_falls_back_to_default_target(minimal_config: AppConfig) -> None:
    ami = FakeAmi()
    caller, server = _caller(VoiceChannelConfig(default_target="333"), ami)
    tool = CallPhoneTool(caller)
    ctx = ToolContext(session_id="s", workspace=Path("."), config=minimal_config)

    result = await tool.execute({}, ctx)

    assert "333" in result
    assert tool.spec.name == "call_phone"
    assert server.pending


async def test_voice_sink_speaks_text() -> None:
    ami = FakeAmi()
    config = VoiceChannelConfig(default_target="333")
    caller, server = _caller(config, ami)
    sink = VoiceSink(caller, config)

    await sink.send("promemoria delle 9")

    assert list(server.pending.values()) == ["promemoria delle 9"]
