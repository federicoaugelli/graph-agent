from __future__ import annotations

import asyncio

import pytest

from graph_agent.config import VoiceChannelConfig
from graph_agent.transports.voice.ami import AmiClient, AmiError


async def _read_block(reader: asyncio.StreamReader) -> dict[str, str] | None:
    fields: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if not line:
            return fields or None
        text = line.decode().rstrip("\r\n")
        if not text:
            if fields:
                return fields
            continue
        key, _, value = text.partition(":")
        fields[key.strip()] = value.strip()


class FakeAmiServer:
    def __init__(self) -> None:
        self.actions: list[dict[str, str]] = []
        self._server: asyncio.AbstractServer | None = None

    @property
    def port(self) -> int:
        assert self._server is not None
        sockets = self._server.sockets
        assert sockets
        return int(sockets[0].getsockname()[1])

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(b"Asterisk Call Manager/5.0.4\r\n")
        await writer.drain()
        while True:
            block = await _read_block(reader)
            if block is None:
                break
            self.actions.append(block)
            action = block.get("Action")
            if action == "Login":
                reply = b"Response: Success\r\nMessage: Authentication accepted\r\n\r\n"
            elif action == "Events":
                reply = b"Response: Success\r\n\r\n"
            elif action == "Originate":
                reply = b"Response: Success\r\nMessage: Originate successfully queued\r\n\r\n"
            else:
                reply = b"Response: Error\r\nMessage: unknown action\r\n\r\n"
            writer.write(reply)
            await writer.drain()
        writer.close()


def _config(port: int, secret_env: str = "TEST_AMI_SECRET") -> VoiceChannelConfig:
    return VoiceChannelConfig(
        enabled=True,
        advertise_host="192.0.2.5",
        port=8101,
        trunk="test-trunk",
        caller_id="5550000",
        ami_host="127.0.0.1",
        ami_port=port,
        ami_username="graphagent",
        ami_secret_env=secret_env,
    )


async def test_originate_sends_expected_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_AMI_SECRET", "s3cr3t")
    server = FakeAmiServer()
    await server.start()
    try:
        result = await AmiClient(_config(server.port)).originate("15551234567", "abc-uuid")
    finally:
        await server.stop()

    assert "queued" in result
    login = next(action for action in server.actions if action["Action"] == "Login")
    assert login["Username"] == "graphagent"
    assert login["Secret"] == "s3cr3t"

    originate = next(action for action in server.actions if action["Action"] == "Originate")
    assert originate["Channel"] == "PJSIP/15551234567@test-trunk"
    assert originate["Application"] == "AudioSocket"
    assert originate["Data"] == "abc-uuid,192.0.2.5:8101"
    assert originate["CallerID"] == "5550000"
    assert originate["Async"] == "true"


async def test_originate_requires_advertise_host() -> None:
    with pytest.raises(AmiError):
        await AmiClient(VoiceChannelConfig(ami_secret_env="X")).originate("1", "u")


async def test_login_fails_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE", raising=False)
    server = FakeAmiServer()
    await server.start()
    try:
        with pytest.raises(AmiError):
            await AmiClient(_config(server.port, secret_env="NOPE")).originate("1", "u")
    finally:
        await server.stop()
