from __future__ import annotations

import asyncio
import contextlib
import os

from graph_agent.config import VoiceChannelConfig
from graph_agent.logging import get_logger

logger = get_logger(__name__)

AMI_TIMEOUT_SECONDS = 10.0


class AmiError(RuntimeError):
    """Raised when the Asterisk Manager Interface rejects an action."""


class AmiClient:
    """Minimal Asterisk Manager Interface client: login + Originate.

    One short-lived TCP connection per outbound call keeps the transport
    dependency-free; the dialplan then bridges the answered leg back to this
    process through AudioSocket.
    """

    def __init__(self, config: VoiceChannelConfig) -> None:
        self._config = config

    async def originate(self, number: str, call_uuid: str) -> str:
        """Ask Asterisk to call ``number`` and bridge it to our AudioSocket server."""
        target = self._audiosocket_target()
        reader, writer = await asyncio.open_connection(self._config.ami_host, self._config.ami_port)
        try:
            await self._read_banner(reader)
            await self._login(reader, writer)
            await self._disable_events(reader, writer)
            await self._send(
                writer,
                {
                    "Action": "Originate",
                    "Channel": f"PJSIP/{number}@{self._config.trunk}",
                    "Application": "AudioSocket",
                    "Data": f"{call_uuid},{target}",
                    "CallerID": self._config.caller_id or "",
                    "Async": "true",
                    "ActionID": call_uuid,
                },
            )
            response = await self._read_response(reader, "Originate")
            if response.get("Response") != "Success":
                raise AmiError(response.get("Message") or "Originate failed")
            return response.get("Message", "originate queued")
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    def _audiosocket_target(self) -> str:
        host = (self._config.advertise_host or "").strip()
        if not host:
            raise AmiError("channels.voice.advertise_host is required for outbound calls")
        return f"{host}:{self._config.port}"

    def _secret(self) -> str:
        secret = os.environ.get(self._config.ami_secret_env, "")
        if not secret:
            raise AmiError(f"AMI secret not set (env {self._config.ami_secret_env})")
        return secret

    async def _read_banner(self, reader: asyncio.StreamReader) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(reader.readline(), timeout=AMI_TIMEOUT_SECONDS)

    async def _login(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await self._send(
            writer,
            {
                "Action": "Login",
                "Username": self._config.ami_username,
                "Secret": self._secret(),
            },
        )
        response = await self._read_response(reader, "Login")
        if response.get("Response") != "Success":
            raise AmiError(response.get("Message") or "AMI login failed")

    async def _disable_events(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await self._send(writer, {"Action": "Events", "EventMask": "off"})
        await self._read_response(reader, "Events")

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, fields: dict[str, str]) -> None:
        lines = "".join(f"{key}: {value}\r\n" for key, value in fields.items())
        writer.write(f"{lines}\r\n".encode())
        await writer.drain()

    async def _read_response(self, reader: asyncio.StreamReader, action: str) -> dict[str, str]:
        while True:
            try:
                block = await asyncio.wait_for(
                    self._read_block(reader), timeout=AMI_TIMEOUT_SECONDS
                )
            except TimeoutError as exc:
                raise AmiError(f"timeout waiting for AMI {action} response") from exc
            if block is None:
                raise AmiError(f"AMI connection closed while waiting for {action}")
            if "Response" in block:
                return block

    @staticmethod
    async def _read_block(reader: asyncio.StreamReader) -> dict[str, str] | None:
        fields: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if not line:
                return fields or None
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not text:
                if fields:
                    return fields
                continue
            key, _, value = text.partition(":")
            fields[key.strip()] = value.strip()
