from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from graph_agent.config import VoiceChannelConfig
from graph_agent.tools.base import ToolContext, ToolSpec
from graph_agent.transports.voice.ami import AmiClient

if TYPE_CHECKING:
    from graph_agent.transports.voice.server import AudioSocketServer


class OutboundCaller:
    """Places outbound calls through Asterisk AMI and pre-arms the greeting.

    The AMI ``Originate`` dials ``PJSIP/<number>@<trunk>`` with
    ``Application=AudioSocket``; when the callee answers, Asterisk reconnects to
    this process and the previously registered greeting makes the brain speak
    first.
    """

    def __init__(
        self,
        server: AudioSocketServer,
        config: VoiceChannelConfig,
        ami: AmiClient | None = None,
    ) -> None:
        self._server = server
        self._config = config
        self._ami = ami or AmiClient(config)

    async def call(self, number: str | None = None, greeting: str | None = None) -> str:
        target = (number or self._config.default_target or "").strip()
        if not target:
            raise ValueError(
                "no number to call: pass 'number' or set channels.voice.default_target"
            )

        call_uuid = str(uuid4())
        self._server.register_outbound(call_uuid, greeting or self._config.greeting)
        try:
            message = await self._ami.originate(target, call_uuid)
        except Exception:
            self._server.cancel_outbound(call_uuid)
            raise
        return f"call to {target} placed ({message})"


class CallPhoneTool:
    """Agent-facing tool that starts an outbound voice conversation."""

    def __init__(self, caller: OutboundCaller) -> None:
        self._caller = caller

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="call_phone",
            description=(
                "Call a phone number and talk to the person through the voice channel. "
                "Use it to proactively reach the user or a third party. If 'number' is "
                "omitted the configured default target is dialed. Use 'greeting' to "
                "control what is said as soon as the callee answers."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "number": {
                        "type": "string",
                        "description": "Phone number to dial (national or international).",
                    },
                    "greeting": {
                        "type": "string",
                        "description": "Instruction for the opening line once answered.",
                    },
                },
                "required": [],
            },
            requires_approval=False,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        number = args.get("number")
        greeting = args.get("greeting")
        return await self._caller.call(
            number if isinstance(number, str) and number.strip() else None,
            greeting if isinstance(greeting, str) and greeting.strip() else None,
        )


class VoiceSink:
    """Scheduler output sink: place a call and speak the produced text.

    Lets a cron/heartbeat job initiate a voice conversation (see ``CronScheduler``).
    """

    def __init__(self, caller: OutboundCaller, config: VoiceChannelConfig) -> None:
        self._caller = caller
        self._config = config

    async def send(self, text: str) -> None:
        await self._caller.call(greeting=text)

    async def send_file(self, path: str, caption: str | None = None) -> None:
        await self._caller.call(greeting=caption or f"I have a file to tell you about: {path}")
