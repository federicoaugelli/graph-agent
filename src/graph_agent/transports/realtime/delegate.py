from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from graph_agent.core.service import AgentService
from graph_agent.events import DoneEvent, ErrorEvent, TokenEvent, ToolResultEvent
from graph_agent.logging import get_logger
from graph_agent.models.realtime import RealtimeBackend, RealtimeCapabilities
from graph_agent.transports.realtime.text import normalize_for_speech

logger = get_logger(__name__)

DELEGATE_TOOL_NAME = "delegate_to_brain"

DELEGATE_INSTRUCTION = (
    "You are the realtime voice front-end of an autonomous agent called the 'brain'. "
    "The brain owns persistent memory, files, tools and multi-step reasoning; you do "
    "not. RULE: whenever the user asks about personal facts, previous context, files, "
    "the web, or asks you to do, remember, find, send or check anything, you MUST call "
    "the `delegate_to_brain` tool with a self-contained prompt instead of answering. "
    "The tool output is the brain's reply: relay it to the user in natural speech. "
    "NEVER say that you cannot remember, access or do something: delegate instead. "
    "Chat directly only for pure small talk that needs neither memory nor tools."
)

DELEGATE_REQUEST_SUFFIX = (
    "\n\n[Reply in plain spoken language for a voice assistant: short conversational "
    "prose. No markdown, no lists, no code blocks, no emojis, no asterisks.]"
)

FILLER_INSTRUCTION = (
    "The main agent is now working on the request. Say one very short filler phrase "
    "in the user's language to acknowledge it (for example 'un attimo', 'ci sto "
    "pensando', 'sto lavorando'), then stop. Do not call any tool and do not answer "
    "the request yet."
)

PROGRESS_INSTRUCTION = (
    "The main agent is still working. Say one very short phrase in the user's language "
    "to reassure them (for example 'ci sto lavorando', 'quasi fatto'), then stop. Do not "
    "call any tool and do not give the answer yet."
)

DELEGATE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": DELEGATE_TOOL_NAME,
    "description": (
        "Delegate a task, question or long-running request to the main agent. "
        "Use it whenever the user needs tools, memory, files, or multi-step reasoning. "
        "Pass a self-contained prompt with all the context the main agent needs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Self-contained instruction for the main agent.",
            }
        },
        "required": ["prompt"],
    },
}

_EMPTY_PROMPT_ERROR = (
    "delegate_to_brain requires a non-empty 'prompt' argument "
    "describing the task for the main agent."
)

_ASSISTANT_TRANSCRIPT_DONE = frozenset(
    {
        "response.output_audio_transcript.done",
        "response.audio_transcript.done",
        "response.output_text.done",
    }
)

_MAX_CONTEXT_MESSAGES = 40
_DEFAULT_PROGRESS_INTERVAL = 6.0


def _filler_response() -> dict[str, Any]:
    return {
        "type": "response.create",
        "response": {"instructions": FILLER_INSTRUCTION, "tool_choice": "none"},
    }


def _progress_response() -> dict[str, Any]:
    return {
        "type": "response.create",
        "response": {"instructions": PROGRESS_INSTRUCTION, "tool_choice": "none"},
    }


def _is_delegate_tool(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    if tool.get("name") == DELEGATE_TOOL_NAME:
        return True
    nested = tool.get("function")
    return isinstance(nested, dict) and nested.get("name") == DELEGATE_TOOL_NAME


def _ensure_delegate_tool(session: dict[str, Any]) -> None:
    tools = session.setdefault("tools", [])
    if isinstance(tools, list) and not any(_is_delegate_tool(tool) for tool in tools):
        tools.append(DELEGATE_TOOL)


def _ensure_delegate_instructions(session: dict[str, Any]) -> None:
    existing = session.get("instructions")
    if not isinstance(existing, str) or not existing:
        session["instructions"] = DELEGATE_INSTRUCTION
    elif DELEGATE_INSTRUCTION not in existing:
        session["instructions"] = f"{existing}\n\n{DELEGATE_INSTRUCTION}"


def _ensure_voice(session: dict[str, Any], voice: str | None) -> None:
    if voice:
        session["voice"] = voice


def _normalize_turn_detection(session: dict[str, Any], capabilities: RealtimeCapabilities) -> None:
    """Downgrade semantic VAD to server VAD for backends that lack it.

    Handles both the beta (``session.turn_detection``) and GA
    (``session.audio.input.turn_detection``) shapes so the same client works
    against Qwen and OpenAI.
    """
    if capabilities.supports_semantic_vad:
        return
    containers: list[dict[str, Any]] = [session]
    audio = session.get("audio")
    if isinstance(audio, dict):
        audio_input = audio.get("input")
        if isinstance(audio_input, dict):
            containers.append(audio_input)
    for container in containers:
        detection = container.get("turn_detection")
        if isinstance(detection, dict) and detection.get("type") == "semantic_vad":
            detection["type"] = "server_vad"


def _delegate_session_update(instructions: str, voice: str | None = None) -> dict[str, Any]:
    session: dict[str, Any] = {"tools": [DELEGATE_TOOL], "instructions": instructions}
    _ensure_voice(session, voice)
    return {"type": "session.update", "session": session}


def _log_session_update(update: dict[str, Any]) -> None:
    session = update.get("session")
    if not isinstance(session, dict):
        return
    tools = [
        tool.get("name")
        for tool in session.get("tools", [])
        if isinstance(tool, dict) and tool.get("name")
    ]
    logger.info(
        "realtime session configured",
        tools=tools,
        instructions_chars=len(session.get("instructions") or ""),
    )


@dataclass(slots=True)
class _DelegateCall:
    call_id: str
    arguments: str


def _delegate_call(event: dict[str, Any]) -> _DelegateCall | None:
    """Extract a delegate invocation across OpenAI/Qwen function-call dialects."""
    etype = event.get("type")
    if etype == "response.function_call_arguments.done":
        if event.get("name") == DELEGATE_TOOL_NAME:
            return _DelegateCall(
                str(event.get("call_id") or ""),
                str(event.get("arguments") or "{}"),
            )
    elif etype == "response.output_item.done":
        item = event.get("item")
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") == DELEGATE_TOOL_NAME
        ):
            arguments = str(item.get("arguments") or "")
            if arguments:
                return _DelegateCall(str(item.get("call_id") or ""), arguments)
    return None


def _parse_prompt(arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return ""
    if not isinstance(args, dict):
        return ""
    return str(args.get("prompt") or args.get("task") or "").strip()


def _text_from_item(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text") or part.get("input_text") or part.get("transcript")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return " ".join(parts)


@dataclass(slots=True)
class SessionContext:
    """Rolling transcript and delegation state for one realtime connection.

    The buffer holds user/assistant turns recorded since the last delegation and is
    drained into the brain, so it complements (never duplicates) the LangGraph
    checkpointer that owns cross-turn memory.
    """

    buffer: list[BaseMessage] = field(default_factory=list)
    handled_calls: set[str] = field(default_factory=set)
    revision: int = 0
    active: asyncio.Task[None] | None = None

    def record_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "conversation.item.input_audio_transcription.completed":
            self._append(HumanMessage, event.get("transcript"))
        elif etype in _ASSISTANT_TRANSCRIPT_DONE:
            self._append(AIMessage, event.get("transcript") or event.get("text"))
        elif etype == "conversation.item.created":
            item = event.get("item")
            if not isinstance(item, dict):
                return
            text = _text_from_item(item)
            if item.get("role") == "user":
                self._append(HumanMessage, text)
            elif item.get("role") == "assistant":
                self._append(AIMessage, text)

    def _append(self, cls: type[BaseMessage], text: Any) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        self.buffer.append(cls(content=text.strip()))
        if len(self.buffer) > _MAX_CONTEXT_MESSAGES:
            del self.buffer[:-_MAX_CONTEXT_MESSAGES]

    def take_context(self) -> list[BaseMessage]:
        messages = self.buffer
        self.buffer = []
        return messages

    def next_revision(self) -> int:
        self.revision += 1
        return self.revision

    def is_current(self, revision: int) -> bool:
        return revision == self.revision


class DelegationSupervisor:
    """Runs the brain on behalf of the voice front-end, Live-style.

    One supervisor per realtime connection. It acknowledges the delegate call,
    supersedes stale work, streams progress while the brain runs and speaks the
    verified result, dropping output that a newer request has made obsolete.
    """

    def __init__(
        self,
        service: AgentService,
        backend: RealtimeBackend,
        session_id: str,
        *,
        capabilities: RealtimeCapabilities | None = None,
        instructions: str | None = None,
        voice: str | None = None,
        progress_interval: float = _DEFAULT_PROGRESS_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.service = service
        self.backend = backend
        self.session_id = session_id
        self.capabilities = capabilities or RealtimeCapabilities()
        self.instructions = instructions
        self.voice = voice
        self.progress_interval = progress_interval
        self.context = SessionContext()
        self._clock = clock
        self._last_progress = clock()

    def record_event(self, event: dict[str, Any]) -> None:
        self.context.record_event(event)

    def session_update(self) -> dict[str, Any]:
        instructions = self.instructions or DELEGATE_INSTRUCTION
        return _delegate_session_update(instructions, self.voice)

    async def handle_call(self, call: _DelegateCall) -> None:
        if call.call_id in self.context.handled_calls:
            return
        self.context.handled_calls.add(call.call_id)

        prompt = _parse_prompt(call.arguments)
        if not prompt:
            await self._send_output(call.call_id, _EMPTY_PROMPT_ERROR)
            return

        await self._cancel_active()
        revision = self.context.next_revision()
        context = self.context.take_context()
        logger.info(
            "delegating to brain",
            call_id=call.call_id,
            revision=revision,
            transcript_messages=len(context),
            prompt_chars=len(prompt),
        )

        await self.backend.send(_filler_response())
        self._last_progress = self._clock()
        task = asyncio.create_task(self._finish(call.call_id, revision, prompt, context))
        self.context.active = task

    async def on_speech_started(self) -> None:
        task = self.context.active
        if task is None or task.done() or not self.capabilities.supports_cancel:
            return
        logger.info("barge-in while delegating, cancelling speech")
        await self.backend.send({"type": "response.cancel"})

    async def aclose(self) -> None:
        await self._cancel_active()

    async def wait(self) -> None:
        task = self.context.active
        if task is None:
            return
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _cancel_active(self) -> None:
        task = self.context.active
        if task is None:
            return
        self.context.active = None
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _finish(
        self, call_id: str, revision: int, prompt: str, context: list[BaseMessage]
    ) -> None:
        try:
            output = await self._run_brain(prompt, context)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("delegate_to_brain failed", call_id=call_id)
            return

        if not self.context.is_current(revision):
            logger.info(
                "dropping stale delegate output",
                call_id=call_id,
                revision=revision,
                current=self.context.revision,
            )
            return
        await self._send_output(call_id, output)

    async def _run_brain(self, prompt: str, context: list[BaseMessage]) -> str:
        messages = [*context, HumanMessage(content=f"{prompt}{DELEGATE_REQUEST_SUFFIX}")]
        parts: list[str] = []
        final_text: str | None = None
        async for event in self.service.run(self.session_id, messages, approval_mode="auto"):
            if isinstance(event, TokenEvent):
                parts.append(event.delta)
            elif isinstance(event, ErrorEvent):
                parts.append(f"[error] {event.message}")
            elif isinstance(event, DoneEvent):
                final_text = event.final_text
            elif isinstance(event, ToolResultEvent):
                await self._maybe_progress()
        text = "".join(parts) or final_text or ""
        return normalize_for_speech(text) or "(no response)"

    async def _maybe_progress(self) -> None:
        if not self.capabilities.supports_progress:
            return
        now = self._clock()
        if now - self._last_progress < self.progress_interval:
            return
        self._last_progress = now
        await self.backend.send(_progress_response())

    async def _send_output(self, call_id: str, output: str) -> None:
        await self.backend.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            }
        )
        await self.backend.send({"type": "response.create"})
