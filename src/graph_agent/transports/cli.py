from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator

from graph_agent.core.service import AgentService
from graph_agent.core.sessions import SessionManager
from graph_agent.events import (
    ApprovalRequestEvent,
    DoneEvent,
    ErrorEvent,
    Event,
    FileEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

NEW_COMMAND = "/new"


async def run_cli(
    service: AgentService,
    sessions: SessionManager,
    channel: str = "cli",
    identifier: str = "local",
) -> None:
    session_id = await sessions.thread_id(channel, identifier)
    print(f"graph-agent CLI - session '{session_id}' (Ctrl-D to exit, {NEW_COMMAND} to reset)")
    loop = asyncio.get_running_loop()
    while True:
        try:
            user_input = (await loop.run_in_executor(None, lambda: input("you> "))).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input == NEW_COMMAND:
            session_id = await sessions.reset(channel, identifier)
            print(f"[new session] {session_id}")
            continue
        stream = service.run(session_id, user_input)
        await _drain_stream(loop, service, session_id, stream)


async def _drain_stream(
    loop: asyncio.AbstractEventLoop,
    service: AgentService,
    session_id: str,
    stream: AsyncIterator[Event],
) -> None:
    while True:
        approval: ApprovalRequestEvent | None = None
        async for event in stream:
            _render(event)
            if isinstance(event, ApprovalRequestEvent):
                approval = event
        if approval is None:
            return
        approved = await _ask_approval(loop, approval)
        stream = service.resume(session_id, approval.approval_id, approved)


async def _ask_approval(loop: asyncio.AbstractEventLoop, event: ApprovalRequestEvent) -> bool:
    while True:
        try:
            answer = (
                (
                    await loop.run_in_executor(
                        None,
                        lambda: input(f"[approval] run {event.name} {event.args}? [y/N] "),
                    )
                )
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False


def _render(event: object) -> None:
    if isinstance(event, TokenEvent):
        print(event.delta, end="", flush=True)
    elif isinstance(event, ToolCallEvent):
        print(f"\n[tool] {event.name} {event.args}")
    elif isinstance(event, ToolResultEvent):
        print(f"[tool:{'error' if event.is_error else 'ok'}] {event.result}")
    elif isinstance(event, FileEvent):
        caption = f" ({event.caption})" if event.caption else ""
        print(f"\n[file] {event.path}{caption}")
    elif isinstance(event, ApprovalRequestEvent):
        print(f"\n[approval {event.approval_id}] {event.name} {event.args}")
    elif isinstance(event, ErrorEvent):
        print(f"\n[error] {event.message}", file=sys.stderr)
    elif isinstance(event, DoneEvent):
        print()
