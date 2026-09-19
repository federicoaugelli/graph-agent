from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


def _new_id() -> str:
    return uuid4().hex


@dataclass(slots=True)
class TokenEvent:
    delta: str
    event_id: str = field(default_factory=_new_id)


@dataclass(slots=True)
class ToolCallEvent:
    tool_call_id: str
    name: str
    args: dict[str, Any]
    event_id: str = field(default_factory=_new_id)


@dataclass(slots=True)
class ToolResultEvent:
    tool_call_id: str
    name: str
    result: Any
    is_error: bool = False
    event_id: str = field(default_factory=_new_id)


@dataclass(slots=True)
class ApprovalRequestEvent:
    approval_id: str
    tool_call_id: str
    name: str
    args: dict[str, Any]
    event_id: str = field(default_factory=_new_id)


@dataclass(slots=True)
class FileEvent:
    path: str
    caption: str | None = None
    event_id: str = field(default_factory=_new_id)


@dataclass(slots=True)
class ErrorEvent:
    message: str
    code: str | None = None
    event_id: str = field(default_factory=_new_id)


@dataclass(slots=True)
class DoneEvent:
    session_id: str
    final_text: str | None = None
    event_id: str = field(default_factory=_new_id)


Event = (
    TokenEvent
    | ToolCallEvent
    | ToolResultEvent
    | ApprovalRequestEvent
    | FileEvent
    | ErrorEvent
    | DoneEvent
)
