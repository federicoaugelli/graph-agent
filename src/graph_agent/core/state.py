from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict, cast

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

ApprovalMode = Literal["manual", "auto"]

DEFAULT_APPROVAL_MODE: ApprovalMode = "manual"


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    iterations: int
    pending_approval: dict[str, Any] | None
    approval_mode: ApprovalMode
    system_prompt: str | None


def resolve_approval_mode(state: Any) -> ApprovalMode:
    mode = state.get("approval_mode") or DEFAULT_APPROVAL_MODE
    return cast(ApprovalMode, mode)
