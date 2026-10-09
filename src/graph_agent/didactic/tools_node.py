"""Didactic archive: hand-written sequential tools node.

What this module is
    The original ``tools_node`` of the ReAct graph. It iterated over the
    ``tool_calls`` carried by the last ``AIMessage`` **one at a time**, enforced
    approval via ``interrupt()``, emitted ``ToolCallEvent``/``ToolResultEvent``
    (or ``FileEvent``) and built the matching ``ToolMessage`` list.

What built-in replaces it
    ``langgraph.prebuilt.ToolNode``. The production graph now builds it with an
    ``awrap_tool_call`` interceptor that only adds our events, approval and
    loop guards; the dispatch loop itself is the built-in.

Why the built-in is preferred
    ``ToolNode`` executes async tool calls **concurrently** (``asyncio.gather``),
    so when the model asks for several independent tools in one turn they no
    longer run back to back. It also handles sync tools, argument validation,
    error-to-message conversion and the ``Command``/``Send`` control-flow cases
    that the hand-written loop ignored. Less code, correct concurrency.

What to look at while reading
    - The sequential ``for`` loop: one round-trip per tool call.
    - Approval is decided *before* the tool executes and *before* the
      ``ToolCallEvent`` is emitted, so a paused run leaks no "call" event.
    - ``FileEvent`` is a side channel: it replaces the tool result message.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import interrupt

from graph_agent.config import AppConfig
from graph_agent.core.state import AgentState, resolve_approval_mode
from graph_agent.events import FileEvent, ToolCallEvent, ToolResultEvent
from graph_agent.tools.base import ToolContext, ToolRegistry


async def run_tools_node(
    state: AgentState,
    registry: ToolRegistry,
    *,
    workspace: Path,
    config: AppConfig,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> list[BaseMessage]:
    """Execute the last message's tool calls sequentially, returning ToolMessages."""
    notify = emit or (lambda payload: None)
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return []

    manual_mode = resolve_approval_mode(state) == "manual"
    outputs: list[BaseMessage] = []

    for tool_call in last_message.tool_calls:
        call_id = str(tool_call["id"])
        name = str(tool_call["name"])
        args = dict(tool_call["args"])
        result: Any
        is_error = False

        try:
            tool = registry.get(name)
        except KeyError:
            notify({"event": ToolCallEvent(tool_call_id=call_id, name=name, args=args)})
            result = f"unknown tool: {name}"
            is_error = True
        else:
            denied = False
            if tool.spec.requires_approval and manual_mode:
                decision = interrupt(
                    {
                        "approval_id": call_id,
                        "tool_call_id": call_id,
                        "name": name,
                        "args": args,
                    }
                )
                approved = (
                    bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
                )
                denied = not approved

            if denied:
                result = "user denied tool execution"
                is_error = True
            else:
                notify({"event": ToolCallEvent(tool_call_id=call_id, name=name, args=args)})
                try:
                    result = await tool.execute(
                        args,
                        ToolContext(
                            session_id=state["session_id"],
                            workspace=workspace,
                            config=config,
                        ),
                    )
                except Exception as exc:
                    result = str(exc)
                    is_error = True

        if isinstance(result, FileEvent):
            notify({"event": result})
            outputs.append(ToolMessage(content=f"file sent: {result.path}", tool_call_id=call_id))
            continue

        notify(
            {
                "event": ToolResultEvent(
                    tool_call_id=call_id,
                    name=name,
                    result=result,
                    is_error=is_error,
                )
            }
        )
        outputs.append(ToolMessage(content=_tool_message_content(result), tool_call_id=call_id))

    return outputs


def _tool_message_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)
