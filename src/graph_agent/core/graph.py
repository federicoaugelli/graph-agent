from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command, interrupt

from graph_agent.config import AppConfig
from graph_agent.core.compaction import compact
from graph_agent.core.messages import INTERRUPTED_TOOL_RESULT, repair_tool_call_pairs
from graph_agent.core.metrics import RunMetrics, current_metrics
from graph_agent.core.state import AgentState, resolve_approval_mode
from graph_agent.events import ErrorEvent, FileEvent, TokenEvent, ToolCallEvent, ToolResultEvent
from graph_agent.models.llm import LLMBackend
from graph_agent.tools.base import ToolContext, ToolRegistry


def build_agent_graph(
    backend: LLMBackend,
    registry: ToolRegistry,
    config: AppConfig,
    max_iterations: int | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    resolved_max_iterations = (
        max_iterations if max_iterations is not None else config.agent.max_iterations
    )
    resolved_token_limit = config.agent.context_token_limit
    resolved_keep = config.agent.compaction_keep_messages
    output_limit = config.agent.max_tool_output_chars
    repeat_limit = config.agent.repeat_tool_call_limit
    workspace = Path(config.agent.workspace)

    async def compact_node(state: AgentState) -> dict[str, object]:
        replacement = await compact(
            backend,
            list(state["messages"]),
            token_limit=resolved_token_limit,
            keep=resolved_keep,
        )
        if replacement is None:
            return {}
        return {"messages": replacement}

    async def agent_node(state: AgentState) -> dict[str, object]:
        started = time.perf_counter()
        metrics = current_metrics()
        writer = get_stream_writer()
        text_parts: list[str] = []
        tool_calls: list[Any] = []

        messages = repair_tool_call_pairs(list(state["messages"]))

        system_prompt = state.get("system_prompt")
        if system_prompt:
            messages = [SystemMessage(content=system_prompt), *messages]

        try:
            async for chunk in backend.astream(
                messages,
                tools=registry.to_openai_schema(),
            ):
                if chunk.delta_text:
                    text_parts.append(chunk.delta_text)
                    writer({"event": TokenEvent(delta=chunk.delta_text)})

                if chunk.tool_calls:
                    tool_calls = list(chunk.tool_calls)

                if metrics is not None:
                    if chunk.prompt_tokens is not None:
                        metrics.prompt_tokens += chunk.prompt_tokens
                    if chunk.completion_tokens is not None:
                        metrics.completion_tokens += chunk.completion_tokens

        except Exception as exc:
            _record_llm(metrics, started)
            writer({"event": ErrorEvent(message=str(exc), code="llm_error")})
            return {
                "messages": [AIMessage(content=f"LLM error: {exc}")],
                "iterations": state["iterations"] + 1,
            }

        _record_llm(metrics, started)

        ai_message = AIMessage(
            content="".join(text_parts),
            tool_calls=[
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "args": tool_call.args,
                    "type": "tool_call",
                }
                for tool_call in tool_calls
            ],
        )

        return {
            "messages": [ai_message],
            "iterations": state["iterations"] + 1,
        }

    async def execute_tool_call(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Run one tool call, emitting events and enforcing approval/loop guards.

        Executes the registry tool directly (``execute`` is unused: ``ToolNode``
        only provides the parallel scheduling), so file events and error results
        keep the same contract as the hand-written tools node.
        """
        writer = get_stream_writer()
        metrics = current_metrics()

        tool_call = request.tool_call
        call_id = str(tool_call["id"])
        name = str(tool_call["name"])
        args = dict(tool_call.get("args") or {})
        state = request.state if isinstance(request.state, dict) else {}
        manual_mode = resolve_approval_mode(state) == "manual"

        try:
            tool = registry.get(name)
        except KeyError:
            writer({"event": ToolCallEvent(tool_call_id=call_id, name=name, args=args)})
            result = f"unknown tool: {name}"
            writer(
                {
                    "event": ToolResultEvent(
                        tool_call_id=call_id, name=name, result=result, is_error=True
                    )
                }
            )
            return ToolMessage(content=result, tool_call_id=call_id)

        if tool.spec.requires_approval and manual_mode:
            approval_payload = {
                "approval_id": call_id,
                "tool_call_id": call_id,
                "name": name,
                "args": args,
            }
            decision = interrupt(approval_payload)
            approved = (
                bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
            )
            if not approved:
                result = "user denied tool execution"
                writer(
                    {
                        "event": ToolResultEvent(
                            tool_call_id=call_id, name=name, result=result, is_error=True
                        )
                    }
                )
                return ToolMessage(content=result, tool_call_id=call_id)

        if repeat_limit > 0 and metrics is not None:
            signature = (name, json.dumps(args, sort_keys=True, default=str))
            count = metrics.signatures.get(signature, 0) + 1
            metrics.signatures[signature] = count
            if count > repeat_limit:
                result = (
                    f"repeated tool call: {name} was already called {count - 1} times with "
                    "identical arguments. Do not repeat it; change approach or answer now."
                )
                writer(
                    {
                        "event": ToolResultEvent(
                            tool_call_id=call_id, name=name, result=result, is_error=True
                        )
                    }
                )
                return ToolMessage(content=result, tool_call_id=call_id)

        writer({"event": ToolCallEvent(tool_call_id=call_id, name=name, args=args)})
        if metrics is not None:
            metrics.tool_calls += 1

        started = time.perf_counter()
        is_error = False
        try:
            result = await tool.execute(
                args,
                ToolContext(
                    session_id=str(state.get("session_id", "")),
                    workspace=workspace,
                    config=config,
                ),
            )
        except Exception as exc:
            result = str(exc)
            is_error = True
        finally:
            if metrics is not None:
                metrics.tool_ms += (time.perf_counter() - started) * 1000.0

        if isinstance(result, FileEvent):
            writer({"event": result})
            return ToolMessage(content=f"file sent: {result.path}", tool_call_id=call_id)

        writer(
            {
                "event": ToolResultEvent(
                    tool_call_id=call_id,
                    name=name,
                    result=result,
                    is_error=is_error,
                )
            }
        )
        return ToolMessage(
            content=_tool_message_content(result, output_limit), tool_call_id=call_id
        )

    async def limit_node(state: AgentState) -> dict[str, object]:
        writer = get_stream_writer()
        message = (
            f"I stopped after reaching the maximum of {resolved_max_iterations} iterations "
            "without producing a final answer: the last requested tool calls were not executed. "
            "Refine the request, or raise agent.max_iterations."
        )
        writer({"event": TokenEvent(delta=message)})
        outputs: list[BaseMessage] = []
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                outputs.append(
                    ToolMessage(
                        content=INTERRUPTED_TOOL_RESULT,
                        tool_call_id=str(tool_call["id"]),
                    )
                )
        outputs.append(AIMessage(content=message))
        return {"messages": outputs}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            if state["iterations"] >= resolved_max_iterations:
                return "limit"
            return "tools"

        return "end"

    tools_node = ToolNode([], awrap_tool_call=execute_tool_call, name="tools")

    builder = StateGraph(AgentState)
    builder.add_node("compact", compact_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("limit", limit_node)
    builder.add_edge(START, "compact")
    builder.add_edge("compact", "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "limit": "limit", "end": END},
    )
    builder.add_edge("tools", "compact")
    builder.add_edge("limit", END)

    return builder.compile(checkpointer=checkpointer)


def _record_llm(metrics: RunMetrics | None, started: float) -> None:
    if metrics is not None:
        metrics.llm_calls += 1
        metrics.llm_ms += (time.perf_counter() - started) * 1000.0


def _tool_message_content(result: Any, max_chars: int) -> str:
    content = (
        result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    )
    if max_chars > 0 and len(content) > max_chars:
        return f"{content[:max_chars]}\n...[truncated {len(content) - max_chars} chars]"
    return content
