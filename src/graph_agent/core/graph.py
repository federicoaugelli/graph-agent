from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from graph_agent.config import AppConfig
from graph_agent.core.compaction import compact
from graph_agent.core.messages import INTERRUPTED_TOOL_RESULT, repair_tool_call_pairs
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

        except Exception as exc:
            writer({"event": ErrorEvent(message=str(exc), code="llm_error")})
            return {
                "messages": [AIMessage(content=f"LLM error: {exc}")],
                "iterations": state["iterations"] + 1,
            }

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

    async def tools_node(state: AgentState) -> dict[str, object]:
        writer = get_stream_writer()
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

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
                writer({"event": ToolCallEvent(tool_call_id=call_id, name=name, args=args)})
                result = f"unknown tool: {name}"
                is_error = True
            else:
                denied = False
                if tool.spec.requires_approval and manual_mode:
                    approval_payload = {
                        "approval_id": call_id,
                        "tool_call_id": call_id,
                        "name": name,
                        "args": args,
                    }
                    decision = interrupt(approval_payload)

                    if isinstance(decision, dict):
                        approved = bool(decision.get("approved"))
                    else:
                        approved = bool(decision)

                    if not approved:
                        denied = True

                if denied:
                    result = "user denied tool execution"
                    is_error = True
                else:
                    writer(
                        {
                            "event": ToolCallEvent(
                                tool_call_id=call_id,
                                name=name,
                                args=args,
                            )
                        }
                    )
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
                writer({"event": result})
                outputs.append(
                    ToolMessage(content=f"file sent: {result.path}", tool_call_id=call_id)
                )
                continue

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
            outputs.append(ToolMessage(content=_tool_message_content(result), tool_call_id=call_id))

        return {"messages": outputs}

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


def _tool_message_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)
