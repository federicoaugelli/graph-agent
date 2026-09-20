from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from graph_agent.bus import EventBus
from graph_agent.config import AppConfig
from graph_agent.core.graph import build_agent_graph
from graph_agent.core.state import AgentState, ApprovalMode
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
from graph_agent.mcp.client import MCPClientManager
from graph_agent.memory import MemoryStore
from graph_agent.models.llm import LLMBackend, OpenAICompatBackend
from graph_agent.skills import SkillStore
from graph_agent.tools import FilesystemTool, SendFileTool, ToolRegistry, WebSearchTool
from graph_agent.tools.shell import ShellTool

_EVENT_TYPES = (
    ApprovalRequestEvent,
    DoneEvent,
    ErrorEvent,
    FileEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class AgentService:
    """The only core API: transports talk exclusively to this class.

    run() yields a stream of events (token, tool_call, approval, done).
    resume() continues a session waiting for approval (LangGraph interrupt).
    """

    def __init__(self, config: AppConfig, bus: EventBus | None = None) -> None:
        self.config = config
        self.bus = bus or EventBus()
        self._graph: CompiledStateGraph | None = None
        self._backend: LLMBackend | None = None
        self._registry: ToolRegistry | None = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._db_conn: aiosqlite.Connection | None = None
        self._memory: MemoryStore | None = None
        self._skills: SkillStore | None = None
        self._mcp: MCPClientManager | None = None
        self._default_system_prompt: str | None = None

    @property
    def registry(self) -> ToolRegistry:
        if self._registry is None:
            raise RuntimeError("service not initialized")
        return self._registry

    @property
    def default_system_prompt(self) -> str | None:
        """Composed system prompt (persona + skills + memory), if the service is set up."""
        return self._default_system_prompt

    async def setup(self, backend: LLMBackend | None = None) -> None:
        """Initialize LLM backend, tool registry, checkpointer and compile the graph."""
        if self._graph is not None:
            return

        backend = backend or OpenAICompatBackend(self.config.models.brain)
        self._backend = backend

        registry = ToolRegistry()
        self._registry = registry

        workspace = Path(self.config.agent.workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        if self.config.memory.enabled:
            self._memory = MemoryStore(workspace, self.config.memory.dir)
            self._memory.ensure()

        if self.config.skills.enabled:
            self._skills = SkillStore(workspace, self.config.skills.dir)
            self._skills.ensure()

        self._default_system_prompt = self._resolve_system_prompt()

        if self.config.tools.filesystem.enabled:
            root = Path(self.config.tools.filesystem.root)
            root.mkdir(parents=True, exist_ok=True)
            registry.register(FilesystemTool(root))
            registry.register(SendFileTool())

        if self.config.tools.web_search.enabled:
            registry.register(WebSearchTool(self.config.tools.web_search))

        if self.config.tools.shell.enabled:
            registry.register(ShellTool(self.config.sandbox))

        if self.config.mcp.servers:
            self._mcp = MCPClientManager(self.config.mcp.servers)
            await self._mcp.connect_all()
            self._mcp.register_tools(registry)

        checkpoint_path = Path(self.config.agent.checkpointer_db)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_conn = await aiosqlite.connect(str(checkpoint_path))
        self._checkpointer = AsyncSqliteSaver(self._db_conn)
        await self._checkpointer.setup()

        self._graph = build_agent_graph(
            backend=backend,
            registry=registry,
            config=self.config,
            max_iterations=self.config.agent.max_iterations,
            checkpointer=self._checkpointer,
        )

    def _resolve_system_prompt(self) -> str | None:
        return self._compose_prompt(self._base_system_prompt())

    def _compose_prompt(self, base: str | None) -> str | None:
        prompt = base
        if self._skills is not None:
            prompt = self._skills.compose(prompt)
        if self._memory is not None:
            prompt = self._memory.compose(prompt)
        return prompt

    def _base_system_prompt(self) -> str | None:
        persona_file = self.config.agent.persona_file
        if persona_file is None:
            return self.config.agent.system_prompt
        if not persona_file.exists():
            raise ValueError(f"persona file not found: {persona_file}")
        return persona_file.read_text()

    async def shutdown(self) -> None:
        if self._db_conn is not None:
            await self._db_conn.close()
            self._db_conn = None

        if self._mcp is not None:
            await self._mcp.shutdown()
            self._mcp = None

        self._graph = None
        self._checkpointer = None
        self._registry = None
        self._backend = None
        self._memory = None
        self._skills = None
        self._default_system_prompt = None

    async def run(
        self,
        session_id: str,
        user_input: str | list[dict[str, Any]],
        *,
        approval_mode: ApprovalMode | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[Event]:
        if self._graph is None:
            await self.setup()

        input_state: AgentState = {
            "messages": [HumanMessage(content=cast(Any, user_input))],
            "session_id": session_id,
            "iterations": 0,
            "pending_approval": None,
        }
        if approval_mode is not None:
            input_state["approval_mode"] = approval_mode
        prompt: str | None
        if system_prompt is not None:
            prompt = self._compose_prompt(system_prompt)
        else:
            prompt = self._default_system_prompt
        if prompt is not None:
            input_state["system_prompt"] = prompt

        async for event in self._stream_graph(input_state, session_id):
            yield event

    async def set_approval_mode(self, session_id: str, mode: ApprovalMode) -> None:
        """Persist the approval mode for a session (survives restarts)."""
        if self._graph is None:
            await self.setup()
        graph = self._graph
        if graph is None:
            raise RuntimeError("service not initialized")

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        await graph.aupdate_state(config, {"approval_mode": mode}, as_node="__start__")

    async def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and pending writes for a thread (hard reset)."""
        if self._db_conn is None:
            return
        await self._db_conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        await self._db_conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        await self._db_conn.commit()

    async def resume(
        self, session_id: str, approval_id: str, approved: bool
    ) -> AsyncIterator[Event]:
        if self._graph is None:
            await self.setup()

        command: Command[Any] = Command(
            resume={
                "approval_id": approval_id,
                "approved": approved,
            }
        )

        async for event in self._stream_graph(command, session_id):
            yield event

    async def pending_approval(self, session_id: str) -> ApprovalRequestEvent | None:
        """Return the approval event currently blocking a session, if any.

        Used by transports (Phase 3) to expose/validate approvals without
        re-running the graph.
        """
        graph = self._graph
        if graph is None:
            await self.setup()
            graph = self._graph
            if graph is None:
                raise RuntimeError("service not initialized")

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        snapshot = await graph.aget_state(config)

        for item in snapshot.interrupts:
            return _approval_event(item)
        return None

    async def _stream_graph(self, graph_input: Any, session_id: str) -> AsyncIterator[Event]:
        graph = self._graph
        if graph is None:
            raise RuntimeError("service not initialized")

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        done_seen = False

        async for chunk in graph.astream(graph_input, config=config, stream_mode="custom"):
            if not isinstance(chunk, dict):
                continue

            event = chunk.get("event")
            if not isinstance(event, _EVENT_TYPES):
                continue

            if isinstance(event, DoneEvent):
                done_seen = True

            await self.bus.publish(f"session:{session_id}", event)
            yield event

        if done_seen:
            return

        snapshot = await graph.aget_state(config)

        if snapshot.interrupts:
            for item in snapshot.interrupts:
                event = _approval_event(item)
                await self.bus.publish(f"session:{session_id}", event)
                yield event
            return

        final_text = _last_assistant_text(snapshot.values)
        done_event = DoneEvent(session_id=session_id, final_text=final_text)
        await self.bus.publish(f"session:{session_id}", done_event)
        yield done_event


def _approval_event(item: Any) -> ApprovalRequestEvent:
    payload = item.value if isinstance(item.value, dict) else {}
    args = payload.get("args")
    return ApprovalRequestEvent(
        approval_id=str(payload.get("approval_id") or item.id),
        tool_call_id=str(payload.get("tool_call_id") or ""),
        name=str(payload.get("name") or ""),
        args=dict(args) if isinstance(args, dict) else {},
    )


def _last_assistant_text(values: Any) -> str | None:
    if not isinstance(values, dict):
        return None

    messages = values.get("messages")
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            return str(content)

    return None
