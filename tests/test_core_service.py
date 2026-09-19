from __future__ import annotations

from conftest import ScriptedLLMBackend
from graph_agent.config import AppConfig
from graph_agent.core.service import AgentService
from graph_agent.events import DoneEvent, TokenEvent
from graph_agent.models.llm import StreamChunk


async def test_service_run_yields_done_event(app_config: AppConfig) -> None:
    backend = ScriptedLLMBackend([[StreamChunk(delta_text="answer", finish_reason="stop")]])
    service = AgentService(app_config)
    await service.setup(backend)

    events = [event async for event in service.run("s1", "hello")]

    assert isinstance(events[0], TokenEvent)
    assert events[0].delta == "answer"
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].final_text == "answer"

    await service.shutdown()


async def test_service_persists_history_with_checkpointer(app_config: AppConfig) -> None:
    backend1 = ScriptedLLMBackend([[StreamChunk(delta_text="first", finish_reason="stop")]])
    service1 = AgentService(app_config)
    await service1.setup(backend1)
    _ = [event async for event in service1.run("shared", "one")]
    await service1.shutdown()

    backend2 = ScriptedLLMBackend([[StreamChunk(delta_text="second", finish_reason="stop")]])
    service2 = AgentService(app_config)
    await service2.setup(backend2)
    _ = [event async for event in service2.run("shared", "two")]
    await service2.shutdown()

    contents = [message.content for message in backend2.calls[-1]]
    assert "one" in contents
    assert "first" in contents
    assert "two" in contents
