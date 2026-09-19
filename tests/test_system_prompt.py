from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from conftest import MINIMAL_CONFIG_YAML, fake_text_response
from graph_agent.config import AppConfig, load_config
from graph_agent.core.service import AgentService


def _write_persona(app_config: AppConfig, content: str) -> Path:
    path = app_config.agent.checkpointer_db.parent / "persona.md"
    path.write_text(content)
    return path


async def test_config_system_prompt_sent_as_first_message(app_config: AppConfig) -> None:
    app_config.agent.system_prompt = "You are Graph, a concise assistant."
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello")]

    first = backend.calls[-1][0]
    assert isinstance(first, SystemMessage)
    assert first.content == "You are Graph, a concise assistant."

    await service.shutdown()


async def test_persona_file_takes_precedence_over_inline(app_config: AppConfig) -> None:
    persona_path = _write_persona(app_config, "Persona da file.")
    app_config.agent.persona_file = persona_path
    app_config.agent.system_prompt = "Ignored inline prompt."
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello")]

    first = backend.calls[-1][0]
    assert isinstance(first, SystemMessage)
    assert first.content == "Persona da file."

    await service.shutdown()


async def test_explicit_run_override_wins_over_config(app_config: AppConfig) -> None:
    app_config.agent.system_prompt = "Default da config."
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello", system_prompt="One-off override.")]

    first = backend.calls[-1][0]
    assert isinstance(first, SystemMessage)
    assert first.content == "One-off override."

    await service.shutdown()


async def test_no_config_prompt_means_no_system_message(app_config: AppConfig) -> None:
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello")]

    assert isinstance(backend.calls[-1][0], HumanMessage)

    await service.shutdown()


async def test_missing_persona_file_fails_setup(app_config: AppConfig) -> None:
    app_config.agent.persona_file = app_config.agent.checkpointer_db.parent / "missing.md"
    service = AgentService(app_config)

    with pytest.raises(ValueError, match="persona"):
        await service.setup(fake_text_response("ok"))

    await service.shutdown()


def test_load_config_reads_agent_prompt_fields(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        MINIMAL_CONFIG_YAML
        + """
agent:
  system_prompt: Prompt from yaml.
  persona_file: persona.md
"""
    )

    config = load_config(config_file)

    assert config.agent.system_prompt == "Prompt from yaml."
    assert config.agent.persona_file == Path("persona.md")
