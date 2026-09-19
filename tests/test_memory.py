from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from conftest import MINIMAL_CONFIG_YAML, fake_text_response
from graph_agent.config import AppConfig, load_config
from graph_agent.core.service import AgentService
from graph_agent.memory import MEMORY_FILENAME, MemoryStore


def test_ensure_creates_memory_file(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)

    store.ensure()

    assert store.path == tmp_path / "memory" / MEMORY_FILENAME
    assert store.path.exists()
    assert store.path.read_text().strip()


def test_ensure_preserves_existing_content(tmp_path: Path) -> None:
    path = tmp_path / "memory" / MEMORY_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text("existing notes")

    MemoryStore(tmp_path).ensure()

    assert path.read_text() == "existing notes"


def test_instructions_mention_path_and_hygiene(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.ensure()

    text = store.instructions()

    assert store.relative_path == "memory/MEMORY.md"
    assert store.relative_path in text
    lowered = text.lower()
    assert "read" in lowered
    assert "update" in lowered


def test_compose_appends_instructions_after_base(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.ensure()

    composed = store.compose("You are a bot.")

    assert composed.startswith("You are a bot.")
    assert store.instructions() in composed


def test_compose_without_base_is_just_instructions(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.ensure()

    assert store.compose(None) == store.instructions()
    assert store.compose("") == store.instructions()


def test_load_config_reads_memory_fields(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        MINIMAL_CONFIG_YAML
        + """
memory:
  enabled: true
  dir: notes
"""
    )

    config = load_config(config_file)

    assert config.memory.enabled is True
    assert config.memory.dir == Path("notes")


def test_memory_disabled_by_default(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(MINIMAL_CONFIG_YAML)

    assert load_config(config_file).memory.enabled is False


async def test_setup_creates_memory_file(app_config: AppConfig) -> None:
    app_config.memory.enabled = True
    service = AgentService(app_config)
    await service.setup(fake_text_response("ok"))

    memory_file = Path(app_config.agent.workspace) / "memory" / MEMORY_FILENAME
    assert memory_file.exists()

    await service.shutdown()


async def test_memory_instructions_injected_without_prompt(app_config: AppConfig) -> None:
    app_config.memory.enabled = True
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello")]

    first = backend.calls[-1][0]
    assert isinstance(first, SystemMessage)
    assert "memory/MEMORY.md" in first.content

    await service.shutdown()


async def test_default_prompt_composed_with_memory(app_config: AppConfig) -> None:
    app_config.memory.enabled = True
    app_config.agent.system_prompt = "Base prompt."
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello")]

    first = backend.calls[-1][0]
    assert isinstance(first, SystemMessage)
    assert first.content.startswith("Base prompt.")
    assert "memory/MEMORY.md" in first.content

    await service.shutdown()


async def test_run_override_still_gets_memory_instructions(app_config: AppConfig) -> None:
    app_config.memory.enabled = True
    app_config.agent.system_prompt = "Default prompt."
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello", system_prompt="Override.")]

    first = backend.calls[-1][0]
    assert isinstance(first, SystemMessage)
    assert first.content.startswith("Override.")
    assert "memory/MEMORY.md" in first.content

    await service.shutdown()


async def test_memory_disabled_adds_nothing(app_config: AppConfig) -> None:
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello")]

    assert isinstance(backend.calls[-1][0], HumanMessage)
    assert not (Path(app_config.agent.workspace) / "memory").exists()

    await service.shutdown()