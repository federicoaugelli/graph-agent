from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from conftest import MINIMAL_CONFIG_YAML, fake_text_response
from graph_agent.config import AppConfig, load_config
from graph_agent.core.service import AgentService
from graph_agent.skills import SKILL_FILENAME, Skill, SkillStore


def write_skill(
    root: Path,
    folder: str,
    *,
    name: str,
    description: str,
    body: str = "Detailed instructions.",
) -> Path:
    path = root / "skills" / folder / SKILL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")
    return path


def test_ensure_creates_skills_directory(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)

    store.ensure()

    assert store.directory_path == tmp_path / "skills"
    assert store.directory_path.is_dir()


def test_discover_empty_without_directory(tmp_path: Path) -> None:
    assert SkillStore(tmp_path).discover() == []


def test_discover_parses_frontmatter(tmp_path: Path) -> None:
    write_skill(tmp_path, "commit", name="commit", description="Write good commits.")

    skills = SkillStore(tmp_path).discover()

    assert skills == [
        Skill(
            name="commit",
            description="Write good commits.",
            relative_path="skills/commit/SKILL.md",
        )
    ]


def test_discover_uses_frontmatter_name_not_folder(tmp_path: Path) -> None:
    write_skill(tmp_path, "commit-skill", name="commit", description="Write good commits.")

    assert SkillStore(tmp_path).discover()[0].name == "commit"


def test_discover_ignores_directories_without_skill_file(tmp_path: Path) -> None:
    (tmp_path / "skills" / "empty").mkdir(parents=True)

    assert SkillStore(tmp_path).discover() == []


def test_discover_ignores_plain_files(tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir(parents=True)
    (tmp_path / "skills" / "notes.txt").write_text("not a skill")

    assert SkillStore(tmp_path).discover() == []


def test_discover_sorts_by_name(tmp_path: Path) -> None:
    write_skill(tmp_path, "zeta", name="zeta", description="Z.")
    write_skill(tmp_path, "alpha", name="alpha", description="A.")

    assert [skill.name for skill in SkillStore(tmp_path).discover()] == ["alpha", "zeta"]


@pytest.mark.parametrize(
    "content",
    [
        "# No frontmatter at all",
        "---\ndescription: Missing name.\n---\n",
        "---\nname: foo\n---\n",
        "---\nname: foo\ndescription: Unterminated\n",
    ],
)
def test_discover_errors_on_malformed_frontmatter(tmp_path: Path, content: str) -> None:
    path = tmp_path / "skills" / "broken" / SKILL_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text(content)

    with pytest.raises(ValueError):
        SkillStore(tmp_path).discover()


def test_instructions_lists_name_description_and_path(tmp_path: Path) -> None:
    write_skill(tmp_path, "commit", name="commit", description="Write good commits.")

    text = SkillStore(tmp_path).instructions()

    assert "commit" in text
    assert "Write good commits." in text
    assert "skills/commit/SKILL.md" in text


def test_instructions_empty_without_skills(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)

    assert store.instructions() == ""
    assert store.compose(None) == ""
    assert store.compose("Base.") == "Base."


def test_compose_appends_index_after_base(tmp_path: Path) -> None:
    write_skill(tmp_path, "commit", name="commit", description="Write good commits.")
    store = SkillStore(tmp_path)

    composed = store.compose("You are a bot.")

    assert composed.startswith("You are a bot.")
    assert store.instructions() in composed


def test_load_config_reads_skills_fields(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        MINIMAL_CONFIG_YAML
        + """
skills:
  enabled: true
  dir: capabilities
"""
    )

    config = load_config(config_file)

    assert config.skills.enabled is True
    assert config.skills.dir == Path("capabilities")


def test_skills_disabled_by_default(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(MINIMAL_CONFIG_YAML)

    assert load_config(config_file).skills.enabled is False


async def test_setup_creates_skills_directory(app_config: AppConfig) -> None:
    app_config.skills.enabled = True
    service = AgentService(app_config)
    await service.setup(fake_text_response("ok"))

    assert (Path(app_config.agent.workspace) / "skills").is_dir()

    await service.shutdown()


async def test_skills_index_injected_without_prompt(app_config: AppConfig) -> None:
    app_config.skills.enabled = True
    write_skill(
        Path(app_config.agent.workspace),
        "commit",
        name="commit",
        description="Write good commits.",
    )
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello")]

    first = backend.calls[-1][0]
    assert isinstance(first, SystemMessage)
    assert "skills/commit/SKILL.md" in first.content

    await service.shutdown()


async def test_default_prompt_composed_with_skills(app_config: AppConfig) -> None:
    app_config.skills.enabled = True
    app_config.agent.system_prompt = "Base prompt."
    write_skill(
        Path(app_config.agent.workspace),
        "commit",
        name="commit",
        description="Write good commits.",
    )
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello")]

    first = backend.calls[-1][0]
    assert isinstance(first, SystemMessage)
    assert first.content.startswith("Base prompt.")
    assert "skills/commit/SKILL.md" in first.content

    await service.shutdown()


async def test_run_override_still_gets_skills(app_config: AppConfig) -> None:
    app_config.skills.enabled = True
    app_config.agent.system_prompt = "Default prompt."
    write_skill(
        Path(app_config.agent.workspace),
        "commit",
        name="commit",
        description="Write good commits.",
    )
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello", system_prompt="Override.")]

    first = backend.calls[-1][0]
    assert isinstance(first, SystemMessage)
    assert first.content.startswith("Override.")
    assert "skills/commit/SKILL.md" in first.content

    await service.shutdown()


async def test_skills_disabled_adds_nothing(app_config: AppConfig) -> None:
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello")]

    assert isinstance(backend.calls[-1][0], HumanMessage)
    assert not (Path(app_config.agent.workspace) / "skills").exists()

    await service.shutdown()


async def test_skills_come_before_memory(app_config: AppConfig) -> None:
    app_config.skills.enabled = True
    app_config.memory.enabled = True
    app_config.agent.system_prompt = "Base prompt."
    write_skill(
        Path(app_config.agent.workspace),
        "commit",
        name="commit",
        description="Write good commits.",
    )
    backend = fake_text_response("ok")
    service = AgentService(app_config)
    await service.setup(backend)

    _ = [event async for event in service.run("s1", "hello")]

    content = backend.calls[-1][0].content
    assert (
        content.index("Base prompt.")
        < content.index("commit")
        < content.index("memory/MEMORY.md")
    )

    await service.shutdown()