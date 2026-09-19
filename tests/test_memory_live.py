from __future__ import annotations

import os
from pathlib import Path

import pytest

from graph_agent.config import AppConfig, load_config
from graph_agent.core.service import AgentService
from graph_agent.events import DoneEvent
from graph_agent.memory import MEMORY_FILENAME

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_TESTS") != "1",
        reason="live tests need a running model endpoint: RUN_LIVE_TESTS=1",
    ),
]

TOKEN = "ZORBLAX-42"


@pytest.fixture
def live_config(tmp_path: Path) -> AppConfig:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")
    workspace = tmp_path / "workspace"
    config.agent.workspace = workspace
    config.agent.checkpointer_db = tmp_path / "checkpoints.sqlite"
    config.tools.filesystem.root = workspace
    config.tools.web_search.enabled = False
    config.tools.shell.enabled = False
    config.memory.enabled = True
    config.memory.dir = Path("memory")
    return config


def memory_file(config: AppConfig) -> Path:
    return Path(config.agent.workspace) / config.memory.dir / MEMORY_FILENAME


async def test_live_memory_write(live_config: AppConfig) -> None:
    service = AgentService(live_config)
    await service.setup()
    try:
        async for _ in service.run(
            "mem-write",
            "Remember this durable fact about me: my project codename is "
            f"{TOKEN}. Store it in your memory file so you recall it in future sessions.",
        ):
            pass
    finally:
        await service.shutdown()

    content = memory_file(live_config).read_text() if memory_file(live_config).exists() else ""
    assert TOKEN in content, f"model did not persist memory; file was:\n{content}"


async def test_live_memory_recall(live_config: AppConfig) -> None:
    path = memory_file(live_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Memory\n\n- User's project codename is {TOKEN}.\n")

    service = AgentService(live_config)
    await service.setup()
    final: str | None = None
    try:
        async for event in service.run("mem-recall", "What is my project codename?"):
            if isinstance(event, DoneEvent):
                final = event.final_text
    finally:
        await service.shutdown()

    assert final is not None
    assert TOKEN in final