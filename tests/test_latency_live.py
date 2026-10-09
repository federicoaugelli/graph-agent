from __future__ import annotations

import os
from pathlib import Path

import pytest

from graph_agent.config import load_config
from graph_agent.core.service import AgentService
from graph_agent.events import MetricsEvent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_TESTS") != "1",
        reason="live tests need a running model endpoint: RUN_LIVE_TESTS=1",
    ),
]

PROMPT = os.environ.get(
    "GRAPH_AGENT_BENCH_PROMPT",
    "Qual e' la programmazione del festival di Nimes 2026? Cerca sul web e rispondi in breve.",
)


async def test_live_latency_breakdown(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")
    config.agent.latency_metrics = True
    config.agent.workspace = tmp_path / "workspace"
    config.agent.checkpointer_db = tmp_path / "checkpoints.sqlite"
    config.tools.filesystem.root = tmp_path / "fs"

    service = AgentService(config)
    await service.setup()
    try:
        events = [event async for event in service.run("bench", PROMPT, approval_mode="auto")]
    finally:
        await service.shutdown()

    metrics = next(event for event in events if isinstance(event, MetricsEvent))
    print(
        "\n[bench] "
        f"llm_calls={metrics.llm_calls} tool_calls={metrics.tool_calls} "
        f"llm_ms={metrics.llm_duration_ms} tool_ms={metrics.tool_duration_ms} "
        f"total_ms={metrics.total_duration_ms} "
        f"tokens={metrics.prompt_tokens}/{metrics.completion_tokens}"
    )
    assert metrics.llm_calls >= 1
    assert metrics.total_duration_ms > 0
