from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from graph_agent.config import AppConfig, SchedulerConfig
from graph_agent.events import DoneEvent, Event
from graph_agent.scheduler.cron import CronScheduler, JobSpec
from graph_agent.tools.base import ToolContext
from graph_agent.tools.schedule import ScheduleTool


class FakeScheduler:
    def __init__(self) -> None:
        self.added: list[JobSpec] = []
        self.removed: list[str] = []

    async def add_job(self, job: JobSpec) -> None:
        self.added.append(job)

    async def list_jobs(self) -> list[JobSpec]:
        return list(self.added)

    async def remove_job(self, job_id: str) -> None:
        self.removed.append(job_id)
        self.added = [job for job in self.added if job.job_id != job_id]


class EchoService:
    async def run(self, session_id: str, user_input: str, **kwargs: Any) -> AsyncIterator[Event]:
        yield DoneEvent(session_id=session_id, final_text=f"ran {user_input}")


class RecordingSink:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.files: list[tuple[str, str | None]] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def send_file(self, path: str, caption: str | None = None) -> None:
        self.files.append((path, caption))


@pytest.fixture
def ctx(tmp_path: Path, minimal_config: AppConfig) -> ToolContext:
    return ToolContext(session_id="test", workspace=tmp_path, config=minimal_config)


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def tool(fake_scheduler: FakeScheduler) -> ScheduleTool:
    return ScheduleTool(fake_scheduler)  # type: ignore[arg-type]


async def test_schedule_one_shot_delay(
    tool: ScheduleTool, fake_scheduler: FakeScheduler, ctx: ToolContext
) -> None:
    result = await tool.execute(
        {"action": "schedule", "prompt": "avvisami", "trigger": "date", "delay_seconds": 300},
        ctx,
    )

    assert fake_scheduler.added, "schedule must register a job"
    job = fake_scheduler.added[0]
    assert job.prompt == "avvisami"
    assert job.trigger == "date"
    assert "run_date" in job.trigger_args
    assert job.output_channel == "telegram"
    assert job.session_id == f"cron:{job.job_id}"
    assert result["job_id"] == job.job_id


async def test_schedule_respects_explicit_fields(
    tool: ScheduleTool, fake_scheduler: FakeScheduler, ctx: ToolContext
) -> None:
    await tool.execute(
        {
            "action": "schedule",
            "job_id": "mattino",
            "prompt": "buongiorno",
            "trigger": "cron",
            "crontab": "0 9 * * *",
            "output_channel": "telegram",
            "session_id": "cron-mattino",
        },
        ctx,
    )

    job = fake_scheduler.added[0]
    assert job.job_id == "mattino"
    assert job.trigger_args == {"crontab": "0 9 * * *"}
    assert job.session_id == "cron-mattino"


async def test_schedule_interval(
    tool: ScheduleTool, fake_scheduler: FakeScheduler, ctx: ToolContext
) -> None:
    await tool.execute(
        {"action": "schedule", "prompt": "check", "trigger": "interval", "interval_seconds": 60},
        ctx,
    )

    assert fake_scheduler.added[0].trigger_args == {"seconds": 60}


async def test_schedule_without_prompt_raises(tool: ScheduleTool, ctx: ToolContext) -> None:
    with pytest.raises(ValueError):
        await tool.execute({"action": "schedule", "trigger": "date", "delay_seconds": 10}, ctx)


async def test_schedule_date_without_time_raises(tool: ScheduleTool, ctx: ToolContext) -> None:
    with pytest.raises(ValueError):
        await tool.execute({"action": "schedule", "prompt": "x", "trigger": "date"}, ctx)


async def test_schedule_interval_without_seconds_raises(
    tool: ScheduleTool, ctx: ToolContext
) -> None:
    with pytest.raises(ValueError):
        await tool.execute({"action": "schedule", "prompt": "x", "trigger": "interval"}, ctx)


async def test_schedule_cron_without_crontab_raises(tool: ScheduleTool, ctx: ToolContext) -> None:
    with pytest.raises(ValueError):
        await tool.execute({"action": "schedule", "prompt": "x", "trigger": "cron"}, ctx)


async def test_list_returns_jobs(
    tool: ScheduleTool, fake_scheduler: FakeScheduler, ctx: ToolContext
) -> None:
    await tool.execute(
        {
            "action": "schedule",
            "job_id": "j1",
            "prompt": "uno",
            "trigger": "date",
            "delay_seconds": 1,
        },
        ctx,
    )

    listed = await tool.execute({"action": "list"}, ctx)

    assert listed == [
        {
            "job_id": "j1",
            "prompt": "uno",
            "trigger": "date",
            "trigger_args": fake_scheduler.added[0].trigger_args,
            "session_id": "cron:j1",
            "output_channel": "telegram",
        }
    ]


async def test_cancel_removes_job(
    tool: ScheduleTool, fake_scheduler: FakeScheduler, ctx: ToolContext
) -> None:
    await tool.execute(
        {
            "action": "schedule",
            "job_id": "j1",
            "prompt": "uno",
            "trigger": "date",
            "delay_seconds": 1,
        },
        ctx,
    )

    result = await tool.execute({"action": "cancel", "job_id": "j1"}, ctx)

    assert result == {"removed": "j1"}
    assert fake_scheduler.removed == ["j1"]
    assert await tool.execute({"action": "list"}, ctx) == []


async def test_cancel_without_job_id_raises(tool: ScheduleTool, ctx: ToolContext) -> None:
    with pytest.raises(ValueError):
        await tool.execute({"action": "cancel"}, ctx)


async def test_unknown_action_raises(tool: ScheduleTool, ctx: ToolContext) -> None:
    with pytest.raises(ValueError):
        await tool.execute({"action": "teleport"}, ctx)


async def test_tool_spec_exposes_schedule_action(tool: ScheduleTool) -> None:
    spec = tool.spec
    assert spec.name == "schedule"
    assert "schedule" in spec.parameters["properties"]["action"]["enum"]
    assert spec.requires_approval is False


async def test_tool_with_real_scheduler_roundtrip(tmp_path: Path, ctx: ToolContext) -> None:
    config = SchedulerConfig(enabled=True, jobstore_db=tmp_path / "jobs.sqlite")
    scheduler = CronScheduler(EchoService(), config)  # type: ignore[arg-type]
    await scheduler.start()
    sink = RecordingSink()
    scheduler.register_sink("telegram", sink)
    real_tool = ScheduleTool(scheduler)
    try:
        result = await real_tool.execute(
            {
                "action": "schedule",
                "prompt": "ciao",
                "trigger": "interval",
                "interval_seconds": 3600,
            },
            ctx,
        )
        job_id = result["job_id"]

        assert [job["job_id"] for job in await real_tool.execute({"action": "list"}, ctx)] == [
            job_id
        ]

        await scheduler.run_once(job_id)
        assert sink.sent == ["ran ciao"]

        await real_tool.execute({"action": "cancel", "job_id": job_id}, ctx)
        assert await real_tool.execute({"action": "list"}, ctx) == []
    finally:
        await scheduler.shutdown()
