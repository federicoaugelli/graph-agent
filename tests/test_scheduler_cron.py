from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from graph_agent.config import SchedulerConfig
from graph_agent.events import DoneEvent, ErrorEvent, Event, FileEvent
from graph_agent.scheduler.cron import CronScheduler, JobSpec


class FakeService:
    def __init__(
        self, final_text: str | None = "cron done", events: list[Event] | None = None
    ) -> None:
        self.final_text = final_text
        self.events = events
        self.runs: list[tuple[str, str]] = []
        self.calls: list[dict[str, Any]] = []

    async def run(self, session_id: str, user_input: str, **kwargs: Any) -> AsyncIterator[Event]:
        self.runs.append((session_id, user_input))
        self.calls.append(kwargs)
        if self.events is not None:
            for event in self.events:
                yield event
            return
        yield DoneEvent(session_id=session_id, final_text=self.final_text)


class FakeSink:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.files: list[tuple[str, str | None]] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def send_file(self, path: str, caption: str | None = None) -> None:
        self.files.append((path, caption))


@pytest.fixture
def jobstore_db(tmp_path: Path) -> Path:
    return tmp_path / "jobs.sqlite"


@pytest.fixture
def scheduler_config(jobstore_db: Path) -> SchedulerConfig:
    return SchedulerConfig(enabled=True, jobstore_db=jobstore_db)


def make_scheduler(config: SchedulerConfig) -> tuple[CronScheduler, FakeService]:
    service = FakeService()
    return CronScheduler(service, config), service  # type: ignore[arg-type]


def interval_job(job_id: str = "every-hour", prompt: str = "riepilogo") -> JobSpec:
    return JobSpec(
        job_id=job_id,
        prompt=prompt,
        trigger="interval",
        trigger_args={"seconds": 3600},
    )


async def test_add_list_remove_job(scheduler_config: SchedulerConfig) -> None:
    sched, _ = make_scheduler(scheduler_config)
    await sched.start()
    try:
        await sched.add_job(interval_job())
        specs = await sched.list_jobs()
        assert [s.job_id for s in specs] == ["every-hour"]
        assert specs[0].prompt == "riepilogo"

        await sched.remove_job("every-hour")
        assert await sched.list_jobs() == []
    finally:
        await sched.shutdown()


async def test_add_job_twice_replaces(scheduler_config: SchedulerConfig) -> None:
    sched, _ = make_scheduler(scheduler_config)
    await sched.start()
    try:
        await sched.add_job(interval_job(prompt="prima"))
        await sched.add_job(interval_job(prompt="seconda"))
        specs = await sched.list_jobs()
        assert len(specs) == 1
        assert specs[0].prompt == "seconda"
    finally:
        await sched.shutdown()


async def test_unknown_trigger_raises(scheduler_config: SchedulerConfig) -> None:
    sched, _ = make_scheduler(scheduler_config)
    await sched.start()
    try:
        job = JobSpec(job_id="j", prompt="p", trigger="sundial", trigger_args={})
        with pytest.raises(ValueError):
            await sched.add_job(job)
    finally:
        await sched.shutdown()


async def test_operations_before_start_raise(scheduler_config: SchedulerConfig) -> None:
    sched, _ = make_scheduler(scheduler_config)
    with pytest.raises(RuntimeError):
        await sched.add_job(interval_job())


async def test_jobs_persist_across_restart(
    jobstore_db: Path, scheduler_config: SchedulerConfig
) -> None:
    sched, _ = make_scheduler(scheduler_config)
    await sched.start()
    await sched.add_job(
        JobSpec(
            job_id="mattino",
            prompt="buongiorno",
            trigger="cron",
            trigger_args={"crontab": "0 9 * * *"},
            session_id="cron",
            output_channel="telegram",
        )
    )
    await sched.shutdown()

    restarted, _ = make_scheduler(scheduler_config)
    await restarted.start()
    try:
        specs = await restarted.list_jobs()
        assert len(specs) == 1
        assert specs[0].job_id == "mattino"
        assert specs[0].prompt == "buongiorno"
        assert specs[0].trigger == "cron"
        assert specs[0].trigger_args == {"crontab": "0 9 * * *"}
        assert specs[0].output_channel == "telegram"
    finally:
        await restarted.shutdown()


async def test_run_once_executes_prompt_and_sends_output(scheduler_config: SchedulerConfig) -> None:
    sched, service = make_scheduler(scheduler_config)
    await sched.start()
    sink = FakeSink()
    sched.register_sink("test", sink)
    try:
        job = interval_job(job_id="j1", prompt="prompt di prova")
        job.session_id = "cron-sess"
        job.output_channel = "test"
        await sched.add_job(job)

        await sched.run_once("j1")
        assert service.runs == [("cron-sess", "prompt di prova")]
        assert sink.sent == ["cron done"]
    finally:
        await sched.shutdown()


async def test_run_once_without_final_text_sends_nonempty(
    scheduler_config: SchedulerConfig,
) -> None:
    service = FakeService(final_text=None)
    sched = CronScheduler(service, scheduler_config)  # type: ignore[arg-type]
    await sched.start()
    sink = FakeSink()
    sched.register_sink("test", sink)
    try:
        job = interval_job(job_id="j1")
        job.output_channel = "test"
        await sched.add_job(job)
        await sched.run_once("j1")
        assert len(sink.sent) == 1
        assert sink.sent[0].strip()
    finally:
        await sched.shutdown()


async def test_run_once_uses_auto_approval(scheduler_config: SchedulerConfig) -> None:
    sched, service = make_scheduler(scheduler_config)
    await sched.start()
    sink = FakeSink()
    sched.register_sink("test", sink)
    try:
        job = interval_job(job_id="j1")
        job.output_channel = "test"
        await sched.add_job(job)

        await sched.run_once("j1")

        assert service.calls[-1]["approval_mode"] == "auto"
    finally:
        await sched.shutdown()


async def test_run_once_delivers_file_events(scheduler_config: SchedulerConfig) -> None:
    events: list[Event] = [
        FileEvent(path="/ws/report.txt", caption="ecco"),
        DoneEvent(session_id="cron", final_text="fatto"),
    ]
    service = FakeService(events=events)
    sched = CronScheduler(service, scheduler_config)  # type: ignore[arg-type]
    await sched.start()
    sink = FakeSink()
    sched.register_sink("test", sink)
    try:
        job = interval_job(job_id="j1")
        job.output_channel = "test"
        await sched.add_job(job)

        await sched.run_once("j1")

        assert sink.files == [("/ws/report.txt", "ecco")]
        assert sink.sent == ["fatto"]
    finally:
        await sched.shutdown()


async def test_run_once_skips_output_on_error(scheduler_config: SchedulerConfig) -> None:
    events: list[Event] = [
        ErrorEvent(message="boom"),
        DoneEvent(session_id="cron", final_text="partial"),
    ]
    service = FakeService(events=events)
    sched = CronScheduler(service, scheduler_config)  # type: ignore[arg-type]
    await sched.start()
    sink = FakeSink()
    sched.register_sink("test", sink)
    try:
        job = interval_job(job_id="j1")
        job.output_channel = "test"
        await sched.add_job(job)

        await sched.run_once("j1")

        assert sink.sent == []
        assert sink.files == []
    finally:
        await sched.shutdown()


async def test_run_once_file_only_does_not_send_no_output(
    scheduler_config: SchedulerConfig,
) -> None:
    events: list[Event] = [
        FileEvent(path="/ws/report.txt", caption=None),
        DoneEvent(session_id="cron", final_text=""),
    ]
    service = FakeService(events=events)
    sched = CronScheduler(service, scheduler_config)  # type: ignore[arg-type]
    await sched.start()
    sink = FakeSink()
    sched.register_sink("test", sink)
    try:
        job = interval_job(job_id="j1")
        job.output_channel = "test"
        await sched.add_job(job)

        await sched.run_once("j1")

        assert sink.files == [("/ws/report.txt", None)]
        assert sink.sent == []
    finally:
        await sched.shutdown()


async def test_run_once_unknown_channel_still_executes(scheduler_config: SchedulerConfig) -> None:
    sched, service = make_scheduler(scheduler_config)
    await sched.start()
    try:
        job = interval_job(job_id="j1")
        job.output_channel = "nonexistent-channel"
        await sched.add_job(job)
        await sched.run_once("j1")
        assert service.runs == [("cron", "riepilogo")]
    finally:
        await sched.shutdown()


async def test_run_once_without_output_channel_skips_sink(
    scheduler_config: SchedulerConfig,
) -> None:
    sched, service = make_scheduler(scheduler_config)
    await sched.start()
    sink = FakeSink()
    sched.register_sink("test", sink)
    try:
        await sched.add_job(interval_job(job_id="j1"))
        await sched.run_once("j1")
        assert service.runs == [("cron", "riepilogo")]
        assert sink.sent == []
    finally:
        await sched.shutdown()


async def test_run_once_unknown_job_raises(scheduler_config: SchedulerConfig) -> None:
    sched, _ = make_scheduler(scheduler_config)
    await sched.start()
    try:
        with pytest.raises(KeyError):
            await sched.run_once("inesistente")
    finally:
        await sched.shutdown()
