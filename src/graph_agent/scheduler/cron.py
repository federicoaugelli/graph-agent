from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from graph_agent.config import SchedulerConfig
from graph_agent.core.service import AgentService
from graph_agent.events import DoneEvent, ErrorEvent, FileEvent
from graph_agent.logging import get_logger

log = get_logger(__name__)


class ChannelSink(Protocol):
    async def send(self, text: str) -> None: ...

    async def send_file(self, path: str, caption: str | None = None) -> None: ...


@dataclass(slots=True)
class JobSpec:
    job_id: str
    prompt: str
    trigger: str
    trigger_args: dict[str, Any] = field(default_factory=dict)
    session_id: str = "cron"
    output_channel: str | None = None


_RUNTIME: CronScheduler | None = None


async def fire_job(job: JobSpec) -> None:
    """Module-level entrypoint so the jobstore can serialize jobs by reference."""
    if _RUNTIME is None:
        raise RuntimeError("no CronScheduler running in this process")
    await _RUNTIME.execute_job(job)


class CronScheduler:
    """Scheduled jobs via APScheduler (persistent SQLite jobstore)."""

    def __init__(self, service: AgentService, config: SchedulerConfig) -> None:
        self.service = service
        self.config = config
        self._scheduler: AsyncIOScheduler | None = None
        self._sinks: dict[str, ChannelSink] = {}

    def register_sink(self, name: str, sink: ChannelSink) -> None:
        self._sinks[name] = sink

    async def start(self) -> None:
        global _RUNTIME
        db_path = Path(self.config.jobstore_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        scheduler = AsyncIOScheduler(
            jobstores={"default": SQLAlchemyJobStore(url=f"sqlite:///{db_path}")},
            timezone="UTC",
        )
        scheduler.start()
        self._scheduler = scheduler
        _RUNTIME = self

    async def shutdown(self) -> None:
        global _RUNTIME
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        if _RUNTIME is self:
            _RUNTIME = None

    async def add_job(self, job: JobSpec) -> None:
        self._require().add_job(
            fire_job,
            trigger=_build_trigger(job),
            args=[job],
            id=job.job_id,
            replace_existing=True,
        )

    async def remove_job(self, job_id: str) -> None:
        self._require().remove_job(job_id)

    async def list_jobs(self) -> list[JobSpec]:
        return [job.args[0] for job in self._require().get_jobs()]

    async def run_once(self, job_id: str) -> None:
        job = self._require().get_job(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id}")
        await self.execute_job(job.args[0])

    async def execute_job(self, job: JobSpec) -> None:
        final_text: str | None = None
        files: list[FileEvent] = []
        had_error = False

        async for event in self.service.run(
            job.session_id, job.prompt, approval_mode="auto"
        ):
            if isinstance(event, DoneEvent):
                final_text = event.final_text
            elif isinstance(event, FileEvent):
                files.append(event)
            elif isinstance(event, ErrorEvent):
                had_error = True

        if job.output_channel is None:
            return

        sink = self._sinks.get(job.output_channel)
        if sink is None:
            log.warning(
                "cron_output_channel_missing",
                job_id=job.job_id,
                channel=job.output_channel,
            )
            return

        if had_error:
            log.warning("cron_job_error", job_id=job.job_id)
            return

        for file in files:
            await sink.send_file(file.path, file.caption)
        if final_text:
            await sink.send(final_text)
        elif not files:
            await sink.send("(no output)")

    def _require(self) -> AsyncIOScheduler:
        if self._scheduler is None:
            raise RuntimeError("scheduler not started")
        return self._scheduler


def _build_trigger(job: JobSpec) -> Any:
    args = dict(job.trigger_args)
    if job.trigger == "cron":
        crontab = args.pop("crontab", None)
        if crontab is not None:
            return CronTrigger.from_crontab(crontab, timezone="UTC")
        return CronTrigger(timezone="UTC", **args)
    if job.trigger == "interval":
        return IntervalTrigger(**args)
    if job.trigger == "date":
        run_date = args.get("run_date")
        if isinstance(run_date, str):
            args["run_date"] = datetime.fromisoformat(run_date)
        return DateTrigger(**args)
    raise ValueError(f"unknown trigger: {job.trigger}")
