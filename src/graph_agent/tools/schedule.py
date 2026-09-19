from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from graph_agent.scheduler.cron import CronScheduler, JobSpec
from graph_agent.tools.base import ToolContext, ToolSpec

DEFAULT_TRIGGER = "date"
DEFAULT_OUTPUT_CHANNEL = "telegram"


class ScheduleTool:
    """Let the agent schedule future or recurring runs of itself.

    Jobs are persisted in the scheduler jobstore and, when they fire, run the
    agent unattended (approval disabled) and deliver the result to their
    output channel.
    """

    def __init__(self, scheduler: CronScheduler) -> None:
        self.scheduler = scheduler

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="schedule",
            description=(
                "Schedule future or recurring tasks that run this agent with a prompt. "
                "Use trigger 'date' with delay_seconds for one-shot reminders, "
                "'interval' with interval_seconds for repeated tasks, or 'cron' with a "
                "crontab for calendar schedules. Jobs run unattended (tool approval is "
                "disabled) and report their result to the output channel. Also lists and "
                "cancels existing jobs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["schedule", "list", "cancel"]},
                    "job_id": {
                        "type": "string",
                        "description": "Job identifier; generated if omitted on schedule.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt the agent runs when the job fires.",
                    },
                    "trigger": {"type": "string", "enum": ["date", "interval", "cron"]},
                    "delay_seconds": {
                        "type": "integer",
                        "description": "For trigger 'date': run once after N seconds.",
                    },
                    "run_at": {
                        "type": "string",
                        "description": "For trigger 'date': ISO 8601 datetime.",
                    },
                    "interval_seconds": {
                        "type": "integer",
                        "description": "For trigger 'interval': period in seconds.",
                    },
                    "crontab": {
                        "type": "string",
                        "description": "For trigger 'cron': crontab expression, e.g. '0 9 * * *'.",
                    },
                    "output_channel": {
                        "type": "string",
                        "description": "Where the result is sent (default 'telegram').",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Thread id for the job (default 'cron:{job_id}').",
                    },
                },
                "required": ["action"],
            },
            requires_approval=False,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        action = args.get("action")
        match action:
            case "schedule":
                return await self._schedule(args)
            case "list":
                return await self._list()
            case "cancel":
                return await self._cancel(args)
            case _:
                raise ValueError(f"unknown action: {action}")

    async def _schedule(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt = args.get("prompt")
        if not prompt:
            raise ValueError("schedule requires 'prompt'")

        job_id = args.get("job_id") or f"job-{uuid4().hex[:8]}"
        trigger = args.get("trigger") or DEFAULT_TRIGGER
        trigger_args = _trigger_args(trigger, args)
        job = JobSpec(
            job_id=job_id,
            prompt=prompt,
            trigger=trigger,
            trigger_args=trigger_args,
            session_id=args.get("session_id") or f"cron:{job_id}",
            output_channel=args.get("output_channel") or DEFAULT_OUTPUT_CHANNEL,
        )
        await self.scheduler.add_job(job)
        return {"job_id": job.job_id, "trigger": trigger, "trigger_args": trigger_args}

    async def _list(self) -> list[dict[str, Any]]:
        jobs = await self.scheduler.list_jobs()
        return [
            {
                "job_id": job.job_id,
                "prompt": job.prompt,
                "trigger": job.trigger,
                "trigger_args": job.trigger_args,
                "session_id": job.session_id,
                "output_channel": job.output_channel,
            }
            for job in jobs
        ]

    async def _cancel(self, args: dict[str, Any]) -> dict[str, str]:
        job_id = args.get("job_id")
        if not job_id:
            raise ValueError("cancel requires 'job_id'")
        await self.scheduler.remove_job(job_id)
        return {"removed": job_id}


def _trigger_args(trigger: str, args: dict[str, Any]) -> dict[str, Any]:
    if trigger == "date":
        delay = args.get("delay_seconds")
        if delay is not None:
            run_date = datetime.now(UTC) + timedelta(seconds=int(delay))
            return {"run_date": run_date.isoformat()}
        run_at = args.get("run_at")
        if not run_at:
            raise ValueError("trigger 'date' requires 'delay_seconds' or 'run_at'")
        return {"run_date": str(run_at)}
    if trigger == "interval":
        seconds = args.get("interval_seconds")
        if seconds is None:
            raise ValueError("trigger 'interval' requires 'interval_seconds'")
        return {"seconds": int(seconds)}
    if trigger == "cron":
        crontab = args.get("crontab")
        if not crontab:
            raise ValueError("trigger 'cron' requires 'crontab'")
        return {"crontab": str(crontab)}
    raise ValueError(f"unknown trigger: {trigger}")