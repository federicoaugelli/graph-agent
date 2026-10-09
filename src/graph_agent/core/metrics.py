from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field

_current: ContextVar[RunMetrics | None] = ContextVar("graph_agent_run_metrics", default=None)


@dataclass
class RunMetrics:
    """Per-run latency counters, collected via contextvar during a graph run.

    Nodes and the tool wrapper mutate the instance returned by
    ``start_run_metrics``; ``AgentService`` reads it after the stream to build a
    ``MetricsEvent``. Kept out of ``AgentState`` so nothing non-serializable
    reaches the checkpointer.
    """

    llm_calls: int = 0
    tool_calls: int = 0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    signatures: dict[tuple[str, str], int] = field(default_factory=dict)


def start_run_metrics() -> tuple[RunMetrics, Token[RunMetrics | None]]:
    metrics = RunMetrics()
    return metrics, _current.set(metrics)


def reset_run_metrics(token: Token[RunMetrics | None]) -> None:
    _current.reset(token)


def current_metrics() -> RunMetrics | None:
    return _current.get()
