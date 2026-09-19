from __future__ import annotations

import logging
import sys

import structlog

from graph_agent.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, config.level.upper(), logging.INFO),
    )
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if config.json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors, wrapper_class=structlog.make_filtering_bound_logger(0)
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
