"""
Structured logging setup using structlog.

All log entries carry context fields (request_id, stage, latency) that
make log aggregation and debugging much easier in production.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from app.core.config import get_settings


def _add_app_info(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Inject application-level context into every log record."""
    settings = get_settings()
    event_dict.setdefault("app", settings.app_name)
    event_dict.setdefault("env", settings.tripgenie_env)
    return event_dict


def configure_logging() -> None:
    """
    Configure structlog and the standard library logging backend.

    In development we emit coloured console output; in production we emit
    JSON lines suitable for log aggregation systems.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level, logging.INFO)
    is_dev = settings.tripgenie_env in ("development", "test")

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if is_dev:
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processor=renderer,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Quieten noisy libraries
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = "tripgenie") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)
