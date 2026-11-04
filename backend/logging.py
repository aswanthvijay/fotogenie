"""Small logging helpers so every request can be traced by request id."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from backend import config

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s")
    )
    root = logging.getLogger("fotogenie")
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def get_logger(name: str = "fotogenie") -> logging.Logger:
    configure_logging()
    logger = logging.getLogger(name)
    if not any(isinstance(f, _RequestIdFilter) for f in logger.filters):
        logger.addFilter(_RequestIdFilter())
    return logger


def bind_request_id(logger: logging.Logger, request_id: Optional[str]) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, {"request_id": request_id or "-"})
