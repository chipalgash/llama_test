"""Readable logging helpers for research scripts."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rest:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {rest:.0f}s"


def log_stage(logger: logging.Logger, title: str) -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)


def log_kv(logger: logging.Logger, values: Mapping[str, Any]) -> None:
    if not values:
        return
    width = max(len(str(key)) for key in values)
    for key, value in values.items():
        logger.info("%-*s : %s", width, key, value)


class StepTimer:
    def __init__(self) -> None:
        self.started_at = time.monotonic()

    def elapsed(self) -> str:
        return format_seconds(time.monotonic() - self.started_at)
