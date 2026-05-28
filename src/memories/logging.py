from __future__ import annotations

from typing import TextIO

from loguru import logger

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}"


def configure_logging(stderr: TextIO, *, level: str = "INFO") -> None:
    """Configure process logging for CLI commands."""
    logger.remove()
    logger.add(stderr, level=level, format=LOG_FORMAT, colorize=False)
