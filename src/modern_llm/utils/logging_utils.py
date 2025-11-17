"""Project-specific logging helpers."""

from __future__ import annotations

import logging
from typing import Optional


def create_logger(name: str, level: int = logging.INFO, propagate: bool = False) -> logging.Logger:
    """Create a standard logger with a concise formatter.

    Pre:
        - name is non-empty.
    Post:
        - returns a logger with at most one handler attached by this helper.
    Complexity:
        - O(1); operations are metadata-only.
    """

    if not name:
        raise ValueError("Logger name must be non-empty.")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger

