"""Structured logging configuration for the healer pipeline.

Centralises logging setup so every submodule just calls ``get_logger(__name__)``
and inherits consistent formatting, verbosity, and (optional) file output bound
to the experiment declared in :mod:`healer.config`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from healer.config import HealerConfig, LogLevel

_CONFIGURED: bool = False


def configure_logging(config: HealerConfig) -> None:
    """Initialise the root logger from experiment config.

    Idempotent — safe to call repeatedly across pipeline phases; after the
    first invocation subsequent calls are no-ops to avoid stacking handlers.

    Args:
        config: The fully validated :class:`HealerConfig` instance. The
            log level is taken from ``experiment.log_level`` and any output
            directory from ``experiment.output_dir``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, config.experiment.log_level.value)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    out_dir = Path(config.experiment.output_dir)
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{config.experiment.name}.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger. Thin wrapper for uniform access."""
    return logging.getLogger(name)


__all__ = ["LogLevel", "configure_logging", "get_logger"]
