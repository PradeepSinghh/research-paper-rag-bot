"""
Logging configuration for the project.

Every module should call get_logger(__name__) to obtain a configured logger.
All log output goes to stdout so it appears in the Streamlit terminal.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a named logger with a consistent format.

    Args:
        name:  Logger name — use __name__ in each module.
        level: Logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the module is reloaded (e.g., Streamlit hot-reload)
    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent log messages from propagating to the root logger and being printed twice
        logger.propagate = False

    return logger
