"""Application logging configuration."""

import logging
import sys
from typing import Optional


def setup_logging(log_level: str = "INFO", log_format: str = "standard") -> None:
    """Configure root and application loggers."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    if log_format.lower() == "json":
        fmt = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
    else:
        fmt = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]

    # Suppress verbose third-party logging
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve logger for a given component or module."""
    return logging.getLogger(name or "medvision")
