"""Logging setup.

Every agent run carries a ``run_id`` so that the lines belonging to one question
can be pulled out of a shared log, which is the minimum needed to debug a
multi-step agent after the fact.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

# Marked on the root logger rather than held in a module global, so that a
# reimported module cannot reconfigure handlers a second time.
_CONFIGURED_ATTR = "_agentic_rag_logging_configured"


class JsonFormatter(logging.Formatter):
    """Render records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record``, preserving any extra fields attached to it."""
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord("", 0, "", 0, "", None, None).__dict__:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Configure the root logger once per process."""
    root = logging.getLogger()
    if getattr(root, _CONFIGURED_ATTR, False):
        return

    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.__dict__[_CONFIGURED_ATTR] = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
