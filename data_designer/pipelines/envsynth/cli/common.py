"""Shared worker protocol helpers for the environment-synthesis CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


class WorkerError(RuntimeError):
    """Input envelope is invalid (exit 2)."""


def _reject_json_constant(name: str) -> float:
    raise ValueError("Invalid JSON constant {}".format(name))


def load_json(path: str) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as error:
        raise WorkerError("Input file is not readable {}: {}".format(path, error)) from error
    except ValueError as error:
        raise WorkerError("Input is not valid JSON: {}".format(error)) from error


def absolute_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("{} must be a non-empty string".format(field))
    path = Path(value)
    if not path.is_absolute():
        raise WorkerError("{} must be an absolute path".format(field))
    return str(path)


__all__ = [
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_USAGE",
    "SCHEMA_VERSION",
    "WorkerError",
    "absolute_path",
    "load_json",
]
