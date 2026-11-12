"""Spawned-child result protocol and lifecycle helpers."""
from __future__ import annotations

import json
import multiprocessing
import os
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .document_common import DocumentParseError

_POLL_INTERVAL_S = 0.05
_TERMINATE_GRACE_S = 2.0


def write_result(path: str, payload: Mapping[str, Any]) -> None:
    """Atomically publish a JSON result for the parent process."""
    temp = f"{path}.tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_result(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("child result must be a JSON object")
    return payload


def start_child(target: Callable[..., None], args: Sequence[Any]) -> Any:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=target, args=tuple(args), daemon=True)
    process.start()
    return process


def stop_child(process: Any) -> None:
    if process.is_alive():
        process.terminate()
        process.join(_TERMINATE_GRACE_S)
    if process.is_alive():
        process.kill()
        process.join(_TERMINATE_GRACE_S)


def wait_for_result(path: str, process: Any, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if os.path.exists(path):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not process.is_alive():
            return os.path.exists(path)
        time.sleep(min(_POLL_INTERVAL_S, remaining))


def result_error(
    payload: Mapping[str, Any], *, default_stage: str = "parse"
) -> Optional[DocumentParseError]:
    if payload.get("ok") is True:
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return DocumentParseError(
            "RESOURCE_LIMIT",
            "child process produced an invalid result",
            stage=default_stage,
            retryable=True,
        )
    return DocumentParseError(
        str(error.get("code", "RESOURCE_LIMIT")),
        str(error.get("message", "child process failed")),
        stage=str(error.get("stage", default_stage)),
        retryable=bool(error.get("retryable")),
        doc_id=error.get("doc_id"),
    )


def run_json_child(
    target: Callable[..., None],
    args: Sequence[Any],
    *,
    result_path: str,
    timeout_s: float,
    timeout_message: str,
    stage: str,
    missing_message: str,
) -> Dict[str, Any]:
    process = start_child(target, args)
    try:
        process.join(timeout_s)
        if process.is_alive():
            stop_child(process)
            raise DocumentParseError(
                "PARSE_TIMEOUT", timeout_message, stage=stage, retryable=True
            )
        if not os.path.exists(result_path):
            raise DocumentParseError(
                "RESOURCE_LIMIT", missing_message, stage=stage, retryable=True
            )
        try:
            return read_result(result_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise DocumentParseError(
                "RESOURCE_LIMIT",
                missing_message,
                stage=stage,
                retryable=True,
            ) from exc
    finally:
        stop_child(process)


__all__ = [
    "read_result",
    "result_error",
    "run_json_child",
    "start_child",
    "stop_child",
    "wait_for_result",
    "write_result",
]
