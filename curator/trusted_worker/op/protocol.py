"""op mode protocol layer (Evaluator §4.3 invocation protocol + §4.5 failure semantics).

Single JSON request on stdin → single JSON response on stdout (written only when exit code is 0);
diagnostics go to stderr (one line ``{'code', 'message'}``). Exit codes: 0 success / 1 known failure
(op not in registry, data unreadable, operator raised an error, non-finite score) / 2 invalid input.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, TextIO

from curator.trusted_worker.errors import (
    EXIT_KNOWN_FAILURE,
    WorkerError,
    diagnostic_line,
)
from curator.trusted_worker.op.registry import OP_MODE_REGISTRY

SCHEMA_VERSION = 1

_REQUEST_KEYS = {"schemaVersion", "op", "params", "dataPath", "outputDir"}


def _invalid(message: str) -> WorkerError:
    return WorkerError("REQUEST_INVALID", message, exit_code=2)


def _reject_unknown(record: Dict[str, Any], allowed: set, where: str) -> None:
    unknown = set(record) - allowed
    if unknown:
        raise _invalid(f"{where} has unknown keys: {sorted(unknown)}")


def parse_request(request_text: str) -> Dict[str, Any]:
    try:
        request = json.loads(request_text)
    except json.JSONDecodeError as exc:
        raise _invalid(f"request is not valid JSON: {exc}") from exc
    if not isinstance(request, dict):
        raise _invalid("request must be a JSON object")
    if request.get("schemaVersion") != SCHEMA_VERSION:
        raise _invalid(f"schemaVersion must be {SCHEMA_VERSION}")
    _reject_unknown(request, _REQUEST_KEYS, "request")
    if not isinstance(request.get("op"), str):
        raise _invalid("op must be a string")
    if not isinstance(request.get("dataPath"), str):
        raise _invalid("dataPath must be a string")
    output_dir = request.get("outputDir")
    if output_dir is not None and not isinstance(output_dir, str):
        raise _invalid("outputDir must be a string when present")
    return request


def read_rows(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read the dataPath JSONL and validate the row structure by op; an empty file is invalid input (exit 2)."""
    data_path = Path(request["dataPath"])
    try:
        text = data_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkerError("DATA_UNREADABLE", f"dataPath not readable: {data_path} ({exc})")

    required = OP_MODE_REGISTRY[request["op"]]["required_fields"]
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _invalid(f"dataPath line {line_no} is not valid JSON: {exc}")
        if not isinstance(row, dict):
            raise _invalid(f"dataPath line {line_no} must be a JSON object")
        for field, expected in required.items():
            value = row.get(field)
            if not isinstance(value, expected):
                raise _invalid(
                    f"dataPath line {line_no}: field '{field}' must be {expected.__name__}"
                )
        rows.append(row)
    if not rows:
        raise _invalid("dataPath contains no records")
    return rows


def _assert_finite_metrics(metrics: Dict[str, Any]) -> None:
    """Red line §4.4-1: non-finite score → exit 1; never produce a zero score."""
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise WorkerError(
                "NON_FINITE_METRIC",
                f"non-finite metric '{key}': refusing to fabricate metrics",
            )


def handle(request_text: str, stdout: TextIO, stderr: TextIO) -> int:
    try:
        request = parse_request(request_text)
        op = request["op"]
        if op not in OP_MODE_REGISTRY:
            raise WorkerError(
                "OP_NOT_FOUND",
                f"unknown op: {op}; supported: {sorted(OP_MODE_REGISTRY)}",
            )
        rows = read_rows(request)
        output_dir = None
        if request.get("outputDir") is not None:
            output_dir = Path(request["outputDir"])
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise WorkerError(
                    "OUTPUT_DIR_UNWRITABLE", f"outputDir not writable: {output_dir} ({exc})"
                )
        entry = OP_MODE_REGISTRY[op]
        context_runner = entry.get("run_with_context")
        if context_runner is not None:
            body = context_runner(
                op,
                request.get("params", {}),
                rows,
                output_dir,
                Path(request["dataPath"]),
            )
        else:
            body = entry["run"](op, request.get("params", {}), rows, output_dir)
        _assert_finite_metrics(body["metrics"])
    except WorkerError as error:
        print(diagnostic_line(error), file=stderr)
        return error.exit_code
    except Exception as error:  # noqa: BLE001 - red line: operator exceptions must not produce stdout
        print(
            diagnostic_line(WorkerError("OP_EXECUTION_FAILED", str(error))),
            file=stderr,
        )
        return EXIT_KNOWN_FAILURE

    response = {"schemaVersion": SCHEMA_VERSION, "op": request["op"], **body}
    stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    return 0


def main(stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    """Process body: read stdin → handle; stdin errors are normalized to exit code 2 + stderr diagnostics."""
    try:
        request_text = stdin.read()
    except Exception as error:  # noqa: BLE001
        print(diagnostic_line(WorkerError("STDIN_UNREADABLE", str(error))), file=stderr)
        return 2
    return handle(request_text, stdout, stderr)
