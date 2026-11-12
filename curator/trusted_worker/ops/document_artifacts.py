"""Shared request, artifact and response helpers for document OCR ops."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from curator.operations.loading.document_common import DocumentParseError
from curator.operations.loading.document_parsing import row_payload
from curator.operations.loading.parsed_document import (
    ErrorInfo,
    make_error_row,
)
from curator.trusted_worker.errors import WorkerError

MAX_INLINE_BYTES = 16 * 1024 * 1024


def request_error(exc: DocumentParseError) -> WorkerError:
    exit_code = (
        2
        if exc.stage == "validate" and exc.code != "DEPENDENCY_MISSING"
        else 1
    )
    return WorkerError(exc.code, exc.message, exit_code=exit_code)


def resolve_source_path(raw: Any, manifest_path: Path) -> str:
    if not isinstance(raw, str) or not raw:
        raise DocumentParseError(
            "INVALID_PARAMETER",
            "sourcePath must be a non-empty string",
            stage="validate",
        )
    path = raw if os.path.isabs(raw) else os.path.join(str(manifest_path.parent), raw)
    return os.path.abspath(path)


def write_artifact(
    output_dir: Path, filename: str, rows: List[Dict[str, Any]]
) -> None:
    fd = -1
    temp_name = ""
    try:
        fd, temp_name = tempfile.mkstemp(
            dir=str(output_dir), prefix=f".{filename}.", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            fd = -1
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output_dir / filename)
        temp_name = ""
    except OSError as exc:
        raise WorkerError(
            "OUTPUT_WRITE_FAILED", "artifact could not be written"
        ) from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def error_row(
    input_index: int, exc: DocumentParseError, output_key: str
) -> Dict[str, Any]:
    error = ErrorInfo(
        code=exc.code,
        stage=exc.stage,
        message=exc.message,
        retryable=exc.retryable,
    )
    return row_payload(
        make_error_row(input_index, error, output_key, doc_id=exc.doc_id),
        output_key,
    )


def rows_response(
    rows: List[Dict[str, Any]],
    *,
    input_count: int,
    success_count: int,
    error_count: int,
    filename: str,
    output_dir: Optional[Path],
    max_inline_bytes: int = MAX_INLINE_BYTES,
) -> Dict[str, Any]:
    metrics = {
        "input_count": float(input_count),
        "document_count": float(success_count),
        "error_count": float(error_count),
    }
    if output_dir is not None:
        write_artifact(output_dir, filename, rows)
        return {
            "metrics": metrics,
            "artifacts": [{"relativePath": filename}],
        }
    try:
        encoded = json.dumps(
            rows,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkerError(
            "OUTPUT_TOO_LARGE", "items are not JSON serializable"
        ) from exc
    if len(encoded) > max_inline_bytes:
        raise WorkerError(
            "OUTPUT_TOO_LARGE",
            "inline output exceeds 16 MiB; provide outputDir",
            exit_code=1,
        )
    return {"metrics": metrics, "items": rows}


__all__ = [
    "MAX_INLINE_BYTES",
    "error_row",
    "request_error",
    "resolve_source_path",
    "rows_response",
    "write_artifact",
]
