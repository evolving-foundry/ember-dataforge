"""``parse_document`` op mode: JSONL manifest -> C-IR-2 rows."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from curator.operations.loading.document_common import (
    DocumentParseError,
    OP_DOC_PARSE_PARAM_KEYS,
    REQUEST_LEVEL_CODES,
    build_parse_options,
)
from curator.operations.loading.document_parsing import (
    canonicalize_roots,
    parse_in_subprocess,
    row_payload,
    validate_source_file,
)
from curator.operations.loading.parsed_document import make_success_row
from curator.trusted_worker.errors import WorkerError
from curator.trusted_worker.ops.document_artifacts import (
    MAX_INLINE_BYTES,
    error_row,
    request_error,
    resolve_source_path,
    rows_response,
)


def run_parse_document(
    op: str,
    params: Dict[str, Any],
    rows: List[Dict[str, Any]],
    output_dir: Optional[Path],
    manifest_path: Path,
) -> Dict[str, Any]:
    try:
        options = build_parse_options(
            params,
            require_allowed_roots=True,
            allowed_keys=OP_DOC_PARSE_PARAM_KEYS,
        )
        roots = canonicalize_roots(options.allowed_roots)
    except DocumentParseError as exc:
        raise request_error(exc) from exc

    output_rows: List[Dict[str, Any]] = []
    success_count = 0
    error_count = 0
    for input_index, manifest_row in enumerate(rows):
        try:
            source_path = validate_source_file(
                resolve_source_path(manifest_row.get("sourcePath"), manifest_path),
                roots,
            )
        except DocumentParseError as exc:
            if exc.code == "INVALID_PARAMETER":
                raise WorkerError(exc.code, exc.message, exit_code=2) from exc
            if options.on_error == "raise":
                raise WorkerError(exc.code, exc.message, exit_code=1) from exc
            output_rows.append(error_row(input_index, exc, options.output_key))
            error_count += 1
            continue
        try:
            document = parse_in_subprocess(source_path, options)
        except DocumentParseError as exc:
            if exc.code in REQUEST_LEVEL_CODES:
                raise WorkerError(exc.code, exc.message, exit_code=1) from exc
            if options.on_error == "raise":
                raise WorkerError(exc.code, exc.message, exit_code=1) from exc
            output_rows.append(error_row(input_index, exc, options.output_key))
            error_count += 1
            continue
        output_rows.append(
            row_payload(
                make_success_row(input_index, document, options.output_key),
                options.output_key,
            )
        )
        success_count += 1
    return rows_response(
        output_rows,
        input_count=len(rows),
        success_count=success_count,
        error_count=error_count,
        filename="parsed-documents.jsonl",
        output_dir=output_dir,
        max_inline_bytes=MAX_INLINE_BYTES,
    )


__all__ = ["run_parse_document"]
