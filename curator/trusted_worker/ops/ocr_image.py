"""``ocr_image`` op mode: JSONL manifest -> C-IR-2 rows."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from curator.operations.loading.document_common import (
    DocumentParseError,
    REQUEST_LEVEL_CODES,
)
from curator.operations.loading.document_io import canonicalize_roots
from curator.operations.loading.document_parsing import (
    row_payload,
    validate_source_file,
)
from curator.operations.loading.image_ocr import (
    IMAGE_EXTENSIONS,
    OP_IMAGE_OCR_PARAM_KEYS,
    build_image_ocr_options,
    iter_image_results_in_subprocess,
)
from curator.operations.loading.ocr_routing import select_ocr_routes
from curator.operations.loading.parsed_document import make_success_row
from curator.trusted_worker.errors import WorkerError
from curator.trusted_worker.ops.document_artifacts import (
    MAX_INLINE_BYTES,
    error_row,
    request_error,
    resolve_source_path,
    rows_response,
)


def _validated_entry(source_path: str, roots: List[str]) -> str:
    path = validate_source_file(source_path, roots)
    if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
        raise DocumentParseError(
            "UNSUPPORTED_FORMAT", "unsupported image extension", stage="validate"
        )
    return path


def run_ocr_image(
    op: str,
    params: Dict[str, Any],
    rows: List[Dict[str, Any]],
    output_dir: Optional[Path],
    manifest_path: Path,
) -> Dict[str, Any]:
    try:
        options = build_image_ocr_options(
            params,
            allowed_keys=OP_IMAGE_OCR_PARAM_KEYS,
            require_allowed_roots=True,
        )
        roots = list(canonicalize_roots(options.allowed_roots))
        source_paths = [
            resolve_source_path(row.get("sourcePath"), manifest_path) for row in rows
        ]
        routes = select_ocr_routes(
            backend=options.backend,
            device=options.device,
            lang=options.lang,
            layout_analysis=options.layout_analysis,
            model_paths=options.model_path_map,
            allow_model_download=options.allow_model_download,
            backend_ranking_path=options.backend_ranking_path,
        )
        for route in routes:
            route.adapter.close()
    except DocumentParseError as exc:
        raise request_error(exc) from exc

    entries: List[Union[str, DocumentParseError]] = []
    for source_path in source_paths:
        try:
            entries.append(_validated_entry(source_path, roots))
        except DocumentParseError as exc:
            if exc.code in REQUEST_LEVEL_CODES:
                raise request_error(exc) from exc
            if options.on_error == "raise":
                raise WorkerError(exc.code, exc.message, exit_code=1) from exc
            entries.append(exc)

    output_rows: List[Dict[str, Any]] = []
    success_count = 0
    error_count = 0
    index = 0
    while index < len(entries):
        entry = entries[index]
        if isinstance(entry, DocumentParseError):
            output_rows.append(error_row(index, entry, options.output_key))
            error_count += 1
            index += 1
            continue
        run_start = index
        valid_paths: List[str] = []
        while index < len(entries) and isinstance(entries[index], str):
            valid_paths.append(entries[index])
            index += 1
        for offset, result in enumerate(
            iter_image_results_in_subprocess(valid_paths, options)
        ):
            result_index = run_start + offset
            if isinstance(result, DocumentParseError):
                if result.code in REQUEST_LEVEL_CODES:
                    raise request_error(result) from result
                if options.on_error == "raise":
                    raise WorkerError(
                        result.code, result.message, exit_code=1
                    ) from result
                output_rows.append(
                    error_row(result_index, result, options.output_key)
                )
                error_count += 1
                continue
            output_rows.append(
                row_payload(
                    make_success_row(
                        result_index, result, options.output_key
                    ),
                    options.output_key,
                )
            )
            success_count += 1

    return rows_response(
        output_rows,
        input_count=len(rows),
        success_count=success_count,
        error_count=error_count,
        filename="ocr-documents.jsonl",
        output_dir=output_dir,
        max_inline_bytes=MAX_INLINE_BYTES,
    )


__all__ = ["run_ocr_image"]
