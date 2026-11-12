"""Shared constants, parameter validation, and deterministic conversion for document parsing."""
from __future__ import annotations

import datetime as _dt
import math
import re
import unicodedata
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .parsed_document import Provenance, READING_ORDER_VERSION, validate_output_key

DEFAULT_MAX_FILE_MB = 100.0
DEFAULT_MAX_PAGES = 500
DEFAULT_PDF_MIN_TEXT_CHARS = 20
DEFAULT_MAX_TABLE_CELLS = 1_000_000
DEFAULT_MAX_MEMORY_MB = 2048
DEFAULT_PARSE_TIMEOUT_S = 60.0

DOC_PARSE_PARAM_KEYS = frozenset(
    {
        "data_paths",
        "output_key",
        "extract_style",
        "max_file_mb",
        "max_pages",
        "pdf_min_text_chars_per_page",
        "parse_timeout_s",
        "max_memory_mb",
        "on_error",
        "allowed_roots",
    }
)
OP_DOC_PARSE_PARAM_KEYS = DOC_PARSE_PARAM_KEYS - {"data_paths"}

REQUEST_LEVEL_CODES = frozenset(
    {
        "INVALID_PARAMETER",
        "DEPENDENCY_MISSING",
        "OCR_BACKEND_UNAVAILABLE",
        "OCR_MODEL_UNAVAILABLE",
        "OCR_RANKING_INVALID",
        "OUTPUT_TOO_LARGE",
        "OUTPUT_WRITE_FAILED",
    }
)


class DocumentParseError(Exception):
    """Stable error-code carrier for parse failures; message must not contain the source path."""

    def __init__(
        self,
        code: str,
        message: str,
        stage: str = "parse",
        retryable: bool = False,
        doc_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.retryable = retryable
        self.doc_id = doc_id


@dataclass(frozen=True)
class ParseOptions:
    max_file_mb: float = DEFAULT_MAX_FILE_MB
    max_pages: int = DEFAULT_MAX_PAGES
    pdf_min_text_chars_per_page: int = DEFAULT_PDF_MIN_TEXT_CHARS
    extract_style: bool = False
    max_table_cells: int = DEFAULT_MAX_TABLE_CELLS
    max_memory_mb: int = DEFAULT_MAX_MEMORY_MB
    parse_timeout_s: float = DEFAULT_PARSE_TIMEOUT_S
    on_error: str = "raise"
    output_key: str = "parsed_doc"
    allowed_roots: Tuple[str, ...] = field(default_factory=tuple)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _require_number(name: str, value: Any, *, integer: bool = False, minimum: float = 0) -> Any:
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise DocumentParseError("INVALID_PARAMETER", f"{name} must be a number", stage="validate")
    if integer and not isinstance(value, int):
        raise DocumentParseError("INVALID_PARAMETER", f"{name} must be an integer", stage="validate")
    if not math.isfinite(float(value)) or value < minimum:
        raise DocumentParseError(
            "INVALID_PARAMETER", f"{name} must be finite and >= {minimum}", stage="validate"
        )
    return value


def build_parse_options(
    params: Dict[str, Any],
    *,
    require_allowed_roots: bool = False,
    allowed_keys: frozenset = DOC_PARSE_PARAM_KEYS,
) -> ParseOptions:
    if not isinstance(params, dict):
        raise DocumentParseError("INVALID_PARAMETER", "params must be an object", stage="validate")
    unknown = set(params) - allowed_keys
    if unknown:
        raise DocumentParseError(
            "INVALID_PARAMETER", f"unknown params: {sorted(unknown)}", stage="validate"
        )
    output_key = params.get("output_key", "parsed_doc")
    try:
        validate_output_key(output_key)
    except ValueError as exc:
        raise DocumentParseError("INVALID_PARAMETER", str(exc), stage="validate") from exc
    max_file_mb = float(
        _require_number("max_file_mb", params.get("max_file_mb", DEFAULT_MAX_FILE_MB), minimum=0.000001)
    )
    max_pages = int(
        _require_number("max_pages", params.get("max_pages", DEFAULT_MAX_PAGES), integer=True, minimum=1)
    )
    min_chars = int(
        _require_number(
            "pdf_min_text_chars_per_page",
            params.get("pdf_min_text_chars_per_page", DEFAULT_PDF_MIN_TEXT_CHARS),
            integer=True,
            minimum=1,
        )
    )
    timeout_s = float(
        _require_number(
            "parse_timeout_s", params.get("parse_timeout_s", DEFAULT_PARSE_TIMEOUT_S), minimum=0.000001
        )
    )
    max_memory_mb = int(
        _require_number(
            "max_memory_mb", params.get("max_memory_mb", DEFAULT_MAX_MEMORY_MB), integer=True, minimum=256
        )
    )
    extract_style = params.get("extract_style", False)
    if not isinstance(extract_style, bool):
        raise DocumentParseError("INVALID_PARAMETER", "extract_style must be boolean", stage="validate")
    on_error = params.get("on_error", "raise")
    if on_error not in {"raise", "skip"}:
        raise DocumentParseError("INVALID_PARAMETER", "on_error must be 'raise' or 'skip'", stage="validate")
    roots = params.get("allowed_roots", [])
    if not isinstance(roots, list) or not all(isinstance(root, str) and root for root in roots):
        raise DocumentParseError(
            "INVALID_PARAMETER", "allowed_roots must be an array of non-empty strings", stage="validate"
        )
    if require_allowed_roots and not roots:
        raise DocumentParseError("INVALID_PARAMETER", "allowed_roots must be non-empty", stage="validate")
    return ParseOptions(
        max_file_mb=max_file_mb,
        max_pages=max_pages,
        pdf_min_text_chars_per_page=min_chars,
        extract_style=extract_style,
        max_memory_mb=max_memory_mb,
        parse_timeout_s=timeout_s,
        on_error=on_error,
        output_key=output_key,
        allowed_roots=tuple(roots),
    )


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise DocumentParseError("CORRUPT_DOCUMENT", "text value is not a string")
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\x00", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value.strip()


def join_cell_paragraphs(paragraphs: Sequence[str]) -> str:
    return normalize_text("\n".join(paragraphs))


def detect_language(text: str) -> str:
    han = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = han + latin
    if total < 20:
        return "unknown"
    if han / total >= 0.8:
        return "zh"
    if latin / total >= 0.8:
        return "en"
    return "mixed"


def excel_cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DocumentParseError("CORRUPT_DOCUMENT", "non-finite numeric cell value")
        if value.is_integer():
            return str(int(value))
        return repr(value)
    if isinstance(value, _dt.datetime):
        return value.isoformat(sep="T")
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, _dt.time):
        return value.isoformat()
    return normalize_text(str(value))


def normalize_bbox(
    raw: Optional[Sequence[float]],
    width: float,
    height: float,
    *,
    error_code: str = "CORRUPT_DOCUMENT",
) -> List[float]:
    if raw is None or width <= 0 or height <= 0 or len(raw) != 4:
        raise DocumentParseError(error_code, "invalid bbox source dimensions")
    values = [float(item) for item in raw]
    if not all(math.isfinite(item) for item in values):
        raise DocumentParseError(error_code, "bbox contains non-finite value")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise DocumentParseError(error_code, "bbox is outside page bounds")
    bbox = [round(x0 / width, 6), round(y0 / height, 6), round(x1 / width, 6), round(y1 / height, 6)]
    if not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
        raise DocumentParseError(error_code, "bbox becomes invalid after normalization")
    return bbox


def package_version(distribution: str, fallback: str = "unknown") -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


def make_document_provenance(extractor: str, extractor_version: str) -> Provenance:
    return Provenance(
        extractor=extractor,
        extractor_version=extractor_version,
        operation="document_parsing",
        op_version=package_version("data-forge", "0.2.0"),
        reading_order_version=READING_ORDER_VERSION,
        routing_reason=None,
        ranking_version=None,
        ranking_evaluation_set=None,
        backend=None,
        backend_version=None,
        device=None,
        model_id=None,
    )
