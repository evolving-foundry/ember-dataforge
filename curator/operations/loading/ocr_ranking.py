"""Strict ``ocr-ranking-v1`` loading for OCR auto routing."""
from __future__ import annotations

import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import yaml

from .document_common import DocumentParseError

RANKING_SCHEMA_VERSION = "ocr-ranking-v1"
RANKING_PRIMARY_METRIC = "cer"
MAX_RANKING_FILE_BYTES = 10 * 1024 * 1024
MAX_RANKING_LINES = 10_000
MAX_RANKING_ROWS = 10_000
_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "ranking_version",
        "evaluation_set_version",
        "primary_metric",
        "rows",
    }
)
_ROW_KEYS = frozenset(
    {
        "backend",
        "backend_version",
        "device",
        "model_id",
        "cer",
        "line_accuracy",
        "layout_f1",
        "throughput_pages_s",
        "eligible",
    }
)


@dataclass(frozen=True)
class RankingRow:
    backend: str
    backend_version: str
    device: str
    model_id: str
    cer: float
    line_accuracy: float
    layout_f1: float
    throughput_pages_s: float
    eligible: bool


@dataclass(frozen=True)
class RankingTable:
    ranking_version: str
    evaluation_set_version: str
    rows: Tuple[RankingRow, ...]


def _invalid(message: str) -> DocumentParseError:
    return DocumentParseError("INVALID_PARAMETER", message, stage="validate")


def _ranking_invalid(message: str) -> DocumentParseError:
    return DocumentParseError("OCR_RANKING_INVALID", message, stage="validate")


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: Optional[float] = None,
    exclusive_minimum: bool = False,
) -> float:
    if not _is_number(value):
        raise _ranking_invalid(f"ranking field '{name}' must be a number")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < minimum
        or (exclusive_minimum and result == minimum)
    ):
        raise _ranking_invalid(f"ranking field '{name}' is out of range")
    if maximum is not None and result > maximum:
        raise _ranking_invalid(f"ranking field '{name}' is out of range")
    return result


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _ranking_invalid(f"ranking field '{name}' must be a non-empty string")
    return value


def _parse_row(raw: Any, index: int) -> RankingRow:
    if not isinstance(raw, dict):
        raise _ranking_invalid(f"ranking row {index} must be an object")
    if set(raw) != _ROW_KEYS:
        raise _ranking_invalid(f"ranking row {index} has an invalid field set")
    device = raw["device"]
    if device not in {"npu", "cuda", "cpu"}:
        raise _ranking_invalid(f"ranking row {index} has an invalid device")
    eligible = raw["eligible"]
    if not isinstance(eligible, bool):
        raise _ranking_invalid(f"ranking row {index} eligible must be boolean")
    return RankingRow(
        backend=_non_empty_string(raw["backend"], "backend"),
        backend_version=_non_empty_string(raw["backend_version"], "backend_version"),
        device=device,
        model_id=_non_empty_string(raw["model_id"], "model_id"),
        cer=_number(raw["cer"], "cer", minimum=0.0),
        line_accuracy=_number(raw["line_accuracy"], "line_accuracy", minimum=0.0, maximum=1.0),
        layout_f1=_number(raw["layout_f1"], "layout_f1", minimum=0.0, maximum=1.0),
        throughput_pages_s=_number(
            raw["throughput_pages_s"],
            "throughput_pages_s",
            minimum=0.0,
            exclusive_minimum=True,
        ),
        eligible=eligible,
    )


def parse_ranking_document(payload: Any) -> RankingTable:
    if not isinstance(payload, dict):
        raise _ranking_invalid("ranking table must be an object")
    if set(payload) != _TOP_LEVEL_KEYS:
        raise _ranking_invalid("ranking table has an invalid field set")
    if payload["schema_version"] != RANKING_SCHEMA_VERSION:
        raise _ranking_invalid("ranking schema_version is invalid")
    if payload["primary_metric"] != RANKING_PRIMARY_METRIC:
        raise _ranking_invalid("ranking primary_metric must be 'cer'")
    ranking_version = _non_empty_string(payload["ranking_version"], "ranking_version")
    evaluation_set_version = _non_empty_string(
        payload["evaluation_set_version"], "evaluation_set_version"
    )
    rows_raw = payload["rows"]
    if not isinstance(rows_raw, list) or not rows_raw or len(rows_raw) > MAX_RANKING_ROWS:
        raise _ranking_invalid("ranking rows must be a non-empty bounded array")
    rows = tuple(_parse_row(row, index) for index, row in enumerate(rows_raw))
    seen = set()
    for row in rows:
        key = (row.backend, row.backend_version, row.device, row.model_id)
        if key in seen:
            raise _ranking_invalid("ranking rows contain a duplicate backend tuple")
        seen.add(key)
    return RankingTable(
        ranking_version=ranking_version,
        evaluation_set_version=evaluation_set_version,
        rows=rows,
    )


def _read_file(path: Path) -> RankingTable:
    try:
        info = path.stat()
    except OSError as exc:
        raise _ranking_invalid("ranking file is not readable") from exc
    if not stat.S_ISREG(info.st_mode):
        raise _ranking_invalid("ranking path is not a regular file")
    if info.st_size > MAX_RANKING_FILE_BYTES:
        raise _ranking_invalid("ranking file exceeds 10 MiB")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _ranking_invalid("ranking file is not readable") from exc
    if len(text.splitlines()) > MAX_RANKING_LINES:
        raise _ranking_invalid("ranking file exceeds 10000 lines")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise _ranking_invalid("ranking YAML is invalid") from exc
    return parse_ranking_document(payload)


def _validate_explicit_path(raw: Any, *, parameter: bool) -> Path:
    if not isinstance(raw, str) or not raw:
        raise _invalid("backend_ranking_path must be a non-empty local path")
    if raw.startswith("file:") or _URL_RE.match(raw):
        raise _invalid("backend_ranking_path must not be a URL")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        normalized = path.resolve(strict=False)
    except OSError as exc:
        raise _ranking_invalid("ranking path is not readable") from exc
    if not normalized.exists():
        if parameter:
            raise _invalid("backend_ranking_path does not exist")
        raise _ranking_invalid("ranking file does not exist")
    return normalized


def resolve_ranking_table(
    backend_ranking_path: Optional[str],
    *,
    environ: Optional[Mapping[str, str]] = None,
    package_path: Optional[Path] = None,
) -> Optional[RankingTable]:
    env = os.environ if environ is None else environ
    if backend_ranking_path:
        return _read_file(_validate_explicit_path(backend_ranking_path, parameter=True))
    env_path = env.get("CURATOR_OCR_RANKING")
    if env_path:
        return _read_file(_validate_explicit_path(env_path, parameter=False))
    default_path = package_path or Path(__file__).with_name("ocr_backend_ranking.yaml")
    if not default_path.exists():
        return None
    return _read_file(default_path)


__all__ = [
    "RANKING_SCHEMA_VERSION",
    "RankingRow",
    "RankingTable",
    "parse_ranking_document",
    "resolve_ranking_table",
]
