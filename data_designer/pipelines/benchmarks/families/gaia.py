"""Read-only importer for a locally authorized original GAIA metadata snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..contracts import (
    BenchmarkError,
    canonical_sha256,
    file_sha256,
    require_nonempty,
    require_sha,
    require_split,
)
from ..runtime import lock_runtime

def in_generation_checkout(path: Path) -> bool:
    """Detect this generation repo even when the package is wheel-installed."""
    target = path.resolve()
    return any(
        (ancestor / "data_designer" / "pipelines" / "benchmarks").is_dir()
        and (ancestor / "data_designer" / "sources").is_dir()
        for ancestor in (target, *target.parents)
    )


def _outside_public_checkout(path: Path, name: str) -> None:
    if in_generation_checkout(path):
        raise BenchmarkError(f"restricted {name} cannot live in this repository")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read GAIA runtime registry: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkError("GAIA runtime registry must be an object")
    return value


def _level(value: Any) -> int:
    if type(value) is int and value in (1, 2, 3):
        return value
    if isinstance(value, str) and value in ("1", "2", "3"):
        return int(value)
    raise BenchmarkError("GAIA Level must be 1, 2, or 3")


def _attachment(row: Dict[str, Any], root: Path) -> Dict[str, str] | None:
    raw = row.get("file_path") or row.get("file_name")
    if raw in (None, ""):
        return None
    relative = Path(require_nonempty(raw, "GAIA file_path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise BenchmarkError("GAIA attachment path must stay within snapshot root")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise BenchmarkError(f"GAIA attachment unavailable: {relative}")
    _outside_public_checkout(target, "attachment")
    return {"path": str(target), "sha256": file_sha256(target)}


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot_format(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "gaia-jsonl-v1"
    if path.suffix.lower() == ".parquet":
        return "gaia-parquet-v1"
    raise BenchmarkError("GAIA snapshot must be .jsonl or .parquet")


def _rows(
    path: Path, format_name: str
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if format_name == "gaia-jsonl-v1":
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(
                        raw_line.decode("utf-8"),
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            ValueError(token)
                        ),
                    )
                except (UnicodeError, ValueError) as error:
                    raise BenchmarkError(
                        f"invalid GAIA JSONL row {line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise BenchmarkError(f"GAIA row {line_number} must be an object")
                yield row, {
                    "line": line_number,
                    "sha256": hashlib.sha256(raw_line).hexdigest(),
                }
        return
    if format_name != "gaia-parquet-v1":
        raise BenchmarkError("unsupported GAIA snapshot format")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise BenchmarkError("GAIA Parquet import requires pyarrow") from error
    try:
        table = parquet.read_table(path)
        for row_index, row in enumerate(table.to_pylist()):
            if not isinstance(row, dict):
                raise BenchmarkError(f"GAIA Parquet row {row_index} invalid")
            yield row, {"row_index": row_index, "sha256": canonical_sha256(row)}
    except BenchmarkError:
        raise
    except Exception as error:
        raise BenchmarkError(f"cannot read GAIA Parquet snapshot: {error}") from error


def _row_case(
    row: Dict[str, Any], row_ref: Dict[str, Any], root: Path,
    upstream_split: str,
) -> Dict[str, Any]:
    task_id = require_nonempty(row.get("task_id"), "GAIA task_id")
    question = require_nonempty(row.get("Question"), "GAIA Question")
    level = _level(row.get("Level"))
    answer = row.get("Final answer")
    if upstream_split == "dev":
        answer = require_nonempty(answer, "GAIA Final answer")
    elif answer not in (None, ""):
        raise BenchmarkError("GAIA test answers must not be imported")
    answer_sha = _sha_text(answer) if isinstance(answer, str) else None
    attachment = _attachment(row, root)
    attachments = [attachment] if attachment else []
    return {
        "source_task_id": task_id,
        "task_text": question,
        "initial_state_sha256": canonical_sha256(
            {
                "task_id": task_id,
                "question_sha256": _sha_text(question),
                "attachments": attachments,
            }
        ),
        "expected_state_sha256": canonical_sha256(
            {"answer_sha256": answer_sha, "oracle_available": answer_sha is not None}
        ),
        "source_row": row_ref,
        "gaia": {
            "level": level,
            "answer_sha256": answer_sha,
            "oracle_available": answer_sha is not None,
            "attachments": attachments,
        },
    }


def import_gaia_snapshot(
    snapshot_path: str,
    snapshot_sha256: str,
    *,
    release: str,
    split: str,
    upstream_split: str,
    license_record: Dict[str, Any],
    attachment_root: Optional[str] = None,
    runtime_registry_path: Optional[str] = None,
    runtime_registry_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Import authorized GAIA JSONL or Parquet without exposing answer text."""
    snapshot = Path(require_nonempty(snapshot_path, "snapshot_path"))
    snapshot_sha = require_sha(snapshot_sha256, "snapshot_sha256")
    if not snapshot.is_absolute() or not snapshot.is_file():
        raise BenchmarkError("GAIA snapshot must be an existing absolute file")
    _outside_public_checkout(snapshot, "snapshot")
    if file_sha256(snapshot) != snapshot_sha:
        raise BenchmarkError("GAIA snapshot SHA-256 drift")
    format_name = _snapshot_format(snapshot)
    release = require_nonempty(release, "release")
    split = require_split(split)
    if upstream_split not in ("dev", "test"):
        raise BenchmarkError("GAIA upstream split must be dev or test")
    if upstream_split == "test" and split != "holdout":
        raise BenchmarkError("GAIA test cannot seed development or regression")
    if (
        not isinstance(license_record, dict)
        or not isinstance(license_record.get("id"), str)
        or not license_record["id"].strip()
        or type(license_record.get("derive_allowed")) is not bool
        or license_record.get("redistribute_allowed") is not False
        or not isinstance(license_record.get("access_conditions"), str)
        or not license_record["access_conditions"].strip()
    ):
        raise BenchmarkError("GAIA gated access and redistribution decision required")
    root = Path(attachment_root) if attachment_root else snapshot.parent
    if not root.is_absolute() or not root.is_dir():
        raise BenchmarkError("GAIA attachment root must be an absolute directory")
    _outside_public_checkout(root, "attachment root")
    runtimes: Dict[str, Dict[str, Any]] | None = None
    registry_sha: str | None = None
    if runtime_registry_path is not None or runtime_registry_sha256 is not None:
        registry_path = Path(
            require_nonempty(runtime_registry_path, "runtime_registry_path")
        )
        registry_sha = require_sha(runtime_registry_sha256, "runtime_registry_sha256")
        if not registry_path.is_absolute() or not registry_path.is_file():
            raise BenchmarkError("GAIA runtime registry unavailable")
        if file_sha256(registry_path) != registry_sha:
            raise BenchmarkError("GAIA runtime registry SHA-256 drift")
        registry = _load_json(registry_path)
        if (
            registry.get("schema_version") != "benchmark-runtime-registry-v1"
            or not isinstance(registry.get("cases"), dict)
        ):
            raise BenchmarkError("invalid GAIA runtime registry schema")
        runtimes = registry["cases"]
    cases: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        for row, row_ref in _rows(snapshot, format_name):
            case = _row_case(row, row_ref, root, upstream_split)
            task_id = case["source_task_id"]
            if task_id in seen_ids:
                raise BenchmarkError(f"duplicate GAIA task_id {task_id}")
            seen_ids.add(task_id)
            if runtimes is not None:
                if task_id not in runtimes:
                    raise BenchmarkError(f"missing GAIA runtime for {task_id}")
                runtime = lock_runtime(runtimes[task_id])
                by_path = {asset["path"]: asset for asset in runtime["assets"]}
                for attachment in case["gaia"]["attachments"]:
                    existing = by_path.get(attachment["path"])
                    if existing is not None and existing != attachment:
                        raise BenchmarkError(
                            "GAIA attachment runtime SHA-256 conflict"
                        )
                    by_path[attachment["path"]] = attachment
                runtime["assets"] = list(by_path.values())
                case["runtime"] = runtime
            cases.append(case)
    except OSError as error:
        raise BenchmarkError(f"cannot read GAIA snapshot: {error}") from error
    if not cases:
        raise BenchmarkError("GAIA snapshot has no usable cases")
    if runtimes is not None and set(runtimes) != seen_ids:
        raise BenchmarkError("GAIA runtime registry contains unknown task IDs")
    return {
        "schema_version": "benchmark-source-v1",
        "family": "gaia",
        "release": release,
        "split": split,
        "license": dict(license_record),
        "source_snapshot": {
            "path": str(snapshot),
            "sha256": snapshot_sha,
            "format": format_name,
            "upstream_split": upstream_split,
            "attachment_root": str(root),
        },
        "runtime_registry_sha256": registry_sha,
        "cases": cases,
    }


def verify_gaia_source_manifest(source: Dict[str, Any]) -> None:
    """Recheck every GAIA source row and attachment before native audit."""
    snapshot_record = source.get("source_snapshot")
    if (
        not isinstance(snapshot_record, dict)
        or snapshot_record.get("format") not in ("gaia-jsonl-v1", "gaia-parquet-v1")
        or snapshot_record.get("upstream_split") not in ("dev", "test")
    ):
        raise BenchmarkError("GAIA source needs a supported pinned snapshot")
    license_record = source.get("license")
    if (
        not isinstance(license_record, dict)
        or license_record.get("redistribute_allowed") is not False
        or not isinstance(license_record.get("access_conditions"), str)
        or not license_record["access_conditions"].strip()
    ):
        raise BenchmarkError("GAIA gated redistribution record missing")
    if snapshot_record["upstream_split"] == "test" and source.get("split") != "holdout":
        raise BenchmarkError("GAIA test cannot seed development or regression")
    snapshot = Path(require_nonempty(snapshot_record.get("path"), "snapshot.path"))
    expected_sha = require_sha(snapshot_record.get("sha256"), "snapshot.sha256")
    root = Path(
        require_nonempty(snapshot_record.get("attachment_root"), "attachment_root")
    )
    if not snapshot.is_absolute() or not snapshot.is_file() or not root.is_dir():
        raise BenchmarkError("GAIA source snapshot or attachment root unavailable")
    _outside_public_checkout(snapshot, "snapshot")
    _outside_public_checkout(root, "attachment root")
    if file_sha256(snapshot) != expected_sha:
        raise BenchmarkError("GAIA source snapshot drift")
    format_name = _snapshot_format(snapshot)
    if snapshot_record["format"] != format_name:
        raise BenchmarkError("GAIA source snapshot format drift")
    cases = source.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("GAIA source cases required")
    index_key = "line" if format_name == "gaia-jsonl-v1" else "row_index"
    min_index = 1 if index_key == "line" else 0
    by_index: Dict[int, Dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("source_row"), dict):
            raise BenchmarkError("GAIA source row reference missing")
        index = case["source_row"].get(index_key)
        if type(index) is not int or index < min_index or index in by_index:
            raise BenchmarkError("GAIA source row line invalid or duplicated")
        require_sha(case["source_row"].get("sha256"), "source_row.sha256")
        by_index[index] = case
    found: set[int] = set()
    for row, row_ref in _rows(snapshot, format_name):
        index = row_ref[index_key]
        case = by_index.get(index)
        if case is None:
            continue
        found.add(index)
        if case["source_row"] != row_ref:
            raise BenchmarkError(f"GAIA source row {index} SHA-256 drift")
        expected = _row_case(row, row_ref, root, snapshot_record["upstream_split"])
        for key in (
            "source_task_id", "task_text", "initial_state_sha256",
            "expected_state_sha256", "source_row", "gaia",
        ):
            if case.get(key) != expected[key]:
                raise BenchmarkError(f"GAIA source row {index} metadata drift: {key}")
    if found != set(by_index):
        raise BenchmarkError("GAIA source row reference out of range")
