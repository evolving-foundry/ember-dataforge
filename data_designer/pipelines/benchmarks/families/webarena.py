"""Read-only importer for a pinned WebArena-Verified task JSON snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts import (
    BenchmarkError,
    canonical_sha256,
    file_sha256,
    require_nonempty,
    require_sha,
    require_split,
)
from ..runtime import lock_runtime
from .gaia import in_generation_checkout


def _pin(path_value: Any, sha_value: Any, name: str) -> Dict[str, str]:
    path = Path(require_nonempty(path_value, f"{name}.path"))
    sha = require_sha(sha_value, f"{name}.sha256")
    if not path.is_absolute() or not path.is_file():
        raise BenchmarkError(f"WebArena {name} must be an absolute file")
    if in_generation_checkout(path):
        raise BenchmarkError(f"WebArena {name} cannot live in this repository")
    if file_sha256(path) != sha:
        raise BenchmarkError(f"WebArena {name} SHA-256 drift")
    return {"path": str(path), "sha256": sha}


def _json(path: str, name: str) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read WebArena {name}: {error}") from error


def _strings(value: Any, name: str) -> List[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(part, str) and part.strip() for part in value
    ):
        raise BenchmarkError(f"WebArena {name} must be a nonempty string list")
    return value


def _case(row: Any, index: int) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise BenchmarkError(f"WebArena row {index} must be an object")
    task_id = row.get("task_id")
    revision = row.get("revision")
    template_id = row.get("intent_template_id")
    if type(task_id) is not int or task_id < 0:
        raise BenchmarkError(f"WebArena row {index} task_id invalid")
    if type(revision) is not int or revision < 1:
        raise BenchmarkError(f"WebArena task {task_id} revision invalid")
    if type(template_id) is not int or template_id < 0:
        raise BenchmarkError(f"WebArena task {task_id} template ID invalid")
    intent = require_nonempty(row.get("intent"), "WebArena intent")
    sites = _strings(row.get("sites"), "sites")
    start_urls = _strings(row.get("start_urls"), "start_urls")
    evaluators = row.get("eval")
    if not isinstance(evaluators, list) or not evaluators or not all(
        isinstance(value, dict)
        and isinstance(value.get("evaluator"), str)
        and value["evaluator"]
        for value in evaluators
    ):
        raise BenchmarkError(f"WebArena task {task_id} evaluators invalid")
    return {
        "source_task_id": str(task_id),
        "task_text": intent,
        "initial_state_sha256": canonical_sha256(
            {"sites": sites, "start_urls": start_urls}
        ),
        "expected_state_sha256": canonical_sha256(evaluators),
        "source_row": {"index": index, "sha256": canonical_sha256(row)},
        "webarena": {
            "revision": revision,
            "intent_template_id": template_id,
            "sites": sites,
            "start_urls": start_urls,
            "evaluation_sha256": canonical_sha256(evaluators),
            "evaluator_types": [value["evaluator"] for value in evaluators],
            "format_specification_sha256": canonical_sha256(
                row.get("format_specification")
            ),
        },
    }


def assert_webarena_runtime_assets(
    runtime: Dict[str, Any], source_snapshot: Dict[str, Any]
) -> None:
    assets = runtime.get("assets") if isinstance(runtime, dict) else None
    files = source_snapshot.get("files") if isinstance(source_snapshot, dict) else None
    if not isinstance(assets, list) or not isinstance(files, dict):
        raise BenchmarkError("WebArena runtime/source assets missing")
    for name, pin in files.items():
        if pin not in assets:
            raise BenchmarkError(f"WebArena runtime missing pinned {name} asset")


def import_webarena_snapshot(
    snapshot_path: str,
    snapshot_sha256: str,
    *,
    release: str,
    split: str,
    license_record: Dict[str, Any],
    selection_path: Optional[str] = None,
    selection_sha256: Optional[str] = None,
    selection_use: Optional[str] = None,
    runtime_registry_path: Optional[str] = None,
    runtime_registry_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Import selected original tasks; derived tasks need an explicit selection."""
    files = {"tasks": _pin(snapshot_path, snapshot_sha256, "tasks")}
    release = require_nonempty(release, "WebArena release")
    split = require_split(split)
    if (selection_path is None) != (selection_sha256 is None):
        raise BenchmarkError("WebArena selection path and SHA-256 must be paired")
    if selection_path is not None:
        files["selection"] = _pin(
            selection_path, selection_sha256, "selection"
        )
    if split in ("development", "regression") and (
        "selection" not in files or selection_use != "derivation"
    ):
        raise BenchmarkError("WebArena derivation needs explicit pinned selection")
    if selection_use not in (None, "derivation", "benchmark"):
        raise BenchmarkError("WebArena selection_use invalid")
    if split == "holdout" and selection_use == "derivation":
        raise BenchmarkError("WebArena derivation selection cannot be holdout")
    if (
        not isinstance(license_record, dict)
        or not isinstance(license_record.get("id"), str)
        or not license_record["id"].strip()
        or type(license_record.get("derive_allowed")) is not bool
    ):
        raise BenchmarkError("explicit WebArena license decision required")
    rows = _json(files["tasks"]["path"], "tasks")
    if not isinstance(rows, list) or not rows:
        raise BenchmarkError("WebArena tasks must be a nonempty list")
    all_cases = [_case(row, index) for index, row in enumerate(rows)]
    by_id = {case["source_task_id"]: case for case in all_cases}
    if len(by_id) != len(all_cases):
        raise BenchmarkError("WebArena task IDs duplicated")
    if "selection" in files:
        selection = _json(files["selection"]["path"], "selection")
        ids = selection.get("task_ids") if isinstance(selection, dict) else None
        if not isinstance(ids, list) or not ids or not all(
            type(task_id) is int and task_id >= 0 for task_id in ids
        ) or len(ids) != len(set(ids)):
            raise BenchmarkError("WebArena selection.task_ids invalid")
        selected_ids = [str(task_id) for task_id in ids]
        if not set(selected_ids) <= set(by_id):
            raise BenchmarkError("WebArena selection has unknown task IDs")
    else:
        selected_ids = [case["source_task_id"] for case in all_cases]
    runtimes: Dict[str, Any] | None = None
    registry_sha: str | None = None
    if runtime_registry_path is not None or runtime_registry_sha256 is not None:
        registry_pin = _pin(
            runtime_registry_path, runtime_registry_sha256, "runtime registry"
        )
        registry_sha = registry_pin["sha256"]
        registry = _json(registry_pin["path"], "runtime registry")
        if (
            not isinstance(registry, dict)
            or registry.get("schema_version") != "benchmark-runtime-registry-v1"
            or not isinstance(registry.get("cases"), dict)
        ):
            raise BenchmarkError("invalid WebArena runtime registry schema")
        runtimes = registry["cases"]
        if set(runtimes) != set(selected_ids):
            raise BenchmarkError("WebArena runtimes must match selected task IDs")
    cases: List[Dict[str, Any]] = []
    for task_id in selected_ids:
        case = dict(by_id[task_id])
        if runtimes is not None:
            runtime = lock_runtime(runtimes[task_id])
            by_path = {asset["path"]: asset for asset in runtime["assets"]}
            for pin in files.values():
                existing = by_path.get(pin["path"])
                if existing is not None and existing != pin:
                    raise BenchmarkError("WebArena runtime asset SHA-256 conflict")
                by_path[pin["path"]] = pin
            runtime["assets"] = list(by_path.values())
            case["runtime"] = runtime
        cases.append(case)
    return {
        "schema_version": "benchmark-source-v1",
        "family": "webarena",
        "release": release,
        "split": split,
        "license": dict(license_record),
        "source_snapshot": {
            **files["tasks"],
            "format": "webarena-verified-json-v1",
            "selection_use": selection_use,
            "files": files,
        },
        "runtime_registry_sha256": registry_sha,
        "cases": cases,
    }


def verify_webarena_source_manifest(source: Dict[str, Any]) -> None:
    snapshot = source.get("source_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("format") != "webarena-verified-json-v1"
        or not isinstance(snapshot.get("files"), dict)
    ):
        raise BenchmarkError("WebArena source needs a pinned Verified snapshot")
    files = snapshot["files"]
    if set(files) not in ({"tasks"}, {"tasks", "selection"}):
        raise BenchmarkError("WebArena source files invalid")
    if {key: snapshot.get(key) for key in ("path", "sha256")} != files["tasks"]:
        raise BenchmarkError("WebArena primary snapshot pin mismatch")
    for name, pin in files.items():
        if not isinstance(pin, dict):
            raise BenchmarkError(f"WebArena {name} pin invalid")
        _pin(pin.get("path"), pin.get("sha256"), name)
    expected = import_webarena_snapshot(
        files["tasks"]["path"], files["tasks"]["sha256"],
        release=source.get("release"),
        split=source.get("split"),
        license_record=source.get("license"),
        selection_path=files.get("selection", {}).get("path"),
        selection_sha256=files.get("selection", {}).get("sha256"),
        selection_use=snapshot.get("selection_use"),
    )
    if snapshot != expected["source_snapshot"]:
        raise BenchmarkError("WebArena source snapshot metadata drift")
    cases = source.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected["cases"]):
        raise BenchmarkError("WebArena source cases incomplete")
    for actual, fresh in zip(cases, expected["cases"]):
        if not isinstance(actual, dict):
            raise BenchmarkError("WebArena source case invalid")
        for key, value in fresh.items():
            if actual.get(key) != value:
                raise BenchmarkError(f"WebArena source metadata drift: {key}")
        if "runtime" in actual:
            assert_webarena_runtime_assets(actual["runtime"], snapshot)
