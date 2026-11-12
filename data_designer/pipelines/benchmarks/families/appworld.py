"""Read-only AppWorld task-directory importer; protected data stays local."""

from __future__ import annotations

import json
from datetime import datetime
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

UPSTREAM_SPLITS = frozenset({"train", "dev", "test_normal", "test_challenge"})


def _private_path(path_value: Any, name: str) -> Path:
    path = Path(require_nonempty(path_value, name))
    if not path.is_absolute() or not path.exists():
        raise BenchmarkError(f"AppWorld {name} must be an existing absolute path")
    if in_generation_checkout(path):
        raise BenchmarkError(f"AppWorld {name} cannot live in this repository")
    return path


def _pin(path: Path) -> Dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise BenchmarkError(f"AppWorld source file unavailable or linked: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def _tree(directory: Path, name: str) -> List[Dict[str, str]]:
    if not directory.is_dir() or directory.is_symlink():
        raise BenchmarkError(f"AppWorld {name} directory unavailable")
    paths = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise BenchmarkError(f"AppWorld {name} cannot contain links")
    # Executing the upstream verifier writes Python bytecode next to its
    # protected evaluation.py. It is runtime output, not part of the source
    # snapshot; every other added file remains visible to drift checks.
    files = [
        _pin(path)
        for path in paths
        if path.is_file() and "__pycache__" not in path.relative_to(directory).parts
    ]
    if not files:
        raise BenchmarkError(f"AppWorld {name} must contain files")
    return files


def _json(path: Path, name: str) -> Dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read AppWorld {name}: {error}") from error
    if not isinstance(data, dict):
        raise BenchmarkError(f"AppWorld {name} must be an object")
    return data


def _task_case(task_id: str, root: Path) -> Dict[str, Any]:
    task_dir = root / "tasks" / task_id
    if not task_dir.is_dir() or task_dir.is_symlink():
        raise BenchmarkError(f"AppWorld task directory unavailable: {task_id}")
    specs_path = task_dir / "specs.json"
    specs = _json(specs_path, f"{task_id} specs")
    instruction = require_nonempty(specs.get("instruction"), "AppWorld instruction")
    timestamp = require_nonempty(specs.get("datetime"), "AppWorld datetime")
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise BenchmarkError("AppWorld task datetime invalid") from error
    db_version = require_nonempty(specs.get("db_version"), "AppWorld db_version")
    supervisor = specs.get("supervisor", specs.get("main_user"))
    if not isinstance(supervisor, dict) or not supervisor:
        raise BenchmarkError("AppWorld supervisor required")
    db_files = _tree(task_dir / "dbs", f"{task_id} dbs")
    ground_dir = task_dir / "ground_truth"
    ground_files = _tree(ground_dir, f"{task_id} ground_truth")
    required_ground = {
        "metadata.json", "answer.json", "evaluation.py",
        "public_data.json", "private_data.json", "test_data.json",
    }
    if not required_ground <= {Path(pin["path"]).name for pin in ground_files}:
        raise BenchmarkError(f"AppWorld task {task_id} ground truth incomplete")
    metadata = _json(ground_dir / "metadata.json", f"{task_id} metadata")
    mode = metadata.get("mode", "full")
    if mode not in ("minimal", "partial", "full"):
        raise BenchmarkError(f"AppWorld task {task_id} ground truth mode invalid")
    if type(metadata.get("difficulty")) is not int:
        raise BenchmarkError(f"AppWorld task {task_id} difficulty invalid")
    specs_pin = _pin(specs_path)
    files = [specs_pin, *db_files, *ground_files]
    return {
        "source_task_id": task_id,
        "task_text": instruction,
        "initial_state_sha256": canonical_sha256(
            {
                "datetime": timestamp,
                "db_version": db_version,
                "supervisor_sha256": canonical_sha256(supervisor),
                "db_files": db_files,
            }
        ),
        "expected_state_sha256": canonical_sha256(ground_files),
        "appworld": {
            "datetime": timestamp,
            "db_version": db_version,
            "supervisor_sha256": canonical_sha256(supervisor),
            "ground_truth_mode": mode,
            "difficulty": metadata["difficulty"],
            "db_file_count": len(db_files),
            "ground_truth_file_count": len(ground_files),
            "source_files": files,
        },
    }


def assert_appworld_runtime_assets(
    runtime: Dict[str, Any], source_snapshot: Dict[str, Any], case: Dict[str, Any]
) -> None:
    assets = runtime.get("assets") if isinstance(runtime, dict) else None
    metadata = case.get("appworld") if isinstance(case, dict) else None
    if not isinstance(assets, list) or not isinstance(metadata, dict):
        raise BenchmarkError("AppWorld runtime assets missing")
    required = [
        {"path": source_snapshot["path"], "sha256": source_snapshot["sha256"]},
        *metadata["source_files"],
    ]
    for pin in required:
        if pin not in assets:
            raise BenchmarkError("AppWorld runtime missing pinned source asset")


def import_appworld_snapshot(
    data_root: str,
    dataset_sha256: str,
    *,
    upstream_split: str,
    release: str,
    split: str,
    license_record: Dict[str, Any],
    runtime_registry_path: Optional[str] = None,
    runtime_registry_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Import one official dataset list and hash each selected task directory."""
    root = _private_path(data_root, "data root")
    if not root.is_dir() or root.is_symlink():
        raise BenchmarkError("AppWorld data root must be a real directory")
    if upstream_split not in UPSTREAM_SPLITS:
        raise BenchmarkError("AppWorld upstream split unsupported")
    split = require_split(split)
    if upstream_split.startswith("test_") and split != "holdout":
        raise BenchmarkError("AppWorld test cannot seed development or regression")
    if upstream_split in ("train", "dev") and split == "holdout":
        raise BenchmarkError("AppWorld train/dev cannot be labeled holdout")
    release = require_nonempty(release, "AppWorld release")
    if (
        not isinstance(license_record, dict)
        or not isinstance(license_record.get("id"), str)
        or not license_record["id"].strip()
        or type(license_record.get("derive_allowed")) is not bool
        or license_record.get("redistribute_allowed") is not False
        or not isinstance(license_record.get("access_conditions"), str)
        or not license_record["access_conditions"].strip()
    ):
        raise BenchmarkError("AppWorld protected-data license decision required")
    dataset_path = root / "datasets" / f"{upstream_split}.txt"
    dataset_pin = _pin(dataset_path)
    if dataset_pin["sha256"] != require_sha(dataset_sha256, "dataset_sha256"):
        raise BenchmarkError("AppWorld dataset SHA-256 drift")
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BenchmarkError(f"cannot read AppWorld dataset list: {error}") from error
    task_ids = [line.strip().split(":", 1)[0] for line in lines if line.strip()]
    if (
        not task_ids
        or any(not value for value in task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise BenchmarkError("AppWorld dataset list empty or duplicated")
    if any(
        "/" in task_id or "\\" in task_id or task_id in (".", "..")
        for task_id in task_ids
    ):
        raise BenchmarkError("AppWorld task ID cannot be a path")
    runtimes: Dict[str, Any] | None = None
    registry_sha: str | None = None
    if runtime_registry_path is not None or runtime_registry_sha256 is not None:
        registry_path = _private_path(runtime_registry_path, "runtime registry")
        registry_sha = require_sha(
            runtime_registry_sha256, "runtime_registry_sha256"
        )
        if not registry_path.is_file() or file_sha256(registry_path) != registry_sha:
            raise BenchmarkError("AppWorld runtime registry SHA-256 drift")
        registry = _json(registry_path, "runtime registry")
        if (
            registry.get("schema_version") != "benchmark-runtime-registry-v1"
            or not isinstance(registry.get("cases"), dict)
        ):
            raise BenchmarkError("invalid AppWorld runtime registry schema")
        runtimes = registry["cases"]
        if set(runtimes) != set(task_ids):
            raise BenchmarkError("AppWorld runtimes must match dataset task IDs")
    cases: List[Dict[str, Any]] = []
    for task_id in task_ids:
        case = _task_case(task_id, root)
        if runtimes is not None:
            runtime = lock_runtime(runtimes[task_id])
            by_path = {asset["path"]: asset for asset in runtime["assets"]}
            for pin in [dataset_pin, *case["appworld"]["source_files"]]:
                existing = by_path.get(pin["path"])
                if existing is not None and existing != pin:
                    raise BenchmarkError("AppWorld runtime asset SHA-256 conflict")
                by_path[pin["path"]] = pin
            runtime["assets"] = list(by_path.values())
            case["runtime"] = runtime
        cases.append(case)
    return {
        "schema_version": "benchmark-source-v1",
        "family": "appworld",
        "release": release,
        "split": split,
        "license": dict(license_record),
        "source_snapshot": {
            **dataset_pin,
            "format": "appworld-task-directories-v1",
            "data_root": str(root),
            "upstream_split": upstream_split,
        },
        "runtime_registry_sha256": registry_sha,
        "cases": cases,
    }


def verify_appworld_source_manifest(source: Dict[str, Any]) -> None:
    snapshot = source.get("source_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("format") != "appworld-task-directories-v1"
    ):
        raise BenchmarkError("AppWorld source needs a pinned task directory snapshot")
    expected = import_appworld_snapshot(
        snapshot.get("data_root"), snapshot.get("sha256"),
        upstream_split=snapshot.get("upstream_split"),
        release=source.get("release"),
        split=source.get("split"),
        license_record=source.get("license"),
    )
    if snapshot != expected["source_snapshot"]:
        raise BenchmarkError("AppWorld source snapshot metadata drift")
    cases = source.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected["cases"]):
        raise BenchmarkError("AppWorld source cases incomplete")
    for actual, fresh in zip(cases, expected["cases"]):
        if not isinstance(actual, dict):
            raise BenchmarkError("AppWorld source case invalid")
        for key, value in fresh.items():
            if actual.get(key) != value:
                raise BenchmarkError(f"AppWorld source metadata drift: {key}")
        if "runtime" in actual:
            assert_appworld_runtime_assets(actual["runtime"], snapshot, actual)
