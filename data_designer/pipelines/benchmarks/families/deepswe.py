"""Read-only importer for a pinned DeepSWE-Bench tree snapshot.

DeepSWE-Bench v1.1 is a single immutable Git tree (pinned upstream commit,
113 tasks, task schema 1.3; each task carries ``task.toml`` /
``instruction.md`` / ``environment/`` / ``tests/test.sh``). The trusted-side
tree manifest freezes the task path list and per-file SHA-256 pins; the
importer re-verifies every pinned file against the local authorized checkout
and keeps ``task.toml`` / ``instruction.md`` / ``tests`` in the source data
plane — the generated manifest stores only hashes and references.

Scoring re-expression (RSI_deepswe scoring re-expression addendum v0.1 DS1/DS5): the dataset does not include a gold patch,
so the expected file tree cannot be pre-generated from inside the dataset. Admission requires two criteria: (1) the gold diff can be pinned and acquired
(``gold_diff`` pin in the trusted-side offline registry); (2) the base snapshot can be pinned offline and a real
``git apply`` proves it succeeds. If either is missing, the result is ``excluded/gold_unavailable`` or
``excluded/base_unavailable``--no scoring-derived data enters any stage of taskset→suite→eval.
``environment/`` and ``tests/test.sh`` do not enter the scoring data plane and are provenance only.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
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
from ..runners.deepswe_offline import (
    SPEC_KIND,
    VERSION as OFFLINE_RUNNER_VERSION,
    _apply_patch,
    _clean_relative,
    _extract_snapshot,
    _sha256_file,
)
from ..runtime import lock_runtime

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
TREE_SNAPSHOT_SCHEMA = "deepswe-tree-snapshot-v1"
OFFLINE_REGISTRY_SCHEMA = "deepswe-offline-registry-v1"
SNAPSHOT_FORMAT = "deepswe-tree-v1"
REQUIRED_TREE_FILES = ("task.toml", "instruction.md", "tests/test.sh")
_HEX = frozenset("0123456789abcdef")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must be an object")
    return value


def _pin_path(pin: Any, task_id: Any, label: str) -> Path:
    if not isinstance(pin, dict):
        raise BenchmarkError(f"invalid {label} pin for {task_id}")
    path = Path(require_nonempty(pin.get("path"), f"{label}.path"))
    sha = require_sha(pin.get("sha256"), f"{label}.sha256")
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != sha:
        raise BenchmarkError(f"{label} missing or SHA-256 drift for {task_id}")
    return path


def _tree_sha_map(workdir: Path) -> Dict[str, str]:
    tree: Dict[str, str] = {}
    for target in sorted(workdir.rglob("*")):
        if target.is_symlink() or not target.is_file():
            continue
        tree[target.relative_to(workdir).as_posix()] = _sha256_file(target)
    return tree


def _apply_gold_diff(
    git: str, base_path: Path, gold_bytes: bytes, timeout: float, tmp: Path
) -> Optional[Dict[str, Any]]:
    """Apply the pinned gold diff to a fresh base snapshot; None if inapplicable.

    The expectation is derived empirically by diffing the pre- and post-apply
    trees rather than by parsing patch headers: binary hunks from
    ``git diff --binary`` carry no ``---/+++`` text headers, so header parsing
    would silently miss them.
    """
    workdir = tmp / "workspace"
    workdir.mkdir()
    _extract_snapshot(base_path, workdir)
    pre = _tree_sha_map(workdir)
    patch_file = tmp / "gold.patch"
    patch_file.write_bytes(gold_bytes)
    try:
        _apply_patch(git, workdir, patch_file, timeout, check_only=True)
        _apply_patch(git, workdir, patch_file, timeout, check_only=False)
    except (ValueError, subprocess.SubprocessError, OSError):
        return None
    post = _tree_sha_map(workdir)
    expected_files = {
        relative: sha
        for relative, sha in sorted(post.items())
        if pre.get(relative) != sha
    }
    deleted_paths = sorted(set(pre) - set(post))
    return {
        "expected_files": expected_files,
        "deleted_paths": deleted_paths,
        "affected": len(expected_files) + len(deleted_paths),
    }


def _load_offline_registry(
    offline_registry_path: Optional[str],
    offline_registry_sha256: Optional[str],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if offline_registry_path is None and offline_registry_sha256 is None:
        return None
    registry_path = Path(
        require_nonempty(offline_registry_path, "offline_registry_path")
    )
    registry_sha = require_sha(offline_registry_sha256, "offline_registry_sha256")
    if not registry_path.is_absolute() or not registry_path.is_file():
        raise BenchmarkError("offline registry must be an existing absolute file")
    if file_sha256(registry_path) != registry_sha:
        raise BenchmarkError("deepswe offline registry SHA-256 drift")
    registry = _json_object(registry_path, "deepswe offline registry")
    if (
        registry.get("schema_version") != OFFLINE_REGISTRY_SCHEMA
        or not isinstance(registry.get("cases"), dict)
    ):
        raise BenchmarkError("invalid deepswe offline registry schema")
    cases = registry["cases"]
    for task_id, entry in cases.items():
        if not isinstance(entry, dict):
            raise BenchmarkError(f"invalid deepswe offline entry for {task_id}")
        for label in ("gold_diff", "base_snapshot"):
            pin = entry.get(label)
            if pin is None:
                continue
            _pin_path(pin, task_id, f"offline {label}")
    return cases


def _admit_offline_assets(
    task_id: str, assets: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """DS5 subset admission: gold diff pinnable AND base applies for real."""
    entry = assets.get(task_id)
    if entry is None or entry.get("gold_diff") is None:
        return {"admission": "excluded", "reason": "gold_unavailable"}
    if entry.get("base_snapshot") is None:
        return {"admission": "excluded", "reason": "base_unavailable"}
    base_path = _pin_path(entry["base_snapshot"], task_id, "offline base")
    gold_path = _pin_path(entry["gold_diff"], task_id, "offline gold diff")
    git = shutil.which("git")
    if not git:
        raise BenchmarkError("git binary is required for deepswe offline admission")
    gold_bytes = gold_path.read_bytes()
    git = shutil.which("git")
    if not git:
        raise BenchmarkError("git binary is required for deepswe offline admission")
    with tempfile.TemporaryDirectory(prefix="deepswe-admit-") as tmp:
        try:
            outcome = _apply_gold_diff(git, base_path, gold_bytes, 30.0, Path(tmp))
        except zipfile.BadZipFile:
            return {"admission": "excluded", "reason": "base_unavailable"}
    if outcome is None or outcome["affected"] == 0:
        return {"admission": "excluded", "reason": "base_unavailable"}
    return {
        "admission": "expressible",
        "base_snapshot": {
            "path": str(base_path), "sha256": entry["base_snapshot"]["sha256"],
        },
        "gold_diff": {
            "path": str(gold_path), "sha256": entry["gold_diff"]["sha256"],
        },
    }


def _read_runtime_registry(
    runtime_registry_path: Optional[str],
    runtime_registry_sha256: Optional[str],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if runtime_registry_path is None and runtime_registry_sha256 is None:
        return None
    registry_path = Path(require_nonempty(runtime_registry_path, "runtime_registry_path"))
    registry_sha = require_sha(runtime_registry_sha256, "runtime_registry_sha256")
    if not registry_path.is_absolute() or not registry_path.is_file():
        raise BenchmarkError("runtime registry must be an existing absolute file")
    if file_sha256(registry_path) != registry_sha:
        raise BenchmarkError("deepswe runtime registry SHA-256 drift")
    registry = _json_object(registry_path, "deepswe runtime registry")
    if (
        registry.get("schema_version") != "benchmark-runtime-registry-v1"
        or not isinstance(registry.get("cases"), dict)
    ):
        raise BenchmarkError("invalid deepswe runtime registry schema")
    return registry["cases"]


def import_deepswe_tree(
    tree_path: str,
    tree_sha256: str,
    *,
    release: str,
    split: str,
    license_record: Dict[str, Any],
    runtime_registry_path: Optional[str] = None,
    runtime_registry_sha256: Optional[str] = None,
    offline_registry_path: Optional[str] = None,
    offline_registry_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a pinned local tree snapshot into benchmark-source-v1 data."""
    snapshot = Path(require_nonempty(tree_path, "tree_path"))
    expected_sha = require_sha(tree_sha256, "tree_sha256")
    if not snapshot.is_absolute() or not snapshot.is_file():
        raise BenchmarkError("deepswe tree snapshot must be an existing absolute file")
    if file_sha256(snapshot) != expected_sha:
        raise BenchmarkError("deepswe tree snapshot SHA-256 drift")
    release = require_nonempty(release, "release")
    split = require_split(split)
    if (
        not isinstance(license_record, dict)
        or not isinstance(license_record.get("id"), str)
        or not license_record["id"].strip()
        or type(license_record.get("derive_allowed")) is not bool
    ):
        raise BenchmarkError("explicit source license decision required")
    tree = _json_object(snapshot, "deepswe tree snapshot")
    if tree.get("schema_version") != TREE_SNAPSHOT_SCHEMA:
        raise BenchmarkError("invalid deepswe tree snapshot schema")
    commit = require_nonempty(tree.get("commit"), "tree.commit")
    if not COMMIT_RE.fullmatch(commit):
        raise BenchmarkError("deepswe tree commit must be a 40-hex SHA-1")
    repository_url = require_nonempty(tree.get("repository_url"), "tree.repository_url")
    root = Path(require_nonempty(tree.get("root"), "tree.root"))
    if not root.is_absolute() or not root.is_dir():
        raise BenchmarkError("deepswe tree root must be an existing absolute directory")
    tasks = tree.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise BenchmarkError("deepswe tree snapshot has no tasks")
    runtimes = _read_runtime_registry(runtime_registry_path, runtime_registry_sha256)
    assets = _load_offline_registry(offline_registry_path, offline_registry_sha256)

    cases: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in tasks:
        if not isinstance(entry, dict):
            raise BenchmarkError("deepswe tree task must be an object")
        task_id = require_nonempty(entry.get("task_id"), "task.task_id")
        if task_id in seen_ids:
            raise BenchmarkError(f"duplicate deepswe task_id {task_id}")
        seen_ids.add(task_id)
        base_commit = require_nonempty(
            entry.get("base_commit_hash"), "task.base_commit_hash"
        )
        if not COMMIT_RE.fullmatch(base_commit):
            raise BenchmarkError(f"invalid deepswe base_commit_hash for {task_id}")
        language = require_nonempty(entry.get("language"), "task.language")
        directory = _clean_relative(
            require_nonempty(entry.get("directory"), "task.directory")
        )
        files = entry.get("files")
        if not isinstance(files, dict) or not files:
            raise BenchmarkError(f"deepswe task {task_id} lacks a file pin map")
        for relative, sha in files.items():
            _clean_relative(str(relative))
            if not _is_sha256(sha):
                raise BenchmarkError(f"file pin lacks a sha256: {relative!r}")
        missing = [name for name in REQUIRED_TREE_FILES if name not in files]
        if missing:
            raise BenchmarkError(
                f"deepswe task {task_id} lacks required files: {missing}"
            )
        for relative, sha in sorted(files.items()):
            target = root / directory / relative
            if target.is_symlink() or not target.is_file():
                raise BenchmarkError(
                    f"deepswe tree file unavailable for {task_id}: {relative}"
                )
            if _sha256_file(target) != sha:
                raise BenchmarkError(
                    f"deepswe tree file SHA-256 drift for {task_id}: {relative}"
                )
        instruction_text = (root / directory / "instruction.md").read_text(encoding="utf-8")
        source_case: Dict[str, Any] = {
            "source_task_id": task_id,
            "task_text": instruction_text,
            "initial_state_sha256": canonical_sha256(
                {
                    "repository_url": repository_url,
                    "base_commit_hash": base_commit,
                    "tree_commit": commit,
                    "directory": directory,
                }
            ),
            "expected_state_sha256": canonical_sha256(
                {
                    "instruction_sha256": files["instruction.md"],
                    "tests_sha256": files["tests/test.sh"],
                }
            ),
            "source_tree": {
                "root": str(root),
                "commit": commit,
                "directory": directory,
                "files": dict(files),
            },
            "deepswe": {
                "task_id": task_id,
                "base_commit_hash": base_commit,
                "repository_url": repository_url,
                "language": language,
                "instruction_sha256": files["instruction.md"],
                "tests_sha256": files["tests/test.sh"],
            },
        }
        if runtimes is not None:
            if task_id not in runtimes:
                raise BenchmarkError(f"missing deepswe runtime for {task_id}")
            source_case["runtime"] = lock_runtime(runtimes[task_id])
        if assets is not None:
            source_case["deepswe"]["offline"] = _admit_offline_assets(task_id, assets)
        cases.append(source_case)
    if runtimes is not None and set(runtimes) != seen_ids:
        raise BenchmarkError("deepswe runtime registry contains unknown task IDs")
    if assets is not None and not set(assets) <= seen_ids:
        raise BenchmarkError("deepswe offline registry contains unknown task IDs")
    return {
        "schema_version": "benchmark-source-v1",
        "family": "deepswe",
        "release": release,
        "split": split,
        "license": dict(license_record),
        "source_snapshot": {
            "path": str(snapshot),
            "sha256": expected_sha,
            "format": SNAPSHOT_FORMAT,
            "commit": commit,
            "repository_url": repository_url,
        },
        "runtime_registry_sha256": (
            require_sha(runtime_registry_sha256, "runtime_registry_sha256")
            if runtime_registry_sha256 is not None
            else None
        ),
        "offline_registry_sha256": (
            require_sha(offline_registry_sha256, "offline_registry_sha256")
            if offline_registry_sha256 is not None
            else None
        ),
        "cases": cases,
    }


def verify_deepswe_source_manifest(source: Dict[str, Any]) -> None:
    """Recheck every imported task against the pinned tree before derivation."""
    snapshot_record = source.get("source_snapshot")
    if (
        not isinstance(snapshot_record, dict)
        or snapshot_record.get("format") != SNAPSHOT_FORMAT
    ):
        raise BenchmarkError("deepswe source needs a deepswe-tree-v1 snapshot")
    snapshot = Path(require_nonempty(snapshot_record.get("path"), "snapshot.path"))
    expected_sha = require_sha(snapshot_record.get("sha256"), "snapshot.sha256")
    if not snapshot.is_absolute() or not snapshot.is_file():
        raise BenchmarkError("deepswe source snapshot unavailable")
    if file_sha256(snapshot) != expected_sha:
        raise BenchmarkError("deepswe source snapshot drift")
    tree = _json_object(snapshot, "deepswe tree snapshot")
    if tree.get("schema_version") != TREE_SNAPSHOT_SCHEMA:
        raise BenchmarkError("invalid deepswe tree snapshot schema")
    commit = require_nonempty(tree.get("commit"), "tree.commit")
    repository_url = require_nonempty(tree.get("repository_url"), "tree.repository_url")
    root = Path(require_nonempty(tree.get("root"), "tree.root"))
    if not root.is_absolute() or not root.is_dir():
        raise BenchmarkError("deepswe tree root must be an existing absolute directory")
    tasks_by_id = {
        entry.get("task_id"): entry
        for entry in tree.get("tasks", [])
        if isinstance(entry, dict)
    }
    cases = source.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("deepswe source cases required")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("source_tree"), dict):
            raise BenchmarkError("deepswe source tree reference missing")
        metadata = case.get("deepswe")
        if not isinstance(metadata, dict):
            raise BenchmarkError(
                f"deepswe source case lacks family metadata: "
                f"{case.get('source_task_id')!r}"
            )
        task_id = case["source_task_id"]
        entry = tasks_by_id.get(task_id)
        if entry is None:
            raise BenchmarkError(
                f"deepswe source task {task_id!r} not in tree snapshot"
            )
        files = case["source_tree"].get("files")
        if not isinstance(files, dict) or files != entry.get("files"):
            raise BenchmarkError(f"deepswe file pin drift for {task_id}")
        directory = case["source_tree"].get("directory")
        if directory != entry.get("directory"):
            raise BenchmarkError(f"deepswe directory drift for {task_id}")
        for relative, sha in sorted(files.items()):
            target = root / directory / relative
            if (
                target.is_symlink()
                or not target.is_file()
                or _sha256_file(target) != sha
            ):
                raise BenchmarkError(
                    f"deepswe tree file unavailable or drifted for {task_id}: "
                    f"{relative}"
                )
        instruction = (root / directory / "instruction.md").read_text(encoding="utf-8")
        expected = (
            instruction,
            canonical_sha256(
                {
                    "repository_url": repository_url,
                    "base_commit_hash": entry.get("base_commit_hash"),
                    "tree_commit": commit,
                    "directory": directory,
                }
            ),
            canonical_sha256(
                {
                    "instruction_sha256": files["instruction.md"],
                    "tests_sha256": files["tests/test.sh"],
                }
            ),
            task_id,
            entry.get("base_commit_hash"),
            repository_url,
            entry.get("language"),
            files["instruction.md"],
            files["tests/test.sh"],
        )
        actual = (
            case.get("task_text"),
            case.get("initial_state_sha256"),
            case.get("expected_state_sha256"),
            metadata.get("task_id"),
            metadata.get("base_commit_hash"),
            metadata.get("repository_url"),
            metadata.get("language"),
            metadata.get("instruction_sha256"),
            metadata.get("tests_sha256"),
        )
        if expected != actual:
            raise BenchmarkError(f"deepswe source metadata drift for {task_id}")
        offline = metadata.get("offline")
        if not isinstance(offline, dict) or offline.get("admission") != "expressible":
            continue
        for label in ("base_snapshot", "gold_diff"):
            pin = offline.get(label)
            if not isinstance(pin, dict):
                raise BenchmarkError(
                    f"deepswe offline admission for {task_id!r} lacks a {label} pin"
                )
            _pin_path(pin, task_id, f"offline {label}")


def derive_deepswe_offline_spec(
    source: Dict[str, Any], source_case: Dict[str, Any], directive: Any
) -> Dict[str, Any]:
    """DS5 pregeneration: build the offline verifier_spec from the pinned gold diff.

    The gold diff is read from the admission's pinned file and applied to the
    admitted base snapshot; only the resulting path->sha256 expectation travels
    into the trusted manifest. The gold diff text itself never does.
    """
    if not isinstance(directive, dict) or set(directive) - {"deliverable", "timeout_s"}:
        raise BenchmarkError(
            "deepswe_offline directive accepts only deliverable and timeout_s"
        )
    deliverable = _clean_relative(str(directive.get("deliverable") or "patch.diff"))
    timeout = directive.get("timeout_s", 30.0)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise BenchmarkError("deepswe_offline timeout_s must be a positive number")
    metadata = source_case.get("deepswe")
    task_id = source_case.get("source_task_id")
    if not isinstance(metadata, dict):
        raise BenchmarkError(f"deepswe source case lacks family metadata: {task_id!r}")
    offline = metadata.get("offline")
    if (
        not isinstance(offline, dict)
        or offline.get("admission") != "expressible"
        or not isinstance(offline.get("base_snapshot"), dict)
        or not isinstance(offline.get("gold_diff"), dict)
    ):
        raise BenchmarkError(
            f"deepswe task {task_id!r} is not admitted for offline derivation"
        )
    base_path = _pin_path(offline["base_snapshot"], task_id, "offline base")
    base_sha = offline["base_snapshot"]["sha256"]
    gold_path = _pin_path(offline["gold_diff"], task_id, "offline gold diff")
    gold_bytes = gold_path.read_bytes()
    if hashlib.sha256(gold_bytes).hexdigest() != offline["gold_diff"]["sha256"]:
        raise BenchmarkError(f"gold diff SHA-256 drift for {task_id}")
    git = shutil.which("git")
    if not git:
        raise BenchmarkError("git binary is required for deepswe offline derivation")
    with tempfile.TemporaryDirectory(prefix="deepswe-derive-") as tmp:
        outcome = _apply_gold_diff(git, base_path, gold_bytes, float(timeout), Path(tmp))
    if outcome is None:
        raise BenchmarkError(
            f"gold diff does not apply to the base snapshot for {task_id}"
        )
    if outcome["affected"] == 0:
        raise BenchmarkError("gold diff touches no file")
    expected_files = outcome["expected_files"]
    deleted_paths = outcome["deleted_paths"]
    release = require_nonempty(source.get("release"), "release")
    tree = source_case.get("source_tree") or {}
    spec = {
        "kind": SPEC_KIND,
        "generator_ref": f"deepswe-offline-gold@{release}",
        "payload": {
            "deliverable": deliverable,
            "base_snapshot": {"path": str(base_path), "sha256": base_sha},
            "expected_files": expected_files,
            "deleted_paths": deleted_paths,
            "timeout_s": float(timeout),
        },
    }
    return {
        "verifier_spec": spec,
        "initial_state_sha256": canonical_sha256(
            {
                "repository_url": metadata.get("repository_url"),
                "base_commit_hash": metadata.get("base_commit_hash"),
                "tree_commit": tree.get("commit"),
                "directory": tree.get("directory"),
                "base_snapshot_sha256": base_sha,
            }
        ),
        "expected_state_sha256": canonical_sha256(
            {"expected_files": expected_files, "deleted_paths": deleted_paths}
        ),
    }


def assert_deepswe_offline_build(
    runtime: Dict[str, Any], private: Dict[str, Any], source_case: Dict[str, Any]
) -> None:
    """M2 gate: every new deepswe taskset pins the offline runner and a real spec."""
    metadata = source_case.get("deepswe")
    offline = metadata.get("offline") if isinstance(metadata, dict) else None
    task_id = source_case.get("source_task_id")
    if not isinstance(offline, dict):
        raise BenchmarkError(
            f"deepswe task {task_id!r} lacks an offline admission marker; "
            "import with a deepswe offline registry"
        )
    if offline.get("admission") != "expressible":
        raise BenchmarkError(
            f"deepswe task {task_id!r} is excluded from offline derivation "
            f"({offline.get('reason', 'unknown')})"
        )
    verifier = runtime.get("verifier")
    if (
        not isinstance(verifier, dict)
        or verifier.get("version") != OFFLINE_RUNNER_VERSION
    ):
        raise BenchmarkError(
            "deepswe derivation must pin the offline runner "
            f"({OFFLINE_RUNNER_VERSION})"
        )
    spec = private.get("verifier_spec") if isinstance(private, dict) else None
    if not isinstance(spec, dict) or spec.get("kind") != SPEC_KIND:
        raise BenchmarkError(
            f"deepswe trusted case lacks a {SPEC_KIND} verifier_spec: {task_id}"
        )
