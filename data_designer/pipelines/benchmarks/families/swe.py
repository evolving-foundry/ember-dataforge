"""Read-only importer for a pinned SWE-bench-format JSONL snapshot.

The official row's patch, tests, and hints stay in the source data plane. The
generated manifest contains only hashes and a line reference to that file.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..contracts import (
    BenchmarkError,
    canonical_sha256,
    file_sha256,
    require_nonempty,
    require_sha,
    require_split,
)
from ..runners.swe_offline import (
    VERSION as OFFLINE_RUNNER_VERSION,
    _apply_patch,
    _clean_relative,
    _extract_snapshot,
    _sha256_file,
)
from ..runtime import lock_runtime

COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
UPSTREAM_SPLITS = frozenset({"train", "dev", "test", "candidate"})
OFFLINE_BASE_REGISTRY_SCHEMA = "swe-offline-base-registry-v1"
SPEC_KIND = "swe_offline_patch"


def _json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read runtime registry: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkError("runtime registry must be an object")
    return value


def _tests(value: Any, name: str) -> List[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as error:
            raise BenchmarkError(f"{name} is not a JSON test list") from error
    if not isinstance(value, list) or any(
        not isinstance(test, str) or not test.strip() for test in value
    ):
        raise BenchmarkError(f"{name} must be a list of test names")
    if len(set(value)) != len(value):
        raise BenchmarkError(f"{name} has duplicate test names")
    return value


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _registry_pin_path(pin: Any, instance_id: str, label: str) -> Path:
    if not isinstance(pin, dict):
        raise BenchmarkError(f"invalid {label} pin for {instance_id}")
    path = Path(require_nonempty(pin.get("path"), f"{label}.path"))
    sha = require_sha(pin.get("sha256"), f"{label}.sha256")
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != sha:
        raise BenchmarkError(f"{label} snapshot missing or SHA-256 drift for {instance_id}")
    return path


def _gold_targets(patch_text: str) -> tuple[Set[str], Set[str]]:
    """Repo-relative paths a unified diff touches, split by new-side deletion."""
    touched: Set[str] = set()
    deleted: Set[str] = set()
    old_path: Optional[str] = None
    for line in patch_text.splitlines():
        if line.startswith("--- "):
            token = line[4:].strip().split("\t", 1)[0]
            old_path = (
                None
                if token == "/dev/null"
                else _clean_relative(token[2:] if token.startswith("a/") else token)
            )
        elif line.startswith("+++ "):
            token = line[4:].strip().split("\t", 1)[0]
            if token == "/dev/null":
                if old_path is not None:
                    deleted.add(old_path)
            else:
                touched.add(
                    _clean_relative(token[2:] if token.startswith("b/") else token)
                )
                if old_path is not None:
                    touched.add(old_path)
            old_path = None
    return touched, deleted


def _admit_offline_base(
    instance_id: str, row: Dict[str, Any], offline_bases: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """S5 subset admission: expressible only if the gold patch applies offline."""
    pin = offline_bases.get(instance_id)
    if pin is None:
        return {"admission": "excluded", "reason": "build_required"}
    base_path = _registry_pin_path(pin, instance_id, "offline base")
    git = shutil.which("git")
    if not git:
        raise BenchmarkError("git binary is required for swe offline admission")
    with tempfile.TemporaryDirectory(prefix="swe-admit-") as tmp:
        workdir = Path(tmp) / "workspace"
        workdir.mkdir()
        _extract_snapshot(base_path, workdir)
        patch_file = Path(tmp) / "gold.patch"
        patch_file.write_text(row["patch"], encoding="utf-8")
        try:
            _apply_patch(git, workdir, patch_file, 30.0, check_only=True)
            _apply_patch(git, workdir, patch_file, 30.0, check_only=False)
        except (ValueError, subprocess.SubprocessError):
            return {"admission": "excluded", "reason": "build_required"}
    return {
        "admission": "expressible",
        "base_snapshot": {"path": str(base_path), "sha256": pin["sha256"]},
    }


def import_swe_jsonl(
    snapshot_path: str,
    snapshot_sha256: str,
    *,
    release: str,
    split: str,
    upstream_split: str,
    license_record: Dict[str, Any],
    runtime_registry_path: Optional[str] = None,
    runtime_registry_sha256: Optional[str] = None,
    offline_base_registry_path: Optional[str] = None,
    offline_base_registry_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a local authorized snapshot into benchmark-source-v1 data."""
    snapshot = Path(require_nonempty(snapshot_path, "snapshot_path"))
    expected_sha = require_sha(snapshot_sha256, "snapshot_sha256")
    if not snapshot.is_absolute() or not snapshot.is_file():
        raise BenchmarkError("SWE snapshot must be an existing absolute file")
    if file_sha256(snapshot) != expected_sha:
        raise BenchmarkError("SWE snapshot SHA-256 drift")
    release = require_nonempty(release, "release")
    split = require_split(split)
    if upstream_split not in UPSTREAM_SPLITS:
        raise BenchmarkError("invalid SWE upstream split")
    if upstream_split == "test" and split != "holdout":
        raise BenchmarkError("upstream test cannot seed development or regression")
    if (
        not isinstance(license_record, dict)
        or not isinstance(license_record.get("id"), str)
        or not license_record["id"].strip()
        or type(license_record.get("derive_allowed")) is not bool
    ):
        raise BenchmarkError("explicit source license decision required")
    runtimes: Dict[str, Dict[str, Any]] | None = None
    registry_sha: str | None = None
    if runtime_registry_path is not None or runtime_registry_sha256 is not None:
        registry_path = Path(
            require_nonempty(runtime_registry_path, "runtime_registry_path")
        )
        registry_sha = require_sha(runtime_registry_sha256, "runtime_registry_sha256")
        if not registry_path.is_absolute() or not registry_path.is_file():
            raise BenchmarkError("runtime registry must be an existing absolute file")
        if file_sha256(registry_path) != registry_sha:
            raise BenchmarkError("SWE runtime registry SHA-256 drift")
        registry = _json_object(registry_path)
        if (
            registry.get("schema_version") != "benchmark-runtime-registry-v1"
            or not isinstance(registry.get("cases"), dict)
        ):
            raise BenchmarkError("invalid SWE runtime registry schema")
        runtimes = registry["cases"]
    offline_bases: Dict[str, Dict[str, Any]] | None = None
    offline_registry_sha: str | None = None
    if (
        offline_base_registry_path is not None
        or offline_base_registry_sha256 is not None
    ):
        base_registry_path = Path(
            require_nonempty(
                offline_base_registry_path, "offline_base_registry_path"
            )
        )
        offline_registry_sha = require_sha(
            offline_base_registry_sha256, "offline_base_registry_sha256"
        )
        if not base_registry_path.is_absolute() or not base_registry_path.is_file():
            raise BenchmarkError("offline base registry must be an existing absolute file")
        if file_sha256(base_registry_path) != offline_registry_sha:
            raise BenchmarkError("SWE offline base registry SHA-256 drift")
        base_registry = _json_object(base_registry_path)
        if (
            base_registry.get("schema_version") != OFFLINE_BASE_REGISTRY_SCHEMA
            or not isinstance(base_registry.get("cases"), dict)
        ):
            raise BenchmarkError("invalid SWE offline base registry schema")
        offline_bases = base_registry["cases"]
        for instance_id, pin in offline_bases.items():
            _registry_pin_path(pin, instance_id, "offline base")
    cases: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        with snapshot.open("rb") as stream:
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
                        f"invalid SWE JSONL row {line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise BenchmarkError(f"SWE row {line_number} must be an object")
                instance_id = require_nonempty(row.get("instance_id"), "instance_id")
                if instance_id in seen_ids:
                    raise BenchmarkError(f"duplicate SWE instance_id {instance_id}")
                seen_ids.add(instance_id)
                repo = require_nonempty(row.get("repo"), "repo")
                if repo.count("/") != 1 or any(not part for part in repo.split("/")):
                    raise BenchmarkError(f"invalid SWE repo for {instance_id}")
                base_commit = require_nonempty(row.get("base_commit"), "base_commit")
                if not COMMIT_RE.fullmatch(base_commit):
                    raise BenchmarkError(f"invalid SWE base_commit for {instance_id}")
                statement = require_nonempty(
                    row.get("problem_statement"), "problem_statement"
                )
                gold_patch = require_nonempty(row.get("patch"), "patch")
                test_patch = require_nonempty(row.get("test_patch"), "test_patch")
                fail_to_pass = _tests(row.get("FAIL_TO_PASS"), "FAIL_TO_PASS")
                pass_to_pass = _tests(row.get("PASS_TO_PASS"), "PASS_TO_PASS")
                if not fail_to_pass:
                    raise BenchmarkError(
                        f"SWE instance {instance_id} has no target test"
                    )
                created_at = require_nonempty(row.get("created_at"), "created_at")
                test_patch_sha = _sha_text(test_patch)
                source_case: Dict[str, Any] = {
                    "source_task_id": instance_id,
                    "task_text": statement,
                    "initial_state_sha256": canonical_sha256(
                        {"repo": repo, "base_commit": base_commit}
                    ),
                    "expected_state_sha256": canonical_sha256(
                        {
                            "test_patch_sha256": test_patch_sha,
                            "FAIL_TO_PASS": fail_to_pass,
                            "PASS_TO_PASS": pass_to_pass,
                        }
                    ),
                    "source_row": {
                        "line": line_number,
                        "sha256": hashlib.sha256(raw_line).hexdigest(),
                    },
                    "swe": {
                        "repo": repo,
                        "base_commit": base_commit,
                        "created_at": created_at,
                        "version": row.get("version"),
                        "gold_patch_sha256": _sha_text(gold_patch),
                        "test_patch_sha256": test_patch_sha,
                        "FAIL_TO_PASS": fail_to_pass,
                        "PASS_TO_PASS": pass_to_pass,
                    },
                }
                if runtimes is not None:
                    if instance_id not in runtimes:
                        raise BenchmarkError(f"missing SWE runtime for {instance_id}")
                    source_case["runtime"] = lock_runtime(runtimes[instance_id])
                if offline_bases is not None:
                    source_case["swe"]["offline"] = _admit_offline_base(
                        instance_id, row, offline_bases
                    )
                cases.append(source_case)
    except OSError as error:
        raise BenchmarkError(f"cannot read SWE snapshot: {error}") from error
    if not cases:
        raise BenchmarkError("SWE snapshot has no usable cases")
    if runtimes is not None and set(runtimes) != seen_ids:
        raise BenchmarkError("SWE runtime registry contains unknown instance IDs")
    if offline_bases is not None and not set(offline_bases) <= seen_ids:
        raise BenchmarkError("SWE offline base registry contains unknown instance IDs")
    return {
        "schema_version": "benchmark-source-v1",
        "family": "swe",
        "release": release,
        "split": split,
        "license": dict(license_record),
        "source_snapshot": {
            "path": str(snapshot),
            "sha256": expected_sha,
            "format": "swe-jsonl-v1",
            "upstream_split": upstream_split,
        },
        "runtime_registry_sha256": registry_sha,
        "offline_base_registry_sha256": offline_registry_sha,
        "cases": cases,
    }


def verify_swe_source_manifest(source: Dict[str, Any]) -> None:
    """Recheck every imported row against the frozen JSONL before derivation."""
    snapshot_record = source.get("source_snapshot")
    if (
        not isinstance(snapshot_record, dict)
        or snapshot_record.get("format") != "swe-jsonl-v1"
    ):
        raise BenchmarkError("SWE source needs a swe-jsonl-v1 snapshot")
    snapshot = Path(require_nonempty(snapshot_record.get("path"), "snapshot.path"))
    expected_sha = require_sha(snapshot_record.get("sha256"), "snapshot.sha256")
    if not snapshot.is_absolute() or not snapshot.is_file():
        raise BenchmarkError("SWE source snapshot unavailable")
    if file_sha256(snapshot) != expected_sha:
        raise BenchmarkError("SWE source snapshot drift")
    cases = source.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("SWE source cases required")
    by_line: Dict[int, Dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("source_row"), dict):
            raise BenchmarkError("SWE source row reference missing")
        line = case["source_row"].get("line")
        if type(line) is not int or line < 1 or line in by_line:
            raise BenchmarkError("SWE source row line invalid or duplicated")
        require_sha(case["source_row"].get("sha256"), "source_row.sha256")
        by_line[line] = case
    found: set[int] = set()
    with snapshot.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            case = by_line.get(line_number)
            if case is None:
                continue
            found.add(line_number)
            if hashlib.sha256(raw_line).hexdigest() != case["source_row"]["sha256"]:
                raise BenchmarkError(f"SWE source row {line_number} SHA-256 drift")
            try:
                row = json.loads(raw_line.decode("utf-8"))
            except (UnicodeError, ValueError) as error:
                raise BenchmarkError(f"SWE source row {line_number} invalid") from error
            metadata = case.get("swe")
            if not isinstance(row, dict) or not isinstance(metadata, dict):
                raise BenchmarkError(f"SWE source row {line_number} invalid")
            expected = (
                case.get("source_task_id"),
                case.get("task_text"),
                case.get("initial_state_sha256"),
                case.get("expected_state_sha256"),
                metadata.get("repo"),
                metadata.get("base_commit"),
                metadata.get("created_at"),
                metadata.get("version"),
                metadata.get("gold_patch_sha256"),
                metadata.get("test_patch_sha256"),
                metadata.get("FAIL_TO_PASS"),
                metadata.get("PASS_TO_PASS"),
            )
            actual = (
                row.get("instance_id"),
                row.get("problem_statement"),
                canonical_sha256(
                    {"repo": row.get("repo"), "base_commit": row.get("base_commit")}
                ),
                canonical_sha256(
                    {
                        "test_patch_sha256": _sha_text(
                            require_nonempty(row.get("test_patch"), "test_patch")
                        ),
                        "FAIL_TO_PASS": _tests(row.get("FAIL_TO_PASS"), "FAIL_TO_PASS"),
                        "PASS_TO_PASS": _tests(row.get("PASS_TO_PASS"), "PASS_TO_PASS"),
                    }
                ),
                row.get("repo"),
                row.get("base_commit"),
                row.get("created_at"),
                row.get("version"),
                _sha_text(require_nonempty(row.get("patch"), "patch")),
                _sha_text(require_nonempty(row.get("test_patch"), "test_patch")),
                _tests(row.get("FAIL_TO_PASS"), "FAIL_TO_PASS"),
                _tests(row.get("PASS_TO_PASS"), "PASS_TO_PASS"),
            )
            if expected != actual:
                raise BenchmarkError(f"SWE source row {line_number} metadata drift")
    if found != set(by_line):
        raise BenchmarkError("SWE source row reference out of range")
    for case in cases:
        metadata = case.get("swe") if isinstance(case, dict) else None
        offline = metadata.get("offline") if isinstance(metadata, dict) else None
        if not isinstance(offline, dict) or offline.get("admission") != "expressible":
            continue
        pin = offline.get("base_snapshot")
        if not isinstance(pin, dict):
            raise BenchmarkError(
                f"SWE offline admission for {case.get('source_task_id')!r} "
                "lacks a base snapshot pin"
            )
        _registry_pin_path(pin, case.get("source_task_id"), "offline base")


def _read_pinned_row(
    source: Dict[str, Any], source_case: Dict[str, Any]
) -> Dict[str, Any]:
    snapshot_record = source.get("source_snapshot")
    if (
        not isinstance(snapshot_record, dict)
        or snapshot_record.get("format") != "swe-jsonl-v1"
    ):
        raise BenchmarkError("SWE offline derivation needs a swe-jsonl-v1 snapshot")
    snapshot = Path(require_nonempty(snapshot_record.get("path"), "snapshot.path"))
    if not snapshot.is_absolute() or not snapshot.is_file():
        raise BenchmarkError("SWE source snapshot unavailable")
    if file_sha256(snapshot) != require_sha(
        snapshot_record.get("sha256"), "snapshot.sha256"
    ):
        raise BenchmarkError("SWE source snapshot drift")
    row_ref = source_case.get("source_row")
    if not isinstance(row_ref, dict):
        raise BenchmarkError("SWE source row reference missing")
    line = row_ref.get("line")
    if type(line) is not int or line < 1:
        raise BenchmarkError("SWE source row line invalid")
    with snapshot.open("rb") as stream:
        lines = stream.readlines()
    if line > len(lines):
        raise BenchmarkError("SWE source row reference out of range")
    raw = lines[line - 1]
    if hashlib.sha256(raw).hexdigest() != row_ref.get("sha256"):
        raise BenchmarkError(f"SWE source row {line} SHA-256 drift")
    try:
        row = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError) as error:
        raise BenchmarkError(f"SWE source row {line} invalid") from error
    if not isinstance(row, dict):
        raise BenchmarkError(f"SWE source row {line} must be an object")
    return row


def derive_swe_offline_spec(
    source: Dict[str, Any], source_case: Dict[str, Any], directive: Any
) -> Dict[str, Any]:
    """S4 pregeneration: build the offline verifier_spec from the pinned gold patch.

    The gold patch text is read from the frozen JSONL row and applied to the
    admitted base snapshot; only the resulting path->sha256 expectation travels
    into the trusted manifest. The gold patch text itself never does.
    """
    if not isinstance(directive, dict) or set(directive) - {"deliverable", "timeout_s"}:
        raise BenchmarkError(
            "swe_offline directive accepts only deliverable and timeout_s"
        )
    deliverable = _clean_relative(str(directive.get("deliverable") or "patch.diff"))
    timeout = directive.get("timeout_s", 30.0)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise BenchmarkError("swe_offline timeout_s must be a positive number")
    metadata = source_case.get("swe")
    if not isinstance(metadata, dict):
        raise BenchmarkError("SWE source case lacks swe metadata")
    offline = metadata.get("offline")
    if (
        not isinstance(offline, dict)
        or offline.get("admission") != "expressible"
        or not isinstance(offline.get("base_snapshot"), dict)
    ):
        raise BenchmarkError(
            f"swe instance {source_case.get('source_task_id')!r} is not admitted "
            "for offline derivation"
        )
    base_path = _registry_pin_path(
        offline["base_snapshot"], source_case.get("source_task_id"), "offline base"
    )
    base_sha = offline["base_snapshot"]["sha256"]
    row = _read_pinned_row(source, source_case)
    gold = require_nonempty(row.get("patch"), "row.patch")
    if _sha_text(gold) != metadata.get("gold_patch_sha256"):
        raise BenchmarkError("gold patch SHA-256 drift against source manifest")
    touched, deleted = _gold_targets(gold)
    if not touched and not deleted:
        raise BenchmarkError("gold patch touches no file")
    git = shutil.which("git")
    if not git:
        raise BenchmarkError("git binary is required for swe offline derivation")
    expected_files: Dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="swe-derive-") as tmp:
        workdir = Path(tmp) / "workspace"
        workdir.mkdir()
        _extract_snapshot(base_path, workdir)
        patch_file = Path(tmp) / "gold.patch"
        patch_file.write_text(gold, encoding="utf-8")
        _apply_patch(git, workdir, patch_file, float(timeout), check_only=True)
        _apply_patch(git, workdir, patch_file, float(timeout), check_only=False)
        for relative in sorted(touched - deleted):
            target = workdir / relative
            if target.is_symlink() or not target.is_file():
                raise BenchmarkError(f"gold patch produced no file for {relative!r}")
            expected_files[relative] = _sha256_file(target)
        for relative in sorted(deleted):
            leftover = workdir / relative
            if leftover.is_symlink() or leftover.exists():
                raise BenchmarkError(
                    f"gold patch left deleted path present: {relative!r}"
                )
    release = require_nonempty(source.get("release"), "release")
    spec = {
        "kind": SPEC_KIND,
        "generator_ref": f"swe-offline-gold@{release}",
        "payload": {
            "deliverable": deliverable,
            "base_snapshot": {"path": str(base_path), "sha256": base_sha},
            "expected_files": expected_files,
            "deleted_paths": sorted(deleted),
            "timeout_s": float(timeout),
        },
    }
    return {
        "verifier_spec": spec,
        "initial_state_sha256": canonical_sha256(
            {
                "repo": metadata.get("repo"),
                "base_commit": metadata.get("base_commit"),
                "base_snapshot_sha256": base_sha,
            }
        ),
        "expected_state_sha256": canonical_sha256(
            {"expected_files": expected_files, "deleted_paths": sorted(deleted)}
        ),
    }


def assert_swe_offline_build(
    runtime: Dict[str, Any], private: Dict[str, Any], source_case: Dict[str, Any]
) -> None:
    """M2 gate: every new swe taskset pins the offline runner and a real spec."""
    metadata = source_case.get("swe")
    offline = metadata.get("offline") if isinstance(metadata, dict) else None
    instance_id = source_case.get("source_task_id")
    if not isinstance(offline, dict):
        raise BenchmarkError(
            f"swe instance {instance_id!r} lacks an offline admission marker; "
            "import with an offline base registry"
        )
    if offline.get("admission") != "expressible":
        raise BenchmarkError(
            f"swe instance {instance_id!r} is excluded from offline derivation "
            f"({offline.get('reason', 'unknown')})"
        )
    verifier = runtime.get("verifier")
    if (
        not isinstance(verifier, dict)
        or verifier.get("version") != OFFLINE_RUNNER_VERSION
    ):
        raise BenchmarkError(
            "swe derivation must pin the offline runner "
            f"({OFFLINE_RUNNER_VERSION}); the harness runner is retired"
        )
    spec = private.get("verifier_spec") if isinstance(private, dict) else None
    if not isinstance(spec, dict) or spec.get("kind") != SPEC_KIND:
        raise BenchmarkError(
            f"swe trusted case lacks a {SPEC_KIND} verifier_spec: {instance_id}"
        )
