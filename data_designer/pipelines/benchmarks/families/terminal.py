"""terminal seed → benchmark-source-v1 importer, re-expression, augment adapter.

Input is the trusted seed material produced by ``TB21SeedAdapter`` (either the
in-repo sanitized fixture or a locally-held authorized terminal-bench-2.1
checkout — never redistributed). The importer freezes it into a
``benchmark-source-v1`` manifest; the T8 expressibility classification travels
per case in the family metadata block and gates every derivation entry point.

Re-expression (scoring role): the upstream sandbox acceptance signal is re-expressed
offline by the trusted side into a frozen ``verifier_spec`` — ``unit_test``
(frozen assertion script executed in a scrubbed workspace) preferred,
``artifact_compare`` for JSON-artifact tasks. Upstream ``tests/`` and
``solution/`` stay pinned in the source snapshot; only the re-expressed spec
and the frozen gold/negative deliverables travel into the trusted case
manifest. Upstream ``docker_image`` is archived as provenance only (T7) — the
pipeline never pulls or runs images.

License (T3): upstream licensing is archived, not a gate — internal evaluation
use. ``derive_allowed=true`` is valid for both fixture and upstream releases;
the license decision itself must still be explicit.

Contamination red line: every derived ``task_text`` is screened against the
fingerprint set of the seed texts (exact ``normalized_text_sha256`` match) so
seed text never re-enters a published item.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts import (
    BenchmarkError,
    canonical_sha256,
    file_sha256,
    normalized_text_sha256,
    require_nonempty,
    require_sha,
    require_split,
)
from ..runners.terminal import SUPPORTED_SPEC_KINDS, VERSION as RUNNER_VERSION
from ..runtime import lock_runtime

SUPPORTED_OPERATIONS = ("setup_fault", "constraint_tightening", "failure_probe")
SUPPORTED_SEED_SOURCES = ("synthetic-fixture", "upstream-checkout")
_SNAPSHOT_FORMAT = "terminal-bench-archive-v1"
SPEC_KINDS = SUPPORTED_SPEC_KINDS
UPSTREAM_TASK_COUNT = 89  # harbor-framework/terminal-bench-2-1 @ 7131e437
UPSTREAM_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DERIVED_KEYS = (
    "source_task_id", "task_text", "tools_available", "success_criteria",
    "difficulty", "verifier_spec",
)


def derive_release(
    seed_source: str,
    seed_rows: List[Dict[str, Any]],
    archive_sha256: Optional[str] = None,
    upstream_commit: Optional[str] = None,
) -> str:
    """T2 release semantics: fixture SHA-8 or upstream commit-8 pin."""
    if seed_source == "synthetic-fixture":
        return "synthetic-fixture:" + canonical_sha256(list(seed_rows))[:8]
    if seed_source == "upstream-checkout":
        commit = require_nonempty(upstream_commit, "upstream_commit")
        if not UPSTREAM_COMMIT_RE.fullmatch(commit):
            raise BenchmarkError(
                "upstream-checkout release needs the full 40-hex upstream commit"
            )
        if len(seed_rows) != UPSTREAM_TASK_COUNT:
            raise BenchmarkError(
                "terminal upstream checkout must carry exactly "
                f"{UPSTREAM_TASK_COUNT} tasks; found {len(seed_rows)}"
            )
        return "tb21-" + commit[:8]
    raise BenchmarkError(f"unknown terminal seed_source: {seed_source!r}")


def _seed_case(row: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise BenchmarkError("seed row must be an object")
    task_text = require_nonempty(row.get("task_text"), "seed task_text")
    tools = row.get("tools_available")
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise BenchmarkError("seed tools_available must be a string list")
    expressibility = row.get("expressibility")
    if expressibility not in ("expressible", "excluded"):
        raise BenchmarkError("seed row lacks a valid expressibility marker")
    exclusion_reason = row.get("exclusion_reason")
    if expressibility == "excluded":
        require_nonempty(exclusion_reason, "exclusion_reason")
    elif exclusion_reason not in (None, ""):
        raise BenchmarkError("expressible rows must not carry an exclusion reason")
    setup_files = row.get("setup_files") or {}
    if not isinstance(setup_files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in setup_files.items()
    ):
        raise BenchmarkError("seed setup_files must be a string dictionary")
    source_task_id = require_nonempty(row.get("seed_id"), "seed seed_id")
    return {
        "source_task_id": source_task_id,
        "task_text": task_text,
        "tools_available": list(tools),
        "success_criteria": require_nonempty(
            row.get("success_criteria"), "seed success_criteria"
        ),
        "difficulty": require_nonempty(row.get("difficulty"), "seed difficulty"),
        "initial_state_sha256": canonical_sha256(
            {
                "stage": "seed-workspace",
                "task_text": task_text,
                "tools_available": list(tools),
                "setup_files": setup_files,
            }
        ),
        "expected_state_sha256": canonical_sha256(
            {
                "success_criteria": row.get("success_criteria"),
                "expressibility": expressibility,
            }
        ),
        "terminal": {
            "expressibility": expressibility,
            "exclusion_reason": exclusion_reason,
            "category": str(row.get("category") or "unspecified"),
            "docker_image": str(row.get("docker_image") or "unspecified"),
            "setup_files": dict(setup_files),
            "tb_keywords": list(row.get("tb_keywords") or []),
        },
    }


def import_terminal_source(
    seed_rows: List[Dict[str, Any]],
    *,
    release: str,
    split: str,
    license_record: Dict[str, Any],
    seed_source: str,
    dataset_id: str,
    upstream_commit: Optional[str] = None,
    archive_path: Optional[str] = None,
    archive_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Freeze trusted seed rows into a benchmark-source-v1 manifest."""
    if seed_source not in SUPPORTED_SEED_SOURCES:
        raise BenchmarkError(f"unknown terminal seed_source: {seed_source!r}")
    split = require_split(split)
    release = require_nonempty(release, "release")
    if not isinstance(seed_rows, list) or not seed_rows:
        raise BenchmarkError("terminal source needs non-empty seed rows")
    if not isinstance(license_record, dict) or (
        not require_nonempty(license_record.get("id"), "license.id")
        or type(license_record.get("derive_allowed")) is not bool
    ):
        raise BenchmarkError("terminal license decision required (T3)")
    expected_release = derive_release(
        seed_source, seed_rows, archive_sha256, upstream_commit
    )
    if release != expected_release:
        raise BenchmarkError(
            f"terminal release pin mismatch: {release!r} != {expected_release!r}"
        )
    cases = [_seed_case(row) for row in seed_rows]
    if len({case["source_task_id"] for case in cases}) != len(cases):
        raise BenchmarkError("duplicate terminal seed IDs")
    if not any(case["terminal"]["expressibility"] == "expressible" for case in cases):
        raise BenchmarkError("terminal source has no expressible cases")
    source: Dict[str, Any] = {
        "schema_version": "benchmark-source-v1",
        "family": "terminal",
        "release": release,
        "split": split,
        "license": dict(license_record),
        "seed_source": seed_source,
        "dataset_id": require_nonempty(dataset_id, "dataset_id"),
        "exclusion_text_sha256": sorted(
            {normalized_text_sha256(row.get("task_text")) for row in seed_rows}
        ),
        "cases": cases,
    }
    if upstream_commit is not None:
        source["upstream_commit"] = require_nonempty(
            upstream_commit, "upstream_commit"
        )
    if archive_path is not None or archive_sha256 is not None:
        if archive_path is None or archive_sha256 is None:
            raise BenchmarkError("archive pin needs path and SHA-256 together")
        source["source_snapshot"] = {
            "path": str(archive_path),
            "sha256": require_sha(archive_sha256, "source_snapshot.sha256"),
            "format": _SNAPSHOT_FORMAT,
            "dataset_id": source["dataset_id"],
            "seed_source": seed_source,
        }
    return source


def verify_terminal_source_manifest(source: Dict[str, Any]) -> None:
    """Re-derive the manifest from the pinned archive and compare every pin."""
    from ....sources.terminal_seed import TB21SeedAdapter

    snapshot = source.get("source_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("format") != _SNAPSHOT_FORMAT
        or not isinstance(snapshot.get("path"), str)
    ):
        raise BenchmarkError("terminal source needs a pinned seed archive")
    adapter = TB21SeedAdapter.from_archive(
        snapshot["path"],
        seed_source=snapshot.get("seed_source"),
        upstream_commit=source.get("upstream_commit"),
    )
    if adapter.dataset_id != snapshot.get("dataset_id"):
        raise BenchmarkError("terminal seed archive dataset ID drift")
    expected = import_terminal_source(
        adapter.load_seed_tasks(),
        release=source.get("release"),
        split=source.get("split"),
        license_record=source.get("license"),
        seed_source=snapshot.get("seed_source"),
        dataset_id=snapshot.get("dataset_id"),
        upstream_commit=source.get("upstream_commit"),
        archive_path=snapshot.get("path"),
        archive_sha256=snapshot.get("sha256"),
    )
    if (
        source.get("release") != expected["release"]
        or source.get("license") != expected["license"]
        or source.get("exclusion_text_sha256") != expected["exclusion_text_sha256"]
        or source.get("cases") != expected["cases"]
    ):
        raise BenchmarkError("terminal source metadata drift")


def screen_terminal_contamination(
    derived_rows: List[Dict[str, Any]], seed_rows: List[Dict[str, Any]]
) -> None:
    """Reject derived rows whose task text exactly matches any seed text."""
    seed_fingerprints = {
        normalized_text_sha256(row.get("task_text")) for row in seed_rows
    }
    for row in derived_rows:
        if normalized_text_sha256(row.get("task_text")) in seed_fingerprints:
            raise BenchmarkError(
                "derived task text matches a seed text; seed text must never "
                "re-enter a published item"
            )


def _require_admitted(source: Dict[str, Any], source_case: Dict[str, Any]) -> None:
    metadata = source_case.get("terminal")
    if not isinstance(metadata, dict):
        raise BenchmarkError(
            f"terminal source case lacks family metadata: "
            f"{source_case.get('source_task_id')!r}"
        )
    if metadata.get("expressibility") != "expressible":
        raise BenchmarkError(
            f"terminal task {source_case.get('source_task_id')!r} is excluded "
            f"from derivation ({metadata.get('exclusion_reason', 'unknown')})"
        )


def _pinned_adapter(source: Dict[str, Any]):
    from ....sources.terminal_seed import TB21SeedAdapter

    snapshot = source.get("source_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("format") != _SNAPSHOT_FORMAT
        or not isinstance(snapshot.get("path"), str)
    ):
        raise BenchmarkError("terminal derivation needs a pinned seed archive")
    snapshot_path = Path(snapshot["path"])
    if not snapshot_path.is_absolute() or not snapshot_path.is_file():
        raise BenchmarkError("terminal source snapshot unavailable")
    if file_sha256(snapshot_path) != require_sha(
        snapshot.get("sha256"), "snapshot.sha256"
    ):
        raise BenchmarkError("terminal source snapshot drift")
    return TB21SeedAdapter.from_archive(
        snapshot["path"],
        seed_source=snapshot.get("seed_source"),
        upstream_commit=source.get("upstream_commit"),
    )


def _clean_relative(value: Any) -> str:
    relative = str(value)
    parts = Path(relative).parts
    if (
        not relative.strip()
        or Path(relative).is_absolute()
        or ".." in parts
        or not all(part not in ("", ".", "..") for part in parts)
    ):
        raise BenchmarkError(f"deliverable path must be a clean relative path: {relative!r}")
    return relative


def _zeroed(value: Any) -> Any:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return 0
    if isinstance(value, str):
        return ""
    if isinstance(value, dict):
        return {key: _zeroed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_zeroed(item) for item in value]
    return None


def derive_terminal_spec(
    source: Dict[str, Any], source_case: Dict[str, Any], directive: Any
) -> Dict[str, Any]:
    """Trusted-side re-expression: upstream acceptance signal → verifier_spec.

    ``directive`` accepts ``kind`` (unit_test | artifact_compare), the
    candidate deliverable ``path``, and an optional ``timeout_s`` / ``fields``.
    The upstream ``tests/`` / ``solution/`` content is read from the pinned
    snapshot and re-frozen into the spec and gold deliverables; the snapshot's
    original test text never travels anywhere else.
    """
    if not isinstance(directive, dict) or set(directive) - {
        "kind", "path", "timeout_s", "fields",
    }:
        raise BenchmarkError(
            "terminal directive accepts only kind, path, timeout_s, fields"
        )
    kind = directive.get("kind")
    if kind not in SPEC_KINDS:
        raise BenchmarkError(
            f"terminal directive kind must be one of {SPEC_KINDS}: {kind!r}"
        )
    deliverable = _clean_relative(directive.get("path") or "deliverable/files.json")
    timeout = directive.get("timeout_s", 30.0)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise BenchmarkError("terminal timeout_s must be a positive number")
    _require_admitted(source, source_case)
    adapter = _pinned_adapter(source)
    seed_id = source_case["source_task_id"]
    generator_ref = f"tb21-reexpression@{require_nonempty(source.get('release'), 'release')}"
    if kind == "unit_test":
        setup_files = source_case.get("terminal", {}).get("setup_files") or {}
        spec = {
            "kind": kind,
            "generator_ref": generator_ref,
            "payload": {
                "artifact": {"path": deliverable, "format": "json"},
                "setup_files": dict(setup_files),
                "test_script": adapter.test_script_text(seed_id),
                "timeout_s": float(timeout),
            },
        }
        gold = adapter.solution_files(seed_id)
    else:
        fields = directive.get("fields") or {}
        if not isinstance(fields, dict):
            raise BenchmarkError("terminal directive fields must be an object")
        gold = adapter.gold_document(seed_id)
        spec = {
            "kind": kind,
            "generator_ref": generator_ref,
            "payload": {
                "artifact": {"path": deliverable, "format": "json"},
                "gold": gold,
                "fields": fields,
            },
        }
    negative = _zeroed(gold)
    if negative == gold:
        raise BenchmarkError("degenerate negative deliverable for terminal task")
    return {
        "verifier_spec": spec,
        "gold_deliverable": gold,
        "negative_deliverable": negative,
        "initial_state_sha256": source_case["initial_state_sha256"],
        "expected_state_sha256": canonical_sha256(
            {"verifier_spec": spec, "gold_deliverable": gold}
        ),
    }


def _pin(value: Any, name: str) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{name} must be an object")
    path = Path(require_nonempty(value.get("path"), f"{name}.path"))
    sha = require_sha(value.get("sha256"), f"{name}.sha256")
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != sha:
        raise BenchmarkError(f"terminal {name} missing or SHA-256 drift")
    return {"path": str(path), "sha256": sha}


def _terminal_derived_variant(
    row: Dict[str, Any], index: int, strategy: str,
    environment: Dict[str, str], verifier: Dict[str, Any],
    audit: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise BenchmarkError("derived row must be an object")
    missing = [key for key in _DERIVED_KEYS if row.get(key) in (None, "")]
    if missing:
        raise BenchmarkError(f"derived row missing fields: {missing}")
    spec = row["verifier_spec"]
    if not isinstance(spec, dict) or spec.get("kind") not in SPEC_KINDS:
        raise BenchmarkError(
            f"derived verifier_spec kind must be one of {SPEC_KINDS}"
        )
    if not isinstance(spec.get("payload"), dict):
        raise BenchmarkError("derived verifier_spec invalid")
    artifact = spec["payload"].get("artifact")
    if not isinstance(artifact, dict) or not str(artifact.get("path") or "").strip():
        raise BenchmarkError("derived verifier_spec needs a deliverable path")
    deliverable = _clean_relative(artifact["path"])
    source_task_id = require_nonempty(row["source_task_id"], "source_task_id")
    variant_id = require_nonempty(
        row.get("item_fingerprint") or f"{strategy}-{index:04d}", "variant_id"
    )
    tools = row["tools_available"]
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise BenchmarkError("derived tools_available must be a string list")
    return {
        "source_task_id": source_task_id,
        "variant_id": variant_id,
        "operation": strategy,
        "public": {
            "task_text": require_nonempty(row["task_text"], "task_text"),
            "tools_available": list(tools),
            "success_criteria": require_nonempty(
                row["success_criteria"], "success_criteria"
            ),
            "difficulty": require_nonempty(row["difficulty"], "difficulty"),
        },
        "private": {
            "initial_state_sha256": canonical_sha256(
                {
                    "environment_sha256": environment["sha256"],
                    "task_text": row["task_text"],
                }
            ),
            "expected_state_sha256": require_sha(
                row.get("expected_state_sha256"), "expected_state_sha256"
            ),
            "verifier_spec": spec,
            "deliverable_path": deliverable,
            "strategy": strategy,
        },
        "runtime": {
            "environment": dict(environment),
            "verifier": dict(verifier),
            "assets": [],
        },
        "audit": dict(audit),
    }


def build_terminal_augment_config(
    derived_rows: List[Dict[str, Any]],
    *,
    release: str,
    split: str,
    strategy: str,
    environment: Any,
    verifier_path: str,
    verifier_version: str = RUNNER_VERSION,
    verifier_command: Optional[List[str]] = None,
    audit_roots: Any,
) -> Dict[str, Any]:
    """Wrap re-expressed terminal rows into a benchmark-augment-v1 config.

    ``environment`` is the pinned per-case setup snapshot (instruction +
    exported setup files, never upstream tests/solution); pass one record or
    one per row. The verifier pin locks the native runner shim and its
    command. The trusted-side ``verifier_spec`` (including the re-expressed
    assertions) travels in each variant's ``private`` block only.
    """
    if strategy not in SUPPORTED_OPERATIONS:
        raise BenchmarkError(f"unsupported terminal operation: {strategy!r}")
    split = require_split(split)
    release = require_nonempty(release, "release")
    if not isinstance(derived_rows, list) or not derived_rows:
        raise BenchmarkError("terminal augment needs non-empty derived rows")
    if isinstance(environment, dict) and "path" in environment:
        environments = [environment] * len(derived_rows)
    elif isinstance(environment, list) and len(environment) == len(derived_rows):
        environments = environment
    else:
        raise BenchmarkError("environment must be one record or one per row")
    environment_pins = [_pin(record, "environment") for record in environments]
    runner_path = Path(require_nonempty(verifier_path, "verifier_path"))
    if not runner_path.is_absolute() or not runner_path.is_file():
        raise BenchmarkError("terminal runner shim unavailable")
    command = list(
        verifier_command
        if verifier_command is not None
        else [sys.executable, str(runner_path)]
    )
    verifier = {
        "path": str(runner_path),
        "sha256": file_sha256(runner_path),
        "version": require_nonempty(verifier_version, "verifier_version"),
        "command": command,
    }
    if isinstance(audit_roots, dict):
        audits = [audit_roots] * len(derived_rows)
    elif isinstance(audit_roots, list) and len(audit_roots) == len(derived_rows):
        audits = audit_roots
    else:
        raise BenchmarkError("audit_roots must be one record or one per row")
    variants = [
        _terminal_derived_variant(
            row, index, strategy, environment_pin, verifier, audit
        )
        for index, (row, environment_pin, audit) in enumerate(
            zip(derived_rows, environment_pins, audits)
        )
    ]
    for variant in variants:
        lock_runtime(variant["runtime"])
    return {
        "schema_version": "benchmark-augment-v1",
        "family": "terminal",
        "release": release,
        "split": split,
        "variants": variants,
    }


def assert_terminal_build(
    runtime: Dict[str, Any], private: Dict[str, Any], source_case: Dict[str, Any]
) -> None:
    """M2 gate: every new terminal taskset pins the shim runner and a real spec."""
    metadata = source_case.get("terminal")
    instance_id = source_case.get("source_task_id")
    if not isinstance(metadata, dict):
        raise BenchmarkError(
            f"terminal instance {instance_id!r} lacks family metadata; "
            "import with the TB21 seed adapter"
        )
    if metadata.get("expressibility") != "expressible":
        raise BenchmarkError(
            f"terminal instance {instance_id!r} is excluded from derivation "
            f"({metadata.get('exclusion_reason', 'unknown')})"
        )
    verifier = runtime.get("verifier")
    if (
        not isinstance(verifier, dict)
        or verifier.get("version") != RUNNER_VERSION
    ):
        raise BenchmarkError(
            "terminal derivation must pin the offline runner shim "
            f"({RUNNER_VERSION})"
        )
    spec = private.get("verifier_spec") if isinstance(private, dict) else None
    if not isinstance(spec, dict) or spec.get("kind") not in SPEC_KINDS:
        raise BenchmarkError(
            f"terminal trusted case lacks a {SPEC_KINDS} verifier_spec: "
            f"{instance_id}"
        )


__all__ = [
    "SUPPORTED_OPERATIONS",
    "SUPPORTED_SEED_SOURCES",
    "SPEC_KINDS",
    "UPSTREAM_TASK_COUNT",
    "derive_release",
    "import_terminal_source",
    "verify_terminal_source_manifest",
    "screen_terminal_contamination",
    "derive_terminal_spec",
    "build_terminal_augment_config",
    "assert_terminal_build",
]
