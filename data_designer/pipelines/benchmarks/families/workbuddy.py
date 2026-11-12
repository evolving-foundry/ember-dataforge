"""workbuddy seed → benchmark-source-v1 importer and recipe augment adapter.

Input is the trusted seed material produced by ``WBBenchSeedAdapter`` (either a
synthetic fixture or the locally-held workbuddy-bench Office subset — never
redistributed). The importer freezes it into a ``benchmark-source-v1`` manifest;
the augment adapter wraps wb-task-v1 recipe rows (S1/S2/S3 outputs) into a
``benchmark-augment-v1`` config with per-case runtime pins pointing at the
frozen workspace snapshot and the locked native runner shim.

License red line (D3): ``derive_allowed=true`` is only valid for
``synthetic-fixture`` releases. Real-subset releases must stay
``derive_allowed=false`` until separately reviewed; ``build.py`` refuses any
other value. Real-subset native ingest (ingest as-is: do not rewrite task text, do not derive variants) is a
separate authorization: ``native_ingest_allowed=true`` is only valid for
``wb-bench-`` releases with a complete ``native_ingest_review`` triple, and the
two authorization flags are mutually exclusive.

Contamination red line: every derived ``task_text`` is screened against the
fingerprint set of the seed texts (exact ``normalized_text_sha256`` match) so
seed text never re-enters a published item. Native ingest rows re-enter seed
text *by definition*; they skip the screen and are covered instead by the
per-task ingest report (native-ingest evidence + cross-release fingerprint collision checks, design §5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts import (
    PRIVATE_MARKERS,
    BenchmarkError,
    canonical_sha256,
    file_sha256,
    normalized_text_sha256,
    require_nonempty,
    require_sha,
    require_split,
)
from ..runners.workbuddy import VERSION as RUNNER_VERSION
from ..runtime import lock_runtime

SUPPORTED_OPERATIONS = ("mutation", "taxonomy", "failure_probe", "native_ingest")
SUPPORTED_SEED_SOURCES = ("synthetic-fixture", "real-hf-subset")
_SNAPSHOT_FORMAT = "workbuddy-bench-archive-v1"
_NATIVE_RELEASE_PREFIX = "wb-bench-"
_ROW_METADATA_KEYS = (
    "subset", "role", "category", "deliverable_type", "verifier_form",
    "wb_keywords",
)
_DERIVED_KEYS = (
    "source_task_id", "task_text", "tools_available", "success_criteria",
    "difficulty", "verifier_spec",
)


def derive_release(
    seed_source: str, seed_rows: List[Dict[str, Any]],
    archive_sha256: Optional[str] = None,
) -> str:
    """D2 release semantics pinned to ``WBBenchSeedAdapter.seed_source``."""
    if seed_source == "synthetic-fixture":
        return "synthetic-fixture:" + canonical_sha256(list(seed_rows))[:8]
    if seed_source == "real-hf-subset":
        if not archive_sha256:
            raise BenchmarkError("real-hf-subset release needs the archive SHA-256")
        return "wb-bench-" + require_sha(archive_sha256, "archive sha256")[:8]
    raise BenchmarkError(f"unknown workbuddy seed_source: {seed_source!r}")


def _seed_case(row: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise BenchmarkError("seed row must be an object")
    task_text = require_nonempty(row.get("task_text"), "seed task_text")
    tools = row.get("tools_available")
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise BenchmarkError("seed tools_available must be a string list")
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
            }
        ),
        "expected_state_sha256": canonical_sha256(
            {
                "success_criteria": row.get("success_criteria"),
                "scoring_rubric": row.get("scoring_rubric"),
            }
        ),
        "workbuddy": {
            key: row[key] for key in _ROW_METADATA_KEYS if key in row
        },
    }


def _require_native_ingest_review(license_record: Dict[str, Any]) -> None:
    """D2 review triplet: reviewer/date/evidence complete + evidence pinned by sha256."""
    review = license_record.get("native_ingest_review")
    if not isinstance(review, dict):
        raise BenchmarkError("native ingest requires a native_ingest_review record")
    require_nonempty(review.get("reviewer"), "native_ingest_review.reviewer")
    require_nonempty(review.get("date"), "native_ingest_review.date")
    evidence = review.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(item, str) and item.strip() for item in evidence)
    ):
        raise BenchmarkError(
            "native_ingest_review.evidence must be a non-empty string list"
        )
    require_sha(review.get("evidence_sha256"), "native_ingest_review.evidence_sha256")


def import_workbuddy_source(
    seed_rows: List[Dict[str, Any]],
    *,
    release: str,
    split: str,
    license_record: Dict[str, Any],
    seed_source: str,
    dataset_id: str,
    archive_path: Optional[str] = None,
    archive_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Freeze trusted seed rows into a benchmark-source-v1 manifest."""
    if seed_source not in SUPPORTED_SEED_SOURCES:
        raise BenchmarkError(f"unknown workbuddy seed_source: {seed_source!r}")
    split = require_split(split)
    release = require_nonempty(release, "release")
    if not isinstance(seed_rows, list) or not seed_rows:
        raise BenchmarkError("workbuddy source needs non-empty seed rows")
    if not isinstance(license_record, dict) or (
        type(license_record.get("derive_allowed")) is not bool
    ):
        raise BenchmarkError("workbuddy license decision required")
    native_ingest = license_record.get("native_ingest_allowed", False)
    if type(native_ingest) is not bool:
        raise BenchmarkError("native_ingest_allowed must be a boolean")
    if native_ingest and license_record["derive_allowed"]:
        raise BenchmarkError(
            "native ingest and derivation cannot be authorized together (D3)"
        )
    if (
        license_record["derive_allowed"] is True
        and not release.startswith("synthetic-fixture:")
    ):
        raise BenchmarkError(
            "only synthetic-fixture releases may authorize derivation (D3)"
        )
    if native_ingest:
        if not release.startswith(_NATIVE_RELEASE_PREFIX):
            raise BenchmarkError(
                "only wb-bench- releases may authorize native ingest (D3)"
            )
        _require_native_ingest_review(license_record)
    if license_record.get("redistribute_allowed") is not False:
        raise BenchmarkError("workbuddy data must never be redistributable")
    cases = [_seed_case(row) for row in seed_rows]
    if len({case["source_task_id"] for case in cases}) != len(cases):
        raise BenchmarkError("duplicate workbuddy seed IDs")
    source: Dict[str, Any] = {
        "schema_version": "benchmark-source-v1",
        "family": "workbuddy",
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


def verify_workbuddy_source_manifest(source: Dict[str, Any]) -> None:
    """Re-derive the manifest from the pinned archive and compare every pin."""
    from ....sources.workbuddy_seed import WBBenchSeedAdapter

    snapshot = source.get("source_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("format") != _SNAPSHOT_FORMAT
        or not isinstance(snapshot.get("path"), str)
    ):
        raise BenchmarkError("workbuddy source needs a pinned seed archive")
    adapter = WBBenchSeedAdapter.from_archive(
        snapshot["path"], seed_source=snapshot.get("seed_source")
    )
    if adapter.dataset_id != snapshot.get("dataset_id"):
        raise BenchmarkError("workbuddy seed archive dataset ID drift")
    expected = import_workbuddy_source(
        adapter.load_seed_tasks(),
        release=source.get("release"),
        split=source.get("split"),
        license_record=source.get("license"),
        seed_source=snapshot.get("seed_source"),
        dataset_id=snapshot.get("dataset_id"),
        archive_path=snapshot.get("path"),
        archive_sha256=snapshot.get("sha256"),
    )
    if (
        source.get("release") != expected["release"]
        or source.get("license") != expected["license"]
        or source.get("exclusion_text_sha256") != expected["exclusion_text_sha256"]
        or source.get("cases") != expected["cases"]
    ):
        raise BenchmarkError("workbuddy source metadata drift")


def screen_workbuddy_contamination(
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


def _pin(value: Any, name: str) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{name} must be an object")
    path = Path(require_nonempty(value.get("path"), f"{name}.path"))
    sha = require_sha(value.get("sha256"), f"{name}.sha256")
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != sha:
        raise BenchmarkError(f"workbuddy {name} missing or SHA-256 drift")
    return {"path": str(path), "sha256": sha}


def _clean_relative(value: Any) -> str:
    relative = str(value)
    parts = Path(relative).parts
    if (
        Path(relative).is_absolute()
        or ".." in parts
        or not all(part not in ("", ".", "..") for part in parts)
    ):
        raise BenchmarkError(f"deliverable path must be a clean relative path: {relative!r}")
    return relative


def _composite_deliverable_path(payload: Dict[str, Any]) -> str:
    """wb_composite payload validation; the deliverable path uses the first required artifact."""
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BenchmarkError("wb_composite verifier_spec needs artifacts")
    paths: List[str] = []
    required: List[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise BenchmarkError("wb_composite artifact must be an object")
        relative = str(artifact.get("path") or "").strip()
        if not relative:
            raise BenchmarkError("wb_composite artifact needs a path")
        path = _clean_relative(relative)
        if path in paths:
            raise BenchmarkError("wb_composite artifact paths must be unique")
        paths.append(path)
        if artifact.get("required") is True:
            required.append(path)
    if not required:
        raise BenchmarkError("wb_composite needs at least one required artifact")
    rubrics = payload.get("rubrics")
    if not isinstance(rubrics, list) or not all(
        isinstance(rubric, dict) for rubric in rubrics
    ):
        raise BenchmarkError("wb_composite verifier_spec needs rubric objects")
    score_merge = payload.get("score_merge")
    if not isinstance(score_merge, dict):
        raise BenchmarkError("wb_composite verifier_spec needs score_merge")
    for weight in ("rule_weight", "llm_weight"):
        value = score_merge.get(weight)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise BenchmarkError(f"wb_composite score_merge.{weight} invalid")
    require_sha(payload.get("grading_dir_sha256"), "wb_composite grading_dir_sha256")
    env_pin = payload.get("env_pin")
    if not isinstance(env_pin, dict):
        raise BenchmarkError("wb_composite verifier_spec needs a workspace env_pin")
    _pin(env_pin, "wb_composite env_pin")
    return required[0]


def _derived_variant(
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
    if not isinstance(spec, dict) or not isinstance(spec.get("payload"), dict):
        raise BenchmarkError("derived verifier_spec invalid")
    if spec.get("kind") == "wb_composite":
        deliverable_path = _composite_deliverable_path(spec["payload"])
    else:
        artifact = spec["payload"].get("artifact")
        if not isinstance(artifact, dict) or not str(artifact.get("path") or "").strip():
            raise BenchmarkError("derived verifier_spec needs a deliverable path")
        deliverable_path = _clean_relative(artifact["path"])
    gold = spec["payload"].get("gold")
    if not isinstance(gold, dict) or not gold:
        raise BenchmarkError("derived verifier_spec needs a frozen gold")
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
            "expected_state_sha256": canonical_sha256({"gold": gold}),
            "verifier_spec": spec,
            "deliverable_path": deliverable_path,
            "strategy": strategy,
        },
        "runtime": {
            "environment": dict(environment),
            "verifier": dict(verifier),
            "assets": [],
        },
        "audit": dict(audit),
    }


def build_native_ingest_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Native-ingest row (design §5): construct a ``_DERIVED_KEYS`` row per task instead of mutation.generate.

    ``task_text`` is the original instruction.md text (contamination screening is replaced by the per-task ingest report);
    ``verifier_spec`` kind=``wb_composite``; the payload carries the full judge.yaml
    (artifacts/rubrics/score_merge) and the grading directory sha256. gold and workspace
    env_pin go only into the trusted manifest (private.verifier_spec), not the public item.
    """
    rows: List[Dict[str, Any]] = []
    seen_texts: set = set()
    for record in records:
        if not isinstance(record, dict):
            raise BenchmarkError("native record must be an object")
        judge = record.get("judge")
        if not isinstance(judge, dict) or not isinstance(
            judge.get("llm_judge"), dict
        ):
            raise BenchmarkError("native record needs a parsed judge document")
        task_text = require_nonempty(record.get("task_text"), "native task_text")
        text_sha = normalized_text_sha256(task_text)
        if text_sha in seen_texts:
            raise BenchmarkError("duplicate native task text across seed records")
        seen_texts.add(text_sha)
        payload: Dict[str, Any] = {
            "artifacts": judge.get("artifacts"),
            "rubrics": judge["llm_judge"].get("rubrics") or [],
            "llm_judge_method": judge["llm_judge"].get(
                "method", "rubric_binary_mean"
            ),
            "score_merge": judge.get("score_merge"),
            "grading_dir_sha256": record.get("grading_dir_sha256"),
            "env_pin": dict(record.get("workspace") or {}),
            "gold": record.get("gold"),
        }
        if record.get("verifier_timeout_sec") is not None:
            payload["timeout_sec"] = record["verifier_timeout_sec"]
        verifier_spec = {
            "kind": "wb_composite",
            "generator_ref": f"wb_bench_native_ingest@{record.get('seed_id')}",
            "payload": payload,
        }
        rows.append(
            {
                "source_task_id": require_nonempty(
                    record.get("seed_id"), "native seed_id"
                ),
                "task_text": task_text,
                "tools_available": list(record.get("tools_available") or []),
                "success_criteria": require_nonempty(
                    record.get("success_criteria"), "native success_criteria"
                ),
                "difficulty": require_nonempty(
                    record.get("difficulty"), "native difficulty"
                ),
                "item_fingerprint": canonical_sha256(
                    {"task_text": task_text, "verifier_spec": verifier_spec}
                )[:16],
                "verifier_spec": verifier_spec,
            }
        )
    return rows


def verify_native_ingest_pins(
    config: Dict[str, Any], expected_task_ids: Any
) -> Dict[str, Any]:
    """Validate the augment config's native pins and return the payload pin values.

    ingest report: re-check file sha256 + per-task entries exactly cover expected_task_ids and
    contain no ``excluded`` tasks. public inputs: directory exists, per-file sha256 re-checked, paths exclude
    ``tests/**`` and PRIVATE_MARKERS hits (D5 public surface, design §6.2).
    """
    if not isinstance(expected_task_ids, set):
        expected_task_ids = set(expected_task_ids)
    ingest_report = config.get("ingest_report")
    public_inputs = config.get("public_inputs")
    if not isinstance(ingest_report, dict) or not isinstance(public_inputs, dict):
        raise BenchmarkError("native_ingest config needs ingest and public input pins")
    report_path = Path(
        require_nonempty(ingest_report.get("path"), "ingest_report.path")
    )
    report_sha = require_sha(ingest_report.get("sha256"), "ingest_report.sha256")
    if (
        not report_path.is_absolute()
        or not report_path.is_file()
        or file_sha256(report_path) != report_sha
    ):
        raise BenchmarkError("ingest report missing or SHA-256 drift")
    try:
        report = json.loads(
            report_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"ingest report invalid: {error}") from error
    entries = report.get("tasks") if isinstance(report, dict) else None
    if not isinstance(entries, list):
        raise BenchmarkError("ingest report needs a tasks list")
    covered = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("task_id"), str):
            raise BenchmarkError("ingest report task entry invalid")
        if entry.get("excluded") is True:
            raise BenchmarkError(
                f"ingest report marks task excluded: {entry['task_id']}"
            )
        covered.add(entry["task_id"])
    if covered != expected_task_ids:
        raise BenchmarkError(
            "ingest report must cover exactly the native ingest rows"
        )
    inputs_dir = Path(require_nonempty(public_inputs.get("dir"), "public_inputs.dir"))
    if not inputs_dir.is_absolute() or not inputs_dir.is_dir():
        raise BenchmarkError("public_inputs.dir must be an absolute directory")
    by_task = public_inputs.get("by_task")
    if not isinstance(by_task, dict) or not by_task:
        raise BenchmarkError("public_inputs.by_task must be a non-empty object")
    if set(by_task) != expected_task_ids:
        raise BenchmarkError("public_inputs.by_task must cover exactly the rows")
    for task_id, entries in by_task.items():
        if not isinstance(entries, list) or not entries:
            raise BenchmarkError(f"public_inputs entries invalid: {task_id}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise BenchmarkError(f"public_inputs entry invalid: {task_id}")
            relative = _clean_relative(entry.get("path"))
            marker_hit = any(
                marker in part.lower()
                for part in Path(relative).parts
                for marker in PRIVATE_MARKERS
            )
            if marker_hit or relative.split("/", 1)[0] == "tests":
                raise BenchmarkError(
                    f"public input must not reference private material: {relative}"
                )
            target = inputs_dir / task_id / relative
            if not target.is_file() or file_sha256(target) != entry.get("sha256"):
                raise BenchmarkError(
                    f"public input missing or SHA-256 drift: {task_id}/{relative}"
                )
    return {
        "ingest_report_sha256": report_sha,
        "public_inputs_dir": str(inputs_dir),
        "public_inputs_by_task": by_task,
    }


def build_workbuddy_augment_config(
    derived_rows: List[Dict[str, Any]],
    *,
    release: str,
    split: str,
    strategy: str,
    environment: Dict[str, Any],
    verifier_path: str,
    verifier_version: str = RUNNER_VERSION,
    verifier_command: Optional[List[str]] = None,
    audit_roots: Any,
    ingest_report: Optional[Dict[str, Any]] = None,
    public_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wrap wb-task-v1 recipe rows into a benchmark-augment-v1 config.

    ``environment`` is the pinned initial workspace snapshot file; the verifier
    pin locks the native runner shim script and its executing command. The
    trusted-side ``verifier_spec`` (including gold) travels in each variant's
    ``private`` block and lands in the trusted case manifest only.

    ``strategy="native_ingest"`` additionally pins the per-task ingest report
    (path+sha256) and the public-input manifest (dir + per-task path/sha lists,
    design §6.2) into the config; ``build.py`` re-verifies both and copies the
    pins into the taskset payload.
    """
    if strategy not in SUPPORTED_OPERATIONS:
        raise BenchmarkError(f"unsupported workbuddy operation: {strategy!r}")
    split = require_split(split)
    release = require_nonempty(release, "release")
    if not isinstance(derived_rows, list) or not derived_rows:
        raise BenchmarkError("workbuddy augment needs non-empty derived rows")
    environment_pin = _pin(environment, "environment")
    runner_path = Path(require_nonempty(verifier_path, "verifier_path"))
    if not runner_path.is_absolute() or not runner_path.is_file():
        raise BenchmarkError("workbuddy runner shim unavailable")
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
    lock_runtime(
        {
            "environment": environment_pin,
            "verifier": verifier,
            "assets": [],
        }
    )
    if isinstance(audit_roots, dict):
        audits = [audit_roots] * len(derived_rows)
    elif isinstance(audit_roots, list) and len(audit_roots) == len(derived_rows):
        audits = audit_roots
    else:
        raise BenchmarkError("audit_roots must be one record or one per row")
    # wb_composite payload gains its trusted grading-source pin at config
    # time: the frozen seed archive already pinned as the environment is the
    # only source of tests/grading, and the runner re-verifies the extracted
# tree against grading_dir_sha256 + gold before grading (design §8.1/§8.3).
    prepared_rows: List[Dict[str, Any]] = []
    for row in derived_rows:
        spec = row.get("verifier_spec") if isinstance(row, dict) else None
        if isinstance(spec, dict) and spec.get("kind") == "wb_composite":
            payload = dict(spec.get("payload") or {})
            payload["grading_source"] = {
                "path": environment_pin["path"],
                "sha256": environment_pin["sha256"],
                "member_prefix": (
                    "tasks/"
                    f"{require_nonempty(row['source_task_id'], 'source_task_id')}"
                    "/tests"
                ),
            }
            row = {
                **row,
                "verifier_spec": {**spec, "payload": payload},
            }
        prepared_rows.append(row)
    variants = [
        _derived_variant(row, index, strategy, environment_pin, verifier, audit)
        for index, (row, audit) in enumerate(zip(prepared_rows, audits))
    ]
    config: Dict[str, Any] = {
        "schema_version": "benchmark-augment-v1",
        "family": "workbuddy",
        "release": release,
        "split": split,
        "variants": variants,
    }
    if strategy == "native_ingest":
        if not isinstance(ingest_report, dict):
            raise BenchmarkError("native_ingest requires an ingest report pin")
        if not isinstance(public_inputs, dict):
            raise BenchmarkError("native_ingest requires a public_inputs manifest")
        config["ingest_report"] = dict(ingest_report)
        config["public_inputs"] = dict(public_inputs)
        row_ids = {row["source_task_id"] for row in variants}
        config["public_inputs"]["by_task"] = {
            str(key): value for key, value in config["public_inputs"]["by_task"].items()
        }
        verify_native_ingest_pins(config, row_ids)
    elif ingest_report is not None or public_inputs is not None:
        raise BenchmarkError("ingest pins are native_ingest only")
    return config


__all__ = [
    "SUPPORTED_OPERATIONS",
    "import_workbuddy_source",
    "verify_workbuddy_source_manifest",
    "screen_workbuddy_contamination",
    "build_native_ingest_rows",
    "verify_native_ingest_pins",
    "build_workbuddy_augment_config",
    "derive_release",
]
