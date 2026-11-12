"""workbuddy native integration e2e: seed ingestion → source manifest → derivation → taskset/suite freeze →
dual-arm real evaluation → private readiness.

Following the six-stage structure of ``e2e_terminal_native.py``, this script uses the
benchmark-case-v1 freeze pipeline: ``build_benchmark_taskset`` carries the build audit; dual-arm real
evaluation grades each task directly with the runtime locked in the taskset artifact; readiness uses
``load_suite`` (approved review-evidence pins) as the private bookkeeping gate -- the old worker shell
protocol has been removed.

- The default seed is an in-repo desensitized synthetic fixture (``seed_source=synthetic-fixture``,
  ``derive_allowed=true``, derivable). When ``--seed-dir`` points to the real workbuddy-bench
  Office subset (``seed_source=real-hf-subset``), only the source manifest is frozen before stopping explicitly:
  derivation of the real subset requires separate approval (D3), and build rejects it directly when ``derive_allowed=false``.
- Dual-arm real evaluation (D6 candidate protocol): the reference arm writes the verifier_spec-frozen ``gold_solution``
  to the relative path declared under candidateRoot (expected 1.0); the wrong-baseline arm writes
  ``negative_solution`` (expected 0.0). It runs offline without calling an external solver.
- readiness is ``owned_native`` private bookkeeping: review files and six evidence pins are written to the work directory
  (must be outside the repository, defaults to tempfile), expecting ``selected_evidence_complete=true``,
  ``selected_distribution=private_evaluation_only``.

Usage (scripts/__init__.py has a relative import beyond the top level, so it cannot be run as a module):
  PYTHONPATH=. python3 scripts/e2e_workbuddy_native.py --work-dir <directory outside the repo>

Exit code: 0 = full chain passes; 2 = precondition failure (explicit failure, never silent).
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from data_designer.pipelines.benchmarks.contracts import (
    canonical_sha256,
    file_sha256,
)
from data_designer.pipelines.benchmarks.families.workbuddy import (
    build_workbuddy_augment_config,
    derive_release,
    import_workbuddy_source,
    screen_workbuddy_contamination,
    verify_workbuddy_source_manifest,
)
from data_designer.pipelines.benchmarks.suite import (
    OFFICIAL_EVIDENCE_FIELDS,
    assert_comparable_evaluations,
    create_suite,
    set_publication_status,
)
from data_designer.generation.algorithms.builtin.workbuddy_mutation import WorkbuddyMutationEngine
from data_designer.sources.workbuddy_seed import WBBenchSeedAdapter
from data_designer.storage.catalog import ArtifactCatalog

from _benchmark_e2e import build_taskset, eval_arm, verify_private_readiness

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_DIR = (
    _PROJECT_ROOT / "data_designer" / "sources" / "fixtures"
    / "wbbench_office_fixture"
)
_RUNNER_PATH = (
    _PROJECT_ROOT / "data_designer" / "pipelines" / "benchmarks" / "runners" / "workbuddy.py"
)

_FIXTURE_LICENSE = {
    "id": "local-synthetic-fixture",
    "derive_allowed": True,
    "redistribute_allowed": False,
    "access_conditions": "synthetic fixture, local evaluation only",
}
_REAL_SUBSET_LICENSE = {
    "id": "local-workbuddy-bench-office",
    "derive_allowed": False,
    "redistribute_allowed": False,
    "access_conditions": "local evaluation only; derivation needs separate review",
}


def _write_json(path: Path, value: Dict[str, Any]) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(path)


def _tar_seed_dir(seed_dir: Path, work: Path) -> tuple[str, str]:
    archive = work / "workbuddy-seed.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(seed_dir, arcname=seed_dir.name)
    return str(archive), file_sha256(archive)


def _replace_strings(node: Any, old: str, new: str) -> Any:
    if isinstance(node, dict):
        return {k: _replace_strings(v, old, new) for k, v in node.items()}
    if isinstance(node, list):
        return [_replace_strings(v, old, new) for v in node]
    if isinstance(node, str):
        return node.replace(old, new)
    return node


def _namespace_deliverables(tasks: List[Any]) -> None:
    """Rewrite each task's deliverable path into a unique subdirectory (in-place).

    Same-family variants from the mutation engine share a fixed deliverable path within the family (the task-text template is fixed),
    while dual-arm real evaluation writes gold/negative deliverables into a flat candidateRoot, so identical paths overwrite
    each other -- without this rewrite, deduplication could only be done by path, and variants derived by --expand-count would be
    discarded in batches. The rewrite also covers the same literals in the task text and verifier_spec: candidates write to the path
    declared by the task text, and the verifier resolves by artifact.path; keeping both consistent preserves evaluation semantics.
    """
    for i, task in enumerate(tasks):
        spec = task.verifier_spec
        old = spec["payload"]["artifact"]["path"]
        stem, name = old.rsplit("/", 1)
        new = f"{stem}/t{i:04d}/{name}"
        task.task_text = task.task_text.replace(old, new)
        task.success_criteria = task.success_criteria.replace(old, new)
        task.verifier_spec = _replace_strings(spec, old, new)


def _derived_rows(tasks: List[Any], seed_ids: set) -> tuple[List[Dict[str, Any]], List[Any]]:
    """MutationTask → augment rows; keep only the first task for a duplicate deliverable path (identical
    paths under the same candidateRoot overwrite each other, and dual-arm real evaluation requires pairwise distinct paths). Callers
    first rewrite via _namespace_deliverables so paths are pairwise distinct; this deduplication is only a fallback."""
    rows: List[Dict[str, Any]] = []
    kept: List[Any] = []
    seen_paths: set = set()
    for task in tasks:
        spec = task.verifier_spec
        path = spec["payload"]["artifact"]["path"]
        source_task_id = task.provenance_extra.get("seed_task_id")
        if source_task_id not in seed_ids:
            raise RuntimeError(
                f"Derived task source_task_id {source_task_id!r} is not in the seed manifest"
            )
        if path in seen_paths:
            continue
        seen_paths.add(path)
        kept.append(task)
        rows.append(
            {
                "source_task_id": source_task_id,
                "task_text": task.task_text,
                "tools_available": ["workdir"],
                "success_criteria": task.success_criteria,
                "difficulty": task.difficulty,
                "item_fingerprint": canonical_sha256(
                    {"task_text": task.task_text, "verifier_spec": spec}
                )[:16],
                "verifier_spec": spec,
            }
        )
    return rows, kept


def _write_deliverables(root: Path, tasks: List[Any], field: str) -> None:
    for task in tasks:
        relative = task.verifier_spec["payload"]["artifact"]["path"]
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(getattr(task, field), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="WorkBuddy native ingestion e2e: full frozen-pipeline run at fixture scale"
    )
    ap.add_argument("--seed-dir", default=None,
                    help="Extracted workbuddy-bench Office subset root (contains "
                         "dataset.toml); defaults to the in-repo sanitized synthetic fixture")
    ap.add_argument("--mutation-seed", type=int, default=42)
    ap.add_argument("--expand-count", type=int, default=3,
                    help="Maximum derived-task count; each task is rewritten to a unique deliverable subdirectory,"
                         "so variants with the same path do not collide")
    ap.add_argument("--work-dir", default=None,
                    help="Working directory (source manifest, trusted data, store, evidence; must be outside the repository)")
    ap.add_argument("--judge-calibration-report", default=None,
                    help="Path to the M4 judge calibration report (wb-m4-judge-calibration-v1);"
                         "when provided, its path+sha256 is recorded in suite review evidence_refs"
                         "(section 10 M4: calibration is recorded in suite review)")
    args = ap.parse_args(argv)
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="wb-native-e2e-"))
    work.mkdir(parents=True, exist_ok=True)

    # ── 1. Seed ingestion ──────────────────────────────────────────────
    seed_dir = Path(args.seed_dir) if args.seed_dir else _FIXTURE_DIR
    seed_source = "real-hf-subset" if args.seed_dir else "synthetic-fixture"
    if not (seed_dir / "dataset.toml").is_file():
        raise RuntimeError(f"Seed directory is missing dataset.toml: {seed_dir}")
    adapter = WBBenchSeedAdapter(seed_dir, seed_source=seed_source)
    seed_rows = adapter.load_seed_tasks()
    archive_path, archive_sha = _tar_seed_dir(seed_dir, work)

    # ── 2. Source manifest ─────────────────────────────────────────────
    release = derive_release(seed_source, seed_rows, archive_sha)
    license_record = (
        dict(_REAL_SUBSET_LICENSE) if seed_source == "real-hf-subset"
        else dict(_FIXTURE_LICENSE)
    )
    source = import_workbuddy_source(
        seed_rows,
        release=release,
        split="development",
        license_record=license_record,
        seed_source=seed_source,
        dataset_id=adapter.dataset_id,
        archive_path=archive_path,
        archive_sha256=archive_sha,
    )
    verify_workbuddy_source_manifest(source)
    source_path = _write_json(work / "source.json", source)
    if seed_source == "real-hf-subset":
        report = {
            "stopped_at": "source_stage",
            "reason": (
                "real-hf-subset derivation requires separate approval (D3, derive_allowed=false);"
                "the source manifest is frozen and excluded from derivation and evaluation"
            ),
            "seed_stage": {
                "seed_dir": str(seed_dir), "seed_row_count": len(seed_rows),
            },
            "source_stage": {"release": release, "source_path": source_path},
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    # ── 3. Derivation (S1 mutation engine + contamination fingerprint screening) ──
    engine = WorkbuddyMutationEngine(args.mutation_seed)
    profiles: List[Dict[str, Any]] = []
    for row in seed_rows:
        profile = {"seed_id": row["seed_id"], "category": row.get("category")}
        if row.get("difficulty"):
            profile["difficulty"] = row["difficulty"]
        if profile not in profiles:
            profiles.append(profile)
    tasks = engine.generate(profiles=profiles, count=max(1, args.expand_count))
    _namespace_deliverables(tasks)
    derived, kept_tasks = _derived_rows(
        tasks, {row["seed_id"] for row in seed_rows})
    if not derived:
        raise RuntimeError("Derived result is empty; adjust --expand-count")
    screen_workbuddy_contamination(derived, seed_rows)

    # ── 4. taskset/suite freeze (gold/negative native audit enforced within build) ──
    reference_root = work / "audit-reference"
    negative_root = work / "audit-negative"
    _write_deliverables(reference_root, kept_tasks, "gold_solution")
    _write_deliverables(negative_root, kept_tasks, "negative_solution")
    config = build_workbuddy_augment_config(
        derived,
        release=release,
        split="development",
        strategy="mutation",
        environment={"path": archive_path, "sha256": archive_sha},
        verifier_path=str(_RUNNER_PATH),
        audit_roots={
            "reviewed": True,
            "reference_root": str(reference_root),
            "negative_root": str(negative_root),
        },
    )
    config_path = _write_json(work / "config.json", config)
    snapshot = build_taskset(
        ArtifactCatalog(str(work / "catalog")),
        work,
        source_path,
        config_path,
        "workbuddy-native-e2e",
        family="workbuddy",
        release=release,
        split="development",
    )
    suite_ref_id = "workbuddy-native-e2e-suite"
    store = ArtifactCatalog(str(work / "catalog"))
    try:
        suite = create_suite(store, suite_ref_id, [snapshot["taskset"]], repeats=1)
        # owned_native private bookkeeping: review and six evidence pins in the work directory outside the repo
        evidence_dir = work / "evidence"
        evidence_dir.mkdir(exist_ok=True)
        evidence_pins: Dict[str, Dict[str, str]] = {}
        for label in sorted(OFFICIAL_EVIDENCE_FIELDS):
            path = evidence_dir / f"{label}.txt"
            path.write_text(
                f"workbuddy native e2e evidence: {label}\nrelease={release}\n"
                f"source_sha256={file_sha256(source_path)}\n"
                f"taskset_sha256={snapshot['taskset_sha256']}\n",
                encoding="utf-8",
            )
            evidence_pins[label] = {
                "path": str(path), "sha256": file_sha256(path),
            }
        review_path = work / "suite-review.json"
        calibration_ref: List[str] = []
        if args.judge_calibration_report:
            calibration_path = Path(args.judge_calibration_report).resolve()
            if not calibration_path.is_file():
                raise RuntimeError(
                    f"Judge calibration report does not exist: {calibration_path}"
                )
            calibration_ref.append(
                "judge-calibration:"
                f"{calibration_path}#sha256:{file_sha256(calibration_path)}"
            )
        review = {
            "schema_version": "benchmark-suite-review-v1",
            "scope": "owned_native",
            "suite_sha256": suite["suite_sha256"],
            "tasksets": suite["tasksets"],
            "reviewer": "e2e_workbuddy_native",
            "evidence_refs": [
                f"source:{source_path}", f"config:{config_path}",
                f"audit-roots:{reference_root}|{negative_root}",
                *calibration_ref,
            ],
            "evidence_pins": evidence_pins,
            "native_positive_negative_verified": True,
            "private_boundary_verified": True,
            "license_checked": True,
            "environment_reset_verified": True,
        }
        _write_json(review_path, review)
        set_publication_status(
            store, suite_ref_id, suite["version"], "approved",
            "private owned-native review (synthetic fixture e2e)",
            review_evidence={
                "path": str(review_path), "sha256": file_sha256(review_path),
            },
        )
    finally:
        store.close()

    # ── 5. Dual-arm real evaluation (D6: deliverables written to declared paths under candidateRoot) ──
    _write_deliverables(work / "candidate-reference", kept_tasks, "gold_solution")
    _write_deliverables(work / "candidate-baseline", kept_tasks, "negative_solution")
    reference_eval = eval_arm(store, suite, work / "candidate-reference")
    baseline_eval = eval_arm(store, suite, work / "candidate-baseline")
    assert_comparable_evaluations(reference_eval, baseline_eval)
    if reference_eval["metrics"]["pass_rate_workbuddy"] != 1.0:
        raise RuntimeError("Reference arm did not reach 1.0: gold deliverable and verifier_spec are inconsistent")
    if baseline_eval["metrics"]["pass_rate_workbuddy"] != 0.0:
        raise RuntimeError("Negative baseline arm did not reach zero: negative deliverables must not pass scoring")

    # ── 6. Private readiness: load_suite re-checks suite pins and review evidence, i.e. the bookkeeping gate ──
    readiness = verify_private_readiness(
        store,
        [(suite_ref_id, suite["version"], suite["suite_sha256"])],
    )
    if (
        readiness.get("scope") != "owned_native"
        or readiness.get("selected_evidence_complete") is not True
        or readiness.get("selected_distribution") != "private_evaluation_only"
    ):
        raise RuntimeError(f"readiness did not meet the private accounting expectation: {readiness}")

    print(json.dumps({
        "seed_stage": {
            "seed_dir": str(seed_dir),
            "seed_source": seed_source,
            "seed_row_count": len(seed_rows),
        },
        "source_stage": {
            "release": release,
            "dataset_id": adapter.dataset_id,
            "source_path": source_path,
            "source_sha256": file_sha256(source_path),
        },
        "derive_stage": {
            "strategy": "mutation",
            "mutation_seed": args.mutation_seed,
            "derived_count": len(derived),
            "contamination_screen": "pass",
        },
        "taskset_stage": {
            "taskset": snapshot["taskset"],
            "item_count": snapshot["item_count"],
            "taskset_sha256": snapshot["taskset_sha256"],
            "suite": {
                "ref_id": suite_ref_id,
                "version": suite["version"],
                "sha256": suite["suite_sha256"],
            },
        },
        "eval_stage": {
            "reference_pass_rate": reference_eval["metrics"]["pass_rate_workbuddy"],
            "baseline_pass_rate": baseline_eval["metrics"]["pass_rate_workbuddy"],
            "macro_pass_rate_reference": reference_eval["metrics"]["macro_pass_rate"],
            "comparable": True,
        },
        "readiness_stage": {
            "scope": readiness["scope"],
            "selected_evidence_complete": readiness["selected_evidence_complete"],
            "selected_distribution": readiness["selected_distribution"],
        },
        "work_dir": str(work),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(json.dumps({"error": str(error), "exit_code": 2},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
