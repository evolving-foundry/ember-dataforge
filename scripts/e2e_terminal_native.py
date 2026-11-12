"""terminal native integration e2e: seed ingestion → source manifest → re-expression → taskset/suite freeze →
dual-arm real evaluation → private readiness.

Following the six-stage structure of ``e2e_workbuddy_native.py``, this script uses the benchmark-case-v1
freeze pipeline: ``build_benchmark_taskset`` carries the build audit; dual-arm real evaluation grades each task
directly with the runtime locked in the taskset artifact; readiness uses ``load_suite`` (approved review-evidence
pins) as the private bookkeeping gate -- the old worker shell protocol has been removed. The whole chain is offline: it pulls nothing and
runs no docker images (T7); scoring only recognizes deliverable files under candidateRoot.

- The default seed is an in-repo desensitized synthetic fixture (``seed_source=synthetic-fixture``,
  ``derive_allowed=true``, derivable). When ``--seed-dir`` points to a real terminal-bench-2.1
  checkout (dataset_id does not contain "fixture", ``seed_source=upstream-checkout``), ``--upstream-commit``
  must also be provided (T2 40-hex pin + 89-task assertion): only the source manifest is frozen before
  stopping explicitly; re-expression of the real subset requires separate approval.
- Dual-arm real evaluation (D6 candidate protocol): the reference arm writes the re-expression-frozen ``gold_deliverable`` to
  the relative path declared under candidateRoot (expected 1.0); the wrong-baseline arm writes
  ``negative_deliverable`` (expected 0.0).
- readiness is ``owned_native`` private bookkeeping: review files and six evidence pins are written to the work directory
  (must be outside the repository, defaults to tempfile), expecting ``selected_evidence_complete=true``,
  ``selected_distribution=private_evaluation_only``.

Usage (scripts/__init__.py has a relative import beyond the top level, so it cannot be run as a module):
  PYTHONPATH=. python3 scripts/e2e_terminal_native.py --work-dir <directory outside the repo>

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

from data_designer.pipelines.benchmarks.contracts import file_sha256
from data_designer.pipelines.benchmarks.families.terminal import (
    build_terminal_augment_config,
    derive_release,
    derive_terminal_spec,
    import_terminal_source,
    screen_terminal_contamination,
    verify_terminal_source_manifest,
)
from data_designer.pipelines.benchmarks.suite import (
    OFFICIAL_EVIDENCE_FIELDS,
    assert_comparable_evaluations,
    create_suite,
    set_publication_status,
)
from data_designer.sources.terminal_seed import TB21SeedAdapter
from data_designer.storage.catalog import ArtifactCatalog

from _benchmark_e2e import build_taskset, eval_arm, verify_private_readiness

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_DIR = (
    _PROJECT_ROOT / "data_designer" / "sources" / "fixtures"
    / "tb21_terminal_fixture"
)
_RUNNER_PATH = (
    _PROJECT_ROOT / "data_designer" / "pipelines" / "benchmarks" / "runners" / "terminal.py"
)

_LICENSE = {
    "id": "internal-use",
    "derive_allowed": True,
    "access_conditions": "archived upstream/synthetic licensing; internal evaluation",
}

# Per-case re-expression directives (candidate deliverable declaration).
_DIRECTIVES = {
    "tb-fixture-001-fix-script": {
        "kind": "unit_test", "path": "deliverable/files.json", "timeout_s": 15,
    },
    "tb-fixture-002-rollup": {
        "kind": "artifact_compare", "path": "deliverable/summary.json",
        "timeout_s": 15,
    },
}


def _write_json(path: Path, value: Dict[str, Any]) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(path)


def _tar_seed_dir(seed_dir: Path, work: Path) -> tuple[str, str]:
    archive = work / "terminal-seed.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(seed_dir, arcname=seed_dir.name)
    return str(archive), file_sha256(archive)


def _derived_task_text(task_text: str, deliverable: str) -> str:
    return (
        f"{task_text}\n\nCandidate protocol: write the deliverable "
        f"JSON described above to the declared path '{deliverable}' under "
        f"your candidate root."
    )


def _reexpress(
    source: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-express the expressible subset: augment rows + frozen spec bundle (gold/negative)."""
    rows: List[Dict[str, Any]] = []
    bundles: List[Dict[str, Any]] = []
    for case in source["cases"]:
        if case["terminal"]["expressibility"] != "expressible":
            continue
        seed_id = case["source_task_id"]
        directive = _DIRECTIVES.get(seed_id)
        if directive is None:
            raise RuntimeError(
                f"missing seed_id {seed_id!r} re-expression directive; fixture subsets must declare it per task"
            )
        bundle = derive_terminal_spec(source, case, directive)
        deliverable = bundle["verifier_spec"]["payload"]["artifact"]["path"]
        bundles.append(bundle)
        rows.append(
            {
                "source_task_id": seed_id,
                "task_text": _derived_task_text(case["task_text"], deliverable),
                "tools_available": case["tools_available"],
                "success_criteria": case["success_criteria"],
                "difficulty": case["difficulty"],
                "expected_state_sha256": bundle["expected_state_sha256"],
                "item_fingerprint": seed_id.replace("tb-fixture-", "v") + "-a",
                "verifier_spec": bundle["verifier_spec"],
            }
        )
    if not rows:
        raise RuntimeError("Re-expression result is empty: the seed manifest has no expressible tasks")
    screen_terminal_contamination(rows, source["cases"])
    return rows, bundles


def _write_deliverables(root: Path, bundles: List[Dict[str, Any]], field: str) -> None:
    for bundle in bundles:
        relative = bundle["verifier_spec"]["payload"]["artifact"]["path"]
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(bundle[field], ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Terminal native ingestion e2e: full frozen-pipeline run at fixture scale"
    )
    ap.add_argument("--seed-dir", default=None,
                    help="Extracted terminal-bench-2.1 task root (contains "
                         "dataset.toml); defaults to the in-repo sanitized synthetic fixture."
                         "A real checkout must also provide --upstream-commit")
    ap.add_argument("--upstream-commit", default=None,
                    help="Upstream 40-hex commit for the real checkout (T2 release pin)")
    ap.add_argument("--work-dir", default=None,
                    help="Working directory (source manifest, trusted data, store, evidence; must be outside the repository)")
    args = ap.parse_args(argv)
    if args.upstream_commit is not None and args.seed_dir is None:
        raise ValueError("--upstream-commit may be used only with --seed-dir (real checkout)")
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="tb-native-e2e-"))
    work.mkdir(parents=True, exist_ok=True)

    # ── 1. Seed ingestion ──────────────────────────────────────────────
    seed_dir = Path(args.seed_dir) if args.seed_dir else _FIXTURE_DIR
    if not (seed_dir / "dataset.toml").is_file():
        raise ValueError(f"Seed directory is missing dataset.toml: {seed_dir}")
    adapter = TB21SeedAdapter(seed_dir)
    seed_rows = adapter.load_seed_tasks()
    seed_source = adapter.seed_source
    archive_path, archive_sha = _tar_seed_dir(seed_dir, work)

    # ── 2. Source manifest (T2 release pin; T3 license archive has no gate) ──
    release = derive_release(
        seed_source, seed_rows, upstream_commit=args.upstream_commit
        if seed_source == "upstream-checkout" else None,
    )
    source = import_terminal_source(
        seed_rows,
        release=release,
        split="development",
        license_record=dict(_LICENSE),
        seed_source=seed_source,
        dataset_id=adapter.dataset_id,
        upstream_commit=args.upstream_commit,
        archive_path=archive_path,
        archive_sha256=archive_sha,
    )
    verify_terminal_source_manifest(source)
    source_path = _write_json(work / "source.json", source)
    if seed_source == "upstream-checkout":
        report = {
            "stopped_at": "source_stage",
            "reason": (
                "upstream-checkout re-expression requires separate approval: the source manifest is frozen (T2 release "
                f"{release}) and is excluded from derivation and evaluation"
            ),
            "seed_stage": {
                "seed_dir": str(seed_dir), "seed_row_count": len(seed_rows),
            },
            "source_stage": {"release": release, "source_path": source_path},
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    # ── 3. Re-expression (T6 scoring localization + contamination fingerprint screening) ──
    derived, bundles = _reexpress(source)

    # ── 4. taskset/suite freeze (gold/negative native audit enforced within build) ──
    reference_root = work / "audit-reference"
    negative_root = work / "audit-negative"
    _write_deliverables(reference_root, bundles, "gold_deliverable")
    _write_deliverables(negative_root, bundles, "negative_deliverable")
    environments = []
    for row in derived:
        env_archive = work / f"env-{row['source_task_id']}.tar.gz"
        env_sha = adapter.export_setup_archive(env_archive, row["source_task_id"])
        environments.append({"path": str(env_archive), "sha256": env_sha})
    config = build_terminal_augment_config(
        derived,
        release=release,
        split="development",
        strategy="setup_fault",
        environment=environments,
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
        "terminal-native-e2e",
        family="terminal",
        release=release,
        split="development",
    )
    suite_ref_id = "terminal-native-e2e-suite"
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
                f"terminal native e2e evidence: {label}\nrelease={release}\n"
                f"source_sha256={file_sha256(source_path)}\n"
                f"taskset_sha256={snapshot['taskset_sha256']}\n",
                encoding="utf-8",
            )
            evidence_pins[label] = {
                "path": str(path), "sha256": file_sha256(path),
            }
        review_path = work / "suite-review.json"
        review = {
            "schema_version": "benchmark-suite-review-v1",
            "scope": "owned_native",
            "suite_sha256": suite["suite_sha256"],
            "tasksets": suite["tasksets"],
            "reviewer": "e2e_terminal_native",
            "evidence_refs": [
                f"source:{source_path}", f"config:{config_path}",
                f"audit-roots:{reference_root}|{negative_root}",
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
    _write_deliverables(work / "candidate-reference", bundles, "gold_deliverable")
    _write_deliverables(work / "candidate-baseline", bundles, "negative_deliverable")
    reference_eval = eval_arm(store, suite, work / "candidate-reference")
    baseline_eval = eval_arm(store, suite, work / "candidate-baseline")
    assert_comparable_evaluations(reference_eval, baseline_eval)
    if reference_eval["metrics"]["pass_rate_terminal"] != 1.0:
        raise RuntimeError("Reference arm did not reach 1.0: gold deliverable and verifier_spec are inconsistent")
    if baseline_eval["metrics"]["pass_rate_terminal"] != 0.0:
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
            "strategy": "setup_fault",
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
            "reference_pass_rate": reference_eval["metrics"]["pass_rate_terminal"],
            "baseline_pass_rate": baseline_eval["metrics"]["pass_rate_terminal"],
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
