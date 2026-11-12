"""HF single-turn evaluation integration e2e: seed ingestion → per-family source manifest → re-expression →
taskset/suite freeze → judge red line (when unconfigured) → dual-arm real evaluation → private readiness.

Following the six-stage structure of ``e2e_terminal_native.py``, this script covers 4 single-turn families
(reasoning_qa / knowledge_qa / code_gen / health_qa, corresponding to 11 HF benchmarks).
The refactored pipeline does not go through the worker shell protocol: ``build_benchmark_taskset`` carries the build audit,
and dual-arm real evaluation grades each task directly with the runtime locked in the taskset artifact; readiness uses
``load_suite`` (approved review-evidence pins) as the private bookkeeping gate. The whole chain is offline: scoring only recognizes
per-task deliverable files under candidateRoot (D6:
``deliverables/<source_task_id>/response.txt|solution.py``).

- The seed defaults to an in-repo desensitized synthetic fixture (``seed_source=synthetic-fixture``).
- judge gate (D7 red line): when all three of ``--judge-base-url/--judge-api-key/--judge-model``
  are provided, they are bootstrapped through ``DATA_FORGE_JUDGE_*`` environment variables, all 4 families are evaluated, and the
  ``judge_fingerprint`` is recorded in the results; when not provided, only deterministic families (reasoning_qa /
  code_gen) are run, and one explicit red-line build is performed each for knowledge_qa / health_qa (llm_rubric)
  -- the build audit is expected to fail with ``LlmJudgeConfigError`` (judge not configured) and produce no taskset.
- Dual-arm real evaluation: the reference arm writes ``gold_deliverable`` (expected 1.0), and the wrong-baseline arm writes
  ``negative_deliverable`` (expected 0.0).
- readiness is ``owned_native`` private bookkeeping: review files and six evidence pins are written to the work directory
  (must be outside the repository, defaults to tempfile).

Usage (scripts/__init__.py has a relative import beyond the top level, so it cannot be run as a module):
  PYTHONPATH=. python3 scripts/e2e_single_turn_native.py \
      --work-dir <directory outside the repo> [--judge-base-url U --judge-api-key K --judge-model M]

Exit code: 0 = full chain passes; 2 = precondition failure (explicit failure, never silent).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from data_designer.pipelines.benchmarks.contracts import BenchmarkError, file_sha256
from data_designer.pipelines.benchmarks.families.hf_single_turn import (
    FAMILY_OPERATIONS,
    build_hf_single_turn_augment_config,
    derive_hf_single_turn_spec,
    derive_release,
    import_hf_single_turn_source,
    screen_hf_single_turn_contamination,
    verify_hf_single_turn_source_manifest,
)
from data_designer.pipelines.benchmarks.suite import (
    OFFICIAL_EVIDENCE_FIELDS,
    assert_comparable_evaluations,
    create_suite,
    set_publication_status,
)
from data_designer.sources.hf_single_turn_seed import (
    HfSingleTurnSeedAdapter,
)
from data_designer.storage.catalog import ArtifactCatalog

from _benchmark_e2e import build_taskset, eval_arm, verify_private_readiness

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_DIR = (
    _PROJECT_ROOT / "data_designer" / "sources" / "fixtures"
    / "hf_single_turn_fixture"
)
_RUNNER_PATHS = {
    family: _PROJECT_ROOT / "data_designer" / "pipelines" / "benchmarks" / "runners"
    / f"{family}.py"
    for family in FAMILY_OPERATIONS
}

_LICENSE = {
    "id": "internal-use",
    "derive_allowed": True,
    "access_conditions": "archived upstream/synthetic licensing; internal evaluation",
}

# Per-family re-expression strategy (representative operations per family, at fixture scale).
_STRATEGY = {
    "reasoning_qa": "language_transfer",
    "knowledge_qa": "paraphrase",
    "code_gen": "docstring_perturbation",
    "health_qa": "scenario_variation",
}
_DIRECTIVE = {"timeout_s": 20.0, "pass_threshold": 0.5}


def _write_json(path: Path, value: Dict[str, Any]) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(path)


def _tar_fixture(work: Path) -> tuple[str, str]:
    archive = work / "hf-single-turn-snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(_FIXTURE_DIR, arcname=_FIXTURE_DIR.name)
    return str(archive), file_sha256(archive)


def _candidate_protocol(case: Dict[str, Any]) -> str:
    # D6 per-task deliverable paths are deterministically derived from the public item's source_task_id.
    slug = case["source_task_id"].replace(":", "-")
    file_name = (
        "solution.py"
        if case["hf_single_turn"]["benchmark"] == "humaneval"
        else "response.txt"
    )
    return (
        f"\n\nCandidate protocol: write your final answer to "
        f"deliverables/{slug}/{file_name} under your candidate root."
    )


def _write_deliverables(root: Path, files: Dict[str, str]) -> str:
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return str(root)


def _freeze_family(
    family: str,
    adapter: HfSingleTurnSeedAdapter,
    seed_rows: List[Dict[str, Any]],
    archive_path: str,
    archive_sha: str,
    prompt_archive: Dict[str, str],
    work: Path,
) -> Dict[str, Any]:
    """Single family: source manifest → re-expression → taskset freeze; returns the frozen state and deliverables."""
    family_rows = [row for row in seed_rows if row["family"] == family]
    if not family_rows:
        raise RuntimeError(f"fixture has no {family} seed rows for family ")
    release = derive_release("synthetic-fixture", family_rows)
    source = import_hf_single_turn_source(
        family_rows,
        family=family,
        release=release,
        split="development",
        license_record=dict(_LICENSE),
        seed_source=adapter.seed_source,
        dataset_id=adapter.dataset_id,
        archive_path=archive_path,
        archive_sha256=archive_sha,
    )
    verify_hf_single_turn_source_manifest(source)
    source_path = _write_json(work / f"source-{family}.json", source)

    rows: List[Dict[str, Any]] = []
    deliverables = {"gold_deliverable": {}, "negative_deliverable": {}}
    for case in source["cases"]:
        if case["hf_single_turn"]["admission"] != "expressible":
            continue
        bundle = derive_hf_single_turn_spec(source, case, dict(_DIRECTIVE))
        rows.append(
            {
                "source_task_id": case["source_task_id"],
                "task_text": case["task_text"] + _candidate_protocol(case),
                "tools_available": case["tools_available"],
                "success_criteria": case["success_criteria"],
                "difficulty": case["difficulty"],
                "item_fingerprint": case["source_task_id"].replace(":", "-") + "-a",
                "verifier_spec": bundle["verifier_spec"],
                "gold_deliverable": bundle["gold_deliverable"],
                "negative_deliverable": bundle["negative_deliverable"],
            }
        )
        deliverables["gold_deliverable"].update(bundle["gold_deliverable"])
        deliverables["negative_deliverable"].update(bundle["negative_deliverable"])
    if not rows:
        raise RuntimeError(f"{family} re-expression result is empty: the seed manifest has no expressible tasks")
    screen_hf_single_turn_contamination(rows, family_rows)

    audit_roots = {
        "reviewed": True,
        "reference_root": _write_deliverables(
            work / "audit-reference", deliverables["gold_deliverable"]
        ),
        "negative_root": _write_deliverables(
            work / "audit-negative", deliverables["negative_deliverable"]
        ),
    }
    config = build_hf_single_turn_augment_config(
        rows,
        family=family,
        release=release,
        split="development",
        strategy=_STRATEGY[family],
        prompt_archive=dict(prompt_archive),
        verifier_path=str(_RUNNER_PATHS[family]),
        audit_roots=audit_roots,
    )
    config_path = _write_json(work / f"augment-{family}.json", config)
    snapshot = build_taskset(
        ArtifactCatalog(str(work / "catalog")),
        work,
        source_path,
        config_path,
        f"single-turn-{family}",
        family=family,
        release=release,
        split="development",
    )
    return {
        "family": family,
        "release": release,
        "derived_count": len(rows),
        "taskset": snapshot["taskset"],
        "taskset_sha256": snapshot["taskset_sha256"],
        "item_count": snapshot["item_count"],
        "deliverables": deliverables,
        "source_path": source_path,
    }


def _judge_redline_build(
    family: str,
    adapter: HfSingleTurnSeedAdapter,
    seed_rows: List[Dict[str, Any]],
    archive_path: str,
    archive_sha: str,
    prompt_archive: Dict[str, str],
    work: Path,
) -> Dict[str, Any]:
    """D7 red line: when judge is not configured, building a family containing llm_rubric benchmarks must fail explicitly.

    The native audit within the build actually runs the runner shim (gold/negative both directions), and llm_rubric scoring goes through
    the judge endpoint -- an unconfigured judge raises ``LlmJudgeConfigError`` and produces no fabricated score. The red-line
    check requires the failure message to be specifically about judge being unconfigured (not any other build error).
    """
    try:
        _freeze_family(
            family, adapter, seed_rows, archive_path, archive_sha,
            prompt_archive, work,
        )
    except BenchmarkError as error:
        detail = str(error)
        if "LlmJudgeConfigError" not in detail:
            raise RuntimeError(
                f"judge red-line failure shape is unexpected: {family} build failed for a reason other than judge being unconfigured"
                f" (failure tail: {detail[-400:]!r})"
            ) from None
        return {"enforced": True, "family": family, "category": "judge_unconfigured"}
    raise RuntimeError(f"judge red-line check failed: {family} build was not rejected when judge was unconfigured")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="HF single-turn ingestion e2e: full frozen-pipeline run at fixture scale"
    )
    ap.add_argument("--judge-base-url", default=None,
                    help="LLM judge OpenAI-compatible endpoint (all three values must be provided together)")
    ap.add_argument("--judge-api-key", default=None,
                    help="LLM judge API key (all three values must be provided together)")
    ap.add_argument("--judge-model", default=None,
                    help="LLM judge model name (all three values must be provided together)")
    ap.add_argument("--work-dir", default=None,
                    help="Working directory (source manifest, trusted data, catalog, evidence; must be outside the repository)")
    args = ap.parse_args(argv)
    judge_args = (args.judge_base_url, args.judge_api_key, args.judge_model)
    if any(judge_args) and not all(judge_args):
        raise ValueError(
            "--judge-base-url, --judge-api-key, and --judge-model must be provided together"
            "(D7: judge configuration is all-or-nothing and never falls back silently)"
        )
    judge_configured = all(judge_args)
    if judge_configured:
        os.environ["DATA_FORGE_JUDGE_BASE_URL"] = args.judge_base_url
        os.environ["DATA_FORGE_JUDGE_API_KEY"] = args.judge_api_key
        os.environ["DATA_FORGE_JUDGE_MODEL"] = args.judge_model
        os.environ.setdefault("DATA_FORGE_JUDGE_TEMPERATURE", "0.0")
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="st-native-e2e-"))
    work.mkdir(parents=True, exist_ok=True)

    # ── 1. Seed ingestion ──────────────────────────────────────────────
    adapter = HfSingleTurnSeedAdapter(_FIXTURE_DIR)
    seed_rows = adapter.load_seed_tasks()
    archive_path, archive_sha = _tar_fixture(work)

    # ── 2/3. Per-family source manifest + re-expression + taskset freeze (contamination screening in chain) ──
    # Export the full D5 prompt archive once (gold is not archived) as the environment pin for each family.
    prompt_archive_path = work / "prompts.tar.gz"
    prompt_archive = {
        "path": str(prompt_archive_path),
        "sha256": adapter.export_prompt_archive(prompt_archive_path),
    }
    # Deterministic families (pure unit_test) are always frozen; families containing llm_rubric benchmarks
    # (knowledge_qa/simpleqa, health_qa/healthbench) require a real judge during the build audit,
    # so they are frozen only when judge is configured -- otherwise the red line runs (explicit failure, no scores).
    deterministic_families = ("reasoning_qa", "code_gen")
    judge_families = ("knowledge_qa", "health_qa")
    freezes = [
        _freeze_family(
            family, adapter, seed_rows, archive_path, archive_sha,
            prompt_archive, work,
        )
        for family in deterministic_families
    ]
    by_family = {freeze["family"]: freeze for freeze in freezes}

    # ── judge red line (D7) and judge family freeze ─────────────────────
    run_families = list(deterministic_families)
    if judge_configured:
        for family in judge_families:
            by_family[family] = _freeze_family(
                family, adapter, seed_rows, archive_path, archive_sha,
                prompt_archive, work,
            )
            run_families.append(family)
        judge_redline = {
            "enforced": False, "reason": "judge configured; red line not applicable",
        }
    else:
        judge_redline = {
            "families": [
                _judge_redline_build(
                    family, adapter, seed_rows, archive_path, archive_sha,
                    prompt_archive, work,
                )
                for family in judge_families
            ]
        }

    # ── 4. Per-family suite freeze + owned_native private approval ──────
    # readiness requires a single-family suite (tasksets contain exactly one taskset for that family),
    # and evaluation results must be co-sourced with the corresponding suite (matching tasksets), so suites/evaluations are aligned per family.
    evidence_dir = work / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    suites: Dict[str, Dict[str, Any]] = {}
    store = ArtifactCatalog(str(work / "catalog"))
    try:
        for family in run_families:
            suite_ref_id = f"single-turn-{family}-suite"
            suite = create_suite(
                store, suite_ref_id, [by_family[family]["taskset"]], repeats=1,
            )
            evidence_pins: Dict[str, Dict[str, str]] = {}
            for label in sorted(OFFICIAL_EVIDENCE_FIELDS):
                path = evidence_dir / f"{family}-{label}.txt"
                path.write_text(
                    f"single-turn native e2e evidence: {label}\n"
                    f"family={family}\n"
                    f"source_sha256="
                    f"{file_sha256(by_family[family]['source_path'])}\n"
                    f"suite_sha256={suite['suite_sha256']}\n",
                    encoding="utf-8",
                )
                evidence_pins[label] = {
                    "path": str(path), "sha256": file_sha256(path),
                }
            review_path = work / f"suite-review-{family}.json"
            review = {
                "schema_version": "benchmark-suite-review-v1",
                "scope": "owned_native",
                "suite_sha256": suite["suite_sha256"],
                "tasksets": suite["tasksets"],
                "reviewer": "e2e_single_turn_native",
                "evidence_refs": [
                    f"source:{by_family[family]['source_path']}",
                    f"config:{work / f'augment-{family}.json'}",
                    "audit-roots:"
                    f"{work / 'audit-reference'}|{work / 'audit-negative'}",
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
            suites[family] = suite
    finally:
        store.close()

    # ── 5. Dual-arm real evaluation (D6: per-task deliverables written to declared paths under candidateRoot) ─
    # Multiple families share the same candidateRoot (per-task deliverable paths do not conflict).
    reference_files: Dict[str, str] = {}
    baseline_files: Dict[str, str] = {}
    for family in run_families:
        reference_files.update(by_family[family]["deliverables"]["gold_deliverable"])
        baseline_files.update(
            by_family[family]["deliverables"]["negative_deliverable"]
        )
    _write_deliverables(work / "candidate-reference", reference_files)
    _write_deliverables(work / "candidate-baseline", baseline_files)
    evals: Dict[str, Dict[str, Any]] = {}
    store = ArtifactCatalog(str(work / "catalog"))
    try:
        for family in run_families:
            reference_eval = eval_arm(store, suites[family],
                                      work / "candidate-reference")
            baseline_eval = eval_arm(store, suites[family],
                                     work / "candidate-baseline")
            assert_comparable_evaluations(reference_eval, baseline_eval)
            if reference_eval["metrics"][f"pass_rate_{family}"] != 1.0:
                raise RuntimeError(
                    f"{family} reference arm did not reach 1.0: gold deliverable and verifier_spec are inconsistent"
                )
            if baseline_eval["metrics"][f"pass_rate_{family}"] != 0.0:
                raise RuntimeError(
                    f"{family} negative baseline arm did not reach zero: negative deliverables must not pass scoring"
                )
            evals[family] = {"reference": reference_eval, "baseline": baseline_eval}
    finally:
        store.close()

    # ── 6. Private readiness: load_suite re-checks suite pins and review evidence, i.e. the bookkeeping gate ──
    readiness = verify_private_readiness(
        store,
        [
            (suites[family]["ref_id"], suites[family]["version"],
             suites[family]["suite_sha256"])
            for family in run_families
        ],
    )
    judge_fingerprint = None
    if judge_configured:
        from data_designer.integrations.curator.judge import (
            judge_from_environment,
        )
        judge_fingerprint = judge_from_environment().fingerprint()

    print(json.dumps({
        "seed_stage": {
            "seed_dir": str(_FIXTURE_DIR),
            "seed_source": adapter.seed_source,
            "seed_row_count": len(seed_rows),
            "row_counts": adapter.row_counts,
        },
        "source_stage": {
            "dataset_id": adapter.dataset_id,
            "releases": {
                family: by_family[family]["release"] for family in by_family
            },
            "source_sha256": {
                family: file_sha256(by_family[family]["source_path"])
                for family in by_family
            },
        },
        "derive_stage": {
            "strategies": dict(_STRATEGY),
            "derived_count": {
                family: by_family[family]["derived_count"] for family in by_family
            },
            "contamination_screen": "pass",
        },
        "judge_stage": {
            "configured": judge_configured,
            "run_families": run_families,
            "redline": judge_redline,
        },
        "taskset_stage": {
            "tasksets": {
                family: {
                    "taskset": by_family[family]["taskset"],
                    "item_count": by_family[family]["item_count"],
                    "taskset_sha256": by_family[family]["taskset_sha256"],
                }
                for family in by_family
            },
            "suites": {
                family: {
                    "ref_id": suites[family]["ref_id"],
                    "version": suites[family]["version"],
                    "sha256": suites[family]["suite_sha256"],
                }
                for family in run_families
            },
        },
        "eval_stage": {
            "reference_pass_rate": {
                family: evals[family]["reference"]["metrics"][f"pass_rate_{family}"]
                for family in run_families
            },
            "baseline_pass_rate": {
                family: evals[family]["baseline"]["metrics"][f"pass_rate_{family}"]
                for family in run_families
            },
            "macro_pass_rate_reference": {
                family: evals[family]["reference"]["metrics"]["macro_pass_rate"]
                for family in run_families
            },
            "judge_fingerprints": {
                # unit_test families do not go through judge; llm_rubric family scoring goes through
                # the judge endpoint → the fingerprint is non-empty (environment config co-sourced with evaluation).
                family: judge_fingerprint if family in judge_families else None
                for family in run_families
            },
            "comparable": True,
        },
        "readiness_stage": {
            "scope": {family: readiness["scope"] for family in run_families},
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
