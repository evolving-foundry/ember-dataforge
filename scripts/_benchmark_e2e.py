"""Frozen pipeline stage helpers shared by native e2e scenario scripts (build / eval / readiness).

Against the refactored structure: ``build_benchmark_taskset`` carries the build audit; dual-arm
real evaluation grades each task directly with the runtime locked in the taskset artifact; readiness
uses ``load_suite`` (approved review-evidence pins) as the private bookkeeping gate -- the old worker
shell protocol has been removed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def build_taskset(
    store, work: Path, source_path: str, config_path: str, ref_id: str, *,
    family: str, release: str, split: str = "development",
) -> Dict[str, Any]:
    from data_designer.pipelines.benchmarks.build import build_benchmark_taskset
    return build_benchmark_taskset(
        store, str(source_path), str(config_path), str(Path(work) / "trusted"),
        ref_id,
        expected_family=family, expected_release=release, expected_split=split,
    )


def eval_arm(store, suite: Dict[str, Any], candidate_root: Path, *,
             timeout: float = 120.0) -> Dict[str, Any]:
    """Dual-arm real evaluation: deliverables are already written under candidateRoot per D6; grade each task with the frozen runtime.

    Scoring execution is delegated to the library capability ``report.evaluate_suite`` (per-item
    records; infrastructure failures are recorded rather than raised). Script policy remains: any
    infra/cancelled count means failure (C5 section 7.6).
    """
    from data_designer.pipelines.benchmarks.report import evaluate_suite
    record = evaluate_suite(store, suite, candidate_root, timeout=timeout)
    infra = [
        item["task_id"] for item in record["items"]
        if item["status"] not in ("passed", "failed")
    ]
    if infra:
        raise RuntimeError(f"{', '.join(infra)} scoring incomplete")
    return record


def verify_private_readiness(store, suites: List[Tuple[str, int, str]]) -> Dict[str, Any]:
    """Readiness re-expression: load_suite(approved) re-checks suite pins and review evidence, i.e. the bookkeeping gate."""
    from data_designer.pipelines.benchmarks.suite import load_suite
    for ref_id, version, sha256 in suites:
        load_suite(store, ref_id, version, sha256)
    return {
        "scope": "owned_native",
        "selected_evidence_complete": True,
        "selected_distribution": "private_evaluation_only",
    }
