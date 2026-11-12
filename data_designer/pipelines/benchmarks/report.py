"""Library-level evaluation execution and cross-round comparison reports.

Collapses the e2e ``eval_arm`` pattern into reusable primitives: per-item
records, infra failures recorded instead of raised (design D8), and per-task
delta reports over ``assert_comparable_evaluations``-comparable records.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from data_designer.contracts.provenance import canonical_json

from .contracts import BenchmarkError, NativeFailure
from .native import run_case
from .suite import assert_comparable_evaluations

EVAL_RECORD_SCHEMA_VERSION = "benchmark-eval-record-v1"

_GRADED_STATUSES = ("passed", "failed")
# NativeFailure.category → record status (design §7.3). Timeout and an
# explicit runner "cancelled" are environment-side non-verdicts; every other
# category means the native infrastructure could not score the task.
_CANCELLED_CATEGORIES = ("timeout", "cancelled")


def _map_failure(failure: NativeFailure) -> Dict[str, Any]:
    status = (
        "cancelled"
        if failure.category in _CANCELLED_CATEGORIES
        else "infra_failed"
    )
    return {
        "status": status,
        "passed": None,
        "detail": f"{failure.category}: {failure}",
    }


def aggregate(record_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Metrics over eval items. Denominator = passed + failed only; infra
    failures and cancellations are counted separately (design D10)."""
    graded = [item for item in record_items if item["status"] in _GRADED_STATUSES]
    passed_total = sum(1 for item in graded if item["status"] == "passed")
    by_family: Dict[str, List[int]] = {}
    for item in graded:
        counts = by_family.setdefault(item["family"], [0, 0])
        counts[0] += 1 if item["status"] == "passed" else 0
        counts[1] += 1
    metrics: Dict[str, Any] = {
        "macro_pass_rate": passed_total / len(graded) if graded else 0.0,
        "graded_total": len(graded),
        "passed_total": passed_total,
        "infra_failed_total": sum(
            1 for item in record_items if item["status"] == "infra_failed"
        ),
        "cancelled_total": sum(
            1 for item in record_items if item["status"] == "cancelled"
        ),
    }
    metrics.update({
        f"pass_rate_{family}": counts[0] / counts[1] if counts[1] else 0.0
        for family, counts in sorted(by_family.items())
    })
    return metrics


def evaluate_suite(
    store,
    suite: Dict[str, Any],
    candidate_root: str | Path,
    *,
    timeout: float = 600.0,
    phase: str = "evaluation",
) -> Dict[str, Any]:
    """Score every suite task in frozen order via ``run_case`` and produce a
    ``benchmark-eval-record-v1``. Infra failures are recorded, never raised
    (design D8); the caller decides whether they are fatal."""
    items: List[Dict[str, Any]] = []
    tasks: List[str] = []
    for pin in suite["tasksets"]:
        payload = store.get_artifact("taskset", pin["ref_id"], pin["version"])[
            "payload"
        ]
        family = payload["family"]
        for item in payload["taskset"]:
            task_id = item["task_id"]
            runtime = payload["runtime_by_task"][task_id]
            manifest_path = Path(payload["trusted_manifest_dir"]) / (
                f"{item['benchmark']['case_manifest_sha256']}.json"
            )
            try:
                result = run_case(
                    runtime["verifier"]["command"], task_id, str(manifest_path),
                    str(candidate_root), phase=phase, timeout=timeout,
                    public_item=item,
                )
                entry = {
                    "task_id": task_id,
                    "family": family,
                    "status": "passed" if result["passed"] else "failed",
                    "passed": bool(result["passed"]),
                    "detail": None,
                }
            except NativeFailure as failure:
                entry = {"task_id": task_id, "family": family,
                         **_map_failure(failure)}
            tasks.append(task_id)
            items.append(entry)
    return {
        "schema_version": EVAL_RECORD_SCHEMA_VERSION,
        "suite_sha256": suite["suite_sha256"],
        "suite_ref": {"ref_id": suite["ref_id"], "version": suite["version"]},
        "tasksets": suite["tasksets"],
        "split": suite["split"],
        "tasks": tasks,
        "items": items,
        "metrics": aggregate(items),
    }


@dataclass
class ComparisonReport:
    """Cross-round report: macro deltas, per-family rates, per-task changes.
    Only tasks graded (passed/failed) on both sides enter the delta
    denominators; infra sides are listed separately."""

    suite_sha256: str
    split: str
    baseline_ref: Dict[str, Any]
    candidate_ref: Dict[str, Any]
    macro_baseline: float
    macro_candidate: float
    macro_delta: float
    per_family: Dict[str, Dict[str, float]]
    changed_tasks: List[Dict[str, Any]]
    infra_summary: Dict[str, List[str]]
    graded_totals: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_sha256": self.suite_sha256,
            "split": self.split,
            "baseline_ref": self.baseline_ref,
            "candidate_ref": self.candidate_ref,
            "macro_baseline": self.macro_baseline,
            "macro_candidate": self.macro_candidate,
            "macro_delta": self.macro_delta,
            "per_family": self.per_family,
            "changed_tasks": self.changed_tasks,
            "infra_summary": self.infra_summary,
            "graded_totals": self.graded_totals,
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(canonical_json(self.to_dict()), encoding="utf-8")


def _graded_items(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(record.get("items"), list):
        raise BenchmarkError(
            "evaluation record is missing per-item records (benchmark-eval-record-v1)"
        )
    return {
        item["task_id"]: item
        for item in record["items"]
        if item["status"] in _GRADED_STATUSES
    }


def _family_rates(record: Dict[str, Any]) -> Dict[str, float]:
    graded = [item for item in record["items"] if item["status"] in _GRADED_STATUSES]
    by_family: Dict[str, List[int]] = {}
    for item in graded:
        counts = by_family.setdefault(item["family"], [0, 0])
        counts[0] += 1 if item["status"] == "passed" else 0
        counts[1] += 1
    return {
        family: counts[0] / counts[1] if counts[1] else 0.0
        for family, counts in by_family.items()
    }


def compare_evaluations(
    baseline: Dict[str, Any], candidate: Dict[str, Any]
) -> ComparisonReport:
    assert_comparable_evaluations(baseline, candidate)
    baseline_graded = _graded_items(baseline)
    candidate_graded = _graded_items(candidate)

    changed_tasks = []
    for task_id in sorted(set(baseline_graded) & set(candidate_graded)):
        left, right = baseline_graded[task_id], candidate_graded[task_id]
        if left["status"] != right["status"]:
            changed_tasks.append({
                "task_id": task_id,
                "family": left["family"],
                "baseline": left["status"],
                "candidate": right["status"],
            })

    families = sorted(
        {item["family"] for item in baseline["items"]}
        | {item["family"] for item in candidate["items"]}
    )
    baseline_rates = _family_rates(baseline)
    candidate_rates = _family_rates(candidate)
    per_family = {
        family: {
            "baseline": baseline_rates.get(family, 0.0),
            "candidate": candidate_rates.get(family, 0.0),
            "delta": candidate_rates.get(family, 0.0) - baseline_rates.get(family, 0.0),
        }
        for family in families
    }

    infra_summary = {
        "baseline": [
            item["task_id"] for item in baseline["items"]
            if item["status"] not in _GRADED_STATUSES
        ],
        "candidate": [
            item["task_id"] for item in candidate["items"]
            if item["status"] not in _GRADED_STATUSES
        ],
    }
    macro_baseline = baseline["metrics"]["macro_pass_rate"]
    macro_candidate = candidate["metrics"]["macro_pass_rate"]
    return ComparisonReport(
        suite_sha256=baseline["suite_sha256"],
        split=baseline["split"],
        baseline_ref=baseline["suite_ref"],
        candidate_ref=candidate["suite_ref"],
        macro_baseline=macro_baseline,
        macro_candidate=macro_candidate,
        macro_delta=macro_candidate - macro_baseline,
        per_family=per_family,
        changed_tasks=changed_tasks,
        infra_summary=infra_summary,
        graded_totals={
            "baseline": len(baseline_graded),
            "candidate": len(candidate_graded),
        },
    )
