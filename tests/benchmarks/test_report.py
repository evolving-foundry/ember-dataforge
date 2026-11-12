"""C5 probes: library-level evaluation + cross-round comparison report (design document:
evaluation-extension capability completion, section 7.8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.build import build_benchmark_taskset
from data_designer.pipelines.benchmarks.contracts import (
    BenchmarkError,
    NativeFailure,
    file_sha256,
)
from data_designer.pipelines.benchmarks.report import (
    EVAL_RECORD_SCHEMA_VERSION,
    aggregate,
    compare_evaluations,
    evaluate_suite,
)
from data_designer.pipelines.benchmarks.suite import (
    assert_comparable_evaluations,
    create_suite,
)
from data_designer.storage.catalog import ArtifactCatalog


def build_single_taskset(tmp_path):
    """One-variant workbuddy source/augment pair; returns (source, augment) paths."""
    from data_designer.pipelines.benchmarks.runners import workbuddy
    from data_designer.pipelines.benchmarks.contracts import file_sha256

    runner = Path(workbuddy.__file__).resolve()
    environment = tmp_path / "environment.json"
    environment.write_text("{}")
    reference, negative = tmp_path / "reference", tmp_path / "negative"
    for root, value in ((reference, {"total": 42}), (negative, {"total": 0})):
        (root / "output").mkdir(parents=True, exist_ok=True)
        (root / "output/result.json").write_text(json.dumps(value))
    runtime = {"environment": {"path": str(environment), "sha256": file_sha256(environment)},
               "verifier": {"path": str(runner), "sha256": file_sha256(runner),
                            "command": [sys.executable, str(runner)], "version": workbuddy.VERSION}, "assets": []}
    source = {"schema_version": "benchmark-source-v1", "family": "workbuddy", "release": "fixture",
              "split": "development", "license": {"id": "owned-fixture", "derive_allowed": True},
              "cases": [{"source_task_id": "seed-a", "task_text": "Original source accounting exercise A.",
                         "initial_state_sha256": "a" * 64, "expected_state_sha256": "b" * 64}], **runtime}
    variant = {"source_task_id": "seed-a", "variant_id": "v1", "operation": "mutation",
               "public": {"task_text": "Compute the total from the supplied fictional accounting entries.",
                          "success_criteria": "Write output/result.json with the computed total.",
                          "tools_available": ["files"], "difficulty": "easy"},
               "private": {"initial_state_sha256": "c" * 64, "expected_state_sha256": "d" * 64,
                           "verifier_spec": {"kind": "artifact_compare", "generator_ref": "owned-fixture@1",
                                             "payload": {"artifact": {"path": "output/result.json"}, "gold": {"total": 42}}}},
               "audit": {"reviewed": True, "reference_root": str(reference), "negative_root": str(negative)}}
    config = {"schema_version": "benchmark-augment-v1", "family": "workbuddy", "release": "fixture",
              "split": "development", "variants": [variant]}
    source_path, config_path = tmp_path / "source.json", tmp_path / "augment.json"
    source_path.write_text(json.dumps(source))
    config_path.write_text(json.dumps(config))
    return source_path, config_path


def build_workbuddy_suite(tmp_path, monkeypatch, ref_id="report-fixture"):
    """Two-variant workbuddy taskset + suite (test_package pattern, D6 two-arm judging)."""
    import data_designer.pipelines.benchmarks.build as builder
    monkeypatch.setattr(builder, "assess_taskset",
                        lambda items: {"quality_overall_score": 1.0})
    from data_designer.pipelines.benchmarks.runners import workbuddy

    runner = Path(workbuddy.__file__).resolve()
    environment = tmp_path / "environment.json"
    environment.write_text("{}")
    reference, negative = tmp_path / "reference", tmp_path / "negative"
    for root, value in ((reference, {"total": 42}), (negative, {"total": 0})):
        (root / "output").mkdir(parents=True, exist_ok=True)
        (root / "output/result.json").write_text(json.dumps(value))
    runtime = {"environment": {"path": str(environment), "sha256": file_sha256(environment)},
               "verifier": {"path": str(runner), "sha256": file_sha256(runner),
                            "command": [sys.executable, str(runner)], "version": workbuddy.VERSION}, "assets": []}
    source = {"schema_version": "benchmark-source-v1", "family": "workbuddy", "release": "fixture",
              "split": "development", "license": {"id": "owned-fixture", "derive_allowed": True},
              "cases": [
                  {"source_task_id": "seed-a", "task_text": "Original source accounting exercise A.",
                   "initial_state_sha256": "a" * 64, "expected_state_sha256": "b" * 64},
                  {"source_task_id": "seed-b", "task_text": "Original source accounting exercise B.",
                   "initial_state_sha256": "a" * 64, "expected_state_sha256": "b" * 64},
              ], **runtime}
    variants = [
        {"source_task_id": source_id, "variant_id": "v1", "operation": "mutation",
         "public": {"task_text": f"Compute the total from the supplied fictional {tag} entries.",
                    "success_criteria": "Write output/result.json with the computed total.",
                    "tools_available": ["files"], "difficulty": "easy"},
         "private": {"initial_state_sha256": "c" * 64, "expected_state_sha256": "d" * 64,
                     "verifier_spec": {"kind": "artifact_compare", "generator_ref": "owned-fixture@1",
                                       "payload": {"artifact": {"path": "output/result.json"}, "gold": {"total": 42}}}},
         "audit": {"reviewed": True, "reference_root": str(reference), "negative_root": str(negative)}}
        for source_id, tag in (("seed-a", "accounting"), ("seed-b", "ledger"))
    ]
    config = {"schema_version": "benchmark-augment-v1", "family": "workbuddy", "release": "fixture",
              "split": "development", "variants": variants}
    source_path, config_path = tmp_path / "source.json", tmp_path / "augment.json"
    source_path.write_text(json.dumps(source))
    config_path.write_text(json.dumps(config))
    store = ArtifactCatalog(str(tmp_path / "catalog"))
    snapshot = build_benchmark_taskset(store, str(source_path), str(config_path),
                                       str(tmp_path / "trusted"), ref_id)
    suite = create_suite(store, f"{ref_id}-suite", [snapshot["taskset"]])
    return store, suite


def _candidate(tmp_path, name, total):
    root = tmp_path / name
    (root / "output").mkdir(parents=True, exist_ok=True)
    (root / "output/result.json").write_text(json.dumps({"total": total}))
    return root


def test_reference_and_negative_arms_score_one_and_zero(tmp_path, monkeypatch):
    store, suite = build_workbuddy_suite(tmp_path, monkeypatch)

    reference = evaluate_suite(store, suite, _candidate(tmp_path, "ref", 42))
    baseline = evaluate_suite(store, suite, _candidate(tmp_path, "base", 0))
    assert reference["metrics"]["macro_pass_rate"] == 1.0
    assert baseline["metrics"]["macro_pass_rate"] == 0.0
    assert reference["schema_version"] == EVAL_RECORD_SCHEMA_VERSION
    assert reference["tasks"] == [item["task_id"] for item in reference["items"]]
    assert all(item["status"] == "passed" for item in reference["items"])
    assert all(item["status"] == "failed" for item in baseline["items"])


def test_infra_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    store, suite = build_workbuddy_suite(tmp_path, monkeypatch)
    import data_designer.pipelines.benchmarks.report as report

    failing_task = suite["tasksets"][0]["ref_id"]
    original = report.run_case

    def flaky(command, task_id, manifest_path, candidate_root, **kwargs):
        if "seed-b" in task_id:
            raise NativeFailure("runner_failed", "injected infra failure")
        return original(command, task_id, manifest_path, candidate_root, **kwargs)

    monkeypatch.setattr(report, "run_case", flaky)
    record = evaluate_suite(store, suite, _candidate(tmp_path, "ref", 42))
    assert record["metrics"]["infra_failed_total"] == 1
    assert record["metrics"]["cancelled_total"] == 0
    assert record["metrics"]["graded_total"] == 1
    assert record["metrics"]["macro_pass_rate"] == 1.0
    infra_item = next(item for item in record["items"]
                      if item["status"] == "infra_failed")
    assert infra_item["passed"] is None
    assert "runner_failed" in infra_item["detail"]


def test_timeout_maps_to_cancelled():
    metrics = aggregate([
        {"task_id": "t1", "family": "f", "status": "cancelled", "passed": None,
         "detail": "timeout: slow"},
        {"task_id": "t2", "family": "f", "status": "passed", "passed": True,
         "detail": None},
    ])
    assert metrics["cancelled_total"] == 1
    assert metrics["graded_total"] == 1
    assert metrics["macro_pass_rate"] == 1.0


def _record(tasks, statuses, sha="0" * 64):
    items = [
        {"task_id": task_id, "family": "f", "status": status,
         "passed": status == "passed" if status in ("passed", "failed") else None,
         "detail": None}
        for task_id, status in zip(tasks, statuses)
    ]
    return {
        "schema_version": EVAL_RECORD_SCHEMA_VERSION,
        "suite_sha256": sha,
        "suite_ref": {"ref_id": "suite", "version": 1},
        "tasksets": [{"family": "f", "ref_id": "t", "version": 1,
                      "fingerprint": "x"}],
        "split": "development",
        "tasks": list(tasks),
        "items": items,
        "metrics": aggregate(items),
    }


def test_compare_reports_per_task_deltas():
    baseline = _record(["t1", "t2", "t3", "t4"], ["failed"] * 4)
    candidate = _record(["t1", "t2", "t3", "t4"],
                        ["passed", "passed", "passed", "failed"])
    report = compare_evaluations(baseline, candidate)
    assert [task["task_id"] for task in report.changed_tasks] == ["t1", "t2", "t3"]
    assert report.macro_baseline == 0.0
    assert report.macro_candidate == 0.75
    assert report.macro_delta == pytest.approx(0.75)
    assert report.per_family["f"] == {
        "baseline": 0.0, "candidate": 0.75,
        "delta": pytest.approx(0.75),
    }
    assert report.graded_totals == {"baseline": 4, "candidate": 4}
    assert report.infra_summary == {"baseline": [], "candidate": []}


def test_compare_requires_comparable_records():
    baseline = _record(["t1", "t2"], ["passed", "failed"])
    with pytest.raises(BenchmarkError, match="different suites or tasks"):
        compare_evaluations(baseline, _record(["t1", "t2"], ["passed", "failed"],
                                              sha="1" * 64))
    with pytest.raises(BenchmarkError, match="different suites or tasks"):
        compare_evaluations(baseline, _record(["t2", "t1"], ["failed", "passed"]))
    with pytest.raises(BenchmarkError, match="missing per-item records"):
        legacy = {**baseline, "items": None}
        compare_evaluations(legacy, legacy)


def test_new_record_passes_assert_comparable_evaluations():
    record = _record(["t1", "t2"], ["passed", "failed"])
    assert_comparable_evaluations(record, record)


def test_write_json_roundtrip_discipline(tmp_path):
    report = compare_evaluations(
        _record(["t1", "t2"], ["failed", "failed"]),
        _record(["t1", "t2"], ["passed", "failed"]),
    )
    path = tmp_path / "report.json"
    report.write_json(path)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    restored = json.loads(text)
    assert restored == report.to_dict()


def test_e2e_eval_arm_asserts_zero_infra(tmp_path, monkeypatch):
    store, suite = build_workbuddy_suite(tmp_path, monkeypatch)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    try:
        from _benchmark_e2e import eval_arm
    finally:
        sys.path.pop(0)

    record = eval_arm(store, suite, _candidate(tmp_path, "ref", 42))
    assert record["metrics"]["macro_pass_rate"] == 1.0

    import data_designer.pipelines.benchmarks.report as report
    original = report.run_case

    def flaky(command, task_id, manifest_path, candidate_root, **kwargs):
        raise NativeFailure("timeout", "injected timeout")

    monkeypatch.setattr(report, "run_case", flaky)
    with pytest.raises(RuntimeError, match="scoring incomplete"):
        eval_arm(store, suite, _candidate(tmp_path, "ref", 42))
