"""workbuddy native runner tests (M3): file-protocol scoring, reference/negative/empty deliverable, and out-of-bounds rejection.

Aligned with implementation plan §2 M3 acceptance: the runner returns passed=True/score=1.0 on the reference deliverable,
passed=False/score=0.0 on empty/wrong deliverables, and all results pass ``validate_native_result``
(enforced inside ``run_case``); out-of-bounds deliverable paths are rejected explicitly.
"""

import json
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.native import run_case
from data_designer.pipelines.benchmarks.runners.workbuddy import VERSION
from data_designer.pipelines.benchmarks.contracts import NativeFailure

RUNNER_PATH = Path(
    __file__).resolve().parents[2] / (
    "data_designer/pipelines/benchmarks/runners/workbuddy.py"
)

TASK_ID = "workbuddy:wb-fixture-1:v1"

SPEC = {
    "kind": "artifact_compare",
    "generator_ref": "workbuddy_mutation@fixture",
    "payload": {
        "artifact": {"path": "output/summary.json", "format": "json"},
        "gold": {"total": 1500.0, "top_region": "华东"},
        "fields": {
            "total": {"type": "number", "tolerance": 0.01},
            "top_region": {"type": "string"},
        },
    },
}


def _write_manifest(tmp_path: Path, spec: dict) -> str:
    manifest = {
        "schema_version": "benchmark-case-v1",
        "task_id": TASK_ID,
        "family": "workbuddy",
        "private": {"verifier_spec": spec},
    }
    path = tmp_path / "trusted" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(path)


def _runner_command():
    return [sys.executable, str(RUNNER_PATH)]


def _write_deliverable(root: Path, value, relative="output/summary.json"):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        target.write_text(value, encoding="utf-8")
    else:
        target.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_reference_phase_passes_on_correct_deliverable(tmp_path):
    manifest_path = _write_manifest(tmp_path, SPEC)
    root = tmp_path / "reference"
    _write_deliverable(root, {"total": 1500.0, "top_region": "华东"})
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="reference", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["verifier_version"] == VERSION


def test_negative_phase_fails_on_wrong_deliverable(tmp_path):
    manifest_path = _write_manifest(tmp_path, SPEC)
    root = tmp_path / "negative"
    _write_deliverable(root, {"total": 1.0, "top_region": "华南"})
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="negative", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_evaluation_acknowledges_attempt_seed(tmp_path):
    manifest_path = _write_manifest(tmp_path, SPEC)
    root = tmp_path / "candidate"
    _write_deliverable(root, {"total": 1500.0, "top_region": "华东"})
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
        attempt={"index": 2, "seed": 12345},
    )
    assert result["attempt_index"] == 2
    assert result["seed"] == 12345
    assert result["passed"] is True


def test_missing_deliverable_is_invalid_candidate_not_silent(tmp_path):
    manifest_path = _write_manifest(tmp_path, SPEC)
    root = tmp_path / "candidate"
    root.mkdir()
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "invalid_candidate"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_audit_phase_requires_deliverable(tmp_path):
    manifest_path = _write_manifest(tmp_path, SPEC)
    root = tmp_path / "reference"
    root.mkdir()
    with pytest.raises(NativeFailure, match="native runner failed"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="reference", timeout=30, public_item={},
        )


def test_out_of_root_deliverable_path_is_rejected(tmp_path):
    escaping_spec = json.loads(json.dumps(SPEC))
    escaping_spec["payload"]["artifact"]["path"] = "../escape.json"
    manifest_path = _write_manifest(tmp_path, escaping_spec)
    root = tmp_path / "candidate"
    root.mkdir()
    (tmp_path / "escape.json").write_text("{}", encoding="utf-8")
    with pytest.raises(NativeFailure, match="clean relative path"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="evaluation", timeout=30, public_item={},
        )


def test_symlink_escape_is_rejected(tmp_path):
    manifest_path = _write_manifest(tmp_path, SPEC)
    root = tmp_path / "candidate"
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"total": 1500.0, "top_region": "华东"}), encoding="utf-8")
    target = root / "output" / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "invalid_candidate"
    assert result["passed"] is False


def test_unparseable_deliverable_is_invalid_candidate(tmp_path):
    manifest_path = _write_manifest(tmp_path, SPEC)
    root = tmp_path / "candidate"
    _write_deliverable(root, "{not json")
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "invalid_candidate"
    assert result["score"] == 0.0
