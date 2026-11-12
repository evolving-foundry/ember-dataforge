"""terminal native runner tests (M3): file-protocol scoring, reference/negative/empty deliverable,
out-of-bounds rejection, offline execution of unit_test assertion scripts, and kind allowlisting.

Covers terminal onboarding implementation plan §2 M3 acceptance: fixture-derived tasks on the gold artifact
return passed=True/score=1.0; on empty/wrong artifacts return passed=False/score=0.0; all results pass
``validate_native_result`` (enforced inside ``run_case``); fully offline, no docker prerequisite.
"""

import json
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import NativeFailure
from data_designer.pipelines.benchmarks.native import run_case
from data_designer.pipelines.benchmarks.runners.terminal import (
    SUPPORTED_SPEC_KINDS,
    VERSION,
)

RUNNER_PATH = Path(__file__).resolve().parents[2] / (
    "data_designer/pipelines/benchmarks/runners/terminal.py"
)

TASK_ID = "terminal:tb21-fixture-1:v1"

UNIT_TEST_SPEC = {
    "kind": "unit_test",
    "generator_ref": "tb21-reexpression@fixture",
    "payload": {
        "artifact": {"path": "deliverable/files.json", "format": "json"},
        "setup_files": {
            "input/values.txt": "3\n4\n",
        },
        "test_script": (
            "import pathlib\n"
            "fixed = pathlib.Path('scripts/sum.py').read_text(encoding='utf-8')\n"
            "assert '+' in fixed, 'the fix must add the missing operator'\n"
            "assert 'return a  b' not in fixed, 'broken line must be fixed'\n"
        ),
        "timeout_s": 10,
    },
}

ARTIFACT_SPEC = {
    "kind": "artifact_compare",
    "generator_ref": "tb21-reexpression@fixture",
    "payload": {
        "artifact": {"path": "deliverable/summary.json", "format": "json"},
        "gold": {"total": 1500.0, "mode": "batch"},
        "fields": {
            "total": {"type": "number", "tolerance": 0.01},
            "mode": {"type": "string"},
        },
    },
}

GOLD_UNIT_TEST_FILES = {"scripts/sum.py": "def total(a, b):\n    return a + b\n"}
BROKEN_UNIT_TEST_FILES = {"scripts/sum.py": "def total(a, b):\n    return a  b\n"}


def _write_manifest(tmp_path: Path, spec: dict) -> str:
    manifest = {
        "schema_version": "benchmark-case-v1",
        "task_id": TASK_ID,
        "family": "terminal",
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


def _write_deliverable(root: Path, value, relative):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        target.write_text(value, encoding="utf-8")
    else:
        target.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_reference_phase_passes_on_gold_deliverable_unit_test(tmp_path):
    manifest_path = _write_manifest(tmp_path, UNIT_TEST_SPEC)
    root = tmp_path / "reference"
    _write_deliverable(root, GOLD_UNIT_TEST_FILES, "deliverable/files.json")
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="reference", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["verifier_version"] == VERSION


def test_negative_phase_fails_on_broken_deliverable_unit_test(tmp_path):
    manifest_path = _write_manifest(tmp_path, UNIT_TEST_SPEC)
    root = tmp_path / "negative"
    _write_deliverable(root, BROKEN_UNIT_TEST_FILES, "deliverable/files.json")
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="negative", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_reference_and_negative_pass_with_artifact_compare(tmp_path):
    manifest_path = _write_manifest(tmp_path, ARTIFACT_SPEC)
    reference = tmp_path / "reference"
    negative = tmp_path / "negative"
    _write_deliverable(reference, {"total": 1500.0, "mode": "batch"}, "deliverable/summary.json")
    _write_deliverable(negative, {"total": 1.0, "mode": "live"}, "deliverable/summary.json")
    positive = run_case(
        _runner_command(), TASK_ID, manifest_path, str(reference),
        phase="reference", timeout=30, public_item={},
    )
    failed = run_case(
        _runner_command(), TASK_ID, manifest_path, str(negative),
        phase="negative", timeout=30, public_item={},
    )
    assert (positive["passed"], positive["score"]) == (True, 1.0)
    assert (failed["passed"], failed["score"]) == (False, 0.0)


def test_evaluation_acknowledges_attempt_seed(tmp_path):
    manifest_path = _write_manifest(tmp_path, UNIT_TEST_SPEC)
    root = tmp_path / "candidate"
    _write_deliverable(root, GOLD_UNIT_TEST_FILES, "deliverable/files.json")
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
        attempt={"index": 2, "seed": 12345},
    )
    assert result["attempt_index"] == 2
    assert result["seed"] == 12345
    assert result["passed"] is True


def test_missing_deliverable_is_invalid_candidate_not_silent(tmp_path):
    manifest_path = _write_manifest(tmp_path, UNIT_TEST_SPEC)
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
    manifest_path = _write_manifest(tmp_path, UNIT_TEST_SPEC)
    root = tmp_path / "reference"
    root.mkdir()
    with pytest.raises(NativeFailure, match="native runner failed"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="reference", timeout=30, public_item={},
        )


def test_out_of_root_deliverable_path_is_rejected(tmp_path):
    escaping_spec = json.loads(json.dumps(UNIT_TEST_SPEC))
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
    manifest_path = _write_manifest(tmp_path, UNIT_TEST_SPEC)
    root = tmp_path / "candidate"
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(GOLD_UNIT_TEST_FILES), encoding="utf-8")
    target = root / "deliverable" / "files.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "invalid_candidate"
    assert result["passed"] is False


def test_unparseable_deliverable_is_invalid_candidate(tmp_path):
    manifest_path = _write_manifest(tmp_path, UNIT_TEST_SPEC)
    root = tmp_path / "candidate"
    _write_deliverable(root, "{not json", "deliverable/files.json")
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "invalid_candidate"
    assert result["score"] == 0.0


def test_non_whitelisted_spec_kind_is_rejected(tmp_path):
    rogue = json.loads(json.dumps(UNIT_TEST_SPEC))
    rogue["kind"] = "llm_rubric"
    manifest_path = _write_manifest(tmp_path, rogue)
    root = tmp_path / "candidate"
    _write_deliverable(root, GOLD_UNIT_TEST_FILES, "deliverable/files.json")
    with pytest.raises(NativeFailure, match="native runner failed"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="evaluation", timeout=30, public_item={},
        )


def test_supported_kinds_match_plan_t6():
    assert set(SUPPORTED_SPEC_KINDS) == {"unit_test", "artifact_compare"}
