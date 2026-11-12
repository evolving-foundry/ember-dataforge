"""osworld offline runner tests (osworld follow-up M1): full branch coverage for offline state-artifact scoring.

Aligned with the osworld scoring re-expression follow-up doc (v0.1) M1 acceptance: candidates write the final state as
a ``{filename: content}`` file-dictionary JSON (or a structured JSON object for artifact_compare tasks)
to ``verifier_spec.payload.artifact.path``; the runner parses it and dispatches through ``run_verifier``.
Expected state artifacts return passed=True/1.0; wrong content/missing files return completed/False/0.0;
missing/unparseable/symlinked deliverables return invalid_candidate; a missing audit stage is an explicit
NativeFailure; out-of-bounds deliverable paths / unknown kinds / family or task mismatches fail explicitly.
All results pass ``validate_native_result`` (enforced inside ``run_case``). Fully offline, zero environment.
"""

import json
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import NativeFailure
from data_designer.pipelines.benchmarks.native import run_case
from data_designer.pipelines.benchmarks.runners.osworld_offline import VERSION

RUNNER_PATH = Path(__file__).resolve().parents[2] / (
    "data_designer/pipelines/benchmarks/runners/osworld_offline.py"
)

TASK_ID = "osworld:fixture-state-1:v1"
DELIVERABLE = "state_artifact.json"

EXPECTED_FILES = {
    "docs/report.csv": "a,b\n1,2\n",
    "out/summary.json": '{"total": 3}\n',
}
SETUP_FILES = {
    "docs/report.csv": "a,b\n0,0\n",
}
GOLD = {"total": 3, "top_region": "north"}


def _state_test_script() -> str:
    literal = json.dumps(json.dumps(EXPECTED_FILES, sort_keys=True), ensure_ascii=False)
    return (
        "import json\n"
        f"EXPECTED = json.loads({literal})\n"
        "missing, wrong = [], []\n"
        "for relative, content in sorted(EXPECTED.items()):\n"
        "    try:\n"
        "        with open(relative, encoding='utf-8') as handle:\n"
        "            actual = handle.read()\n"
        "    except OSError:\n"
        "        missing.append(relative)\n"
        "        continue\n"
        "    if actual != content:\n"
        "        wrong.append(relative)\n"
        "assert not missing and not wrong, (\n"
        "    'missing=%r wrong=%r' % (missing, wrong))\n"
    )


def _unit_test_spec() -> dict:
    return {
        "kind": "unit_test",
        "generator_ref": "osworld-offline-state@fixture",
        "payload": {
            "artifact": {"path": DELIVERABLE},
            "setup_files": dict(SETUP_FILES),
            "test_script": _state_test_script(),
            "timeout_s": 30.0,
        },
    }


def _artifact_compare_spec() -> dict:
    return {
        "kind": "artifact_compare",
        "generator_ref": "osworld-offline-state@fixture",
        "payload": {
            "artifact": {"path": DELIVERABLE},
            "gold": dict(GOLD),
            "fields": {
                "total": {"type": "integer"},
                "top_region": {"type": "string"},
            },
        },
    }


def _write_manifest(tmp_path: Path, spec: dict) -> str:
    manifest = {
        "schema_version": "benchmark-case-v1",
        "task_id": TASK_ID,
        "family": "osworld",
        "private": {"verifier_spec": spec},
    }
    manifest_path = tmp_path / "trusted" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(manifest_path)


def _runner_command():
    return [sys.executable, str(RUNNER_PATH)]


def _write_deliverable(root: Path, value) -> None:
    target = root / DELIVERABLE
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        target.write_text(value, encoding="utf-8")
    else:
        target.write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )


def _expected_state() -> dict:
    return dict(EXPECTED_FILES)


def _run(tmp_path: Path, spec: dict, phase: str = "evaluation", root: str = "candidate", **kwargs):
    manifest_path = _write_manifest(tmp_path, spec)
    candidate_root = tmp_path / root
    candidate_root.mkdir(parents=True, exist_ok=True)
    return run_case(
        _runner_command(), TASK_ID, manifest_path, str(candidate_root),
        phase=phase, timeout=30, public_item={}, **kwargs,
    )


def test_reference_phase_passes_on_expected_state(tmp_path):
    root = tmp_path / "reference"
    _write_deliverable(root, _expected_state())
    result = _run(tmp_path, _unit_test_spec(), phase="reference", root="reference")
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["verifier_version"] == VERSION
    assert result["native_metrics"]["artifact_bytes"] > 0


def test_negative_phase_fails_on_wrong_content(tmp_path):
    root = tmp_path / "negative"
    wrong = {**_expected_state(), "docs/report.csv": "a,b\n9,9\n"}
    _write_deliverable(root, wrong)
    result = _run(tmp_path, _unit_test_spec(), phase="negative", root="negative")
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_evaluation_acknowledges_attempt_seed(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, _expected_state())
    result = _run(
        tmp_path, _unit_test_spec(), phase="evaluation",
        attempt={"index": 2, "seed": 4242},
    )
    assert result["attempt_index"] == 2
    assert result["seed"] == 4242
    assert result["passed"] is True


def test_missing_file_in_state_scores_zero(tmp_path):
    root = tmp_path / "candidate"
    partial = {"docs/report.csv": "a,b\n1,2\n"}
    _write_deliverable(root, partial)
    result = _run(tmp_path, _unit_test_spec())
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_initial_state_alone_does_not_pass(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, dict(SETUP_FILES))
    result = _run(tmp_path, _unit_test_spec())
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_artifact_compare_passes_on_gold_object(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, dict(GOLD))
    result = _run(tmp_path, _artifact_compare_spec())
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_artifact_compare_partial_match_scores_fraction(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, {"total": 3, "top_region": "south"})
    result = _run(tmp_path, _artifact_compare_spec())
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert 0.0 < result["score"] < 1.0


def test_missing_deliverable_is_invalid_candidate_not_silent(tmp_path):
    (tmp_path / "candidate").mkdir()
    result = _run(tmp_path, _unit_test_spec())
    assert result["status"] == "invalid_candidate"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_unparseable_deliverable_is_invalid_candidate(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, "{not json")
    result = _run(tmp_path, _unit_test_spec())
    assert result["status"] == "invalid_candidate"
    assert result["score"] == 0.0


def test_symlink_deliverable_is_invalid_candidate(tmp_path):
    root = tmp_path / "candidate"
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(_expected_state(), ensure_ascii=False), encoding="utf-8"
    )
    target = root / DELIVERABLE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)
    result = _run(tmp_path, _unit_test_spec())
    assert result["status"] == "invalid_candidate"
    assert result["passed"] is False


def test_audit_phase_requires_deliverable(tmp_path):
    with pytest.raises(NativeFailure, match="native runner failed"):
        _run(tmp_path, _unit_test_spec(), phase="reference", root="reference")


def test_escape_deliverable_path_is_rejected(tmp_path):
    spec = _unit_test_spec()
    spec["payload"]["artifact"]["path"] = "../escape.json"
    (tmp_path / "candidate").mkdir()
    with pytest.raises(NativeFailure, match="clean relative path"):
        _run(tmp_path, spec)


def test_unsupported_verifier_kind_is_rejected(tmp_path):
    spec = _unit_test_spec()
    spec["kind"] = "swe_offline_patch"
    root = tmp_path / "candidate"
    _write_deliverable(root, _expected_state())
    with pytest.raises(NativeFailure, match="unsupported verifier kind"):
        _run(tmp_path, spec)


def test_family_mismatch_is_rejected(tmp_path):
    spec = _unit_test_spec()
    manifest_path = _write_manifest(tmp_path, spec)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest["family"] = "swe"
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    root = tmp_path / "candidate"
    _write_deliverable(root, _expected_state())
    with pytest.raises(NativeFailure, match="not an osworld case"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="evaluation", timeout=30, public_item={},
        )


def test_task_id_mismatch_is_rejected(tmp_path):
    spec = _unit_test_spec()
    manifest_path = _write_manifest(tmp_path, spec)
    root = tmp_path / "candidate"
    _write_deliverable(root, _expected_state())
    with pytest.raises(NativeFailure, match="disagrees with trusted manifest"):
        run_case(
            _runner_command(), "osworld:other-task:v1", manifest_path, str(root),
            phase="evaluation", timeout=30, public_item={},
        )


def test_evidence_ref_names_deliverable(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, _expected_state())
    result = _run(tmp_path, _unit_test_spec())
    assert result["evidence_refs"] == [f"deliverable-sha256:{DELIVERABLE}"]
