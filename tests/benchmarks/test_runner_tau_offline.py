"""tau offline runner tests (tau follow-up M1): full branch coverage for final-state artifact scoring.

Aligned with the tau scoring re-expression follow-up doc (v0.1) M1 acceptance: candidates write the final state as
a ``{db_state, messages}`` structure to ``verifier_spec.payload.artifact.path``;
the runner validates the deliverable structure and dispatches through ``run_verifier``. Expected final state + required messages
return passed=True/1.0; wrong final values or missing required messages return completed/False/0.0; missing/unparseable/
malformed/symlinked deliverables return invalid_candidate; a missing audit stage is an explicit
NativeFailure; out-of-bounds deliverable paths / unknown kinds / family or task mismatches fail explicitly.
All results pass ``validate_native_result`` (enforced inside ``run_case``). Fully offline, zero environment.
"""

import json
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import NativeFailure
from data_designer.pipelines.benchmarks.native import run_case
from data_designer.pipelines.benchmarks.runners.tau_offline import VERSION

RUNNER_PATH = Path(__file__).resolve().parents[2] / (
    "data_designer/pipelines/benchmarks/runners/tau_offline.py"
)

TASK_ID = "tau:retail-fixture:v1"
DELIVERABLE = "tau_offline_artifact.json"

EXPECTED_DB = {
    "orders": {"o-1": {"status": "exchanged", "item": "mechanical-keyboard"}},
    "customers": {"c-1": {"tier": "gold"}},
}
REQUIRED_MESSAGES = ["您的订单已完成换货", "exchange confirmation 4451"]
INITIAL_DB = {
    "orders": {"o-1": {"status": "placed", "item": "mechanical-keyboard"}},
    "customers": {"c-1": {"tier": "gold"}},
}
GOLD = {"orders": {"o-1": {"status": "exchanged", "item": "mechanical-keyboard"}}}
MESSAGES = [
    {"role": "user", "content": "我想换货"},
    {"role": "agent", "content": "好的，您的订单已完成换货（exchange confirmation 4451）。"},
]


def _deliverable(db_state, messages=None) -> dict:
    return {
        "db_state": db_state,
        "messages": list(MESSAGES if messages is None else messages),
    }


def _unit_test_script(required: list[str]) -> str:
    literal = json.dumps(json.dumps(EXPECTED_DB, sort_keys=True), ensure_ascii=False)
    required_literal = json.dumps(
        json.dumps(required, ensure_ascii=False), ensure_ascii=False
    )
    script = (
        "import json\n"
        f"EXPECTED = json.loads({literal})\n"
        "DB = json.load(open('db_state.json', encoding='utf-8'))\n"
        "assert DB == EXPECTED, 'db_state mismatch: %r' % (DB,)\n"
    )
    if required:
        script += (
            f"REQUIRED = json.loads({required_literal})\n"
            "RECORD = json.dumps(json.load(open('messages.json', "
            "encoding='utf-8')), ensure_ascii=False)\n"
            "missing = [item for item in REQUIRED if item not in RECORD]\n"
            "assert not missing, ('missing required messages', missing)\n"
        )
    return script


def _unit_test_spec(required: list[str] | None = None) -> dict:
    return {
        "kind": "unit_test",
        "generator_ref": "tau-offline-state@fixture",
        "payload": {
            "artifact": {"path": DELIVERABLE},
            "test_script": _unit_test_script(REQUIRED_MESSAGES if required is None else required),
            "timeout_s": 30.0,
        },
    }


def _artifact_compare_spec() -> dict:
    return {
        "kind": "artifact_compare",
        "generator_ref": "tau-offline-state@fixture",
        "payload": {
            "artifact": {"path": DELIVERABLE},
            "gold": dict(GOLD),
            "fields": {
                "orders": {"type": "json"},
            },
        },
    }


def _write_manifest(tmp_path: Path, spec: dict) -> str:
    manifest = {
        "schema_version": "benchmark-case-v1",
        "task_id": TASK_ID,
        "family": "tau",
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


def _run(tmp_path: Path, spec: dict, phase: str = "evaluation", root: str = "candidate", **kwargs):
    manifest_path = _write_manifest(tmp_path, spec)
    candidate_root = tmp_path / root
    candidate_root.mkdir(parents=True, exist_ok=True)
    return run_case(
        _runner_command(), TASK_ID, manifest_path, str(candidate_root),
        phase=phase, timeout=30, public_item={}, **kwargs,
    )


def test_reference_phase_passes_on_expected_state_and_messages(tmp_path):
    root = tmp_path / "reference"
    _write_deliverable(root, _deliverable(EXPECTED_DB))
    result = _run(tmp_path, _unit_test_spec(), phase="reference", root="reference")
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["verifier_version"] == VERSION
    assert result["native_metrics"]["artifact_bytes"] > 0
    assert result["native_metrics"]["message_count"] == float(len(MESSAGES))


def test_negative_phase_fails_on_wrong_db_state(tmp_path):
    root = tmp_path / "negative"
    wrong = {"orders": {"o-1": {"status": "cancelled", "item": "mechanical-keyboard"}}}
    _write_deliverable(root, _deliverable(wrong))
    result = _run(tmp_path, _unit_test_spec(), phase="negative", root="negative")
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_missing_required_message_scores_zero(tmp_path):
    root = tmp_path / "candidate"
    partial_messages = [{"role": "agent", "content": "您的订单已完成换货"}]
    _write_deliverable(root, _deliverable(EXPECTED_DB, partial_messages))
    result = _run(tmp_path, _unit_test_spec())
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_empty_message_record_fails_message_assertion(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, _deliverable(EXPECTED_DB, []))
    result = _run(tmp_path, _unit_test_spec())
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_initial_state_alone_does_not_pass(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, _deliverable(INITIAL_DB))
    result = _run(tmp_path, _unit_test_spec())
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_db_only_assertions_pass_without_messages(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, _deliverable(EXPECTED_DB, []))
    result = _run(tmp_path, _unit_test_spec(required=[]))
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_evaluation_acknowledges_attempt_seed(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, _deliverable(EXPECTED_DB))
    result = _run(
        tmp_path, _unit_test_spec(), phase="evaluation",
        attempt={"index": 3, "seed": 4242},
    )
    assert result["attempt_index"] == 3
    assert result["seed"] == 4242
    assert result["passed"] is True


def test_artifact_compare_passes_on_gold_db_state(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, _deliverable(EXPECTED_DB))
    result = _run(tmp_path, _artifact_compare_spec())
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_artifact_compare_ignores_messages_and_extra_tables(tmp_path):
    root = tmp_path / "candidate"
    db_state = {
        **EXPECTED_DB,
        "audit_log": {"entry": "irrelevant to gold fields"},
    }
    _write_deliverable(root, _deliverable(db_state))
    result = _run(tmp_path, _artifact_compare_spec())
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_artifact_compare_wrong_status_scores_zero(tmp_path):
    root = tmp_path / "candidate"
    wrong = {"orders": {"o-1": {"status": "cancelled", "item": "mechanical-keyboard"}}}
    _write_deliverable(root, _deliverable(wrong))
    result = _run(tmp_path, _artifact_compare_spec())
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_missing_deliverable_is_invalid_candidate_not_silent(tmp_path):
    (tmp_path / "candidate").mkdir()
    result = _run(tmp_path, _unit_test_spec())
    assert result["status"] == "invalid_candidate"
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert result["native_metrics"]["message_count"] == 0.0


def test_unparseable_deliverable_is_invalid_candidate(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, "{not json")
    result = _run(tmp_path, _unit_test_spec())
    assert result["status"] == "invalid_candidate"
    assert result["score"] == 0.0


def test_ill_shaped_deliverable_is_invalid_candidate(tmp_path):
    for bad in (
        {"db_state": EXPECTED_DB},
        {"messages": MESSAGES},
        {"db_state": EXPECTED_DB, "messages": MESSAGES, "extra": 1},
        {"db_state": "not-a-table", "messages": MESSAGES},
        {"db_state": EXPECTED_DB, "messages": "not-a-list"},
        [EXPECTED_DB, MESSAGES],
    ):
        root = tmp_path / "candidate"
        if (root / DELIVERABLE).exists():
            (root / DELIVERABLE).unlink()
        _write_deliverable(root, bad)
        result = _run(tmp_path, _unit_test_spec())
        assert result["status"] == "invalid_candidate", f"shape accepted: {bad!r}"
        assert result["score"] == 0.0


def test_symlink_deliverable_is_invalid_candidate(tmp_path):
    root = tmp_path / "candidate"
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(_deliverable(EXPECTED_DB), ensure_ascii=False), encoding="utf-8"
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


def test_audit_phase_rejects_ill_shaped_deliverable(tmp_path):
    root = tmp_path / "reference"
    _write_deliverable(root, {"db_state": EXPECTED_DB})
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
    _write_deliverable(root, _deliverable(EXPECTED_DB))
    with pytest.raises(NativeFailure, match="unsupported verifier kind"):
        _run(tmp_path, spec)


def test_family_mismatch_is_rejected(tmp_path):
    spec = _unit_test_spec()
    manifest_path = _write_manifest(tmp_path, spec)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest["family"] = "osworld"
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    root = tmp_path / "candidate"
    _write_deliverable(root, _deliverable(EXPECTED_DB))
    with pytest.raises(NativeFailure, match="not a tau case"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="evaluation", timeout=30, public_item={},
        )


def test_task_id_mismatch_is_rejected(tmp_path):
    spec = _unit_test_spec()
    manifest_path = _write_manifest(tmp_path, spec)
    root = tmp_path / "candidate"
    _write_deliverable(root, _deliverable(EXPECTED_DB))
    with pytest.raises(NativeFailure, match="disagrees with trusted manifest"):
        run_case(
            _runner_command(), "tau:other-task:v1", manifest_path, str(root),
            phase="evaluation", timeout=30, public_item={},
        )


def test_evidence_ref_names_deliverable(tmp_path):
    root = tmp_path / "candidate"
    _write_deliverable(root, _deliverable(EXPECTED_DB))
    result = _run(tmp_path, _unit_test_spec())
    assert result["evidence_refs"] == [f"deliverable-sha256:{DELIVERABLE}"]
