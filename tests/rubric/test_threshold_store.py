"""Family threshold anchor persistence (section 4.1): P25 anchoring is reproducible
across rounds, borrowing is refused when library fingerprints are incomparable,
atomic persistence can be reloaded, and minimal integration with RubricGate (anchor
takes precedence over the default threshold; explicit parameters take precedence over
the anchor).
"""
from __future__ import annotations

import pytest

from data_designer.rubric.gate.threshold_store import FamilyThresholdStore
from data_designer.rubric.orchestrate.pipeline import RubricGate

LIBRARY = [
    {"rubric_key": "r0", "title": "t0", "description": "d0", "weight": 0.5},
    {"rubric_key": "r1", "title": "t1", "description": "d1", "weight": 0.5},
]
DIGEST_A = "sha256:aaaaaaaa"
DIGEST_B = "sha256:bbbbbbbb"


def test_p25_anchor_survives_reload(tmp_path):
    path = tmp_path / "thresholds.json"
    store = FamilyThresholdStore(path)
    assert store.anchor_threshold("q1", rubric_library_sha256=DIGEST_A) is None
    store.record_accepted(
        "q1", [0.9, 0.7, 0.5, 0.3], rubric_library_sha256=DIGEST_A, updated_step=3
    )
    reloaded = FamilyThresholdStore(path)
    # Nearest-neighbor rank P25 for 4 samples = position int(0.25*3)=0 after sorting
    assert reloaded.anchor_threshold("q1", rubric_library_sha256=DIGEST_A) == 0.3


def test_scores_not_borrowed_across_library_fingerprints(tmp_path):
    store = FamilyThresholdStore(tmp_path / "thresholds.json")
    store.record_accepted("q1", [0.8, 0.9], rubric_library_sha256=DIGEST_A)
    # After a library upgrade (fingerprint changes after decomposition/rewrite), the old
    # distribution is incomparable: return None instead of borrowing across libraries
    assert store.anchor_threshold("q1", rubric_library_sha256=DIGEST_B) is None


def test_same_family_replacement_keeps_latest_round(tmp_path):
    store = FamilyThresholdStore(tmp_path / "thresholds.json")
    store.record_accepted("q1", [0.2, 0.3], rubric_library_sha256=DIGEST_A, updated_step=1)
    store.record_accepted("q1", [0.8, 0.9], rubric_library_sha256=DIGEST_A, updated_step=2)
    assert store.anchor_threshold("q1", rubric_library_sha256=DIGEST_A) == 0.8


def test_invalid_entries_rejected(tmp_path):
    path = tmp_path / "thresholds.json"
    store = FamilyThresholdStore(path)
    with pytest.raises(ValueError):
        store.record_accepted("q1", [], rubric_library_sha256=DIGEST_A)
    with pytest.raises(ValueError):
        store.record_accepted("q1", [1.5], rubric_library_sha256=DIGEST_A)
    with pytest.raises(ValueError):
        store.record_accepted("q1", [0.5], rubric_library_sha256="")
    path.write_text('{"store_schema": "unknown", "schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        FamilyThresholdStore(path)


class _StaticTransport:
    """Fixed scoring rows per response: r0 always 1, r1 0/1 based on answer text."""

    model = "mock-judge"

    def call_judge(self, *, query, final_answer, descriptions, **kwargs):
        from data_designer.rubric.judge.transport import JudgeResult

        second = 1 if "strong" in final_answer else 0
        return JudgeResult(scores=(1, second), attempts=1, prompt_sha256="x" * 64)


def _responses(n=8):
    return [
        {
            "sample_id": f"s{i}",
            "rollout_index": i,
            "text": "strong answer" if i < 4 else "weak answer",
        }
        for i in range(n)
    ]


def test_gate_uses_family_anchor_and_records_round(tmp_path):
    store = FamilyThresholdStore(tmp_path / "thresholds.json")
    store.record_accepted("q1", [0.0], rubric_library_sha256=DIGEST_A)
    gate = RubricGate(
        rubric_library=LIBRARY, anchor_store=store, group_size=8, transport=_StaticTransport()
    )
    gate.library_sha256 = DIGEST_A  # Fixed test library fingerprint
    decisions = gate.gate_group(query_id="q1", query="Q", responses=_responses())
    scores = {record["sample_id"]: record["score"] for record in decisions}
    strong_score = scores["s0"]
    weak_score = scores["s7"]
    assert strong_score > weak_score
    # Everything is accepted below the anchored threshold 0.0, and the audit marks the
    # threshold source
    assert all(record["decision"] == "accept" for record in decisions)
    assert {record["audit"]["threshold_source"] for record in decisions} == {"family_anchor"}
    gate.record_family_anchor("q1", decisions, updated_step=5)
    # Accepted score distribution this round = 4x1.0(strong) + 4x0.5(weak); nearest-
    # neighbor rank P25 = 0.5
    assert store.anchor_threshold("q1", rubric_library_sha256=DIGEST_A) == 0.5
    reloaded = FamilyThresholdStore(tmp_path / "thresholds.json")
    assert reloaded.anchor_threshold("q1", rubric_library_sha256=DIGEST_A) == 0.5


def test_explicit_threshold_wins_over_anchor_store(tmp_path):
    store = FamilyThresholdStore(tmp_path / "thresholds.json")
    store.record_accepted("q1", [0.0], rubric_library_sha256=DIGEST_A)
    gate = RubricGate(
        rubric_library=LIBRARY,
        threshold=1.0,
        anchor_store=store,
        group_size=8,
        transport=_StaticTransport(),
    )
    gate.library_sha256 = DIGEST_A
    decisions = gate.gate_group(query_id="q1", query="Q", responses=_responses())
    # Explicit threshold=1.0 takes precedence over the family anchor; strong samples
    # with exactly 1.0 (score>=threshold) are accepted
    assert {record["audit"]["threshold_source"] for record in decisions} == {"static"}
    gate_scores = {record["sample_id"]: record["score"] for record in decisions}
    assert gate_scores["s0"] == 1.0
    assert gate_scores["s7"] < 1.0
    gate.record_family_anchor("q1", decisions)
    assert store.anchor_threshold("q1", rubric_library_sha256=DIGEST_A) == 0.0


def test_anchor_store_miss_falls_back_to_default_threshold(tmp_path):
    gate = RubricGate(
        rubric_library=LIBRARY,
        anchor_store=FamilyThresholdStore(tmp_path / "thresholds.json"),
        group_size=8,
        transport=_StaticTransport(),
    )
    gate.library_sha256 = DIGEST_A
    decisions = gate.gate_group(query_id="q-unknown", query="Q", responses=_responses())
    assert {record["audit"]["threshold_source"] for record in decisions} == {"static"}
