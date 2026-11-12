"""Gate decision tests: threshold anchoring, difficulty buckets, priority, and the
section 6.1 decision contract.
"""

from __future__ import annotations

import pytest

from data_designer.rubric.gate.acceptance import (
    FamilyThresholdAnchor,
    decide,
    difficulty_bucket,
    priority_of,
    threshold_from_scores,
)
from data_designer.rubric.gate.decisions import (
    BUCKET_MID,
    BUCKET_SATURATED,
    BUCKET_UNREACHABLE,
    DECISION_ACCEPT,
    DECISION_HOLDOUT,
    DECISION_REJECT,
    RULE_CURRICULUM,
    RULE_THRESHOLD,
)

IDS = ("r0", "r1", "r2")
WEIGHTS = (0.5, 0.25, 0.25)


def test_threshold_from_scores_p25():
    scores = [0.4, 0.6, 0.8, 1.0, 0.5, 0.7, 0.9, 0.55]
    theta = threshold_from_scores(scores)
    assert 0.4 <= theta <= 0.6
    with pytest.raises(ValueError):
        threshold_from_scores([])
    with pytest.raises(ValueError):
        threshold_from_scores([0.5], percentile=1.5)


def test_family_threshold_anchor():
    anchor = FamilyThresholdAnchor([0.2, 0.4, 0.6, 0.8])
    assert anchor.theta == pytest.approx(threshold_from_scores([0.2, 0.4, 0.6, 0.8]))


def test_difficulty_bucket_and_priority():
    assert difficulty_bucket([1, 1, 1], WEIGHTS) == BUCKET_SATURATED
    assert difficulty_bucket([0, 0, 0], WEIGHTS) == BUCKET_UNREACHABLE
    assert difficulty_bucket([1, 0, 1], WEIGHTS) == BUCKET_MID
    assert priority_of([1, 0, 1], WEIGHTS) == pytest.approx(1 - abs(2 * (2 / 3) - 1))
    assert priority_of([1, 1, 1], WEIGHTS) == 0.0
    assert priority_of([0, 0, 0], WEIGHTS) == 0.0
    with pytest.raises(ValueError):
        difficulty_bucket([1, 1], [-0.5, -0.5])


def _decide(scores, **kwargs):
    return decide(
        sample_id=kwargs.pop("sample_id", "s1"),
        query_id="q1",
        rubric_ids=IDS,
        rubric_scores=scores,
        rubric_weights=WEIGHTS,
        group_pass_counts=kwargs.pop("group_pass_counts", [4, 4, 4]),
        threshold=kwargs.pop("threshold", 0.6),
        rubric_library_sha256=kwargs.pop("rubric_library_sha256", "sha256:abc"),
        **kwargs,
    )


def test_decide_accept_by_threshold():
    decision = _decide([1, 1, 0])
    assert decision.decision == DECISION_ACCEPT
    assert decision.difficulty_bucket == BUCKET_MID
    assert decision.score == pytest.approx(0.75)
    assert decision.audit["decision_rule"] == RULE_THRESHOLD
    assert decision.audit["rubric_library_sha256"] == "sha256:abc"


def test_decide_reject_below_threshold():
    decision = _decide([0, 1, 0])
    assert decision.decision == DECISION_REJECT
    assert decision.audit["decision_rule"] == RULE_THRESHOLD
    assert decision.priority == pytest.approx(priority_of([0, 1, 0], WEIGHTS))


def test_decide_saturated_accept_low_priority():
    decision = _decide([1, 1, 1])
    assert decision.decision == DECISION_ACCEPT
    assert decision.difficulty_bucket == BUCKET_SATURATED
    assert decision.priority == 0.0


def test_decide_unreachable_holdout_curriculum():
    decision = _decide([0, 0, 0])
    assert decision.decision == DECISION_HOLDOUT
    assert decision.audit["decision_rule"] == RULE_CURRICULUM


def test_decide_evidence_contract():
    decision = _decide([1, 0, 1], group_pass_counts=[8, 3, 5])
    evidence = decision.rubric_evidence.as_dict()
    assert evidence["rubric_ids"] == list(IDS)
    assert evidence["rubric_scores"] == [1, 0, 1]
    assert evidence["rubric_weights"] == list(WEIGHTS)
    assert evidence["group_pass_counts"] == [8, 3, 5]
    record = decision.as_dict()
    assert set(record) == {
        "sample_id",
        "query_id",
        "decision",
        "priority",
        "score",
        "threshold",
        "difficulty_bucket",
        "rubric_evidence",
        "audit",
    }
    assert set(record["audit"]) == {
        "rubric_library_sha256",
        "judge_route",
        "decomposition_events",
        "decision_rule",
    }
