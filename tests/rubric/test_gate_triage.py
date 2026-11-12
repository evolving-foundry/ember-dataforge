"""P4: trap-set/positive-set triage -- negative-criterion discriminative audit and two-
set library classification (no silent dropping).
"""
from __future__ import annotations

import pytest

from data_designer.rubric.gate.triage import (
    POSITIVE_SET,
    REVIEW_SET,
    TRAP_SET,
    classify_expansion_samples,
    negative_criterion_review,
    split_expansion_sets,
)
from data_designer.rubric.protocol.schemas import PositiveRefinementProtocolError


def test_negative_criterion_review_keeps_mid_pass_rate():
    assert negative_criterion_review(3) == {
        "pass_count": 3,
        "pass_rate": 3 / 8,
        "discriminative": True,
        "action": "keep",
        "reason": "mid_pass_rate",
    }
    assert negative_criterion_review(6)["action"] == "keep"
    assert negative_criterion_review(2)["action"] == "keep"


def test_negative_criterion_review_flags_constant_and_dead_patterns():
    over_broad = negative_criterion_review(8)  # Constant trigger: error mode is too
                                               # broad
    assert over_broad["action"] == "review_rewrite"
    assert over_broad["reason"] == "over_broad_constant_violation"
    assert over_broad["discriminative"] is False
    dead = negative_criterion_review(0)  # Never triggers: expansion samples did not
                                         # reproduce the error
    assert dead["action"] == "review_rewrite"
    assert dead["reason"] == "never_triggers"
    weak = negative_criterion_review(1)
    assert weak["reason"] == "weak_imbalanced_pattern"
    assert negative_criterion_review(7)["reason"] == "weak_imbalanced_pattern"


def test_negative_criterion_review_rejects_invalid_counts():
    with pytest.raises(PositiveRefinementProtocolError, match="pass_count"):
        negative_criterion_review(-1)
    with pytest.raises(PositiveRefinementProtocolError, match="exceeds"):
        negative_criterion_review(9)
    with pytest.raises(PositiveRefinementProtocolError, match="bounds"):
        negative_criterion_review(4, filter_low=6, filter_high=2)
    with pytest.raises(PositiveRefinementProtocolError, match="bounds"):
        negative_criterion_review(4, filter_high=16)
    with pytest.raises(PositiveRefinementProtocolError, match="pass_count"):
        negative_criterion_review(True)  # bool is treated as an invalid integer


def test_classify_expansion_samples_splits_trap_positive_review():
    positives = ("r_ok_1", "r_ok_2")
    negatives = ("n_err",)
    rows = {
        # Reproduction class: violation appears -> trap set (positive criteria need not
        # all pass)
        "resp-trap": {"n_err": 1, "r_ok_1": 0, "r_ok_2": 1},
        # Repair class: no violation and all positive criteria pass -> positive set
        "resp-fixed": {"n_err": 0, "r_ok_1": 1, "r_ok_2": 1},
        # Mixed: violation absent but positive criteria not met -> explicit review, no
        # silent dropping
        "resp-mixed": {"n_err": 0, "r_ok_1": 0, "r_ok_2": 1},
    }
    classified = classify_expansion_samples(
        rows, positive_rubric_ids=positives, negative_rubric_ids=negatives
    )
    assert classified["resp-trap"] == {
        "set": TRAP_SET,
        "negative_hits": ["n_err"],
        "missing_positive": ["r_ok_1"],
        "reason": "negative_violation_present",
    }
    assert classified["resp-fixed"]["set"] == POSITIVE_SET
    assert classified["resp-fixed"]["missing_positive"] == []
    assert classified["resp-mixed"]["set"] == REVIEW_SET
    assert classified["resp-mixed"]["reason"] == "violation_absent_but_positives_incomplete"
    assert split_expansion_sets(classified) == {
        "trap": ["resp-trap"],
        "positive": ["resp-fixed"],
        "review": ["resp-mixed"],
    }


def test_classify_expansion_samples_rejects_misaligned_evidence():
    positives = ("r_ok",)
    negatives = ("n_err",)
    with pytest.raises(PositiveRefinementProtocolError, match="negative criterion"):
        classify_expansion_samples({}, positive_rubric_ids=positives, negative_rubric_ids=())
    with pytest.raises(PositiveRefinementProtocolError, match="disjoint"):
        classify_expansion_samples(
            {"r": {"n_err": 0}},
            positive_rubric_ids=("n_err",),
            negative_rubric_ids=("n_err",),
        )
    with pytest.raises(PositiveRefinementProtocolError, match="exactly the merged"):
        classify_expansion_samples(
            {"r": {"n_err": 0}},  # Missing positive-criterion score
            positive_rubric_ids=positives,
            negative_rubric_ids=negatives,
        )
    with pytest.raises(PositiveRefinementProtocolError, match="exactly the merged"):
        classify_expansion_samples(
            {"r": {"n_err": 0, "r_ok": 1, "ghost": 0}},  # Undeclared criterion
            positive_rubric_ids=positives,
            negative_rubric_ids=negatives,
        )


def test_split_expansion_sets_rejects_unknown_bucket():
    with pytest.raises(PositiveRefinementProtocolError, match="unknown expansion set"):
        split_expansion_sets({"resp": {"set": "curriculum"}})
