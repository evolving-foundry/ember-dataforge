"""Threshold gate, difficulty buckets, and priority (design document sections 4.1-4.4).
Semantic conversion: - Reward values (binary weighted sum / positive-weight sum,
range [0,1]) are used directly as gate scores; the scoring formula is unchanged. -
Acceptance thresholds do not use a global constant: per rubric family, theta is
initialized from the P25 of the previous round's accepted-sample score distribution
(`FamilyThresholdAnchor`), avoiding manual tuning across business domains. -
Difficulty buckets use the sample's own positive-rubric pass pattern: all pass =
saturated (no variation at this difficulty; downsample), none pass = unreachable
(likely beyond current capability; keep as a curriculum target and do not reject at
this round's gate), and in between = mid (most informative; prioritize admission).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..judge.score import aggregate_weighted_reward
from .decisions import (
    BUCKET_MID,
    BUCKET_SATURATED,
    BUCKET_UNREACHABLE,
    DECISION_ACCEPT,
    DECISION_HOLDOUT,
    DECISION_REJECT,
    RULE_CURRICULUM,
    RULE_DISCRIMINATIVE,
    RULE_THRESHOLD,
    GateDecision,
    evidence_from_rows,
)


def threshold_from_scores(
    accepted_scores: Sequence[float], *, percentile: float = 0.25
) -> float:
    """Nearest-rank P25: theta is initialized from the previous round's accepted-sample
    score distribution for the family.
    """
    if not accepted_scores:
        raise ValueError("anchor scores must be non-empty")
    if not 0.0 < percentile < 1.0:
        raise ValueError("percentile must be in (0,1)")
    ordered = sorted(float(value) for value in accepted_scores)
    rank = int(percentile * (len(ordered) - 1))
    return ordered[rank]


class FamilyThresholdAnchor:
    """Family-relative threshold anchor (P2 provides the in-memory implementation;
    registry persistence integration is left to P3).
    """

    def __init__(self, accepted_scores: Sequence[float]) -> None:
        self.threshold = threshold_from_scores(accepted_scores)

    @property
    def theta(self) -> float:
        return self.threshold


def difficulty_bucket(
    rubric_scores: Sequence[int], rubric_weights: Sequence[float]
) -> str:
    """Bucket samples by their positive-rubric pass pattern: saturated / mid /
    unreachable.
    """
    positive = [
        (int(score), float(weight))
        for score, weight in zip(rubric_scores, rubric_weights)
        if float(weight) > 0.0
    ]
    if not positive:
        raise ValueError("difficulty bucket requires at least one positive rubric")
    passes = sum((score for score, _ in positive))
    if passes == len(positive):
        return BUCKET_SATURATED
    if passes == 0:
        return BUCKET_UNREACHABLE
    return BUCKET_MID


def priority_of(rubric_scores: Sequence[int], rubric_weights: Sequence[float]) -> float:
    """Admission priority: a positive-rubric pass rate closer to 0.5 carries more
    information; both extremes are zero.
    """
    positive = [
        int(score)
        for score, weight in zip(rubric_scores, rubric_weights)
        if float(weight) > 0.0
    ]
    pass_rate = sum(positive) / len(positive)
    return round(1.0 - abs(2.0 * pass_rate - 1.0), 6)


def decide(
    *,
    sample_id: str,
    query_id: str,
    rubric_scores: Sequence[int],
    rubric_weights: Sequence[float],
    rubric_ids: Sequence[str],
    group_pass_counts: Sequence[int],
    threshold: float,
    rubric_library_sha256: str,
    judge_route: Mapping[str, Any] | None = None,
    decomposition_events: Sequence[Mapping[str, Any]] = (),
) -> GateDecision:
    """Produce an accept/reject/holdout decision for one sample (section 6.1 contract).
    Decision rules: - unreachable bucket -> holdout (decision_rule=curriculum):
    likely beyond current generation capability; keep for the next curriculum target,
    not for immediate gating. - score >= theta -> accept; saturated bucket (all pass)
    has priority=0 and is downsampled by the caller. - score < theta -> reject
    (decision_rule=threshold).
    """
    bucket = difficulty_bucket(rubric_scores, rubric_weights)
    score = aggregate_weighted_reward(list(rubric_scores), list(rubric_weights))
    if bucket == BUCKET_UNREACHABLE:
        decision, rule = DECISION_HOLDOUT, RULE_CURRICULUM
    elif score >= threshold:
        decision = DECISION_ACCEPT
        rule = RULE_DISCRIMINATIVE if bucket == BUCKET_SATURATED else RULE_THRESHOLD
    else:
        decision, rule = DECISION_REJECT, RULE_THRESHOLD
    priority = 0.0 if bucket != BUCKET_MID else priority_of(rubric_scores, rubric_weights)
    audit: dict[str, Any] = {
        "rubric_library_sha256": rubric_library_sha256,
        "judge_route": dict(judge_route or {}),
        "decomposition_events": [dict(event) for event in decomposition_events],
        "decision_rule": rule,
    }
    return GateDecision(
        sample_id=sample_id,
        query_id=query_id,
        decision=decision,
        priority=priority,
        score=score,
        threshold=threshold,
        difficulty_bucket=bucket,
        rubric_evidence=evidence_from_rows(
            rubric_ids, rubric_scores, rubric_weights, group_pass_counts
        ),
        audit=audit,
    )
