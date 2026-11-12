"""Scenario B expansion-sample routing (design document section 5.2 steps 2-4): trap and
positive sample sets. - Trap set: negative rubrics pass (violation present) --
reproduction samples for RSI evaluation rounds to test generalization; positive
rubrics need not all pass (erroneous samples are inherently invalid). - Positive set:
all negative rubrics fail (no violation) and all positive rubrics pass -- repair
samples for training rounds. - Everything else (mixed: violation absent but positive
rubrics also below threshold) goes explicitly into the review bucket rather than
being silently dropped. - Negative-rubric discriminative filtering uses the same
[filter_low, filter_high] anchor as positive rubrics (default [2,6] for group size
8); constant firing (all samples violate) means the error pattern is too broad ->
send for rewrite; never firing is also sent for review.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..protocol.schemas import PositiveRefinementProtocolError

DEFAULT_FILTER_LOW = 2
DEFAULT_FILTER_HIGH = 6
DEFAULT_GROUP_SIZE = 8

TRAP_SET = "trap"
POSITIVE_SET = "positive"
REVIEW_SET = "review"


def negative_criterion_review(
    pass_count: int,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    filter_low: int = DEFAULT_FILTER_LOW,
    filter_high: int = DEFAULT_FILTER_HIGH,
) -> dict[str, Any]:
    """Discriminative audit for one negative rubric over the expansion set (section 5.2
    step 3 / risk table 8.3). Return {pass_count, pass_rate, discriminative, action,
    reason}; action is `keep` or `review_rewrite` (constant firing / never firing /
    weak imbalance all go to review, with reason distinguishing the form).
    """
    for name, value in (
        ("pass_count", pass_count),
        ("group_size", group_size),
        ("filter_low", filter_low),
        ("filter_high", filter_high),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PositiveRefinementProtocolError(
                f"negative criterion review requires non-negative integer {name}"
            )
    if not filter_low <= filter_high <= group_size:
        raise PositiveRefinementProtocolError(
            "negative criterion filter bounds must satisfy 0 <= low <= high <= group_size"
        )
    if pass_count > group_size:
        raise PositiveRefinementProtocolError(
            "negative criterion pass_count exceeds the frozen group size"
        )
    if filter_low <= pass_count <= filter_high:
        action, reason = "keep", "mid_pass_rate"
    elif pass_count == group_size:
        action, reason = "review_rewrite", "over_broad_constant_violation"
    elif pass_count == 0:
        action, reason = "review_rewrite", "never_triggers"
    else:
        action, reason = "review_rewrite", "weak_imbalanced_pattern"
    return {
        "pass_count": pass_count,
        "pass_rate": pass_count / group_size,
        "discriminative": action == "keep",
        "action": action,
        "reason": reason,
    }


def classify_expansion_samples(
    score_rows: Mapping[str, Mapping[str, int]],
    *,
    positive_rubric_ids: Sequence[str],
    negative_rubric_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Route expansion samples into trap / positive / review using merged scores (never
    silently drop). `score_rows`: response_id -> {rubric_id: 0|1}; the two id
    sequences must be disjoint and jointly cover every row's score keys. Return
    response_id -> {set, negative_hits, missing_positive, reason}.
    """
    positives = list(positive_rubric_ids)
    negatives = list(negative_rubric_ids)
    if not negatives:
        raise PositiveRefinementProtocolError(
            "expansion triage requires at least one negative criterion"
        )
    if len(set(positives) & set(negatives)) or len(set(positives + negatives)) != len(
        positives
    ) + len(negatives):
        raise PositiveRefinementProtocolError(
            "positive and negative rubric ids must be disjoint and unique"
        )
    classification: dict[str, dict[str, Any]] = {}
    for response_id, scores in score_rows.items():
        missing = [
            rubric_id
            for rubric_id in positives + negatives
            if rubric_id not in scores
        ]
        unknown = [key for key in scores if key not in set(positives) | set(negatives)]
        if missing or unknown:
            raise PositiveRefinementProtocolError(
                "expansion scores do not cover exactly the merged rubric set"
            )
        negative_hits = [
            rubric_id for rubric_id in negatives if int(scores[rubric_id]) == 1
        ]
        missing_positive = [
            rubric_id for rubric_id in positives if int(scores[rubric_id]) != 1
        ]
        if negative_hits:
            bucket, reason = TRAP_SET, "negative_violation_present"
        elif not missing_positive:
            bucket, reason = POSITIVE_SET, "no_violation_and_positives_pass"
        else:
            bucket, reason = REVIEW_SET, "violation_absent_but_positives_incomplete"
        classification[str(response_id)] = {
            "set": bucket,
            "negative_hits": negative_hits,
            "missing_positive": missing_positive,
            "reason": reason,
        }
    return classification


def split_expansion_sets(
    classification: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Convert classification results into a two-set admission view: {trap: [...],
    positive: [...], review: [...]}.
    """
    sets: dict[str, list[str]] = {TRAP_SET: [], POSITIVE_SET: [], REVIEW_SET: []}
    for response_id, entry in classification.items():
        bucket = str(entry.get("set"))
        if bucket not in sets:
            raise PositiveRefinementProtocolError(
                "unknown expansion set bucket"
            )
        sets[bucket].append(str(response_id))
    return sets


__all__ = [
    "DEFAULT_FILTER_HIGH",
    "DEFAULT_FILTER_LOW",
    "DEFAULT_GROUP_SIZE",
    "POSITIVE_SET",
    "REVIEW_SET",
    "TRAP_SET",
    "classify_expansion_samples",
    "negative_criterion_review",
    "split_expansion_sets",
]
