"""Saturated-parent selection and group-level observation statistics (migrated from the
pure-function portion of RuACT refinement/protocol.py). Semantic conversion (design
document section 4.2): the RL statement 'pass_count == 8 yields no gradient' becomes,
on the inference side, 'the current generation distribution has no variation on this
rubric and no information for self-iteration'. Selection semantics match the source:
- Positive-weight rubrics with pass_count == K are saturated parent candidates (in
`selection_mode="unreachable"`, use pass_count == 0 instead, matching the source
RUACT_CRITIC_SELECTION=unreachable analysis switch). - Negative-weight rubrics
participate only when `include_negative_weight` is enabled; saturation matches the
source (default mode: violation always present, pass_count == K; unreachable mode:
violation never present, pass_count == 0). - Sort by (-|weight|, rubric_index,
rubric_id) and take the first `max_parents_per_query`.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .fingerprint import parent_fingerprint
from .schemas import (
    PositiveRefinementProtocolError,
    raw_matrix,
    required_text,
    validated_group,
)

SELECTION_MODES = ("saturated", "unreachable")


def group_pass_counts(matrix: Sequence[Sequence[int]]) -> list[int]:
    return [sum(row[index] for row in matrix) for index in range(len(matrix[0]))]


def classify_rubric_observation(
    weight: float, pass_count: int, group_size: int
) -> str:
    """Group-level observation class for one rubric: saturated / unreachable /
    informative. Positive weight: all pass = saturated, none pass = unreachable.
    Negative weight (scores when violated): violation always present = saturated,
    violation never present = unreachable.
    """
    if pass_count == group_size:
        return "saturated"
    if pass_count == 0:
        return "unreachable"
    return "informative"


def _selection_threshold(selection_mode: str, expected_group_size: int) -> int:
    if selection_mode not in SELECTION_MODES:
        raise PositiveRefinementProtocolError(
            f"selection_mode must be one of {SELECTION_MODES!r}"
        )
    return 0 if selection_mode == "unreachable" else expected_group_size


def analyze_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_group_size: int = 8,
    selection_mode: str = "saturated",
    include_negative_weight: bool = False,
    max_parents_per_query: int = 3,
    zero_std_epsilon: float = 1e-08,
) -> dict[str, Any]:
    """Compute observation statistics for one complete rollout group and select
    saturated parent candidates. Return group-level statistics (pass_counts / score
    distribution / saturated weight share) and `selected_parents` (with fingerprint,
    pass_count, polarity, and sort truncation).
    """
    if expected_group_size != 8:
        raise PositiveRefinementProtocolError(
            "the refinement protocol requires K=8"
        )
    if not 1 <= max_parents_per_query <= 3:
        raise PositiveRefinementProtocolError(
            "at most three saturated parents may be refined per query"
        )
    group = validated_group(rows, expected_group_size)
    rubric_ids, titles, descriptions, weights, matrix = raw_matrix(group)
    pass_counts = group_pass_counts(matrix)
    threshold = _selection_threshold(selection_mode, expected_group_size)

    positive_weight_total = sum((weight for weight in weights if weight > 0.0))
    if positive_weight_total <= 0.0:
        raise PositiveRefinementProtocolError(
            "group must contain at least one positive rubric"
        )
    saturated_positive_weight = sum(
        (
            weights[index]
            for index in range(len(rubric_ids))
            if weights[index] > 0.0 and pass_counts[index] == threshold
        )
    )
    saturated_positive_fraction = saturated_positive_weight / positive_weight_total
    saturated_negative_weight = 0.0
    saturated_negative_count = 0
    if include_negative_weight:
        saturated_negative_weight = sum(
            (
                -weights[index]
                for index in range(len(rubric_ids))
                if weights[index] < 0.0 and pass_counts[index] == threshold
            )
        )
        saturated_negative_count = sum(
            (
                1
                for index in range(len(rubric_ids))
                if weights[index] < 0.0 and pass_counts[index] == threshold
            )
        )
    saturated_weight_fraction = (
        saturated_positive_weight + saturated_negative_weight
    ) / positive_weight_total

    group_rewards = [
        sum((score * weight for score, weight in zip(row, weights)))
        / positive_weight_total
        for row in matrix
    ]
    group_mean = sum(group_rewards) / len(group_rewards)
    group_std = math.sqrt(
        sum(((value - group_mean) ** 2 for value in group_rewards))
        / len(group_rewards)
    )

    eligible = [
        {
            "rubric_index": index,
            "rubric_id": rubric_ids[index],
            "title": titles[index],
            "description": descriptions[index],
            "weight": weights[index],
            "pass_count": pass_counts[index],
            "parent_fingerprint": parent_fingerprint(
                rubric_ids[index],
                titles[index],
                descriptions[index],
                weights[index],
            ),
            **(
                {"polarity": "negative" if weights[index] < 0.0 else "positive"}
                if include_negative_weight
                else {}
            ),
        }
        for index in range(len(rubric_ids))
        if (weights[index] > 0.0 or (include_negative_weight and weights[index] < 0.0))
        and pass_counts[index] == threshold
    ]
    eligible.sort(
        key=lambda item: (
            -abs(float(item["weight"])),
            int(item["rubric_index"]),
            str(item["rubric_id"]),
        )
    )
    selected_parents = eligible[:max_parents_per_query]
    return {
        "rubric_ids": rubric_ids,
        "titles": titles,
        "descriptions": descriptions,
        "weights": weights,
        "matrix": matrix,
        "pass_counts": pass_counts,
        "group_rewards": group_rewards,
        "group_mean": group_mean,
        "group_std": group_std,
        "zero_std": group_std <= float(zero_std_epsilon),
        "positive_weight_total": positive_weight_total,
        "saturated_positive_weight_fraction": float(saturated_positive_fraction),
        "saturated_negative_count": saturated_negative_count,
        "saturated_weight_fraction": float(saturated_weight_fraction),
        "eligible_parents": eligible,
        "selected_parents": selected_parents,
    }


def query_id_of(rows: Sequence[Mapping[str, Any]]) -> str:
    """All rows in a group must share the same query_id (protocol invariant, reused by
    the orchestration layer).
    """
    query_ids = {required_text(row.get("query_id"), "query_id") for row in rows}
    if len(query_ids) != 1:
        raise PositiveRefinementProtocolError(
            "query identity changed within one group"
        )
    return next(iter(query_ids))
