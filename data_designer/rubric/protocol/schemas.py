"""Data contracts for the rubric protocol (migrated from the validation portions of
RuACT refinement/protocol.py and verl_reward.py). Covers three contracts: - Rollout-
group completeness: `validated_group` -- group size exactly K, rollout_index 0..K-1
each once, no duplicate original_index / response_id. - In-group rubric matrix:
`raw_matrix` -- rubric metadata consistent across rows, score vectors binary and
aligned. - Rubric definitions: `validate_rubrics` -- description/title/rubric_key
non-empty, weights finite and non-zero, keys unique.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Mapping, Sequence


class PositiveRefinementProtocolError(ValueError):
    pass


def required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PositiveRefinementProtocolError(f"{field} must be a non-empty string")
    return value.strip()


def validated_group(
    rows: Sequence[Mapping[str, Any]], expected_group_size: int
) -> list[Mapping[str, Any]]:
    if len(rows) != expected_group_size:
        raise PositiveRefinementProtocolError(
            f"complete group requires {expected_group_size} rows, got {len(rows)}"
        )
    rollout_indices: list[int] = []
    original_indices: list[int] = []
    response_ids: list[str] = []
    for row in rows:
        rollout_index = row.get("rollout_index")
        original_index = row.get("original_index")
        if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
            raise PositiveRefinementProtocolError("rollout_index must be an integer")
        if isinstance(original_index, bool) or not isinstance(original_index, int):
            raise PositiveRefinementProtocolError("original_index must be an integer")
        rollout_indices.append(rollout_index)
        original_indices.append(original_index)
        response_ids.append(required_text(row.get("response_id"), "response_id"))
    if sorted(rollout_indices) != list(range(expected_group_size)):
        raise PositiveRefinementProtocolError(
            "group must contain rollout_index 0..K-1 exactly once"
        )
    if len(set(original_indices)) != expected_group_size:
        raise PositiveRefinementProtocolError("group contains duplicate original_index")
    if len(set(response_ids)) != expected_group_size:
        raise PositiveRefinementProtocolError("group contains duplicate response_id")
    return sorted(rows, key=lambda row: int(row["rollout_index"]))


def raw_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str], list[float], list[list[int]]]:
    first = rows[0]
    rubric_ids = [str(value) for value in first.get("rubric_ids") or []]
    titles = [str(value) for value in first.get("rubric_titles") or []]
    descriptions = [str(value) for value in first.get("rubric_descriptions") or []]
    weights = [float(value) for value in first.get("rubric_weights") or []]
    if not rubric_ids or not len(rubric_ids) == len(titles) == len(descriptions) == len(
        weights
    ):
        raise PositiveRefinementProtocolError(
            "rubric metadata must be non-empty and aligned"
        )
    if any((not title.strip() for title in titles)):
        raise PositiveRefinementProtocolError("rubric titles must be non-empty")
    if len(set(rubric_ids)) != len(rubric_ids):
        raise PositiveRefinementProtocolError("rubric ids must be unique")
    if any((not isfinite(weight) or weight == 0.0 for weight in weights)):
        raise PositiveRefinementProtocolError(
            "rubric weights must be finite and non-zero"
        )
    matrix: list[list[int]] = []
    for row in rows:
        if [str(value) for value in row.get("rubric_ids") or []] != rubric_ids:
            raise PositiveRefinementProtocolError(
                "rubric ids changed within one rollout group"
            )
        if [str(value) for value in row.get("rubric_titles") or []] != titles:
            raise PositiveRefinementProtocolError(
                "rubric titles changed within one rollout group"
            )
        if [
            str(value) for value in row.get("rubric_descriptions") or []
        ] != descriptions:
            raise PositiveRefinementProtocolError(
                "rubric descriptions changed within one rollout group"
            )
        if [float(value) for value in row.get("rubric_weights") or []] != weights:
            raise PositiveRefinementProtocolError(
                "rubric weights changed within one rollout group"
            )
        scores = list(row.get("rubric_scores") or [])
        if len(scores) != len(rubric_ids) or any(
            (isinstance(score, bool) or score not in (0, 1) for score in scores)
        ):
            raise PositiveRefinementProtocolError(
                "rubric score vector is incomplete or non-binary"
            )
        matrix.append([int(score) for score in scores])
    return (rubric_ids, titles, descriptions, weights, matrix)


def validate_rubrics(
    rubrics: Any,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate a rubric definition list and return (descriptions, weights, rubric_ids,
    titles).
    """
    if not isinstance(rubrics, list) or not rubrics:
        raise PositiveRefinementProtocolError("rubrics must be a non-empty list")
    descriptions: list[str] = []
    weights: list[float] = []
    rubric_ids: list[str] = []
    titles: list[str] = []
    for index, rubric in enumerate(rubrics):
        if not isinstance(rubric, dict):
            raise PositiveRefinementProtocolError("each rubric must be a mapping")
        description = required_text(
            rubric.get("description"), f"rubric {index} description"
        )
        title = required_text(rubric.get("title"), f"rubric {index} title")
        try:
            weight = float(rubric["weight"])
        except (KeyError, TypeError, ValueError) as error:
            raise PositiveRefinementProtocolError(
                f"rubric {index} weight must be numeric"
            ) from error
        if not isfinite(weight) or weight == 0.0:
            raise PositiveRefinementProtocolError(
                f"rubric {index} weight must be finite and non-zero"
            )
        rubric_id = required_text(
            rubric.get("rubric_key"), f"rubric {index} rubric_key"
        )
        descriptions.append(description)
        weights.append(weight)
        rubric_ids.append(rubric_id)
        titles.append(title)
    if len(set(rubric_ids)) != len(rubric_ids):
        raise PositiveRefinementProtocolError("rubric keys must be unique")
    return (tuple(descriptions), tuple(weights), tuple(rubric_ids), tuple(titles))
