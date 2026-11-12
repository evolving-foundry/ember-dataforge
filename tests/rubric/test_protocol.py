"""Rubric protocol contract tests: group integrity, matrix validation, rubric-definition
validation, fingerprints, and saturated-parent selection.
"""

from __future__ import annotations

import pytest

from data_designer.contracts.provenance import content_digest
from data_designer.rubric.protocol.fingerprint import (
    ordered_response_set_hash,
    parent_fingerprint,
    rubric_library_digest,
)
from data_designer.rubric.protocol.schemas import (
    PositiveRefinementProtocolError,
    raw_matrix,
    required_text,
    validate_rubrics,
    validated_group,
)
from data_designer.rubric.protocol.selection import (
    analyze_group,
    classify_rubric_observation,
    query_id_of,
)

K = 8


def _group_row(rollout_index: int, scores: list[int], *, response_id: str | None = None):
    return {
        "response_id": response_id or f"resp-{rollout_index}",
        "rollout_index": rollout_index,
        "original_index": rollout_index,
        "query_id": "q-1",
        "rubric_ids": ["r0", "r1", "r2"],
        "rubric_titles": ["t0", "t1", "t2"],
        "rubric_descriptions": ["d0", "d1", "d2"],
        "rubric_weights": [0.5, 0.25, 0.25],
        "rubric_scores": list(scores),
    }


def _full_group(scores_per_row):
    return [
        _group_row(index, scores_per_row[index]) for index in range(len(scores_per_row))
    ]


def test_required_text_rejects_blank():
    assert required_text(" x ", "f") == "x"
    with pytest.raises(PositiveRefinementProtocolError):
        required_text("  ", "f")
    with pytest.raises(PositiveRefinementProtocolError):
        required_text(7, "f")


def test_validated_group_happy_and_size():
    rows = _full_group([[1, 0, 1]] * K)
    ordered = validated_group(rows, K)
    assert [row["rollout_index"] for row in ordered] == list(range(K))


def test_validated_group_rejects_bad_slots():
    rows = _full_group([[1, 0, 1]] * K)
    with pytest.raises(PositiveRefinementProtocolError):
        validated_group(rows[:-1], K)
    duplicated = _full_group([[1, 0, 1]] * K)
    duplicated[3]["rollout_index"] = duplicated[2]["rollout_index"]
    with pytest.raises(PositiveRefinementProtocolError):
        validated_group(duplicated, K)
    bad_type = _full_group([[1, 0, 1]] * K)
    bad_type[0]["rollout_index"] = True
    with pytest.raises(PositiveRefinementProtocolError):
        validated_group(bad_type, K)
    dup_id = _full_group([[1, 0, 1]] * K)
    dup_id[5]["response_id"] = dup_id[0]["response_id"]
    with pytest.raises(PositiveRefinementProtocolError):
        validated_group(dup_id, K)
    dup_orig = _full_group([[1, 0, 1]] * K)
    dup_orig[5]["original_index"] = dup_orig[0]["original_index"]
    with pytest.raises(PositiveRefinementProtocolError):
        validated_group(dup_orig, K)


def test_raw_matrix_happy_and_mismatch():
    rows = _full_group([[1, 0, 1], [0, 1, 0]] + [[1, 1, 0]] * (K - 2))
    rubric_ids, titles, descriptions, weights, matrix = raw_matrix(rows)
    assert rubric_ids == ["r0", "r1", "r2"]
    assert weights == [0.5, 0.25, 0.25]
    assert len(matrix) == K
    changed = [dict(row) for row in rows]
    changed[4]["rubric_ids"] = ["r0", "r1", "rX"]
    with pytest.raises(PositiveRefinementProtocolError):
        raw_matrix(changed)
    non_binary = [dict(row) for row in rows]
    non_binary[2]["rubric_scores"] = [1, 2, 0]
    with pytest.raises(PositiveRefinementProtocolError):
        raw_matrix(non_binary)
    dup_ids = [dict(row) for row in rows]
    dup_ids[0]["rubric_ids"] = ["r0", "r0", "r2"]
    with pytest.raises(PositiveRefinementProtocolError):
        raw_matrix(dup_ids)


def test_validate_rubrics_contract():
    rubrics = [
        {"rubric_key": "a", "title": "A", "description": "da", "weight": 0.5},
        {"rubric_key": "b", "title": "B", "description": "db", "weight": -0.5},
    ]
    descriptions, weights, rubric_ids, titles = validate_rubrics(rubrics)
    assert descriptions == ("da", "db")
    assert weights == (0.5, -0.5)
    with pytest.raises(PositiveRefinementProtocolError):
        validate_rubrics([])
    with pytest.raises(PositiveRefinementProtocolError):
        validate_rubrics([{"title": "A", "description": "d", "weight": 1.0}])
    with pytest.raises(PositiveRefinementProtocolError):
        validate_rubrics(
            [
                {"rubric_key": "a", "title": "A", "description": "d", "weight": 1.0},
                {"rubric_key": "a", "title": "B", "description": "d", "weight": 1.0},
            ]
        )
    with pytest.raises(PositiveRefinementProtocolError):
        validate_rubrics(
            [{"rubric_key": "a", "title": "A", "description": "d", "weight": 0.0}]
        )


def test_parent_fingerprint_matches_provenance_digest():
    expected = content_digest(
        {
            "rubric_id": "r0",
            "title": "t0",
            "description": "d0",
            "weight": 0.5,
        }
    ).removeprefix("sha256:")
    assert parent_fingerprint("r0", "t0", "d0", 0.5) == expected
    other = parent_fingerprint("r0", "t0", "d0-changed", 0.5)
    assert other != expected
    with pytest.raises(PositiveRefinementProtocolError):
        parent_fingerprint("r0", "t0", "d0", 0.0)


def test_response_set_hash_is_order_sensitive():
    members = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    assert ordered_response_set_hash(members) != ordered_response_set_hash(
        list(reversed(members))
    )


def test_rubric_library_digest_stable():
    library = [{"rubric_key": "a", "weight": 1.0}]
    assert rubric_library_digest(library) == rubric_library_digest(
        [{"weight": 1.0, "rubric_key": "a"}]
    )


def test_classify_rubric_observation():
    assert classify_rubric_observation(0.5, 8, 8) == "saturated"
    assert classify_rubric_observation(0.5, 0, 8) == "unreachable"
    assert classify_rubric_observation(0.5, 4, 8) == "informative"
    # Negative weight: violation always present = saturated; violation never present =
    # unreachable
    assert classify_rubric_observation(-0.5, 8, 8) == "saturated"
    assert classify_rubric_observation(-0.5, 0, 8) == "unreachable"


def test_analyze_group_selects_saturated_parent_by_weight():
    # r0: all pass (saturated, max weight 0.5); r1: middle; r2: none pass
    scores = {
        0: [1, 0, 1],
        1: [1, 1, 0],
        2: [1, 1, 1],
        3: [1, 0, 0],
        4: [1, 1, 1],
        5: [1, 0, 1],
        6: [1, 1, 0],
        7: [1, 1, 1],
    }
    rows = _full_group([scores[i] for i in range(K)])
    result = analyze_group(rows)
    assert result["pass_counts"] == [8, 5, 5]
    assert [p["rubric_id"] for p in result["selected_parents"]] == ["r0"]
    assert result["selected_parents"][0]["parent_fingerprint"] == parent_fingerprint(
        "r0", "t0", "d0", 0.5
    )
    assert result["saturated_positive_weight_fraction"] == pytest.approx(0.5)
    assert result["zero_std"] is False


def test_analyze_group_unreachable_mode_flips_threshold():
    scores = {
        0: [0, 0, 1],
        1: [0, 1, 0],
        2: [0, 1, 1],
        3: [0, 0, 0],
        4: [0, 1, 1],
        5: [0, 0, 1],
        6: [0, 1, 0],
        7: [0, 1, 1],
    }
    rows = _full_group([scores[i] for i in range(K)])
    result = analyze_group(rows, selection_mode="unreachable")
    assert [p["rubric_id"] for p in result["selected_parents"]] == ["r0"]
    saturated = analyze_group(rows)
    assert saturated["selected_parents"] == []


def test_analyze_group_negative_weights_when_enabled():
    rows = [
        _group_row(
            index,
            scores,
        )
        for index, scores in enumerate(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
             [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        )
    ]
    for row in rows:
        row["rubric_weights"] = [0.5, -0.25, 0.5]
    result = analyze_group(rows, include_negative_weight=True)
    # r1 has negative weight and violation always present (score all 1) -> saturated
    # candidate; sort by -|weight|
    assert [p["rubric_id"] for p in result["selected_parents"]] == ["r0", "r2", "r1"]
    assert result["saturated_negative_count"] == 1
    plain = analyze_group(rows)
    assert [p["rubric_id"] for p in plain["selected_parents"]] == ["r0", "r2"]


def test_analyze_group_requires_positive_weight_and_k8():
    rows = _full_group([[0, 1, 0]] * K)
    with pytest.raises(PositiveRefinementProtocolError):
        analyze_group(rows, expected_group_size=6)
    for row in rows:
        row["rubric_weights"] = [-0.5, -0.25, -0.25]
    with pytest.raises(PositiveRefinementProtocolError):
        analyze_group(rows)
    with pytest.raises(PositiveRefinementProtocolError):
        analyze_group(rows, max_parents_per_query=4)


def test_query_id_of_invariant():
    rows = _full_group([[1, 0, 1]] * K)
    assert query_id_of(rows) == "q-1"
    rows[3]["query_id"] = "q-2"
    with pytest.raises(PositiveRefinementProtocolError):
        query_id_of(rows)
