"""Binary scoring contract and weighted-aggregation tests (matching source reward.py
semantics).
"""

from __future__ import annotations

import pytest

from data_designer.rubric.judge.score import (
    RewardContractError,
    aggregate_weighted_reward,
    parse_batched_scores,
    rubric_passed,
)


def test_parse_batched_scores_full_coverage():
    parsed = parse_batched_scores(
        [{"id": 2, "score": 1}, {"id": 0, "score": 0}, {"id": 1, "score": 1}], 3
    )
    assert parsed.scores == (0, 1, 1)


def test_parse_batched_scores_rejects_incomplete_or_duplicate_ids():
    with pytest.raises(RewardContractError):
        parse_batched_scores([{"id": 0, "score": 1}], 2)
    with pytest.raises(RewardContractError):
        parse_batched_scores(
            [{"id": 0, "score": 1}, {"id": 0, "score": 0}], 2
        )
    with pytest.raises(RewardContractError):
        parse_batched_scores([{"id": 0, "score": 1}, {"id": 2, "score": 0}], 2)


def test_parse_batched_scores_requires_binary_and_exact_keys():
    with pytest.raises(RewardContractError):
        parse_batched_scores([{"id": 0, "score": 2}], 1)
    with pytest.raises(RewardContractError):
        parse_batched_scores([{"id": 0, "score": True}], 1)
    with pytest.raises(RewardContractError):
        parse_batched_scores([{"id": 0, "score": 1, "extra": 1}], 1)
    with pytest.raises(RewardContractError):
        parse_batched_scores([{"score": 1}], 1)
    with pytest.raises(RewardContractError):
        parse_batched_scores([{"id": 0, "score": 1}], 0)
    with pytest.raises(RewardContractError):
        parse_batched_scores([{"id": 0, "score": 1}], 1, score_scale=5.0)


def test_rubric_passed_polarity():
    assert rubric_passed(0.5, 1.0) is True
    assert rubric_passed(0.5, 0.0) is False
    # Negative-weight criterion 'scores when violated': score 0 (no violation) means
    # pass
    assert rubric_passed(-0.25, 0.0) is True
    assert rubric_passed(-0.25, 1.0) is False
    with pytest.raises(RewardContractError):
        rubric_passed(0.0, 1.0)
    with pytest.raises(RewardContractError):
        rubric_passed(0.5, 0.7)


def test_aggregate_weighted_reward_positive_denominator():
    score = aggregate_weighted_reward([1, 0, 1], [0.5, -0.25, 0.25])
    assert score == pytest.approx(1.0)
    score = aggregate_weighted_reward([0, 1, 0], [0.5, -0.25, 0.25])
    assert score == pytest.approx(-1 / 3)
    score = aggregate_weighted_reward([1, 0, 1], [0.5, 0.25, 0.25])
    assert score == pytest.approx(0.75)


def test_aggregate_weighted_reward_rejects_invalid_inputs():
    with pytest.raises(RewardContractError):
        aggregate_weighted_reward([], [0.5])
    with pytest.raises(RewardContractError):
        aggregate_weighted_reward([1, 0], [0.5])
    with pytest.raises(RewardContractError):
        aggregate_weighted_reward([1, 2], [0.5, 0.5])
    with pytest.raises(RewardContractError):
        aggregate_weighted_reward([True, False], [0.5, 0.5])
    with pytest.raises(RewardContractError):
        aggregate_weighted_reward([1, 0], [0.0, 0.5])
    with pytest.raises(RewardContractError):
        aggregate_weighted_reward([1, 0], [-0.5, -0.5])
