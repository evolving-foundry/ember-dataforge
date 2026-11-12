"""P3: build_refinement_jobs -- saturation detection, seed injection, transaction
identity, and cross-query selection.
"""
from __future__ import annotations

import pytest

from data_designer.rubric.evolve.decompose import build_refinement_jobs
from data_designer.rubric.protocol.fingerprint import parent_fingerprint
from data_designer.rubric.protocol.schemas import PositiveRefinementProtocolError
from tests.rubric.evolve_fixture import RUBRICS, QUERY, group_rows


def _build(rows, **kwargs):
    return build_refinement_jobs(rows, run_id="run-1", global_step=7, **kwargs)


def test_saturated_parent_provides_job_with_exact_identity():
    jobs = _build(group_rows())
    assert len(jobs) == 1
    job = jobs[0]
    assert job["job_id"] == job["transaction_id"]
    assert job["query_id"] == "q1"
    assert job["query"] == QUERY
    assert job["query_version"] == 0
    assert job["global_step"] == 7
    assert job["run_id"] == "run-1"
    assert job["gate_mode"] == "saturated_positive_weight_fraction"
    assert job["generator_input_mode"] == "query_and_parent_only"
    assert job["child_activation_mode"] == "all_meta_valid_fixed_weights"
    assert job["rollout_pass_rate_filtering"] is False
    assert job["reward_mode"] == "parent_preserved_multi_child_50_50"
    assert job["blend_lambda"] == 0.5
    assert [m["id"] for m in job["group_members"]] == [
        f"g1-resp-{i}" for i in range(8)
    ]
    parents = job["selected_parents"]
    assert len(parents) == 1
    assert parents[0]["rubric_id"] == "r0"
    assert parents[0]["pass_count"] == 8
    assert parents[0]["weight"] == 0.6
    assert parents[0]["parent_fingerprint"] == parent_fingerprint(
        "r0", "title r0", "criterion zero", 0.6
    )
    assert parents[0]["seed_candidates"] == []
    assert job["zero_std_gate"] is False
    assert job["min_candidate_count"] == 5 and job["max_candidate_count"] == 5
    assert job["min_children_per_parent"] == 1 and job["max_children_per_parent"] == 3


def test_no_saturated_rubric_yields_no_job():
    scores = [[1 if i % 2 == 0 else 0, i % 2] for i in range(8)]
    assert _build(group_rows(scores_matrix=scores)) == ()


def test_blocked_query_id_is_skipped():
    rows = group_rows() + group_rows(group_id="g2", query_id="q2")
    jobs = _build(rows, blocked_query_ids={"q1"})
    assert [job["query_id"] for job in jobs] == ["q2"]


def test_seed_injection_dedupes_canonical_points_and_requires_provenance():
    fingerprint = parent_fingerprint("r0", "title r0", "criterion zero", 0.6)
    seeds = [
        {
            "title": "seed a",
            "completion_point": "First Point",
            "source_job_id": "seed-job-1",
            "source_step": 3,
            "source_pass_count": 4,
        },
        {
            "title": "seed dup",
            "completion_point": "  first   POINT ",
            "source_job_id": "seed-job-2",
        },
        {
            "title": "seed b",
            "completion_point": "Second point",
            "source_job_id": "seed-job-3",
        },
    ]
    jobs = _build(
        group_rows(),
        seed_candidates_by_parent={("q1", "r0", fingerprint): seeds},
    )
    injected = jobs[0]["selected_parents"][0]["seed_candidates"]
    assert [candidate["completion_point"] for candidate in injected] == [
        "First Point",
        "Second point",
    ]
    assert injected[0]["source_pass_count"] == 4
    assert injected[1]["source_pass_count"] == -1


def test_invalid_settings_rejected():
    with pytest.raises(PositiveRefinementProtocolError):
        _build(group_rows(), blend_lambda=0.6)
    with pytest.raises(PositiveRefinementProtocolError):
        _build(group_rows(), expected_group_size=4)
    with pytest.raises(PositiveRefinementProtocolError):
        _build(group_rows(), selection_mode="bogus")
    with pytest.raises(PositiveRefinementProtocolError):
        build_refinement_jobs(group_rows(), run_id="", global_step=0)
    with pytest.raises(PositiveRefinementProtocolError):
        build_refinement_jobs(group_rows(), run_id="r", global_step=-1)


def test_incomplete_group_rejected():
    rows = group_rows()[:-1]
    with pytest.raises(PositiveRefinementProtocolError):
        _build(rows)


def test_cross_group_selection_prefers_higher_saturated_fraction_and_caps():
    rows = group_rows()
    rich = group_rows(
        group_id="g2",
        query_id="q2",
        rubrics=RUBRICS + [
            {"rubric_id": "r2", "title": "t2", "description": "criterion two", "weight": 0.2}
        ],
        scores_matrix=[[1, 1, 1] if i < 4 else [1, 0, 1] for i in range(8)],
    )
    jobs = _build(rows + rich, max_queries=1)
    assert len(jobs) == 1
    assert jobs[0]["query_id"] == "q2"
    jobs = _build(rows + rich)
    assert [job["query_id"] for job in jobs] == ["q2", "q1"]


def test_include_negative_weight_branch():
    rubrics = [
        {"rubric_id": "r0", "title": "t0", "description": "positive saturated", "weight": 0.5},
        {"rubric_id": "r1", "title": "t1", "description": "pitfall saturated", "weight": -0.5},
    ]
    rows = group_rows(rubrics=rubrics, scores_matrix=[[1, 1] for _ in range(8)])
    jobs = _build(rows, zero_std_epsilon=0.0)
    assert [p["rubric_id"] for p in jobs[0]["selected_parents"]] == ["r0"]
    assert jobs[0]["gate_mode"] == "saturated_positive_weight_fraction"
    assert "include_negative_weight" not in jobs[0]
    jobs = _build(rows, include_negative_weight=True, zero_std_epsilon=0.0)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["include_negative_weight"] is True
    assert job["gate_mode"] == "saturated_signed_weight_fraction"
    assert [p["rubric_id"] for p in job["selected_parents"]] == ["r0", "r1"]
    assert job["selected_parents"][1]["polarity"] == "negative"
    assert job["negative_saturated_count"] == 1
