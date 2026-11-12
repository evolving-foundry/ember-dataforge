"""P3: candidate generation primitives -- schema, prompt discipline, structural
validation, discriminative filtering, and job validation.
"""
from __future__ import annotations

import pytest

from data_designer.rubric.generator.candidates import (
    attach_judge_scores,
    canonical_completion_point,
    fixed_meta_valid_children,
    generator_prompt,
    generator_schema,
    reindex_candidates,
    validate_generated,
    validate_refinement_job,
    validate_seed_candidates,
)
from data_designer.rubric.protocol.schemas import PositiveRefinementProtocolError
from data_designer.rubric.evolve.decompose import build_refinement_jobs
from tests.rubric.evolve_fixture import group_rows

RESPONSE_IDS = [f"resp-{i}" for i in range(8)]


def _candidate(index):
    return {
        "candidate_index": index,
        "title": f"child {index}",
        "completion_point": f"completion point {index}.",
        "description": f"completion point {index}.",
        "polarity": "positive",
        "candidate_source": "online_generation",
        "source_candidate_index": index,
    }


def test_generator_schema_bounds():
    schema = generator_schema(2, 5)
    assert schema["properties"]["candidates"]["minItems"] == 2
    assert schema["properties"]["candidates"]["maxItems"] == 5


def test_generator_prompt_discipline():
    job = {
        "query": "target query?",
        "min_candidate_count": 5,
        "max_candidate_count": 5,
    }
    parent = {"title": "parent title", "description": "parent description"}
    prompt = generator_prompt(job, parent)
    assert "deliberately unavailable" in prompt
    assert "target query?" in prompt
    assert "parent description" in prompt
    assert "split the parent's original weight budget equally" in prompt
    converted = generator_prompt(job, parent, polarity_direction="convert_to_positive")
    assert "Flip the shell and decompose positively" in converted
    negative = generator_prompt(job, parent, polarity_direction="keep_negative")
    assert "Decompose the error negatively" in negative
    feedback = generator_prompt(job, parent, rejection_feedback=["rejected point"])
    assert "Previously rejected completion points" in feedback
    assert "- rejected point" in feedback
    partial = generator_prompt(job, parent, partial_weight_split=True)
    assert "shares a fixed portion of the parent's original weight budget" in partial


def test_validate_generated_structure():
    parent = {"title": "t", "description": "d"}
    value = {
        "candidates": [
            {"title": f"title {i}", "completion_point": f"point {i}."} for i in range(5)
        ]
    }
    candidates = validate_generated(value, min_candidate_count=5, max_candidate_count=5, parent=parent)
    assert len(candidates) == 5
    assert candidates[0]["candidate_source"] == "online_generation"
    assert candidates[0]["description"] == "point 0."
    dup = {"candidates": [
        {"title": "same", "completion_point": "same point."},
        {"title": "SAME", "completion_point": "  same   POINT. "},
    ]}
    with pytest.raises(PositiveRefinementProtocolError, match="duplicate"):
        validate_generated(dup, min_candidate_count=2, max_candidate_count=2, parent=parent)
    few = {"candidates": [{"title": "t", "completion_point": "p."}]}
    with pytest.raises(PositiveRefinementProtocolError, match="between"):
        validate_generated(few, min_candidate_count=2, max_candidate_count=5, parent=parent)
    empty = {"candidates": [{"title": "  ", "completion_point": "p."}]}
    with pytest.raises(PositiveRefinementProtocolError, match="non-empty"):
        validate_generated(empty, min_candidate_count=1, max_candidate_count=1, parent=parent)


def test_validate_seed_candidates_defaults_and_provenance():
    parent = {
        "title": "t",
        "description": "d",
        "seed_candidates": [
            {"title": "s0", "completion_point": "p0.", "source_job_id": "job-1",
             "source_step": 2, "source_pass_count": 5, "polarity": "negative"},
            {"title": "s1", "completion_point": "p1.", "source_job_id": "job-1"},
        ],
    }
    seeds = validate_seed_candidates(parent)
    assert seeds[0]["polarity"] == "negative"
    assert seeds[1]["polarity"] == "positive"  # Default polarity is backfilled to
                                               # positive per the source
    assert seeds[1]["source_pass_count"] == -1
    incomplete = {"title": "t", "description": "d", "seed_candidates": [
        {"title": "s", "completion_point": "p."}]}
    with pytest.raises(PositiveRefinementProtocolError, match="incomplete"):
        validate_seed_candidates(incomplete)


def test_attach_judge_scores_discriminative_filter():
    candidates = [_candidate(i) for i in range(4)]
    grid = [
        [1, 1, 1, 1],  # Column 0 pass=3 retained; column 1 pass=8 full -> rejected;
        [1, 1, 1, 0],  # Column 2 pass=2 retained; column 3 pass=1 weak -> rejected
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ]
    selected, audit = attach_judge_scores(
        candidates, grid, response_ids=RESPONSE_IDS, max_children=5
    )
    assert [c["candidate_index"] for c in selected] == [0, 2]
    reasons = {a["candidate_index"]: a["rejection_reason"] for a in audit}
    assert reasons[0] is None and reasons[2] is None
    assert reasons[1] == "full_pass_pattern"
    assert reasons[3] == "weak_imbalanced_pattern"
    zero = [row[:1] for row in [[0]] * 8]
    selected_zero, audit_zero = attach_judge_scores(
        [_candidate(0)], [[0] for _ in range(8)], response_ids=RESPONSE_IDS, max_children=5
    )
    assert selected_zero == []
    assert audit_zero[0]["rejection_reason"] == "zero_pass_pattern"


def test_attach_judge_scores_no_filter_and_misalignment():
    candidates = [_candidate(0)]
    grid = [[1] for _ in range(8)]
    selected, audit = attach_judge_scores(
        candidates, grid, response_ids=RESPONSE_IDS, max_children=5, no_filter=True
    )
    assert len(selected) == 1 and audit[0]["retained"] is True
    with pytest.raises(PositiveRefinementProtocolError, match="misaligned"):
        attach_judge_scores(candidates, grid, response_ids=RESPONSE_IDS[:4], max_children=5)
    with pytest.raises(PositiveRefinementProtocolError, match="cap"):
        attach_judge_scores([_candidate(i) for i in range(4)], [[1] * 4] * 8,
                            response_ids=RESPONSE_IDS, max_children=2)
    non_binary = [[2] for _ in range(8)]
    with pytest.raises(PositiveRefinementProtocolError, match="non-binary"):
        attach_judge_scores(candidates, non_binary, response_ids=RESPONSE_IDS, max_children=5)


def test_fixed_meta_valid_children_dedup_and_cap():
    candidates = [
        {**_candidate(0), "candidate_index": 0},
        {**_candidate(1), "completion_point": "  COMPLETION   POINT 0. ", "candidate_index": 1},
        {**_candidate(2), "candidate_index": 2},
    ]
    selected, duplicates = fixed_meta_valid_children(candidates, max_children=3)
    assert [c["candidate_index"] for c in selected] == [0, 2]
    assert duplicates == [{
        "candidate_index": 1,
        "candidate_source": "online_generation",
        "reason": "canonical_completion_point_duplicate",
    }]
    reindexed = reindex_candidates(candidates)
    assert [c["candidate_index"] for c in reindexed] == [0, 1, 2]


def test_validate_refinement_job_accepts_built_job_and_rejects_tampering():
    job = build_refinement_jobs(group_rows(), run_id="run-1", global_step=0)[0]
    validate_refinement_job(job)  # Passing means no exception
    tampered = dict(job)
    tampered["generator_input_mode"] = "query_and_parent_and_rollouts"
    with pytest.raises(PositiveRefinementProtocolError, match="query and one parent"):
        validate_refinement_job(tampered)
    tampered = dict(job)
    tampered["reward_mode"] = "legacy_reward"
    with pytest.raises(PositiveRefinementProtocolError, match="reward mode"):
        validate_refinement_job(tampered)
    tampered = dict(job)
    tampered["group_members"] = job["group_members"][:4]
    with pytest.raises(PositiveRefinementProtocolError, match="exactly eight"):
        validate_refinement_job(tampered)
    tampered = dict(job)
    tampered["seed_bundle_hash"] = "deadbeef"
    with pytest.raises(PositiveRefinementProtocolError, match="seed bundle hash"):
        validate_refinement_job(tampered)
    tampered = dict(job)
    tampered["selected_parents"] = [dict(job["selected_parents"][0], pass_count=3)]
    with pytest.raises(PositiveRefinementProtocolError, match="selection and weight polarity"):
        validate_refinement_job(tampered)
    unreachable = build_refinement_jobs(
        group_rows(scores_matrix=[[0, i % 2] for i in range(8)]),
        run_id="run-1",
        global_step=0,
        selection_mode="unreachable",
    )[0]
    validate_refinement_job(unreachable, selection_mode="unreachable")
    with pytest.raises(PositiveRefinementProtocolError, match="selection and weight polarity"):
        validate_refinement_job(unreachable)  # Under saturated semantics, pass_count=0
                                              # is invalid
