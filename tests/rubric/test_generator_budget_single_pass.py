"""P3: parent budget retries and single-pass job orchestration -- 3 attempts/40s/120s
budget, explicit failure, and audit structure.
"""
from __future__ import annotations

import pytest

import data_designer.rubric.generator.budget as budget_module
from data_designer.rubric.evolve.decompose import build_refinement_jobs
from data_designer.rubric.generator.budget import (
    PARENT_ATTEMPTS,
    RefinementBudgetExhausted,
    prepare_parent,
)
from data_designer.rubric.generator.client import WorkerConfig
from data_designer.rubric.generator.single_pass import (
    CANDIDATE_BUDGET,
    EXECUTION_MODE,
    prepare_parent_candidates,
    process_refinement_job,
)
from data_designer.rubric.judge.transport import JudgeResult
from data_designer.rubric.protocol.schemas import PositiveRefinementProtocolError
from tests.rubric.evolve_fixture import group_rows


@pytest.fixture()
def job():
    return build_refinement_jobs(group_rows(), run_id="run-1", global_step=0)[0]


def _config(**overrides):
    return WorkerConfig(
        base_url="http://mock",
        api_key="key",
        model="mock-model",
        retry_backoff_seconds=0.0,
        **overrides,
    )


class _ColumnPatternTransport:
    """Assign a deterministic pattern by column: column j has pass_count = fn(j); rows i
    < pass_count are marked 1. Response row index is parsed from final_answer
    (fixture's 'answer text {i} ...').
    """

    def __init__(self, pass_count_for_column=lambda column: 3):
        self.fn = pass_count_for_column
        self.calls = 0

    def call_judge(self, *, query, final_answer, descriptions, **kwargs):
        self.calls += 1
        row = int(str(final_answer).split("answer text ")[1].split(" ")[0])
        scores = tuple(
            (1 if row < self.fn(column) else 0) for column in range(len(descriptions))
        )
        return JudgeResult(scores=scores, attempts=1, prompt_sha256="0" * 64)


def _patch_generator(monkeypatch, *, candidate_prefix="generated"):
    def fake_generate_json(prompt, schema, *, config, deadline=None):
        count = schema["properties"]["candidates"]["minItems"]
        return (
            {
                "candidates": [
                    {
                        "title": f"{candidate_prefix} {i}",
                        "completion_point": f"{candidate_prefix} completion point {i}: detail {i}.",
                    }
                    for i in range(count)
                ]
            },
            {"attempts": 1},
        )

    monkeypatch.setattr(budget_module, "generate_json", fake_generate_json)


def test_prepare_parent_candidates_seed_pool_skips_generation(job):
    parent = dict(job["selected_parents"][0])
    parent["seed_candidates"] = [
        {
            "title": f"seed {i}",
            "completion_point": f"seed completion point {i}.",
            "source_job_id": "old-job",
            "source_step": 1,
            "source_pass_count": 4,
        }
        for i in range(5)
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("generator must not be called")

    pool, audit = prepare_parent_candidates(job, parent, _config(), forbidden)
    assert len(pool) == CANDIDATE_BUDGET
    assert all(c["candidate_source"] == "historical_seed" for c in pool)
    assert audit["generation_skipped"] is True
    assert audit["preverified_reused_count"] == 5
    assert audit["polarity_direction"] == "positive"
    assert pool[0]["meta_verification"]["decision"] == "not_performed"


def test_prepare_parent_candidates_topup_and_prompt_hygiene(job):
    parent = dict(job["selected_parents"][0])
    parent["seed_candidates"] = [
        {
            "title": "seed 0",
            "completion_point": "seed completion point zero.",
            "source_job_id": "old-job",
        }
    ]
    captured = {}

    def fake_generator_call(prompt, schema, *, config=None, deadline=None):
        captured["prompt"] = prompt
        captured["schema"] = schema
        count = schema["properties"]["candidates"]["minItems"]
        return (
            {
                "candidates": [
                    {
                        "title": f"gen {i}",
                        "completion_point": f"generated completion point {i}.",
                    }
                    for i in range(count)
                ]
            },
            {"attempts": 1, "latency_seconds": 0.1},
        )

    pool, audit = prepare_parent_candidates(job, parent, _config(), fake_generator_call)
    assert len(pool) == CANDIDATE_BUDGET
    assert audit["generated_candidate_count"] == 4
    assert audit["preverified_reused_count"] == 1
    assert audit["generator_call_count"] == 1
    prompt = captured["prompt"]
    # prompt discipline: no response text/rollout statistics; existing points declared
    # distinct; weight wording uses fixed share
    assert "answer text" not in prompt
    assert "Existing completion points; generate distinct additional points" in prompt
    assert "- seed completion point zero." in prompt
    assert "shares a fixed portion" in prompt
    assert captured["schema"]["properties"]["candidates"]["minItems"] == 4
    assert audit["generator"]["prompt_sha256"]
    assert audit["generator_prompt_character_count"] > 0


def test_prepare_parent_retry_backfills_error_then_succeeds(job, monkeypatch):
    captured = {}

    def fake_generate_json(prompt, schema, *, config, deadline=None):
        captured["prompt"] = prompt
        return {"candidates": [{"title": "t", "completion_point": "p."}]}, {}

    monkeypatch.setattr(budget_module, "generate_json", fake_generate_json)
    prepare_calls = []

    def prepare(j, p, config, generator_call):
        prepare_calls.append(1)
        if len(prepare_calls) == 1:
            generator_call("first prompt", {}, config=config)
            raise PositiveRefinementProtocolError("structural problem X")
        generator_call("second prompt", {}, config=config)
        return ([], {"generator_call_count": 0})

    pool, audit = prepare_parent(job, dict(job["selected_parents"][0]), _config(), prepare)
    assert pool == []
    assert audit["preparation_attempts"] == 2
    assert audit["preparation_errors"] == ["structural problem X"]
    # Failure text is fed into the next prompt (injected by the budget closure)
    assert "structural problem X" in captured["prompt"]
    assert "Return exactly the requested number of distinct candidates" in captured["prompt"]


def test_prepare_parent_budget_exhaustion_is_explicit_failure(job):
    parent = dict(job["selected_parents"][0])
    attempts = []

    def prepare(j, p, config, generator_call):
        attempts.append(1)
        raise PositiveRefinementProtocolError("always invalid")

    with pytest.raises(RefinementBudgetExhausted, match="budget exhausted"):
        prepare_parent(job, parent, _config(), prepare)
    assert len(attempts) == PARENT_ATTEMPTS


def test_prepare_parent_bounded_single_call_budget(job, monkeypatch):
    parent = dict(job["selected_parents"][0])
    captured = {}

    def fake_generate_json(prompt, schema, *, config, deadline=None):
        captured["timeout"] = config.timeout_seconds
        captured["max_attempts"] = config.max_attempts
        captured["deadline"] = deadline
        return {"candidates": [{"title": "t", "completion_point": "p."}]}, {}

    monkeypatch.setattr(budget_module, "generate_json", fake_generate_json)

    def prepare(j, p, config, generator_call):
        generator_call("prompt", {}, config=config)
        return ([], {"generator_call_count": 1})

    prepare_parent(
        job,
        parent,
        WorkerConfig(base_url="http://x", model="m", timeout_seconds=999.0),
        prepare,
    )
    assert captured["timeout"] == 40.0
    assert captured["max_attempts"] == 1
    assert captured["deadline"] is not None


def test_process_refinement_job_end_to_end_with_fakes(job, monkeypatch):
    _patch_generator(monkeypatch)
    transport = _ColumnPatternTransport()
    result = process_refinement_job(job, config=_config(), parent_workers=1, judge_transport=transport)
    assert result["status"] == "ok"
    assert result["job_id"] == job["job_id"]
    assert result["query_id"] == "q1"
    decompositions = result["decompositions"]
    assert len(decompositions) == 1
    children = decompositions[0]["children"]
    assert 1 <= len(children) <= 3
    for child in children:
        assert child["description"] == child["completion_point"]
        assert child["pass_count"] == 3
        assert len(child["judge_scores"]) == 8
    audit = result["audit"]
    assert audit["execution_mode"] == EXECUTION_MODE
    assert audit["generator_received_rollouts"] is False
    assert audit["child_pass_count_range"] == [2, 6]
    assert audit["constant_children_rejected"] is True
    assert audit["fixed_judge_input_mode"] == "query_answer_descriptions_only"
    assert audit["selected_child_count"] == len(children)
    assert audit["judge_route"] == {}
    assert len(audit["judge_grid"]) == 8
    assert len(audit["judge_column_map"]) == len(audit["parents"][0]["judged_candidates"])
    parent_audit = audit["parents"][0]
    assert parent_audit["status"] == "selected"
    assert parent_audit["generator"]["prompt_sha256"]


def test_process_refinement_job_constant_pattern_rejected_or_exempt(job, monkeypatch):
    _patch_generator(monkeypatch)
    # All-1 pattern (full): this parent is skipped under default filtering; retained
    # when the exemption switch is on
    transport = _ColumnPatternTransport(lambda column: 8)
    strict = process_refinement_job(job, config=_config(), judge_transport=transport)
    assert strict["decompositions"] == []
    parent_audit = strict["audit"]["parents"][0]
    assert parent_audit["status"] == "no_nonconstant_child_pattern"
    assert all(
        item["rejection_reason"] == "full_pass_pattern"
        for item in parent_audit["candidate_pass_counts"]
    )
    permissive = process_refinement_job(
        job, config=_config(), judge_transport=transport, keep_constant_children=True
    )
    assert len(permissive["decompositions"]) == 1
    assert permissive["audit"]["child_pass_count_range"] == [0, 8]
    assert permissive["audit"]["constant_children_retained"] is True


def test_process_refinement_job_below_min_children_marks_parent_skipped(job, monkeypatch):
    _patch_generator(monkeypatch)
    # Only one column is discriminative, but min_children_per_parent=2 -> skip the whole
    # parent
    job = dict(job)
    job["min_children_per_parent"] = 2
    transport = _ColumnPatternTransport(lambda column: 3 if column == 0 else 8)
    result = process_refinement_job(job, config=_config(), judge_transport=transport)
    assert result["decompositions"] == []
    parent_audit = result["audit"]["parents"][0]
    assert parent_audit["status"] == "no_nonconstant_child_pattern"
    reasons = {
        item["candidate_index"]: item["rejection_reason"]
        for item in parent_audit["candidate_pass_counts"]
    }
    # The sole discriminative candidate is retained by child_cap first; when below_min
    # triggers, the whole parent is rolled back and marked
    assert reasons[0] == "below_min_children"
    assert set(reasons.values()) == {"below_min_children", "full_pass_pattern"}


def test_process_refinement_job_rejects_tampered_job(job):
    tampered = dict(job)
    tampered["job_id"] = "forged"
    with pytest.raises(PositiveRefinementProtocolError):
        process_refinement_job(
            tampered, config=_config(), judge_transport=_ColumnPatternTransport()
        )
