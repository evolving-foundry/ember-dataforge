"""P3 acceptance: two-round closed-loop rubric_evolve drill (mock Generator endpoint +
column-based Judge transport). Round one: saturated parent -> Generator candidates ->
discriminative filtering -> registry commit -> overlay + score-redistribution audit;
round two: under a fresh run (fresh registry, persisted seed library), the same query
and same parent fingerprint match exactly -> seed pool is sufficient (Generator
endpoint is not called even if unreachable) -> explicit commit; negative path: a
parent without seeds exhausts its budget when Generator is interrupted and fails
explicitly; the job is cancelled and 'no candidates' must never be fabricated as
success.
"""
from __future__ import annotations

import json
import threading

import pytest

import data_designer.rubric.generator.budget as budget_module
from data_designer.rubric.evolve.decompose import (
    apply_decompositions_to_scores,
    build_refinement_jobs,
    overlay_decompositions,
)
from data_designer.rubric.evolve.registry import PositiveRefinementRegistry
from data_designer.rubric.evolve.seed_library import PositiveRefinementSeedLibrary
from data_designer.rubric.generator.budget import RefinementBudgetExhausted
from data_designer.rubric.generator.client import WorkerConfig
from data_designer.rubric.generator.single_pass import process_refinement_job
from tests.rubric.evolve_fixture import RUBRICS, group_rows, mock_generator_server
from tests.rubric.test_generator_budget_single_pass import _ColumnPatternTransport


def _worker_config(server):
    return WorkerConfig(
        base_url=f"http://127.0.0.1:{server.server_address[1]}" if server else "http://127.0.0.1:1",
        api_key="k",
        model="mock-model",
        retry_backoff_seconds=0.0,
    )


@pytest.fixture()
def generator_server():
    server = mock_generator_server("ok")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def test_two_round_evolution_with_seed_reuse(tmp_path, generator_server):
    registry = PositiveRefinementRegistry(tmp_path / "state")
    config = _worker_config(generator_server)
    transport = _ColumnPatternTransport()

    # -- Round one: q1 saturated r0 -> generate -> commit --
    round_one_rows = group_rows()
    jobs = build_refinement_jobs(round_one_rows, run_id="rsi-1", global_step=1)
    assert len(jobs) == 1
    job = jobs[0]
    assert registry.reserve_job(job) is True
    result = process_refinement_job(job, config=config, judge_transport=transport)
    assert result["audit"]["generator_received_rollouts"] is False
    committed = registry.commit_result(result, created_step=1)
    assert committed["decomposition_count"] == 1
    snapshot = registry.snapshot("q1")
    assert snapshot["query_version"] == 1
    decomposition = snapshot["decompositions"][0]

    # Score redistribution: family weight is conserved after overlay, dcd_* audit
    # complete (applied_decompositions keep
    # rubric_id + judge_scores; registry snapshots strip judge rows after commit)
    rubrics = [
        {"rubric_key": "r0", "title": "title r0", "description": "criterion zero", "weight": 0.6},
        {"rubric_key": "r1", "title": "title r1", "description": "criterion one", "weight": 0.4},
    ]
    overlaid, diagnostics = overlay_decompositions(rubrics, snapshot["decompositions"])
    assert diagnostics["overlay_child_count"] == len(decomposition["children"])
    evidence = {
        row["response_id"]: {
            "rubric_ids": row["rubric_ids"],
            "rubric_scores": row["rubric_scores"],
            "rubric_weights": row["rubric_weights"],
        }
        for row in round_one_rows
    }
    scored = apply_decompositions_to_scores(
        evidence, committed["applied_decompositions"]
    )
    assert set(scored) == set(evidence)
    assert all("dcd_reward_delta" in row for row in scored.values())

    # -- Round-one decomposition frozen into the seed library (exact parent fingerprint)
    # --
    # Full seed pool (CANDIDATE_BUDGET=5) comes from cross-run aggregation in the frozen
    # layer: 3 children from
    # round-one commit, 2 from an earlier run-0 historical decomposition (provenance
    # metadata preserved).
    committed_child = decomposition["children"][0]
    historical_children = [
        {
            "title": f"run0 child {index}",
            "completion_point": f"run0 completion point {index}: earlier detail {index}.",
            "source_job_id": "run-0-job",
            "source_step": 0,
            "source_pass_count": 3,
        }
        for index in range(2)
    ]
    seed_entry = {
        "query_id": "q1",
        "parent_rubric_id": decomposition["parent_rubric_id"],
        "parent_title": decomposition["parent_title"],
        "parent_description": decomposition["parent_description"],
        "parent_weight": decomposition["parent_weight"],
        "parent_fingerprint": decomposition["parent_fingerprint"],
        "created_step": 1,
        "children": [
            {
                "title": child["title"],
                "completion_point": child["completion_point"],
                "source_job_id": decomposition["source_job_id"],
                "source_step": child["source_step"] if "source_step" in child else 1,
                "source_pass_count": child["pass_count"],
            }
            for child in decomposition["children"]
        ]
        + historical_children,
    }
    seed_path = tmp_path / "seed_library.json"
    seed_path.write_text(
        json.dumps(
            {
                "library_schema": "refinement-seed-library",
                "schema_version": 1,
                "entries": [seed_entry],
                "metadata": {"origin": "round-1"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    seed_library = PositiveRefinementSeedLibrary(seed_path)
    seeds = seed_library.lookup(
        query_id="q1",
        rubric_id=decomposition["parent_rubric_id"],
        title=decomposition["parent_title"],
        description=decomposition["parent_description"],
        weight=decomposition["parent_weight"],
        limit=5,
    )
    assert len(seeds) == len(decomposition["children"]) + len(historical_children)

    # -- Round two: under a fresh run (fresh registry, persisted seed library), q1 same-
    # fingerprint parent
    # becomes saturated again -> exact seed hit, Generator is not called --
    generator_server.mode = "down"  # Even if the endpoint is down, a full seed pool
                                    # means no call is needed
    fresh_registry = PositiveRefinementRegistry(tmp_path / "state-run-2")
    jobs = build_refinement_jobs(
        group_rows(group_id="g2", version=0),
        run_id="rsi-2",
        global_step=2,
        blocked_query_ids=fresh_registry.blocked_query_ids(),
        seed_candidates_by_parent=seed_library.as_lookup_mapping(limit=5),
    )
    assert len(jobs) == 1
    job = jobs[0]
    assert len(job["selected_parents"][0]["seed_candidates"]) == 5
    assert fresh_registry.reserve_job(job) is True
    result = process_refinement_job(job, config=config, judge_transport=transport)
    audit = result["audit"]["parents"][0]
    assert audit["generation_skipped"] is True  # Seed pool is full; zero API calls
    assert audit["preverified_reused_count"] == 5
    committed = fresh_registry.commit_result(result, created_step=2)
    assert committed["decomposition_count"] == 1
    assert fresh_registry.snapshot("q1")["query_version"] == 1
    assert fresh_registry.max_observed_step() == 2


def test_budget_exhaustion_cancels_job_without_fallback(tmp_path, generator_server):
    generator_server.mode = "down"
    registry = PositiveRefinementRegistry(tmp_path / "state")
    job = build_refinement_jobs(
        group_rows(), run_id="rsi-1", global_step=1, blocked_query_ids=frozenset()
    )[0]
    assert registry.reserve_job(job) is True
    with pytest.raises(RefinementBudgetExhausted):
        process_refinement_job(
            job,
            config=_worker_config(generator_server),
            judge_transport=_ColumnPatternTransport(),
        )
    # Explicit failure: no candidates -> cancel job, roll back pending, query is not
    # polluted
    registry.cancel_job(job["query_id"], job["job_id"])
    assert registry.blocked_query_ids() == frozenset()
    assert registry.snapshot("q1")["decompositions"] == []
    assert registry.snapshot("q1")["query_version"] == 0
