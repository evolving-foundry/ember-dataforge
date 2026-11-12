"""Section 6.3 single-round evolution orchestration: run_evolution_round outputs the
evolution_result.json contract, noreplace persistence, failure rollback without
writing a result file, and a valid result for an empty round with no saturated
parents.
"""
from __future__ import annotations

import json
import threading

import pytest

from data_designer.rubric.evolve.registry import PositiveRefinementRegistry
from data_designer.rubric.evolve.round import (
    EVOLUTION_RESULT_SCHEMA,
    run_evolution_round,
)
from data_designer.rubric.generator.budget import RefinementBudgetExhausted
from data_designer.rubric.generator.client import WorkerConfig
from tests.rubric.evolve_fixture import RUBRICS, group_rows, mock_generator_server
from tests.rubric.test_generator_budget_single_pass import _ColumnPatternTransport


BASE_RUBRICS = [
    {"rubric_key": rubric["rubric_id"], **{k: v for k, v in rubric.items() if k != "rubric_id"}}
    for rubric in RUBRICS
]


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


def test_round_end_to_end_writes_evolution_result(tmp_path, generator_server):
    registry = PositiveRefinementRegistry(tmp_path / "state")
    result_path = tmp_path / "evolution_result.json"
    payload, effects = run_evolution_round(
        group_rows(),
        rubrics=BASE_RUBRICS,
        registry=registry,
        config=_worker_config(generator_server),
        judge_transport=_ColumnPatternTransport(),
        run_id="rsi-1",
        global_step=1,
        result_path=result_path,
    )
    # Section 6.3 contract: status / selected_parents / decompositions / skipped_parents
    # / audit
    assert payload["schema"] == EVOLUTION_RESULT_SCHEMA
    assert payload["status"] == "ok"
    assert len(payload["selected_parents"]) == 1
    selected = payload["selected_parents"][0]
    assert selected["rubric_id"] == "r0" and selected["pass_count"] == 8
    assert selected["weight"] == pytest.approx(0.6)
    assert len(payload["decompositions"]) == 1
    decomposition = payload["decompositions"][0]
    assert decomposition["parent_rubric_id"] == "r0"
    assert decomposition["blend_lambda"] == pytest.approx(0.5)
    assert decomposition["child_count"] == len(decomposition["children"]) >= 1
    assert payload["skipped_parents"] == []

    # Audit: version write-back chain + job audit follows the single_pass structure
    audit = payload["audit"]
    assert audit["committed_count"] == 1 and audit["jobs"][0]["status"] == "committed"
    assert audit["jobs"][0]["generator_received_rollouts"] is False
    assert audit["previous_rubric_library_sha256"]
    new_digest = audit["rubric_library_sha256_by_query"]["q1"]
    assert new_digest != audit["previous_rubric_library_sha256"]

    # Overlaid library: parent weight w*(1-lambda) + children appended; total family
    # weight conserved
    overlaid = effects["overlaid_libraries"]["q1"]
    parent_entry = next(rubric for rubric in overlaid if rubric["rubric_key"] == "r0")
    assert parent_entry["weight"] == pytest.approx(0.6 * 0.5)
    family_total = parent_entry["weight"] + sum(
        rubric["weight"]
        for rubric in overlaid
        if rubric.get("dcd_parent_rubric_key") == "r0"
    )
    assert family_total == pytest.approx(0.6)

    # Score redistribution: per-sample dcd_* correction evidence
    scored = effects["score_adjustments"]
    assert set(scored) == {row["response_id"] for row in group_rows()}
    assert all("dcd_reward_delta" in row for row in scored.values())

    # Result file is written with noreplace and content matches; a second export to the
    # same path fails explicitly
    on_disk = json.loads(result_path.read_text(encoding="utf-8"))
    assert on_disk == payload
    with pytest.raises(FileExistsError):
        run_evolution_round(
            group_rows(group_id="g2"),
            rubrics=BASE_RUBRICS,
            registry=registry,
            config=_worker_config(generator_server),
            judge_transport=_ColumnPatternTransport(),
            run_id="rsi-1",
            global_step=1,
            result_path=result_path,
        )


def test_budget_exhaustion_rolls_back_and_writes_no_result(tmp_path, generator_server):
    generator_server.mode = "down"
    registry = PositiveRefinementRegistry(tmp_path / "state")
    result_path = tmp_path / "evolution_result.json"
    with pytest.raises(RefinementBudgetExhausted):
        run_evolution_round(
            group_rows(),
            rubrics=BASE_RUBRICS,
            registry=registry,
            config=_worker_config(generator_server),
            judge_transport=_ColumnPatternTransport(),
            run_id="rsi-1",
            global_step=1,
            result_path=result_path,
        )
    # Explicit failure: job rolled back, query not polluted, no fabricated success
    # result file written
    assert registry.blocked_query_ids() == frozenset()
    assert registry.snapshot("q1")["decompositions"] == []
    assert registry.snapshot("q1")["query_version"] == 0
    assert not result_path.exists()


def test_round_without_saturated_parents_emits_empty_result(tmp_path, generator_server):
    registry = PositiveRefinementRegistry(tmp_path / "state")
    # r0 all 0 (not saturated), r1 mixed -> no selected parent; an empty round still
    # produces a valid contract result
    rows = group_rows(
        scores_matrix=[[0, i % 2] for i in range(8)],
    )
    result_path = tmp_path / "evolution_result.json"
    payload, effects = run_evolution_round(
        rows,
        rubrics=BASE_RUBRICS,
        registry=registry,
        config=_worker_config(generator_server),
        judge_transport=_ColumnPatternTransport(),
        run_id="rsi-1",
        global_step=1,
        result_path=result_path,
    )
    assert payload["status"] == "ok"
    assert payload["selected_parents"] == []
    assert payload["decompositions"] == []
    assert payload["skipped_parents"] == []
    assert payload["audit"]["job_count"] == 0
    assert effects["overlaid_libraries"] == {}
    assert effects["score_adjustments"] == {}
    assert result_path.exists()
