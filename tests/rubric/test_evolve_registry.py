"""P3: PositiveRefinementRegistry -- reserve/commit state machine, idempotency, and
rollback.
"""
from __future__ import annotations

import json

import pytest

from data_designer.rubric.evolve.decompose import build_refinement_jobs
from data_designer.rubric.evolve.registry import META_CHECKS, PositiveRefinementRegistry
from data_designer.rubric.protocol.schemas import PositiveRefinementProtocolError
from tests.rubric.evolve_fixture import child_for, decomposition_for, group_rows


@pytest.fixture()
def job():
    return build_refinement_jobs(group_rows(), run_id="run-1", global_step=3)[0]


def _valid_children(job, *, count=2, source="online_generation"):
    response_ids = [m["id"] for m in job["group_members"]]
    parent = job["selected_parents"][0]
    return [
        child_for(index=index, response_ids=response_ids, source=source)
        for index in range(count)
    ]


def _result(job, *, children=None, child_count=2):
    parent = job["selected_parents"][0]
    return {
        "schema_version": job["schema_version"],
        "status": "ok",
        "job_id": job["job_id"],
        "transaction_id": job["transaction_id"],
        "query_id": job["query_id"],
        "source_query_version": job["query_version"],
        "response_set_hash": job["response_set_hash"],
        "seed_bundle_hash": job["seed_bundle_hash"],
        "decompositions": [
            decomposition_for(
                parent=parent,
                children=children if children is not None else _valid_children(job, count=child_count),
                response_ids=[m["id"] for m in job["group_members"]],
            )
        ],
    }


def test_reserve_commit_lifecycle(tmp_path, job):
    registry = PositiveRefinementRegistry(tmp_path / "state")
    assert registry.registry_version == 0
    assert registry.blocked_query_ids() == frozenset()
    assert registry.reserve_job(job) is True
    assert registry.blocked_query_ids() == frozenset({"q1"})
    # A second reserve for the same query is rejected idempotently; different jobs are
    # also rejected while the query is occupied
    assert registry.reserve_job(job) is False
    registry.cancel_job(job["query_id"], job["job_id"])
    assert registry.blocked_query_ids() == frozenset()
    assert registry.reserve_job(job) is True
    summary = registry.commit_result(_result(job), created_step=3)
    assert summary["status"] == "committed"
    assert summary["decomposition_count"] == 1
    assert summary["selected_child_count"] == 2
    assert registry.blocked_query_ids() == frozenset({"q1"})
    snapshot = registry.snapshot("q1")
    assert snapshot["query_version"] == 1
    decomposition = snapshot["decompositions"][0]
    assert decomposition["parent_effective_weight"] == 0.3
    assert decomposition["child_total_effective_weight"] == 0.3
    assert all(child["effective_weight"] == 0.15 for child in decomposition["children"])
    assert decomposition["children"][0]["rubric_id"].startswith("r0:completion:")
    assert registry.max_observed_step() == 3


def test_duplicate_processed_job_idempotent(tmp_path, job):
    registry = PositiveRefinementRegistry(tmp_path / "state")
    registry.reserve_job(job)
    registry.commit_result(_result(job), created_step=3)
    # Reserving the same job again -> processed hit is False; commit replay is rejected
    assert registry.reserve_job(job) is False
    with pytest.raises(PositiveRefinementProtocolError):
        registry.commit_result(_result(job), created_step=3)


def test_stale_query_version_rejected(tmp_path, job):
    registry = PositiveRefinementRegistry(tmp_path / "state")
    registry.reserve_job(job)
    result = _result(job)
    result["source_query_version"] = 5
    with pytest.raises(PositiveRefinementProtocolError, match="stale query version"):
        registry.commit_result(result, created_step=3)


def test_identity_tampering_rejected(tmp_path, job):
    registry = PositiveRefinementRegistry(tmp_path / "state")
    tampered = dict(job)
    tampered["response_set_hash"] = "deadbeef"
    with pytest.raises(PositiveRefinementProtocolError):
        registry.reserve_job(tampered)
    tampered = dict(job)
    tampered["selected_parents"] = [
        {**job["selected_parents"][0], "weight": 0.99}
    ]
    with pytest.raises(PositiveRefinementProtocolError):
        registry.reserve_job(tampered)


def test_decomposition_validation_rules(tmp_path, job):
    registry = PositiveRefinementRegistry(tmp_path / "state")
    registry.reserve_job(job)
    response_ids = [m["id"] for m in job["group_members"]]
    parent = job["selected_parents"][0]

    # Constant-pattern child (all 1) is rejected by default
    constant = [child_for(index=0, response_ids=response_ids, pass_pattern=[1] * 8)]
    with pytest.raises(PositiveRefinementProtocolError, match="distinguish"):
        registry.commit_result(_result(job, children=constant), created_step=3)

    # child description must equal its atomic completion point
    broken = _valid_children(job)
    broken[0]["description"] = "different from completion point"
    with pytest.raises(PositiveRefinementProtocolError, match="atomic completion point"):
        registry.commit_result(_result(job, children=broken), created_step=3)

    # A positive-weight parent must not carry a negative child
    negative = _valid_children(job)
    for child in negative:
        child["polarity"] = "negative"
    with pytest.raises(PositiveRefinementProtocolError, match="negative children"):
        registry.commit_result(_result(job, children=negative), created_step=3)

    # Invalid meta_verification provenance
    forged = _valid_children(job)
    forged[0]["meta_verification"] = {"decision": "accept", "factually_valid": True}
    with pytest.raises(PositiveRefinementProtocolError, match="provenance"):
        registry.commit_result(_result(job, children=forged), created_step=3)

    # pass_count and Judge score are inconsistent
    miscount = _valid_children(job)
    miscount[0]["pass_count"] = 5
    with pytest.raises(PositiveRefinementProtocolError, match="pass count"):
        registry.commit_result(_result(job, children=miscount), created_step=3)
    del parent  # For contract checks only


def test_keep_constant_children_escape_hatch(tmp_path, job):
    registry = PositiveRefinementRegistry(tmp_path / "state", keep_constant_children=True)
    registry.reserve_job(job)
    children = [
        child_for(index=0, response_ids=[m["id"] for m in job["group_members"]],
                  pass_pattern=[1] * 8)
    ]
    summary = registry.commit_result(_result(job, children=children), created_step=3)
    assert summary["decomposition_count"] == 1
    assert registry.snapshot("q1")["decompositions"][0]["children"][0]["is_constant"] is True


def test_state_file_persists_and_reloads(tmp_path, job):
    state_dir = tmp_path / "state"
    registry = PositiveRefinementRegistry(state_dir)
    registry.reserve_job(job)
    registry.commit_result(_result(job), created_step=3)
    raw = json.loads((state_dir / "positive-refinement-registry.v7.json").read_text())
    assert raw["registry_version"] == 2
    assert "q1" in raw["queries"] and job["job_id"] in raw["processed_jobs"]
    reloaded = PositiveRefinementRegistry(state_dir)
    assert reloaded.registry_version == 2
    assert reloaded.snapshot("q1")["query_version"] == 1
    # Old schema version is refused
    raw["schema_version"] = 6
    (state_dir / "positive-refinement-registry.v7.json").write_text(json.dumps(raw))
    with pytest.raises(PositiveRefinementProtocolError, match="fresh v7"):
        PositiveRefinementRegistry(state_dir)


def test_synchronize_to_checkpoint_rolls_back(tmp_path, job):
    registry = PositiveRefinementRegistry(tmp_path / "state")
    registry.reserve_job(job)
    registry.commit_result(_result(job), created_step=3)
    # require_empty: a non-empty registry rejects a fresh-run declaration
    with pytest.raises(PositiveRefinementProtocolError, match="empty"):
        registry.synchronize_to_checkpoint(0, require_empty=True)
    summary = registry.synchronize_to_checkpoint(2)
    assert summary == {
        "removed_pending": 0,
        "removed_processed": 1,
        "removed_decompositions": 1,
    }
    assert registry.blocked_query_ids() == frozenset()
    assert registry.snapshot("q1")["query_version"] == 0
    assert registry.registry_version == 3


def test_meta_checks_tuple_shape():
    assert META_CHECKS == (
        "factually_valid",
        "preserves_parent",
        "binary_judgeable",
        "relevant_to_query",
        "substantive",
    )
