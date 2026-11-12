"""P3: frozen seed-library exact matching + strict child-candidate score recovery (chunk
down-drill).
"""
from __future__ import annotations

import json

import pytest

from data_designer.rubric.evolve.seed_library import (
    PositiveRefinementSeedLibrary,
    seed_lookup_key,
)
from data_designer.rubric.judge.prompt import JudgeGiveUpError
from data_designer.rubric.judge.recovery import judge_candidates
from data_designer.rubric.protocol.fingerprint import parent_fingerprint
from data_designer.rubric.protocol.schemas import PositiveRefinementProtocolError


SEED_ENTRY = {
    "query_id": "q1",
    "parent_rubric_id": "r0",
    "parent_title": "title r0",
    "parent_description": "criterion zero",
    "parent_weight": 0.6,
    "created_step": 4,
    "children": [
        {"title": "seed child 0", "completion_point": "point zero.",
         "source_job_id": "job-a", "source_step": 4, "source_pass_count": 3},
        {"title": "seed child 1", "completion_point": "point one."},
    ],
    "metadata": None,
}


def _library_payload(entry, *, fingerprint=None):
    entry = dict(entry)
    if fingerprint is not None:
        entry["parent_fingerprint"] = fingerprint
    elif "parent_fingerprint" not in entry:
        entry["parent_fingerprint"] = parent_fingerprint(
            entry["parent_rubric_id"],
            entry["parent_title"],
            entry["parent_description"],
            entry["parent_weight"],
        )
    return {
        "library_schema": "refinement-seed-library",
        "schema_version": 1,
        "entries": [entry],
        "metadata": {"origin": "test"},
    }


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_seed_library_exact_lookup(tmp_path):
    path = _write(tmp_path, "seed.json", _library_payload(SEED_ENTRY))
    library = PositiveRefinementSeedLibrary(path)
    fingerprint = parent_fingerprint("r0", "title r0", "criterion zero", 0.6)
    found = library.lookup(
        query_id="q1", rubric_id="r0", title="title r0",
        description="criterion zero", weight=0.6, limit=5,
    )
    assert len(found) == 2
    assert found[0]["source_job_id"] == "job-a"
    assert found[1]["source_pass_count"] == -1
    assert found[0]["source_step"] == 4
    # Exact key miss -> empty pool
    assert library.lookup(
        query_id="qX", rubric_id="r0", title="title r0",
        description="criterion zero", weight=0.6, limit=5,
    ) == ()
    assert library.lookup(
        query_id="q1", rubric_id="r0", title="title r0",
        description="criterion zero", weight=0.9, limit=5,
    ) == ()
    assert seed_lookup_key("q1", "r0", fingerprint) == ("q1", "r0", fingerprint)
    assert len(library) == 1
    assert library.as_lookup_mapping(limit=1)[("q1", "r0", fingerprint)][0]["title"] == "seed child 0"


def test_seed_library_fingerprint_mismatch_rejected(tmp_path):
    path = _write(
        tmp_path,
        "seed.json",
        _library_payload(SEED_ENTRY, fingerprint="deadbeef"),
    )
    with pytest.raises(PositiveRefinementProtocolError, match="root parent"):
        PositiveRefinementSeedLibrary(path)


def test_seed_library_rejects_duplicate_points_and_zero_weight(tmp_path):
    dup = json.loads(json.dumps(SEED_ENTRY))
    dup["children"].append({"title": "seed child dup", "completion_point": "  POINT   ZERO. "})
    path = _write(tmp_path, "dup.json", _library_payload(dup))
    with pytest.raises(PositiveRefinementProtocolError, match="duplicate completion points"):
        PositiveRefinementSeedLibrary(path)
    zero_entry = {
        **SEED_ENTRY,
        "parent_weight": 0.0,
        "parent_fingerprint": "not-a-real-fingerprint",
    }
    path = _write(tmp_path, "zero.json", _library_payload(zero_entry))
    with pytest.raises(PositiveRefinementProtocolError, match="root parent"):
        PositiveRefinementSeedLibrary(path)


class _RowFlakyTransport:
    """Deterministically fail an entire column review by response row; after splitting
    into chunks (5/1), individual columns succeed.
    """

    def __init__(self, *, fail_full_rows=(), wrong_length_rows=()):
        self.fail_full_rows = set(fail_full_rows)
        self.wrong_length_rows = set(wrong_length_rows)
        self.full_calls = 0
        self.chunk_calls = 0

    def call_judge(self, *, query, final_answer, descriptions, **kwargs):
        row = int(str(final_answer).split("answer ")[1])
        if len(descriptions) > 1:
            self.full_calls += 1
            if row in self.fail_full_rows:
                raise JudgeGiveUpError("malformed strict reply")
        else:
            self.chunk_calls += 1
        scores = [1 if i % 2 == 0 else 0 for i in range(len(descriptions))]
        if row in self.wrong_length_rows:
            scores = scores + [0]
        return type(
            "R", (), {"scores": tuple(scores), "attempts": 1}
        )()


def _job():
    return {
        "query": "target query?",
        "group_members": [
            {"id": f"resp-{i}", "text": f"answer {i}", "rollout_index": i} for i in range(8)
        ],
    }


def test_judge_candidates_grid_shape_and_audit():
    transport = _RowFlakyTransport()
    candidates = [{"description": f"desc {i}."} for i in range(4)]
    grid, audit = judge_candidates(_job(), candidates, transport=transport)
    assert len(grid) == 8 and all(len(row) == 4 for row in grid)
    assert grid[0] == [1, 0, 1, 0]
    assert audit["recovery_policy"] == "strict_json_chunk_recovery"
    assert audit["retry_until_success"] is False
    assert len(audit["per_response"]) == 8
    assert transport.full_calls == 8 and transport.chunk_calls == 0


def test_judge_candidates_split_recovery_preserves_success_rows():
    transport = _RowFlakyTransport(fail_full_rows={0})
    candidates = [{"description": f"desc {i}."} for i in range(3)]
    grid, audit = judge_candidates(_job(), candidates, transport=transport)
    assert len(grid) == 8 and all(len(row) == 3 for row in grid)
    assert grid[0] == [1, 1, 1]  # After splitting failed rows into single columns, each
                                 # column succeeds (single-column score is always 1)
    assert transport.chunk_calls == 3
    assert audit["response_task_attempts"] > 8
    assert audit["per_response"][0]["split_count"] == 1
    assert all(
        item["split_count"] == 0 for item in audit["per_response"][1:]
    )


def test_judge_candidates_empty_pool_and_misaligned_grid():
    empty_grid, audit = judge_candidates(_job(), [], transport=_RowFlakyTransport())
    assert empty_grid == [[] for _ in range(8)]
    assert audit["response_task_attempts"] == 0
    transport = _RowFlakyTransport(wrong_length_rows={2})
    with pytest.raises(JudgeGiveUpError, match="misaligned"):
        judge_candidates(_job(), [{"description": "d."}], transport=transport)


def test_judge_candidates_never_recovers_fails_explicitly():
    class _AlwaysFailsTransport:
        def call_judge(self, **kwargs):
            raise JudgeGiveUpError("judge total attempt/time budget exhausted")

    with pytest.raises(JudgeGiveUpError):
        judge_candidates(
            _job(),
            [{"description": "d."}],
            transport=_AlwaysFailsTransport(),
            attempts_per_call=1,
        )
