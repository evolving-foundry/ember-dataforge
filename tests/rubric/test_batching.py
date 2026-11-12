"""8xN grid batch scoring tests: successful rows preserved, failed rows retried, budget
exhaustion fails explicitly.
"""

from __future__ import annotations

import pytest

from data_designer.rubric.judge.batching import configure_judge_concurrency, score_group
from data_designer.rubric.judge.prompt import JudgeGiveUpError
from data_designer.rubric.judge.transport import JudgeResult

MEMBERS = [{"id": f"resp-{i}", "text": f"text {i}"} for i in range(8)]
DESCRIPTIONS = ("criterion A", "criterion B")


class _StubTransport:
    def __init__(self, fail_first_for=None, always_fail_rows=()):
        self.calls = []
        self.fail_first_for = fail_first_for or set()
        self.always_fail_rows = set(always_fail_rows)

    def call_judge(self, *, query, final_answer, descriptions, **kwargs):
        index = int(final_answer.split()[-1])
        self.calls.append(index)
        if index in self.always_fail_rows:
            raise JudgeGiveUpError("stub give up")
        if index in self.fail_first_for:
            self.fail_first_for.discard(index)
            raise JudgeGiveUpError("transient stub failure")
        return JudgeResult(
            scores=(1, 0), attempts=1, prompt_sha256="0" * 64
        )


def test_score_group_full_grid():
    transport = _StubTransport()
    rows, audit = score_group("q", MEMBERS, DESCRIPTIONS, transport=transport)
    assert len(rows) == 8
    assert all(row == [1, 0] for row in rows)
    assert audit["complete_group_attempts"] == 1
    assert audit["response_task_attempts"] == 8
    assert len(audit["per_response"]) == 8


def test_score_group_retries_failed_row():
    transport = _StubTransport(fail_first_for={3})
    rows, audit = score_group("q", MEMBERS, DESCRIPTIONS, transport=transport)
    assert rows[3] == [1, 0]
    assert audit["retried_response_count"] == 1
    assert audit["complete_group_attempts"] == 2


def test_score_group_explicit_failure_on_exhausted_budget():
    transport = _StubTransport(always_fail_rows={5})
    with pytest.raises(JudgeGiveUpError) as error:
        score_group("q", MEMBERS, DESCRIPTIONS, transport=transport, attempts=2)
    # Successful rows are preserved; failed rows are listed in the error.
    assert "5" in str(error.value)
    assert transport.calls.count(0) == 1


def test_score_group_empty_candidates_is_zero_cost():
    rows, audit = score_group("q", MEMBERS, (), transport=_StubTransport())
    assert rows == [[] for _ in MEMBERS]
    assert audit["complete_group_attempts"] == 0


def test_score_group_rejects_bad_attempts():
    with pytest.raises(Exception):
        score_group("q", MEMBERS, DESCRIPTIONS, transport=_StubTransport(), attempts=4)


def test_judge_concurrency_limit():
    configure_judge_concurrency(2)
    transport = _StubTransport()
    rows, _ = score_group("q", MEMBERS, DESCRIPTIONS, transport=transport, workers=8)
    assert len(rows) == 8
    with pytest.raises(Exception):
        configure_judge_concurrency(0)
