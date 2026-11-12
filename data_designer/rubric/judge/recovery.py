"""Strict scoring recovery for child candidates (migrated from RuACT
refinement/bounded_retry.py, decoupled from asyncio). Source policies retained: -
Strict JSON scoring: key set exactly {1..count}, labels exactly PRESENT/NOT_PRESENT;
malformed responses are sent back in a corrective conversation request. - Chunked
recovery: when multiple columns in the same response fail format checks
consecutively, split into smaller chunks (>5 columns split into 5, otherwise 1) and
rescore; successful rows are kept and failed rows are drilled into. - Budget
exhaustion fails explicitly: the source `retry_until_success` infinite loop is
bounded on the inference side by a deadline (default JUDGE_BUDGET_SECONDS=120s) --
exhaustion raises :class:`JudgeGiveUpError`; **never fabricate score rows**.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

from ..judge.prompt import JudgeGiveUpError
from ..judge.transport import JudgeTransport, get_judge_transport

JUDGE_BUDGET_SECONDS = 120.0


def judge_candidates(
    job: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    transport: JudgeTransport | None = None,
    workers: int = 8,
    deadline: float | None = None,
    attempts_per_call: int | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Score all candidates by response row; return (grid[response][candidate], audit).
    `job` uses only query and group_members id/text; candidates enter the Judge
    prompt only by description (`descriptions_only` policy; rollout statistics are
    invisible).
    """
    active_transport = transport or get_judge_transport()
    if deadline is None:
        deadline = time.monotonic() + JUDGE_BUDGET_SECONDS
    started = time.monotonic()
    if not candidates:
        members = list(job["group_members"])
        return (
            [[] for _ in members],
            {
                "response_task_attempts": 0,
                "per_response": [
                    {"response_id": str(m["id"]), "task_attempts": 0, "split_count": 0}
                    for m in members
                ],
                "latency_seconds": 0.0,
                "retry_until_success": False,
                "recovery_policy": "strict_json_chunk_recovery",
            },
        )
    descriptions = tuple((str(candidate["description"]) for candidate in candidates))
    members = list(job["group_members"])
    counts = [0] * len(members)
    split_counts = [0] * len(members)

    def score_columns(index: int, indices: list[int]) -> list[int]:
        """Score several candidate columns for one response row; on format failure drill
        down into chunks, and on budget exhaustion fail explicitly.
        """
        counts[index] += 1
        try:
            result = active_transport.call_judge(
                query=str(job["query"]),
                final_answer=str(members[index]["text"]),
                descriptions=tuple((descriptions[i] for i in indices)),
                deadline=deadline,
                max_total_attempts=attempts_per_call,
                strict_response=True,
            )
            return list(result.scores)
        except JudgeGiveUpError:
            if len(indices) <= 1:
                raise
            split_counts[index] += 1
            size = 5 if len(indices) > 5 else 1
            rows: list[int] = []
            for start in range(0, len(indices), size):
                rows.extend(score_columns(index, indices[start : start + size]))
            return rows

    worker_count = max(1, min(int(workers), len(members)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(score_columns, index, list(range(len(descriptions))))
            for index in range(len(members))
        ]
        rows = [future.result() for future in futures]
    if any((len(row) != len(descriptions) for row in rows)):
        raise JudgeGiveUpError("recovered judge grid is misaligned")
    return (
        rows,
        {
            "response_task_attempts": sum(counts),
            "per_response": [
                {"response_id": str(m["id"]), "task_attempts": n, "split_count": s}
                for m, n, s in zip(members, counts, split_counts)
            ],
            "latency_seconds": round(time.monotonic() - started, 3),
            "retry_until_success": False,
            "recovery_policy": "strict_json_chunk_recovery",
        },
    )


__all__ = [
    "JUDGE_BUDGET_SECONDS",
    "JudgeGiveUpError",
    "judge_candidates",
]
