"""8xN grid batch scoring (migrated from RuACT refinement/transport.py
`_judge_candidates_forever` and verl_reward.py row-level parallelism). One Judge call
scores all candidate rubrics for a row by column: N rows in a group are submitted in
parallel, successful rows are kept, and failed rows are retried as a group within the
round; when the group-level attempt budget is exhausted, raise explicitly (listing
missing rows and the last error) -- **never fill with all-zero rows**.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Sequence

from .prompt import JudgeGiveUpError, JudgeProtocolError
from .transport import JudgeResult, JudgeTransport, get_judge_transport

_judge_semaphore: threading.BoundedSemaphore | None = None


def configure_judge_concurrency(limit: int) -> None:
    global _judge_semaphore
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise JudgeProtocolError("global Judge concurrency must be positive")
    _judge_semaphore = threading.BoundedSemaphore(limit)


def score_group(
    query: str,
    group_members: Sequence[Mapping[str, Any]],
    candidate_descriptions: Sequence[str],
    *,
    transport: JudgeTransport | None = None,
    workers: int = 8,
    attempts: int = 3,
    deadline: float | None = None,
    retry_until_success: bool | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Score a complete group row by row and return (rows, audit). `rows[i]` is the
    binary score vector for sample i over all candidate rubrics; audit follows the
    source statistics: group attempts / row-task attempts / successful HTTP attempts
    / retried rows / latency.
    """
    if not 1 <= attempts <= 3:
        raise JudgeProtocolError("fixed Judge attempts must be between one and three")
    if not candidate_descriptions:
        return (
            [[] for _ in group_members],
            {
                "complete_group_attempts": 0,
                "response_task_attempts": 0,
                "successful_http_attempts": 0,
                "retried_response_count": 0,
                "latency_seconds": 0.0,
                "per_response": [],
            },
        )
    if retry_until_success is None:
        retry_until_success = os.environ.get("Refinement_JUDGE_RETRY_UNTIL_SUCCESS") == "1"
    descriptions = tuple((str(item) for item in candidate_descriptions))
    members = list(group_members)
    rows: list[list[int] | None] = [None] * len(members)
    response_task_attempts = [0] * len(members)
    successful_http_attempts = [0] * len(members)
    response_latency = [0.0] * len(members)
    last_errors: list[str | None] = [None] * len(members)
    group_attempt = 0
    started = time.monotonic()
    active_transport = transport or get_judge_transport()

    def score(index: int) -> tuple[int, list[int], int, float]:
        request_started = time.monotonic()
        if _judge_semaphore is None:
            result: JudgeResult = active_transport.call_judge(
                query=query,
                final_answer=str(members[index]["text"]),
                descriptions=descriptions,
            )
        else:
            with _judge_semaphore:
                result = active_transport.call_judge(
                    query=query,
                    final_answer=str(members[index]["text"]),
                    descriptions=descriptions,
                )
        return (
            index,
            list(result.scores),
            int(result.attempts),
            time.monotonic() - request_started,
        )

    while any((row is None for row in rows)) and (
        retry_until_success or group_attempt < attempts
    ):
        group_attempt += 1
        missing = [index for index, row in enumerate(rows) if row is None]
        failed = False
        with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            futures = {}
            for index in missing:
                response_task_attempts[index] += 1
                futures[pool.submit(score, index)] = index
            for future in as_completed(futures):
                try:
                    index, scores, http_attempts, latency = future.result()
                    if len(scores) != len(descriptions):
                        raise JudgeProtocolError(
                            "fixed Judge returned a misaligned candidate row"
                        )
                    rows[index] = scores
                    successful_http_attempts[index] += http_attempts
                    response_latency[index] += latency
                except Exception as error:
                    index = futures[future]
                    last_errors[index] = (
                        f"{type(error).__name__}: {' '.join(str(error).split())[:1000]}"
                    )
                    failed = True
        if (
            failed
            and any((row is None for row in rows))
            and (retry_until_success or group_attempt < attempts)
        ):
            print(
                f"RUBRIC_JUDGE_MISSING_ROWS_RETRY round={group_attempt} missing_rows={[i for i, row in enumerate(rows) if row is None]} kept_rows={sum((row is not None for row in rows))}",
                flush=True,
            )
            time.sleep(min(30.0, 2 ** min(group_attempt - 1, 4)))
    if any((row is None for row in rows)):
        missing = [index for index, row in enumerate(rows) if row is None]
        raise JudgeGiveUpError(
            f"fixed Judge left response rows {missing} incomplete after {attempts} attempts; last_errors={{{', '.join((f'{index}: {last_errors[index]}' for index in missing))}}}"
        )
    completed_rows = [list(row or []) for row in rows]
    return (
        completed_rows,
        {
            "complete_group_attempts": group_attempt,
            "response_task_attempts": sum(response_task_attempts),
            "successful_http_attempts": sum(successful_http_attempts),
            "retried_response_count": sum(
                (value > 1 for value in response_task_attempts)
            ),
            "latency_seconds": round(time.monotonic() - started, 3),
            "per_response": [
                {
                    "response_id": str(member["id"]),
                    "task_attempts": response_task_attempts[index],
                    "successful_http_attempts": successful_http_attempts[index],
                    "latency_seconds": round(response_latency[index], 3),
                }
                for index, member in enumerate(members)
            ],
        },
    )
