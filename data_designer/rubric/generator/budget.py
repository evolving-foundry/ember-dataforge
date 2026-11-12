"""Parent-level budget retries (migrated from RuACT refinement/bounded_retry.py
`prepare_parent`). Source policies retained: - A single Generator call is at most 40s
(`timeout_seconds=min(config.timeout_seconds, 40)`) with `max_attempts=1` -- retries
are counted explicitly by this layer (default 3), and failure text is fed into the
next prompt ('previous attempt failed structural validation'). - Whole-parent
preparation is bounded by `PARENT_BUDGET_SECONDS` (120s, deadline passed to
generate_json, shrinking synchronously for each call). - Budget exhaustion raises
:class:`RefinementBudgetExhausted` -- **no parent fallback and no fabricated 'no
candidates' success** (source bounded_retry.py:97-99). Deviations from the source
(intentional on the inference side, decoupled from asyncio): - `asyncio.wait_for` ->
synchronous deadline checks (generate_json natively supports a deadline). - Removed
global concurrency gates `per_job_limit` / `legacy._parent_semaphore` -- parent-level
parallelism is determined by the caller's thread-pool width (see
single_pass.process_refinement_job).
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable, Mapping

from ..protocol.schemas import PositiveRefinementProtocolError
from .client import WorkerConfig, WorkerError, generate_json

PARENT_ATTEMPTS = 3
PARENT_BUDGET_SECONDS = 120.0
PARENT_SINGLE_CALL_SECONDS = 40.0


class RefinementBudgetExhausted(RuntimeError):
    """Parent candidate preparation exhausted its budget -- fail explicitly; never fall
    back to 'no decomposition'.
    """


def prepare_parent(
    job: Mapping[str, Any],
    parent: Mapping[str, Any],
    config: WorkerConfig,
    prepare: Callable[
        [Mapping[str, Any], Mapping[str, Any], WorkerConfig, Callable[..., Any]],
        tuple[list[dict[str, Any]], dict[str, Any]],
    ],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run candidate preparation for one parent with budgeted retries. `prepare(job,
    parent, config, generator_call)` returns (candidate pool, audit);
    `generator_call(prompt, schema, config=...)` is injected by this layer -- it
    appends the previous failure text and forwards to :func:`client.generate_json`
    (deadline shrinks synchronously).
    """
    started = time.monotonic()
    deadline = started + PARENT_BUDGET_SECONDS
    bounded_config = replace(
        config,
        max_attempts=1,
        timeout_seconds=min(config.timeout_seconds, PARENT_SINGLE_CALL_SECONDS),
    )
    errors: list[str] = []
    attempts = 0

    def generator_call(
        prompt: str,
        schema: Mapping[str, Any],
        *,
        config: WorkerConfig = bounded_config,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if errors:
            prompt += (
                "\nThe previous attempt failed structural validation: "
                + errors[-1]
                + "\nReturn exactly the requested number of distinct candidates in the candidates array. Do not return an explanation or a different JSON shape."
            )
        return generate_json(prompt, schema, config=config, deadline=deadline)

    last_error: BaseException | None = None
    for attempt in range(1, PARENT_ATTEMPTS + 1):
        attempts = attempt
        try:
            result = prepare(job, parent, bounded_config, generator_call)
            pool, audit = result
            audit.update(
                preparation_attempts=attempts,
                preparation_errors=errors,
                preparation_seconds=time.monotonic() - started,
            )
            audit["generator_call_count"] = int(audit.get("generator_call_count", 0)) + attempts - 1
            return (pool, audit)
        except (WorkerError, PositiveRefinementProtocolError) as error:
            last_error = error
            errors.append(str(error)[:500])
            if attempt < PARENT_ATTEMPTS:
                time.sleep(min(2.0, config.retry_backoff_seconds))
    raise RefinementBudgetExhausted(
        f"parent attempt budget exhausted after {attempts} attempts "
        f"({type(last_error).__name__}: {errors[-1] if errors else last_error})"
    ) from last_error


__all__ = [
    "PARENT_ATTEMPTS",
    "PARENT_BUDGET_SECONDS",
    "PARENT_SINGLE_CALL_SECONDS",
    "RefinementBudgetExhausted",
    "prepare_parent",
]
