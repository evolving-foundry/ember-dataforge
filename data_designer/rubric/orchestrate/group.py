"""Group orchestration (semantically aligned with RuACT group_reward_loop.py's
GroupMatrixRewardLoopManager; independent of verl: inputs are plain dict rows).
Source discipline (README.md:137): - group by query and deduplicate rollout_index
slots (0..n-1 exactly once each); - trigger callback scoring as soon as a group is
full; - group-level timeout (default 600s): on timeout clear the whole group and
raise explicitly -- **never degrade to all-zero scores**; when Judge is unavailable,
the iteration round must fail explicitly rather than silently accepting/rejecting
every sample. Group size defaults to 8 (the basis for the discriminative range
[2,6]); if business sampling cost is constrained, n in {6,8,10,16} is allowed
(section 6.2), but it must be fixed within each business domain and immutable across
rounds, otherwise pass_count distributions lose comparability.
"""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Callable, Mapping

ALLOWED_GROUP_SIZES = (6, 8, 10, 16)
DEFAULT_GROUP_SIZE = 8
DEFAULT_GROUP_TIMEOUT_SECONDS = 600.0


class GroupTimeoutError(RuntimeError):
    pass


class GroupSlotError(ValueError):
    pass


def _raise_group_timeout(group_key: str, rows: list) -> None:
    raise GroupTimeoutError(
        f"rubric group {group_key!r} did not reach full size within timeout; "
        f"cleared {len(rows)} stranded rows (no zero-score fallback)"
    )


def resolve_group_size(group_size: int | None = None) -> int:
    """Resolve group size: explicit parameter wins; otherwise read
    RUBRIC_INFER_GROUP_SIZE (default 8). Group size must be fixed within each
    business domain and immutable across rounds (section 6.2); the env var only
    provides a process-level default.
    """
    if group_size is None:
        group_size = int(os.environ.get("RUBRIC_INFER_GROUP_SIZE", str(DEFAULT_GROUP_SIZE)))
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size not in ALLOWED_GROUP_SIZES
    ):
        raise ValueError(
            f"group_size must be one of {ALLOWED_GROUP_SIZES}, got {group_size!r}"
        )
    return group_size


def discriminative_filter_range(group_size: int) -> tuple[int, int]:
    """Discriminative filter range for the group size [ceil(0.25n), floor(0.75n)]; [2,6]
    when n=8. RUBRIC_INFER_FILTER_LOW/HIGH may explicitly override it (both must be
    set together and fall within the group).
    """
    low_env = os.environ.get("RUBRIC_INFER_FILTER_LOW")
    high_env = os.environ.get("RUBRIC_INFER_FILTER_HIGH")
    if (low_env is None) != (high_env is None):
        raise ValueError(
            "RUBRIC_INFER_FILTER_LOW and RUBRIC_INFER_FILTER_HIGH must be set together"
        )
    if low_env is not None:
        low, high = int(low_env), int(high_env)
        if not 0 <= low <= high <= group_size:
            raise ValueError(
                f"filter bounds must satisfy 0 <= low <= high <= {group_size}: got [{low}, {high}]"
            )
        return (low, high)
    return (math.ceil(0.25 * group_size), math.floor(0.75 * group_size))


class GroupCollector:
    """Synchronous collector that groups rows by group_key. `on_ready(group_key, rows)`
    runs when the group fills, in ascending rollout_index order;
    `on_timeout(group_key, rows)` defaults to raising :class:`GroupTimeoutError`.
    """

    def __init__(
        self,
        *,
        group_size: int | None = None,
        timeout_seconds: float = DEFAULT_GROUP_TIMEOUT_SECONDS,
        on_ready: Callable[[str, list[Mapping[str, Any]]], Any] | None = None,
        on_timeout: Callable[[str, list[Mapping[str, Any]]], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.group_size = resolve_group_size(group_size)
        if timeout_seconds <= 0:
            raise ValueError("group timeout must be positive")
        self.timeout_seconds = timeout_seconds
        self._on_ready = on_ready
        self._on_timeout = on_timeout or _raise_group_timeout
        self._clock = clock
        # The source group_reward_loop.py protects grouping state with asyncio.Lock
        # (L66/L181-200);
        # this synchronous form keeps equivalent semantics with threading.Lock:
        # submit/expire_stale may be called concurrently by multiple threads.
        self._lock = threading.Lock()
        self._pending: dict[str, dict[int, Mapping[str, Any]]] = {}
        self._deadlines: dict[str, float] = {}

    def pending_group_keys(self) -> list[str]:
        return sorted(self._pending)

    def submit(self, group_key: str, rollout_index: int, row: Mapping[str, Any]) -> None:
        """Accept one completed sample; when the group fills, invoke on_ready and clear
        the group. Thread safety: grouping state is updated while holding the lock,
        and the on_ready callback runs outside the lock (the callback may safely call
        submit/expire_stale again, and a blocked callback does not hold the lock).
        """
        if (
            isinstance(rollout_index, bool)
            or not isinstance(rollout_index, int)
            or not 0 <= rollout_index < self.group_size
        ):
            raise GroupSlotError(
                f"invalid rollout_index {rollout_index!r} for group {group_key!r}"
            )
        ready: list[Mapping[str, Any]] | None = None
        with self._lock:
            group = self._pending.setdefault(group_key, {})
            if rollout_index in group:
                raise GroupSlotError(
                    f"duplicate rollout_index {rollout_index} for group {group_key!r}"
                )
            self._deadlines.setdefault(group_key, self._clock() + self.timeout_seconds)
            group[rollout_index] = row
            if len(group) == self.group_size:
                ready = self._pop(group_key)
        if ready is not None and self._on_ready is not None:
            self._on_ready(group_key, ready)

    def expire_stale(self) -> list[str]:
        """Clear groups that timed out before filling and trigger on_timeout; return the
        cleared group keys.
        """
        now = self._clock()
        expired: list[tuple[str, list[Mapping[str, Any]]]] = []
        with self._lock:
            expired_keys = [
                key for key, deadline in self._deadlines.items() if now >= deadline
            ]
            expired = [(key, self._pop(key)) for key in expired_keys]
        for key, rows in expired:
            if self._on_timeout is not None:
                self._on_timeout(key, rows)
        return [key for key, _ in expired]

    def _pop(self, group_key: str) -> list[Mapping[str, Any]]:
        group = self._pending.pop(group_key, {})
        self._deadlines.pop(group_key, None)
        return [group[slot] for slot in sorted(group)]
