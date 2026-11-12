"""Group orchestration tests: slot deduplication, score-on-full, explicit error on group
timeout.
"""

from __future__ import annotations

import threading

import pytest

from data_designer.rubric.orchestrate.group import (
    ALLOWED_GROUP_SIZES,
    GroupCollector,
    GroupSlotError,
    GroupTimeoutError,
    discriminative_filter_range,
    resolve_group_size,
)


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _ready_collector(group_size=8):
    ready = []
    collector = GroupCollector(
        group_size=group_size,
        on_ready=lambda key, rows: ready.append((key, rows)),
    )
    return collector, ready


def test_submit_completes_group_and_orders_slots():
    collector, ready = _ready_collector()
    for slot in reversed(range(8)):
        collector.submit("g1", slot, {"response_id": f"r{slot}"})
    assert len(ready) == 1
    key, rows = ready[0]
    assert key == "g1"
    assert [row["response_id"] for row in rows] == [f"r{i}" for i in range(8)]
    assert collector.pending_group_keys() == []


def test_duplicate_slot_rejected():
    collector, _ = _ready_collector()
    collector.submit("g1", 0, {"response_id": "r0"})
    with pytest.raises(GroupSlotError):
        collector.submit("g1", 0, {"response_id": "r0-dup"})


def test_invalid_slot_rejected():
    collector, _ = _ready_collector()
    with pytest.raises(GroupSlotError):
        collector.submit("g1", 8, {})
    with pytest.raises(GroupSlotError):
        collector.submit("g1", -1, {})
    with pytest.raises(GroupSlotError):
        collector.submit("g1", True, {})


def test_invalid_group_size_rejected():
    for size in (7, 0, -1):
        with pytest.raises(ValueError):
            GroupCollector(group_size=size)
    for size in ALLOWED_GROUP_SIZES:
        GroupCollector(group_size=size)


def test_timeout_clears_group_and_raises_by_default():
    clock = _Clock()
    collector = GroupCollector(
        group_size=8, timeout_seconds=10.0, clock=clock, on_ready=None
    )
    collector.submit("g1", 0, {"response_id": "r0"})
    clock.now += 11.0
    with pytest.raises(GroupTimeoutError):
        collector.expire_stale()
    assert collector.pending_group_keys() == []
    # After clearing a group, the same key can group again
    collector.submit("g1", 0, {"response_id": "r0"})
    assert collector.pending_group_keys() == ["g1"]


def test_timeout_callback_instead_of_raise():
    clock = _Clock()
    timed_out = []
    collector = GroupCollector(
        group_size=4 if 4 in ALLOWED_GROUP_SIZES else 6,
        timeout_seconds=5.0,
        clock=clock,
        on_timeout=lambda key, rows: timed_out.append((key, len(rows))),
    )
    size = collector.group_size
    for slot in range(size - 1):
        collector.submit("g1", slot, {"response_id": f"r{slot}"})
    clock.now += 6.0
    cleared = collector.expire_stale()
    assert cleared == ["g1"]
    assert timed_out == [("g1", size - 1)]


def test_groups_are_independent():
    collector, ready = _ready_collector()
    for key in ("g1", "g2"):
        for slot in range(8):
            collector.submit(key, slot, {"response_id": f"{key}-r{slot}"})
    assert [key for key, _ in ready] == ["g1", "g2"]


def _locked_ready_sink(ready: list):
    callback_lock = threading.Lock()

    def on_ready(key, rows):
        with callback_lock:
            ready.append((key, rows))

    return on_ready


def test_concurrent_submit_delivers_each_row_exactly_once():
    """Concurrent grouping regression: 200 threads contend for the last empty slot;
    there must be exactly one callback and one successful submission. The source
    group_reward_loop.py protects grouping state with asyncio.Lock (L66/L181-200); an
    unlocked check-then-act race lets multiple threads pass slot deduplication
    simultaneously, overwrite each other, and each trigger the full-group callback
    (rows silently lost) -- this test guards the equivalent threading.Lock semantics.
    """
    ready: list[tuple[str, list]] = []
    collector = GroupCollector(group_size=8, on_ready=_locked_ready_sink(ready))
    for slot in range(7):
        collector.submit("g1", slot, {"response_id": f"r{slot}"})

    accepted = []
    accepted_lock = threading.Lock()

    def worker(i: int) -> None:
        try:
            collector.submit("g1", 7, {"response_id": f"last-{i}"})
            with accepted_lock:
                accepted.append(i)
        except GroupSlotError:
            pass  # Slot already occupied: late thread exits legitimately

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ready) == 1, f"expected exactly one ready callback, got {len(ready)}"
    assert len(accepted) >= 1, "no thread managed to fill the last slot"
    key, rows = ready[0]
    assert key == "g1"
    last_rows = [row for row in rows if row["response_id"].startswith("last-")]
    assert [row["response_id"] for row in rows[:7]] == [f"r{i}" for i in range(7)]
    assert len(rows) == 8
    assert len(last_rows) == 1, (
        "the last slot must be delivered exactly once — duplicates mean the "
        "dedup check raced and a row was silently overwritten"
    )


def test_concurrent_submit_across_many_groups():
    """40 groups x 8 slots (slots differ within each group) fill concurrently: exactly
    one callback per group, no rows lost.
    """
    ready: list[tuple[str, list]] = []
    collector = GroupCollector(group_size=8, on_ready=_locked_ready_sink(ready))

    def worker(i: int) -> None:
        collector.submit(f"g{i % 40}", i // 40, {"response_id": f"r{i}"})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(320)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ready) == 40, f"expected one callback per group, got {len(ready)}"
    assert {key for key, _ in ready} == {f"g{i}" for i in range(40)}
    for key, rows in ready:
        assert len({row["response_id"] for row in rows}) == 8
    assert collector.pending_group_keys() == []


def test_discriminative_filter_range():
    assert discriminative_filter_range(8) == (2, 6)
    assert discriminative_filter_range(6) == (2, 4)
    assert discriminative_filter_range(10) == (3, 7)
    assert discriminative_filter_range(16) == (4, 12)


def test_env_group_size_resolution(monkeypatch):
    assert resolve_group_size() == 8
    assert resolve_group_size(6) == 6
    monkeypatch.setenv("RUBRIC_INFER_GROUP_SIZE", "16")
    assert resolve_group_size() == 16
    assert GroupCollector().group_size == 16
    monkeypatch.setenv("RUBRIC_INFER_GROUP_SIZE", "7")
    with pytest.raises(ValueError):
        resolve_group_size()
    with pytest.raises(ValueError):
        resolve_group_size(True)


def test_env_filter_bounds_override(monkeypatch):
    monkeypatch.setenv("RUBRIC_INFER_FILTER_LOW", "1")
    monkeypatch.setenv("RUBRIC_INFER_FILTER_HIGH", "5")
    assert discriminative_filter_range(6) == (1, 5)
    monkeypatch.delenv("RUBRIC_INFER_FILTER_HIGH")
    with pytest.raises(ValueError):
        # Only one bound set -> both must be set together
        discriminative_filter_range(6)
    monkeypatch.setenv("RUBRIC_INFER_FILTER_HIGH", "0")
    with pytest.raises(ValueError):
        # low > high
        discriminative_filter_range(6)
    monkeypatch.setenv("RUBRIC_INFER_FILTER_LOW", "0")
    monkeypatch.setenv("RUBRIC_INFER_FILTER_HIGH", "9")
    with pytest.raises(ValueError):
        # Exceeds group size
        discriminative_filter_range(6)
