"""JobManager tests — lifecycle, events, cancellation, capacity (§19/§20/§22)."""

import asyncio

import pytest

from curator.runtime.job_manager import JobManager


@pytest.mark.asyncio
async def test_successful_job_lifecycle():
    mgr = JobManager()
    events = []
    job = mgr.create(operation="op")

    async def work():
        return {"datasetId": "ds-1"}

    async def on_event(event, job):
        events.append((event, job.status))

    await mgr.spawn(job, work, on_event)
    assert job.status == "completed"
    assert job.result == {"datasetId": "ds-1"}
    assert [e[0] for e in events] == ["job.started", "job.completed"]


@pytest.mark.asyncio
async def test_failed_job_reports_error():
    mgr = JobManager()
    job = mgr.create(operation="op")

    async def work():
        raise RuntimeError("boom")

    async def on_event(event, job):
        pass

    await mgr.spawn(job, work, on_event)
    assert job.status == "failed"
    assert "boom" in job.error


@pytest.mark.asyncio
async def test_cancel_marks_job_cancelled():
    mgr = JobManager()
    events = []
    job = mgr.create(operation="op")

    async def work():
        await asyncio.sleep(10)

    async def on_event(event, job):
        events.append(event)

    task = mgr.spawn(job, work, on_event)
    await asyncio.sleep(0.05)
    assert await mgr.cancel(job.job_id) is True
    await asyncio.wait_for(task, timeout=2)
    assert job.status == "cancelled"
    assert "job.cancelled" in events


@pytest.mark.asyncio
async def test_capacity_model_rejects_overflow():
    mgr = JobManager(max_concurrent_jobs=1)

    async def work():
        await asyncio.sleep(10)

    async def on_event(event, job):
        pass

    first = mgr.create(operation="a")
    mgr.spawn(first, work, on_event)
    with pytest.raises(RuntimeError, match="capacity"):
        mgr.create(operation="c")


def test_snapshot_counts():
    mgr = JobManager()
    mgr.create(operation="a")
    snapshot = mgr.snapshot()
    assert snapshot["totalJobs"] == 1
    assert snapshot["activeJobs"] == 1  # queued counts as in-flight
    assert snapshot["queueDepth"] == 1
