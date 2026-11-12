"""JobManager (doc §19/§20) — async job lifecycle for the Python runtime.

Because the protocol is async, ``operation.execute`` no longer means "wait
until all computation finishes". Jobs carry a state machine
(queued → running → completed | failed | cancelled) and can be cancelled
best-effort at runtime level (§20: not by killing the process).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

JOB_STATUSES = ("queued", "running", "completed", "failed", "cancelled")


@dataclass
class Job:
    job_id: str
    operation: Optional[str] = None
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None

    def to_ref(self) -> Dict[str, Any]:
        return {
            "jobId": self.job_id,
            "operation": self.operation,
            "status": self.status,
        }


class JobManager:
    def __init__(self, max_concurrent_jobs: int = 4) -> None:
        self.max_concurrent_jobs = max_concurrent_jobs
        self._jobs: Dict[str, Job] = {}

    def create(self, operation: Optional[str] = None, job_id: Optional[str] = None) -> Job:
        if len(self.active_jobs()) >= self.max_concurrent_jobs:
            raise RuntimeError(
                f"runtime at capacity: maxConcurrentJobs={self.max_concurrent_jobs}"
            )
        job = Job(job_id=job_id or f"job-{uuid.uuid4()}", operation=operation)
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def active_jobs(self) -> Dict[str, Job]:
        return {
            job_id: job
            for job_id, job in self._jobs.items()
            if job.status in ("queued", "running")
        }

    def queue_depth(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "queued")

    def spawn(
        self,
        job: Job,
        work: Callable[[], Awaitable[Dict[str, Any]]],
        on_event: Callable[[str, Job], Awaitable[None]],
    ) -> asyncio.Task:
        """Run ``work`` under the job state machine, emitting lifecycle events."""

        async def _run() -> None:
            job.status = "running"
            await on_event("job.started", job)
            try:
                job.result = await work()
                job.status = "completed"
                await on_event("job.completed", job)
            except asyncio.CancelledError:
                job.status = "cancelled"
                await on_event("job.cancelled", job)
                # Swallow: cancellation is a normal outcome, not a failure.
            except Exception as exc:  # noqa: BLE001 — boundary: report via protocol
                job.status = "failed"
                job.error = str(exc)
                await on_event("job.failed", job)
                logger.exception("job %s failed", job.job_id)
            finally:
                job.finished_at = time.time()

        job.task = asyncio.create_task(_run(), name=f"dataforge-job-{job.job_id}")
        return job.task

    async def cancel(self, job_id: str) -> bool:
        """Best-effort cancellation (doc §20). Returns True if a running
        job was found and cancellation was initiated."""
        job = self._jobs.get(job_id)
        if job is None or job.task is None:
            return False
        if job.status in ("completed", "failed", "cancelled"):
            return False
        job.task.cancel()
        return True

    def snapshot(self) -> Dict[str, Any]:
        return {
            "activeJobs": len(self.active_jobs()),
            "queueDepth": self.queue_depth(),
            "totalJobs": len(self._jobs),
        }
