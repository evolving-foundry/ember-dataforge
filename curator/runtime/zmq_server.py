"""ZmqRuntimeServer (doc §17/§18) — transport layer of the Python runtime.

A long-lived DEALER peer connected to the TS supervisor's ROUTER socket.
Message loop per doc §18: recv → dispatch as a task, so long-running
operations never block the loop. All protocol handling lives in the
dispatcher; all execution lives in the executor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any, Dict, Optional

import zmq.asyncio

from . import PROTOCOL, PROTOCOL_VERSION
from .dataset_registry import DatasetRegistry
from .dispatcher import RuntimeDispatcher
from .events import EventPublisher
from .executor import PipelineExecutor
from .job_manager import JobManager

logger = logging.getLogger("runtime.zmq_server")

DEFAULT_COMMAND_ENDPOINT = "ipc:///tmp/dataforge-runtime.sock"
HEARTBEAT_INTERVAL_S = 5.0


class ZmqRuntimeServer:
    def __init__(
        self,
        runtime_id: str,
        command_endpoint: str = DEFAULT_COMMAND_ENDPOINT,
        events_endpoint: Optional[str] = None,
        max_concurrent_jobs: int = 4,
    ) -> None:
        self.runtime_id = runtime_id
        self.command_endpoint = command_endpoint
        self.events_endpoint = events_endpoint
        self.max_concurrent_jobs = max_concurrent_jobs

        self.ctx = zmq.asyncio.Context.instance()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, runtime_id)

        self.publisher = EventPublisher(dealer_socket=None)
        self.jobs = JobManager(max_concurrent_jobs=max_concurrent_jobs)
        self.datasets = DatasetRegistry(
            on_created=self._on_dataset_created,
            on_released=self._on_dataset_released,
        )
        self.executor = PipelineExecutor(self.datasets, self.publisher, runtime_id)
        self.dispatcher = RuntimeDispatcher(runtime_id)
        self._register_handlers()

        self._running = False
        self._pending: set = set()
        self._heartbeat_task: Optional[asyncio.Task] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        self.socket.connect(self.command_endpoint)
        self.publisher = EventPublisher(
            dealer_socket=self.socket, events_endpoint=self.events_endpoint
        )
        self.executor.publisher = self.publisher
        self._running = True
        logger.info(
            "runtime %s connecting to %s (events: %s)",
            self.runtime_id,
            self.command_endpoint,
            self.events_endpoint or "multiplexed over command channel",
        )
        await self._announce()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        await self._run_loop()

    async def stop(self) -> None:
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        for task in list(self._pending):
            task.cancel()
        await self.publisher.close()
        self.socket.close(0)

    # ── registration / heartbeat (doc §15/§21) ───────────────────────────

    async def _announce(self) -> None:
        hello_request = {
            "protocol": PROTOCOL,
            "version": PROTOCOL_VERSION,
            "messageId": f"msg-hello-{self.runtime_id}",
            "type": "request",
            "method": "runtime.hello",
            "runtimeId": self.runtime_id,
            "timestamp": 0,
            "payload": {
                "runtimeId": self.runtime_id,
                "capabilities": self._capabilities(),
                "resources": self._resources(),
            },
        }
        await self.socket.send_multipart(
            [json.dumps(hello_request, default=str).encode("utf-8")]
        )

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            snapshot = self.jobs.snapshot()
            await self.publisher.publish(
                "runtime.heartbeat",
                {
                    "runtimeId": self.runtime_id,
                    "activeJobs": snapshot["activeJobs"],
                    "queueDepth": snapshot["queueDepth"],
                },
                runtime_id=self.runtime_id,
            )

    # ── message loop (doc §18) ───────────────────────────────────────────

    async def _run_loop(self) -> None:
        while self._running:
            try:
                frames = await self.socket.recv_multipart()
            except zmq.error.ZMQError:
                if self._running:
                    raise
                return
            task = asyncio.create_task(self._handle_frames(frames))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    async def _handle_frames(self, frames) -> None:
        raw = frames[0] if frames else b""
        try:
            message = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("undecodable frame dropped (%d bytes)", len(raw))
            return

        # Out-of-band cancellation frames (type "cancel", doc §10/§20).
        if isinstance(message, dict) and message.get("type") == "cancel":
            job_id = message.get("jobId") or (message.get("payload") or {}).get("jobId")
            if job_id:
                await self.jobs.cancel(str(job_id))
            return

        response = await self.dispatcher.dispatch(message)
        if response is not None:
            await self.socket.send_multipart(
                [json.dumps(response, default=str).encode("utf-8")]
            )

    # ── handler wiring (§34 P0 method set) ───────────────────────────────

    def _register_handlers(self) -> None:
        self.dispatcher.register("runtime.health", self._handle_health)
        self.dispatcher.register("operation.execute", self._handle_operation_execute)
        self.dispatcher.register("pipeline.execute", self._handle_pipeline_execute)
        self.dispatcher.register("job.cancel", self._handle_job_cancel)
        self.dispatcher.register("dataset.inspect", self._handle_dataset_inspect)
        self.dispatcher.register("dataset.release", self._handle_dataset_release)

    async def _handle_health(self, message: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = self.jobs.snapshot()
        return {
            "runtimeId": self.runtime_id,
            "healthy": True,
            "activeJobs": snapshot["activeJobs"],
            "queueDepth": snapshot["queueDepth"],
            "datasets": len(self.datasets.ids()),
        }

    async def _handle_operation_execute(self, message: Dict[str, Any]) -> Dict[str, Any]:
        payload = message.get("payload") or {}
        operation = payload.get("operation") or {}
        name = operation.get("name")
        if not name:
            raise ValueError("payload.operation.name is required")
        from curator.core.operation_registry import OperationRegistry

        if OperationRegistry.get_operation_class(name) is None:
            raise ValueError(f"Unknown operation: {name}")
        params = operation.get("params") or {}
        input_ref = payload.get("input") or {}
        input_dataset_id = input_ref.get("datasetId")
        await_result = bool(payload.get("awaitResult", True))
        requested_job_id = payload.get("jobId") or message.get("jobId")

        job = self.jobs.create(operation=name, job_id=requested_job_id)

        async def work() -> Dict[str, Any]:
            return await self.executor.execute_operation(
                name,
                params,
                input_dataset_id,
            )

        async def on_event(event: str, job) -> None:
            await self.publisher.emit_job_event(
                event,
                job.job_id,
                job.operation,
                job.status,
                runtime_id=self.runtime_id,
                result=job.result,
                error={"code": "JOB_FAILED", "message": job.error} if job.error else None,
            )

        if await_result:
            # Synchronous experience for short operations (doc §19).
            await self.jobs.spawn(job, work, on_event)
            return {
                "jobId": job.job_id,
                "status": job.status,
                "output": {"datasetId": (job.result or {}).get("datasetId")}
                if job.result
                else None,
                "metrics": (job.result or {}).get("metrics"),
                "error": job.error,
            }

        self.jobs.spawn(job, work, on_event)
        return {"jobId": job.job_id, "status": job.status}

    async def _handle_pipeline_execute(self, message: Dict[str, Any]) -> Dict[str, Any]:
        payload = message.get("payload") or {}
        config = payload.get("config")
        config_path = payload.get("configPath")
        if config is None and config_path is None:
            raise ValueError("payload.config or payload.configPath is required")

        job = self.jobs.create(operation="pipeline")

        async def work() -> Dict[str, Any]:
            return await self.executor.execute_pipeline(
                config if config is not None else config_path,
                is_path=config is None,
            )

        async def on_event(event: str, job) -> None:
            await self.publisher.emit_job_event(
                event,
                job.job_id,
                job.operation,
                job.status,
                runtime_id=self.runtime_id,
                result=job.result,
                error={"code": "JOB_FAILED", "message": job.error} if job.error else None,
            )

        await self.jobs.spawn(job, work, on_event)
        return {
            "jobId": job.job_id,
            "status": job.status,
            "output": {"datasetId": (job.result or {}).get("datasetId")}
            if job.result
            else None,
            "metrics": (job.result or {}).get("metrics"),
            "error": job.error,
        }

    async def _handle_job_cancel(self, message: Dict[str, Any]) -> Dict[str, Any]:
        payload = message.get("payload") or {}
        job_id = payload.get("jobId")
        if not job_id:
            raise ValueError("payload.jobId is required")
        cancelled = await self.jobs.cancel(str(job_id))
        return {"jobId": str(job_id), "cancelled": bool(cancelled)}

    async def _handle_dataset_inspect(self, message: Dict[str, Any]) -> Dict[str, Any]:
        payload = message.get("payload") or {}
        dataset_id = payload.get("datasetId")
        if not dataset_id:
            raise ValueError("payload.datasetId is required")
        return self.datasets.inspect(str(dataset_id))

    async def _handle_dataset_release(self, message: Dict[str, Any]) -> Dict[str, Any]:
        payload = message.get("payload") or {}
        dataset_id = payload.get("datasetId")
        if not dataset_id:
            raise ValueError("payload.datasetId is required")
        released = self.datasets.release(str(dataset_id))
        return {"datasetId": str(dataset_id), "released": bool(released)}

    # ── dataset lifecycle events (doc §8 dataset.created / released) ─────

    def _on_dataset_created(self, dataset_id: str, backend: str) -> None:
        task = asyncio.create_task(
            self.publisher.publish(
                "dataset.created",
                {"dataset": {"id": dataset_id, "runtimeId": self.runtime_id, "backend": backend}},
                runtime_id=self.runtime_id,
            )
        )
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def _on_dataset_released(self, dataset_id: str, backend: str, reason: str) -> None:
        task = asyncio.create_task(
            self.publisher.publish(
                "dataset.released",
                {
                    "dataset": {"id": dataset_id, "runtimeId": self.runtime_id, "backend": backend},
                    "reason": reason,
                },
                runtime_id=self.runtime_id,
            )
        )
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    # ── env-driven identity (§15/§25: endpoint is deployment config) ─────

    def _capabilities(self) -> list:
        raw = os.environ.get("DATAFORGE_CAPABILITIES")
        try:
            return json.loads(raw) if raw else ["ray-data"]
        except json.JSONDecodeError:
            return ["ray-data"]

    def _resources(self) -> Dict[str, Any]:
        raw = os.environ.get("DATAFORGE_RESOURCES")
        try:
            return json.loads(raw) if raw else {"cpu": 8, "memoryGiB": 16}
        except json.JSONDecodeError:
            return {"cpu": 8, "memoryGiB": 16}


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("DATAFORGE_LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    runtime_id = os.environ.get("DATAFORGE_RUNTIME_ID") or "python-default"
    command_endpoint = os.environ.get("DATAFORGE_COMMAND_ENDPOINT", DEFAULT_COMMAND_ENDPOINT)
    events_endpoint = os.environ.get("DATAFORGE_EVENTS_ENDPOINT") or None
    max_concurrent_jobs = int(os.environ.get("DATAFORGE_MAX_CONCURRENT_JOBS", "4"))

    server = ZmqRuntimeServer(
        runtime_id=runtime_id,
        command_endpoint=command_endpoint,
        events_endpoint=events_endpoint,
        max_concurrent_jobs=max_concurrent_jobs,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop = loop.create_future()

    def _signal(*_args) -> None:
        if not stop.done():
            stop.set_result(None)

    loop.add_signal_handler(signal.SIGTERM, _signal)
    loop.add_signal_handler(signal.SIGINT, _signal)

    async def _serve() -> None:
        serve_task = asyncio.ensure_future(server.start())
        stop_task = asyncio.ensure_future(stop)
        await asyncio.wait(
            {serve_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        serve_task.cancel()
        await server.stop()

    try:
        loop.run_until_complete(_serve())
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
