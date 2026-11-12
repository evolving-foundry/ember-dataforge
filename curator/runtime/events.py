"""EventPublisher (doc §8/§17) — runtime events Python → TS.

Two modes:
  - PUB socket bound to DATAFORGE_EVENTS_ENDPOINT, frames [topic, envelope]
    where topic is the event method name (§8).
  - Fallback (P0): events multiplexed over the DEALER command channel as
    envelopes with type "event" (doc §9).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

import zmq.asyncio

from . import PROTOCOL, PROTOCOL_VERSION

logger = logging.getLogger(__name__)


class EventPublisher:
    def __init__(
        self,
        dealer_socket: Optional[Any] = None,
        events_endpoint: Optional[str] = None,
    ) -> None:
        self._dealer = dealer_socket
        self._pub: Optional[zmq.asyncio.Socket] = None
        if events_endpoint:
            ctx = zmq.asyncio.Context.instance()
            self._pub = ctx.socket(zmq.PUB)
            self._pub.bind(events_endpoint)
            logger.info("event publisher bound to %s", events_endpoint)

    async def publish(
        self,
        method: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        runtime_id: Optional[str] = None,
        job_id: Optional[str] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        envelope: Dict[str, Any] = {
            "protocol": PROTOCOL,
            "version": PROTOCOL_VERSION,
            "messageId": f"msg-{uuid.uuid4()}",
            "type": "event",
            "method": method,
            "timestamp": int(time.time() * 1000),
            "payload": payload or {},
        }
        if runtime_id:
            envelope["runtimeId"] = runtime_id
        if job_id:
            envelope["jobId"] = job_id
        if error:
            envelope["error"] = error

        raw = json.dumps(envelope, ensure_ascii=False, default=str).encode("utf-8")
        if self._pub is not None:
            await self._pub.send_multipart([method.encode("utf-8"), raw])
        elif self._dealer is not None:
            await self._dealer.send_multipart([raw])

    async def emit_job_event(
        self,
        event: str,
        job_id: str,
        operation: Optional[str],
        status: Optional[str] = None,
        *,
        runtime_id: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {"jobId": job_id}
        if operation:
            payload["operation"] = operation
        if status:
            payload["status"] = status
        if result is not None:
            payload["result"] = result
        if metrics is not None:
            payload["metrics"] = metrics
        await self.publish(
            event,
            payload,
            runtime_id=runtime_id,
            job_id=job_id,
            error=error,
        )

    async def close(self) -> None:
        if self._pub is not None:
            self._pub.close(0)
            self._pub = None
