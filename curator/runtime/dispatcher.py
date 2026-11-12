"""RuntimeDispatcher (doc §17) — protocol layer of the Python runtime.

Validates the RuntimeMessage envelope and routes request methods to
handlers. Transport (ZmqRuntimeServer) and execution (PipelineExecutor)
stay separate (doc §17: do not mix the three layers).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from . import PROTOCOL, PROTOCOL_VERSION

logger = logging.getLogger(__name__)


class ProtocolError(Exception):
    def __init__(self, message: str, code: str = "PROTOCOL_ERROR") -> None:
        super().__init__(message)
        self.code = code


def validate_envelope(message: Dict[str, Any]) -> None:
    if not isinstance(message, dict):
        raise ProtocolError("envelope must be a JSON object")
    if message.get("protocol") != PROTOCOL:
        raise ProtocolError(f"unknown protocol: {message.get('protocol')!r}", "UNKNOWN_PROTOCOL")
    if message.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported version: {message.get('version')!r}", "UNSUPPORTED_VERSION")
    if not isinstance(message.get("messageId"), str):
        raise ProtocolError("messageId is required")
    if message.get("type") not in ("request", "response", "event", "cancel", "heartbeat"):
        raise ProtocolError(f"unknown type: {message.get('type')!r}", "UNKNOWN_TYPE")


def build_response(
    request: Dict[str, Any],
    payload: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    runtime_id: str = "",
) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "version": PROTOCOL_VERSION,
        "messageId": f"msg-{uuid.uuid4()}",
        "type": "response",
        "correlationId": request.get("messageId"),
        "method": request.get("method"),
        "timestamp": int(time.time() * 1000),
        "runtimeId": runtime_id,
    }
    if payload is not None:
        response["payload"] = payload
    if error is not None:
        response["error"] = error
    if request.get("jobId"):
        response["jobId"] = request["jobId"]
    return response


class RuntimeDispatcher:
    """Method table: method name → awaitable handler(request) → payload."""

    def __init__(self, runtime_id: str) -> None:
        self.runtime_id = runtime_id
        self._handlers: Dict[str, Any] = {}
        self.register_default_handlers()

    def register(self, method: str, handler) -> None:
        self._handlers[method] = handler

    async def dispatch(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch one incoming envelope. Returns the response envelope,
        or None for frames that require no reply (event/heartbeat/cancel)."""
        try:
            validate_envelope(message)
        except ProtocolError as exc:
            request_id = message.get("messageId") if isinstance(message, dict) else None
            return build_response(
                {"messageId": request_id} if request_id else {},
                error={"code": exc.code, "message": str(exc)},
                runtime_id=self.runtime_id,
            )

        if message["type"] in ("event", "heartbeat"):
            return None
        if message["type"] == "cancel":
            # Cancellation is handled out-of-band by the server via jobId.
            return None

        method = message.get("method")
        handler = self._handlers.get(method) if isinstance(method, str) else None
        if handler is None:
            return build_response(
                message,
                error={"code": "UNKNOWN_METHOD", "message": f"no handler for {method!r}"},
                runtime_id=self.runtime_id,
            )

        try:
            payload = await handler(message)
            return build_response(message, payload=payload or {}, runtime_id=self.runtime_id)
        except Exception as exc:  # noqa: BLE001 — boundary: errors go on the wire
            logger.exception("handler %s failed", method)
            return build_response(
                message,
                error={"code": "RUNTIME_ERROR", "message": str(exc)},
                runtime_id=self.runtime_id,
            )

    def register_default_handlers(self) -> None:
        async def hello(message: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "runtimeId": self.runtime_id,
                "accepted": True,
            }

        self.register("runtime.hello", hello)
