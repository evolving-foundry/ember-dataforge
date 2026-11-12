"""Dispatcher unit tests — envelope validation and method routing (§10/§34)."""

import pytest

from curator.runtime.dispatcher import ProtocolError, RuntimeDispatcher, validate_envelope


def envelope(**overrides):
    base = {
        "protocol": "dataforge-runtime",
        "version": "1",
        "messageId": "msg-1",
        "type": "request",
        "method": "runtime.hello",
        "timestamp": 1,
        "payload": {},
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_validates_unknown_protocol():
    with pytest.raises(ProtocolError):
        validate_envelope(envelope(protocol="other"))


@pytest.mark.asyncio
async def test_validates_unknown_version():
    with pytest.raises(ProtocolError):
        validate_envelope(envelope(version="2"))


@pytest.mark.asyncio
async def test_hello_returns_runtime_id():
    d = RuntimeDispatcher("rt-1")
    response = await d.dispatch(envelope())
    assert response["type"] == "response"
    assert response["correlationId"] == "msg-1"
    assert response["payload"]["runtimeId"] == "rt-1"


@pytest.mark.asyncio
async def test_unknown_method_yields_error_response():
    d = RuntimeDispatcher("rt-1")
    response = await d.dispatch(envelope(method="does.not.exist"))
    assert response["error"]["code"] == "UNKNOWN_METHOD"


@pytest.mark.asyncio
async def test_events_and_heartbeats_need_no_reply():
    d = RuntimeDispatcher("rt-1")
    assert await d.dispatch(envelope(type="event", method="job.progress")) is None
    assert await d.dispatch(envelope(type="heartbeat")) is None
    assert await d.dispatch(envelope(type="cancel")) is None


@pytest.mark.asyncio
async def test_handler_exception_becomes_error_response():
    d = RuntimeDispatcher("rt-1")

    async def boom(message):
        raise ValueError("kaputt")

    d.register("explode", boom)
    response = await d.dispatch(envelope(method="explode"))
    assert response["error"]["code"] == "RUNTIME_ERROR"
    assert "kaputt" in response["error"]["message"]
