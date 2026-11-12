"""End-to-end loopback test over real ZeroMQ (§34 P0 method set).

A python ROUTER client talks to a live ZmqRuntimeServer (DEALER), covering
hello / health / dataset lifecycle / operation.execute with a kernel
operation registered through the real OperationRegistry.
"""

import asyncio
import json
import uuid

import pytest
import pytest_asyncio
import zmq
import zmq.asyncio

from curator.runtime.zmq_server import ZmqRuntimeServer

RUNTIME_ID = "rt-e2e"


def make_envelope(method, payload=None, msg_type="request", **kw):
    return {
        "protocol": "dataforge-runtime",
        "version": "1",
        "messageId": f"msg-{uuid.uuid4().hex[:8]}",
        "type": msg_type,
        "method": method,
        "timestamp": 1,
        "runtimeId": kw.get("runtimeId", "test-client"),
        "payload": payload or {},
    }


@pytest.fixture()
def endpoint(tmp_path):
    return f"ipc://{tmp_path}/cmd-{uuid.uuid4().hex[:6]}.sock"


@pytest_asyncio.fixture()
async def server(endpoint):
    srv = ZmqRuntimeServer(RUNTIME_ID, command_endpoint=endpoint, max_concurrent_jobs=2)
    task = asyncio.ensure_future(srv.start())
    await asyncio.sleep(0.3)  # allow DEALER connect + hello announce
    yield srv
    srv._running = False
    task.cancel()
    await srv.stop()


@pytest_asyncio.fixture()
async def router(endpoint):
    ctx = zmq.asyncio.Context.instance()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(endpoint)
    yield sock
    sock.close(0)


async def request(router, envelope, timeout=10):
    await router.send_multipart(
        [RUNTIME_ID.encode(), json.dumps(envelope).encode()]
    )
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"no response for {envelope['method']}")
        frames = await asyncio.wait_for(router.recv_multipart(), remaining)
        message = json.loads(frames[1].decode())
        # Skip frames the server pushes on its own (announce hello, events).
        if message.get("type") == "response" and message.get("correlationId") == envelope["messageId"]:
            return message


@pytest.mark.asyncio
async def test_hello_request_reaches_server(router, server):
    response = await request(router, make_envelope("runtime.hello", {"runtimeId": RUNTIME_ID}))
    assert response["type"] == "response"
    assert response["payload"]["runtimeId"] == RUNTIME_ID


@pytest.mark.asyncio
async def test_health_reports_empty_state(router, server):
    response = await request(router, make_envelope("runtime.health"))
    assert response.get("error") is None
    assert response["payload"]["runtimeId"] == RUNTIME_ID
    assert response["payload"]["activeJobs"] == 0


@pytest.mark.asyncio
async def test_dataset_inspect_release_roundtrip(router, server):
    """Kernel-side registry is addressable through the wire (§14)."""
    ds_id = server.datasets.register(
        type("D", (), {"dataset": None, "modalities": ["text"]})(),
        backend="ray",
    )
    response = await request(
        router, make_envelope("dataset.inspect", {"datasetId": ds_id})
    )
    assert response["payload"]["datasetId"] == ds_id
    assert response["payload"]["modalities"] == ["text"]

    release = await request(
        router, make_envelope("dataset.release", {"datasetId": ds_id})
    )
    assert release["payload"]["released"] is True
    assert not server.datasets.exists(ds_id)


@pytest.mark.asyncio
async def test_operation_execute_via_kernel_registry(router, server):
    """Full path: dispatcher → JobManager → executor → OperationRegistry."""
    from curator.core.operation import Operation
    from curator.core.operation_registry import (
        register_operation,
    )

    @register_operation("runtime_passthrough_test")
    class PassthroughOp(Operation):
        def __init__(self, params):
            self.params = params or {}

        def execute(self, dataset):
            return dataset

    ds_id = server.datasets.register("sentinel-dataset")
    response = await request(
        router,
        make_envelope(
            "operation.execute",
            {
                "operation": {"name": "runtime_passthrough_test", "params": {}},
                "input": {"datasetId": ds_id},
                "awaitResult": True,
            },
        ),
        timeout=30,
    )
    assert response.get("error") is None, response
    assert response["payload"]["status"] == "completed"
    output_ds = response["payload"]["output"]["datasetId"]
    assert server.datasets.get(output_ds) == "sentinel-dataset"

    # job.cancel on a finished job reports cleanly
    cancel = await request(
        router,
        make_envelope("job.cancel", {"jobId": response["payload"]["jobId"]}),
    )
    assert cancel["payload"]["cancelled"] is False


@pytest.mark.asyncio
async def test_unknown_operation_returns_runtime_error(router, server):
    response = await request(
        router,
        make_envelope(
            "operation.execute",
            {"operation": {"name": "no_such_op_xyz"}, "awaitResult": True},
        ),
        timeout=30,
    )
    assert response["error"]["code"] == "RUNTIME_ERROR"
