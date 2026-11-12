from __future__ import annotations

import os
import sys
import threading
from dataclasses import replace

import pytest

from data_designer.contracts.algorithms import AlgorithmRequest, AlgorithmDescriptor
from data_designer.contracts.datasets import RecordEnvelope
from data_designer.generation.algorithms.adapters.subprocess import SubprocessAlgorithm
from data_designer.generation.algorithms.adapters.service import ServiceAlgorithm
from data_designer.generation.algorithms.builtin.descriptors import AGENT_TASK
from data_designer.generation.algorithms.registry import AlgorithmRegistry
from data_designer.generation.context import AlgorithmContext


def test_registry_conflicts_versions_and_lazy_factories():
    registry = AlgorithmRegistry(discover=False)
    with pytest.raises(ValueError, match="duplicate algorithm id"):
        registry.register(registry.describe(AGENT_TASK.id))
    registry.register(AlgorithmDescriptor(replace(AGENT_TASK, id="test.unused"), "missing_dependency:factory"))
    assert registry.describe("test.unused").spec.id == "test.unused"
    with pytest.raises(RuntimeError, match="dependency unavailable"):
        registry.create("test.unused")
    with pytest.raises(ValueError, match="unknown algorithm"):
        registry.describe("test.absent")


def script(tmp_path, content):
    path = tmp_path / "external.py"
    path.write_text(content)
    return [sys.executable, "-I", str(path)]


def request():
    return AlgorithmRequest((RecordEnvelope("a", {"text": "hello"}),), {}, "run-1")


def test_subprocess_independent_environment_and_version_validation(tmp_path):
    command = script(tmp_path, '''
import json, sys
from pathlib import Path
assert 'data_designer' not in sys.modules
req = json.loads(Path(sys.argv[1]).read_text())
out = {k:req[k] for k in ('protocol_version', 'run_id', 'algorithm', 'implementation_version', 'request_digest')}
out.update(status='completed', records=[{'id':'b', 'payload':{'text':'hello world'}, 'schema':'record-v1', 'parents':['a']}])
Path(sys.argv[2]).write_text(json.dumps(out))
''')
    adapter = SubprocessAlgorithm(replace(AGENT_TASK, execution=("subprocess",)), command)
    result = adapter.run(request(), AlgorithmContext())
    assert list(result.records)[0].parents == ("a",)
    command = script(tmp_path, "import sys; sys.exit(3)")
    with pytest.raises(RuntimeError, match="exit 3"):
        SubprocessAlgorithm(AGENT_TASK, command).run(request(), AlgorithmContext())
    command = script(tmp_path, "import sys; from pathlib import Path; Path(sys.argv[2]).write_text('{}')")
    with pytest.raises(ValueError, match="identity mismatch"):
        SubprocessAlgorithm(AGENT_TASK, command).run(request(), AlgorithmContext())


@pytest.mark.parametrize("cancel", [False, True])
def test_subprocess_timeout_cancel_observes_process_exit(tmp_path, cancel):
    pidfile = tmp_path / "pid"
    command = script(tmp_path, f"import os,time; from pathlib import Path; Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)")
    context = AlgorithmContext()
    timer = threading.Timer(.5, context.cancelled.set) if cancel else None
    if timer:
        timer.start()
    try:
        with pytest.raises(InterruptedError if cancel else TimeoutError):
            SubprocessAlgorithm(AGENT_TASK, command, timeout_seconds=2).run(request(), context)
    finally:
        if timer:
            timer.cancel()
    assert pidfile.exists()
    pid = int(pidfile.read_text())
    if os.name != "nt":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_service_identity_unknown_version_and_no_retry():
    class Provider:
        def describe(self):
            return {"algorithm": AGENT_TASK.id}

        def generate(self, body, **kwargs):
            assert kwargs["timeout_seconds"] == 60
            assert body["idempotency_key"].startswith("sha256:")
            return {"protocol_version": body["protocol_version"], "run_id": body["run_id"],
                    "status": "completed", "records": []}
    context = AlgorithmContext(providers={"model": Provider()})
    result = ServiceAlgorithm(AGENT_TASK, "model").run(request(), context)
    assert result.diagnostics["cache_reusable"] is False
    assert result.diagnostics["automatic_retries"] == 0
    with pytest.raises(ValueError, match="not configured"):
        ServiceAlgorithm(AGENT_TASK, "absent").run(request(), context)
