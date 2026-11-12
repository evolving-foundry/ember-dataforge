"""Versioned file protocol for algorithms in independent environments."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from data_designer.contracts.algorithms import AlgorithmResult
from data_designer.contracts.datasets import RecordEnvelope
from data_designer.contracts.provenance import canonical_json, content_digest

PROTOCOL_VERSION = "dataforge-subprocess-v1"


def terminate_process_tree(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


class SubprocessAlgorithm:
    def __init__(self, spec, command, *, timeout_seconds=60, max_output_bytes=16 * 1024 * 1024):
        if not command or not all(isinstance(part, str) and part for part in command):
            raise ValueError("subprocess command must be an explicit argument list")
        if timeout_seconds <= 0 or max_output_bytes <= 0:
            raise ValueError("subprocess timeout and output limit must be positive")
        self.spec, self.command = spec, tuple(command)
        self.timeout_seconds, self.max_output_bytes = timeout_seconds, max_output_bytes

    def describe(self):
        return self.spec

    def run(self, request, context):
        context.check_cancelled()
        with tempfile.TemporaryDirectory(prefix="dataforge-", dir=context.work_directory) as work:
            root = Path(work)
            request_path, response_path = root / "request.json", root / "result.json"
            body = {"protocol_version": PROTOCOL_VERSION, "run_id": request.run_id,
                    "algorithm": self.spec.id, "implementation_version": self.spec.implementation_version,
                    "records": [asdict(record) for record in request.records],
                    "config": dict(request.config), "seed": context.seed,
                    "target_new_records": request.target_new_records, "max_attempts": request.max_attempts,
                    "work_directory": work, "output_path": str(response_path)}
            body["request_digest"] = content_digest(body)
            request_path.write_text(canonical_json(body), encoding="utf-8")
            options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
                process = subprocess.Popen([*self.command, str(request_path), str(response_path)],
                                           stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           cwd=work, **options)
                deadline = time.monotonic() + self.timeout_seconds
                try:
                    while process.poll() is None:
                        context.check_cancelled()
                        if time.monotonic() >= deadline:
                            raise TimeoutError("algorithm subprocess timed out")
                        for name in ("stdout.log", "stderr.log", "result.json"):
                            path = root / name
                            if path.exists() and path.stat().st_size > self.max_output_bytes:
                                raise RuntimeError("algorithm subprocess output limit exceeded")
                        time.sleep(0.02)
                    if process.returncode:
                        # Raw provider diagnostics may contain credentials; do not echo them.
                        raise RuntimeError(f"algorithm subprocess failed (exit {process.returncode})")
                    for name in ("stdout.log", "stderr.log", "result.json"):
                        path = root / name
                        if path.exists() and path.stat().st_size > self.max_output_bytes:
                            raise RuntimeError("algorithm subprocess output limit exceeded")
                    response = json.loads(response_path.read_text(encoding="utf-8"))
                    for key in ("protocol_version", "run_id", "algorithm", "implementation_version", "request_digest"):
                        if response.get(key) != body[key]:
                            raise ValueError(f"algorithm subprocess response identity mismatch: {key}")
                    if response.get("status") != "completed":
                        raise RuntimeError("algorithm subprocess did not complete")
                    if not isinstance(response.get("records"), list):
                        raise ValueError("algorithm subprocess response records must be a list")
                    records = [RecordEnvelope(**{**row, "parents": tuple(row.get("parents", ()))})
                               for row in response["records"]]
                    return AlgorithmResult(records, response.get("rejected", {}),
                                           {"adapter": "subprocess", "exit_code": process.returncode})
                finally:
                    # Also kill descendants left by an already exited parent on POSIX.
                    if process.poll() is None or os.name != "nt":
                        terminate_process_tree(process)
