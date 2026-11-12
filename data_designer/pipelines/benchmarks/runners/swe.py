"""Run a private SWE-style patch through a pinned official SWE-bench harness.

Machine-specific executable and work paths come from environment variables.
The trusted case manifest pins the local instance file; the candidate supplies
only ``patch.diff``. This runner is intentionally separate from Curator.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


VERSION = "swebench-v2-native-adapter-v1"
MAX_PATCH_BYTES = 2_000_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _machine_settings() -> tuple[str | None, str, str]:
    if os.name == "nt":
        wsl = os.environ.get("SWE_BENCH_WSL_EXE") or shutil.which("wsl")
        python = os.environ.get("SWE_BENCH_PYTHON_WSL")
        workdir = os.environ.get("SWE_BENCH_WORKDIR_WSL")
        if not wsl or not Path(wsl).is_file() or not python or not workdir:
            raise ValueError("SWE_BENCH_WSL_EXE/PYTHON_WSL/WORKDIR_WSL unavailable")
        return wsl, python, workdir
    python = os.environ.get("SWE_BENCH_PYTHON")
    workdir = os.environ.get("SWE_BENCH_WORKDIR")
    if not python or not Path(python).is_file() or not workdir:
        raise ValueError("SWE_BENCH_PYTHON/WORKDIR unavailable")
    return None, python, workdir


def _native_path(path: Path, wsl: str | None) -> str:
    if wsl is None:
        return str(path)
    proc = subprocess.run(
        [wsl, "-e", "wslpath", "-a", str(path)],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


def _host_path(path: str, wsl: str | None) -> Path:
    if wsl is None:
        return Path(path)
    proc = subprocess.run(
        [wsl, "-e", "wslpath", "-w", path],
        capture_output=True, text=True, check=True,
    )
    return Path(proc.stdout.strip())


def score_request(request: dict) -> dict:
    started = time.monotonic()
    manifest = json.loads(Path(request["caseManifestPath"]).read_text(encoding="utf-8"))
    if manifest.get("task_id") != request.get("taskId"):
        raise ValueError("SWE task ID disagrees with trusted manifest")
    spec = manifest["private"]["swe_harness"]
    dataset = Path(spec["dataset_path"])
    if not dataset.is_absolute() or not dataset.is_file() or _sha256(dataset) != spec["dataset_sha256"]:
        raise ValueError("SWE instance snapshot missing or drifted")
    instance_id = spec["instance_id"]
    instance = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(instance, list) or len(instance) != 1 or instance[0].get("instance_id") != instance_id:
        raise ValueError("SWE instance snapshot identity drifted")
    root = Path(request["candidateRoot"]).resolve(strict=True)
    patch = root / "patch.diff"
    if not patch.is_file() or patch.is_symlink() or patch.stat().st_size > MAX_PATCH_BYTES:
        raise ValueError("candidate patch.diff missing or invalid")
    patch_text = patch.read_text(encoding="utf-8")
    if not patch_text.strip():
        raise ValueError("candidate patch.diff is empty")
    wsl, python, workdir = _machine_settings()
    run_id = "rsi-" + uuid.uuid4().hex[:16]
    model_name = "rsi-owned"
    with tempfile.TemporaryDirectory(prefix="swe-native-") as directory:
        prediction = Path(directory) / "prediction.json"
        prediction.write_text(
            json.dumps([{"instance_id": instance_id, "model_patch": patch_text,
                         "model_name_or_path": model_name}]), encoding="utf-8"
        )
        args = [
            python, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", _native_path(dataset, wsl),
            "--split", "dev", "--instance_ids", instance_id,
            "--predictions_path", _native_path(prediction, wsl),
            "--max_workers", "1", "--run_id", run_id,
            "--cache_level", "env", "--clean", "false",
        ]
        command = [wsl, "--cd", workdir, "-e", *args] if wsl else args
        proc = subprocess.run(
            command, cwd=None if wsl else workdir,
            capture_output=True, text=True, timeout=1_800, check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"official SWE-bench harness failed: {proc.stderr[-500:]}")
    report = _host_path(f"{workdir}/{model_name}.{run_id}.json", wsl)
    if not report.is_file():
        raise RuntimeError("official SWE-bench report missing")
    summary = json.loads(report.read_text(encoding="utf-8"))
    if instance_id not in summary.get("completed_ids", []) or summary.get("error_ids"):
        raise RuntimeError("official SWE-bench instance incomplete or errored")
    passed = instance_id in summary.get("resolved_ids", [])
    per_case = _host_path(
        f"{workdir}/logs/run_evaluation/{run_id}/{model_name}/{instance_id}/report.json", wsl
    )
    if not per_case.is_file():
        raise RuntimeError("official SWE-bench per-instance report missing")
    detail = json.loads(per_case.read_text(encoding="utf-8"))[instance_id]
    if detail.get("resolved") is not passed:
        raise RuntimeError("official SWE-bench reports disagree")
    tests = detail["tests_status"]
    f2p = tests["FAIL_TO_PASS"]
    p2p = tests["PASS_TO_PASS"]
    result = {
        "task_id": request["taskId"], "status": "completed",
        "passed": passed, "score": float(passed),
        "native_metrics": {
            "fail_to_pass": float(not f2p["failure"] and bool(f2p["success"])),
            "pass_to_pass": float(not p2p["failure"]),
        },
        "verifier_version": VERSION,
        "evidence_refs": [f"{report}#sha256={_sha256(report)}",
                          f"{per_case}#sha256={_sha256(per_case)}"],
        "runtime_seconds": time.monotonic() - started,
    }
    if "attempt" in request:
        result.update(attempt_index=request["attempt"]["index"], seed=request["attempt"]["seed"])
    return result


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError("usage: swe.py <request.json> <result.json>")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = score_request(request)
    Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
