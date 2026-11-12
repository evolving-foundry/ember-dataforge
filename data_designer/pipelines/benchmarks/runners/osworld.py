"""Evaluate a private OSWorld V2.1 trajectory in the official DesktopEnv."""

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


VERSION = "osworld-v2.1-owned-native-adapter-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wsl_path(path: Path, wsl: str) -> str:
    proc = subprocess.run(
        [wsl, "-e", "wslpath", "-a", str(path)],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


def _child(task_path: str, trajectory_path: str, output_path: str) -> None:
    # Import only inside the external, pinned OSWorld checkout.
    from desktop_env.desktop_env import DesktopEnv
    from task_loader import load_task_from_file

    task = load_task_from_file(task_path)
    trajectory = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    actions = trajectory["actions"]
    if not isinstance(actions, list) or not actions or not all(
        isinstance(action, str) and action.startswith("import pyautogui;")
        for action in actions
    ):
        raise ValueError("OSWorld trajectory must contain GUI actions")
    env = DesktopEnv(
        provider_name="docker", action_space="pyautogui", headless=True,
        require_a11y_tree=False, enable_proxy=False,
        force_disable_vnc=True, force_disable_recording=True,
    )
    try:
        env.reset(task_config=task)
        # VS Code is launched by setup but its editor becomes ready later than
        # the VM control service. Wait before replaying GUI actions.
        time.sleep(25)
        initial = float(env.evaluate())
        for action in actions:
            env.step(action, pause=3)
        time.sleep(5)
        score = float(env.evaluate())
        output_text = None
        if getattr(task, "summary_path", None):
            raw = env.controller.get_file(task.summary_path)
            if isinstance(raw, bytes):
                output_text = raw.decode("utf-8", errors="replace")
        env.reset(task_config=task)
        time.sleep(8)
        reset = float(env.evaluate())
        result = {"initial": initial, "score": score, "reset": reset,
                  "actions": len(actions), "output_text": output_text}
        Path(output_path).write_text(json.dumps(result), encoding="utf-8")
    finally:
        env.close()


def score_request(request: dict) -> dict:
    started = time.monotonic()
    manifest = json.loads(Path(request["caseManifestPath"]).read_text(encoding="utf-8"))
    if manifest.get("task_id") != request.get("taskId"):
        raise ValueError("OSWorld task ID disagrees with trusted manifest")
    spec = manifest["private"]["osworld_task"]
    task_path = Path(spec["path"])
    if not task_path.is_absolute() or not task_path.is_file() or _sha256(task_path) != spec["sha256"]:
        raise ValueError("OSWorld task class missing or drifted")
    root = Path(request["candidateRoot"]).resolve(strict=True)
    trajectory = root / "trajectory.json"
    if not trajectory.is_file() or trajectory.is_symlink() or trajectory.stat().st_size > 100_000:
        raise ValueError("OSWorld candidate trajectory missing or invalid")
    wsl = os.environ.get("OSWORLD_WSL_EXE") or shutil.which("wsl")
    python = os.environ.get("OSWORLD_PYTHON_WSL")
    checkout = os.environ.get("OSWORLD_RELEASE_ROOT_WSL")
    if not wsl or not Path(wsl).is_file() or not python or not checkout:
        raise ValueError("OSWORLD_WSL_EXE/PYTHON_WSL/RELEASE_ROOT_WSL unavailable")
    evidence_refs = [f"trajectory-sha256:{_sha256(trajectory)}",
                     f"task-class-sha256:{_sha256(task_path)}"]
    with tempfile.TemporaryDirectory(prefix="osworld-native-") as directory:
        output = Path(directory) / "native-result.json"
        command = [
            wsl, "--cd", checkout, "-e", "env", f"PYTHONPATH={checkout}",
            python, _wsl_path(Path(__file__).resolve(), wsl), "--child",
            _wsl_path(task_path, wsl), _wsl_path(trajectory, wsl),
            _wsl_path(output, wsl),
        ]
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=700, check=False
        )
        if proc.returncode != 0:
            raise RuntimeError(f"OSWorld native task failed: {proc.stderr[-500:]}")
        result = json.loads(output.read_text(encoding="utf-8"))
    evidence_dir = os.environ.get("OSWORLD_NATIVE_EVIDENCE_DIR")
    if evidence_dir:
        evidence_path = Path(evidence_dir) / f"native-{uuid.uuid4().hex}.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        evidence_refs.append(f"{evidence_path}#sha256={_sha256(evidence_path)}")
    if result["initial"] != 0.0 or result["reset"] != 0.0:
        raise RuntimeError("OSWorld task did not start and reset to zero")
    score = float(result["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError("OSWorld native score outside [0,1]")
    passed = score == 1.0
    response = {
        "task_id": request["taskId"], "status": "completed",
        "passed": passed, "score": score,
        "native_metrics": {"initial_zero": 1.0, "reset_zero": 1.0,
                           "gui_actions": float(result["actions"])},
        "verifier_version": VERSION,
        "evidence_refs": evidence_refs,
        "runtime_seconds": time.monotonic() - started,
    }
    if "attempt" in request:
        response.update(attempt_index=request["attempt"]["index"],
                        seed=request["attempt"]["seed"])
    return response


def main() -> None:
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        _child(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    if len(sys.argv) != 3:
        raise ValueError("usage: osworld.py <request.json> <result.json>")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = score_request(request)
    Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
