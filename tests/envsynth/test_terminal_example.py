"""Regression tests for the formal terminal-profile example."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "envsynth_terminal_e2e" / "run_example.py"


def _gym_root():
    env_root = os.environ.get("DATA_FORGE_GYM")
    candidates = [Path(env_root).resolve()] if env_root else []
    candidates.append(REPO_ROOT.parent / "AscendAgent-Gym")
    for candidate in candidates:
        if (candidate / "overlay/resources_servers/synth_env").is_dir():
            return candidate.resolve()
    return None


def test_formal_terminal_example_freezes_and_runs_with_gym(tmp_path):
    work_dir = tmp_path / "terminal-e2e"
    gym_root = _gym_root()
    command = [sys.executable, str(EXAMPLE), "--work-dir", str(work_dir)]
    if gym_root is not None:
        command.extend(["--gym-root", str(gym_root)])
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    summary = json.loads(
        (work_dir / "example_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "ok"
    assert summary["dataDesigner"]["profile"] == "terminal"
    assert summary["dataDesigner"]["family"] == "terminal"
    assert summary["dataDesigner"]["gatePassed"] is True
    assert "replay/replay.jsonl" in summary["dataDesigner"]["files"]
    assert "state/seed.jsonl" not in summary["dataDesigner"]["files"]
    assert Path(summary["dataDesigner"]["zip"]).is_file()
    assert (
        Path(summary["dataDesigner"]["releaseDir"]) / "manifest.json"
    ).is_file()

    if gym_root is not None:
        assert summary["gym"]["runtimeVerified"] is True
        runtime = summary["gym"]["runtime"]
        assert runtime["fingerprint"] == summary["dataDesigner"]["fingerprint"]
        assert runtime["taskCount"] == summary["dataDesigner"]["taskCount"]
        assert all(item["verified"] for item in runtime["tasks"])
    else:
        assert summary["gym"]["runtimeVerified"] is False