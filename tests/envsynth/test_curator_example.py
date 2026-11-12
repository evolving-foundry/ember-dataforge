"""Regression test for the formal Curator -> DataForge -> Gym example."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "envsynth_curator_e2e" / "run_example.py"


def _gym_root():
    candidates = []
    env_root = os.environ.get("DATA_FORGE_GYM")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(REPO_ROOT.parent / "AscendAgent-Gym")
    for candidate in candidates:
        if (candidate / "overlay/resources_servers/synth_env").is_dir():
            return candidate.resolve()
    return None


def test_formal_curator_example_publishes_gym_native_package(tmp_path):
    work_dir = tmp_path / "curator-e2e"
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
    assert summary["curator"]["schema"] == "parsed-doc-v2"
    assert summary["curator"]["mediaTypes"] == ["docx", "pdf", "png"]
    ocr = summary["curator"]["ocr"]
    assert ocr["requestedBackend"] == "paddleocr"
    assert ocr["backend"] == "paddleocr"
    assert ocr["device"] == "cpu"
    assert ocr["modelId"]
    assert summary["dataDesigner"]["taskset"]["itemCount"] > 0
    assert summary["dataDesigner"]["storeCheck"]["releaseFingerprint"] == (
        summary["dataDesigner"]["release"]["fingerprint"]
    )
    assert summary["gym"]["status"] == "published"
    assert summary["gym"]["runnable"] is True
    assert Path(summary["gym"]["assetZip"]).is_file()
    assert Path(summary["gym"]["prepareScript"]).is_file()
    assert Path(summary["gym"]["generatedData"]).is_file()
    if gym_root is not None:
        assert summary["gym"]["runtimeVerified"] is True
        runtime = summary["gym"]["runtime"]
        assert runtime["fingerprint"] == summary["dataDesigner"]["release"]["fingerprint"]
        assert runtime["taskCount"] == summary["dataDesigner"]["taskset"]["itemCount"]
        assert all(item["verified"] for item in runtime["tasks"])
    else:
        assert summary["gym"]["runtimeVerified"] is False

    assert (work_dir / "artifact-store").is_dir()
    assert (work_dir / "snapshot.json").is_file()
    assert (work_dir / "seed-import" / "parsed" / "documents.jsonl").is_file()
    published_root = Path(summary["gym"]["root"])
    assert (
        published_root / "overlay" / "benchmarks" / summary["gym"]["benchmark"]
    ).is_dir()