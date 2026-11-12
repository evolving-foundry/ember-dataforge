"""workbuddy native e2e full-chain tests (M5): offline fixture-scale serial run + D3 real-subset stop.

Aligned with implementation plan §2 M5 acceptance: the new ``scripts/e2e_workbuddy_native.py`` follows the older
five-stage structure through seed ingestion → source manifest → derivation → taskset/suite freezing → two-arm real evaluation →
private readiness; the real-subset path stops explicitly after freezing the source manifest (D3).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = _PROJECT_ROOT / "scripts" / "e2e_workbuddy_native.py"
FIXTURE_DIR = (
    _PROJECT_ROOT / "data_designer" / "sources" / "fixtures"
    / "wbbench_office_fixture"
)


def _run(args: list, work_dir: Path) -> tuple[int, dict, str]:
    existing = os.environ.get("PYTHONPATH")
    pythonpath = os.pathsep.join(filter(None, [str(_PROJECT_ROOT), existing]))
    env = dict(os.environ, PYTHONPATH=pythonpath)
    process = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--work-dir", str(work_dir)],
        capture_output=True, text=True, env=env, cwd=str(_PROJECT_ROOT),
    )
    return process.returncode, json.loads(process.stdout), process.stderr


def test_full_chain_offline_on_synthetic_fixture(tmp_path):
    code, report, stderr = _run([], tmp_path / "work")
    assert code == 0, stderr
    assert report["seed_stage"]["seed_source"] == "synthetic-fixture"
    assert report["seed_stage"]["seed_row_count"] >= 1
    assert report["source_stage"]["release"].startswith("synthetic-fixture:")
    assert report["derive_stage"]["derived_count"] >= 1
    assert report["derive_stage"]["contamination_screen"] == "pass"
    taskset_stage = report["taskset_stage"]
    assert taskset_stage["item_count"] == report["derive_stage"]["derived_count"]
    assert taskset_stage["suite"]["sha256"]
    eval_stage = report["eval_stage"]
    assert eval_stage["reference_pass_rate"] == 1.0
    assert eval_stage["baseline_pass_rate"] == 0.0
    assert eval_stage["comparable"] is True
    readiness = report["readiness_stage"]
    assert readiness["scope"] == "owned_native"
    assert readiness["selected_evidence_complete"] is True
    assert readiness["selected_distribution"] == "private_evaluation_only"
    work = Path(report["work_dir"])
    assert (work / "trusted" / "cases").is_dir()
    assert (work / "source.json").is_file()


def test_expand_count_yields_all_variants(tmp_path):
    # After each derived task is rewritten into a unique deliverable subdirectory, --expand-count variants
    # are no longer discarded by same-path deduplication (before the fix, only the first 2 of 5 variants survived).
    code, report, stderr = _run(["--expand-count", "5"], tmp_path / "work")
    assert code == 0, stderr
    derive = report["derive_stage"]
    assert derive["derived_count"] == 5
    assert derive["contamination_screen"] == "pass"
    assert report["taskset_stage"]["item_count"] == 5
    assert report["eval_stage"]["reference_pass_rate"] == 1.0
    assert report["eval_stage"]["baseline_pass_rate"] == 0.0


def test_real_subset_stops_at_source_stage(tmp_path):
    local_copy = tmp_path / "real-subset"
    shutil.copytree(FIXTURE_DIR, local_copy)
    code, report, stderr = _run(["--seed-dir", str(local_copy)], tmp_path / "work")
    assert code == 0, stderr
    assert report["stopped_at"] == "source_stage"
    assert "D3" in report["reason"]
    assert report["source_stage"]["release"].startswith("wb-bench-")
    assert "derive_stage" not in report
