"""terminal native e2e full-chain tests (M5): offline fixture-scale serial run + T2 real-subset stop.

Aligned with the terminal onboarding implementation plan §2 M5 acceptance: the new ``scripts/rsi_e2e_terminal_native.py``
follows the workbuddy six-stage structure through seed ingestion → source manifest → re-expression → taskset/suite freezing →
two-arm real evaluation → private readiness; the real upstream-checkout path (requires a pinned 40-hex commit)
stops explicitly after freezing the source manifest (T2/T5).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = _PROJECT_ROOT / "scripts" / "e2e_terminal_native.py"
UPSTREAM_COMMIT = "7131e4375048a0e408a8fb404b5f499d726b695b"


def _run(args: list, work_dir: Path) -> tuple[int, dict, str]:
    existing = os.environ.get("PYTHONPATH")
    pythonpath = os.pathsep.join(filter(None, [str(_PROJECT_ROOT), existing]))
    env = dict(os.environ, PYTHONPATH=pythonpath)
    process = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--work-dir", str(work_dir)],
        capture_output=True, text=True, env=env, cwd=str(_PROJECT_ROOT),
    )
    report = json.loads(process.stdout) if process.stdout.strip() else {}
    return process.returncode, report, process.stderr


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


def _upstream_checkout(tmp_path: Path) -> Path:
    """Minimal 89-task terminal-bench-shaped task root (used for the T2 count assertion)."""
    root = tmp_path / "terminal-bench-2-1-local"
    (root / "tasks").mkdir(parents=True)
    (root / "dataset.toml").write_text(
        '[dataset]\nid = "terminal-bench-2-1-local"\ntask_count = 89\n',
        encoding="utf-8",
    )
    for index in range(89):
        name = f"tb-up-{index:03d}"
        task_dir = root / "tasks" / name
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            f'[task]\nname = "terminal/{name}"\n\n[metadata]\n'
            'difficulty = "easy"\n\n[environment]\n',
            encoding="utf-8",
        )
        (task_dir / "instruction.md").write_text(
            f"Sum the two numbers in input.txt and write the total to "
            f"output-{name}.txt.",
            encoding="utf-8",
        )
    return root


def test_upstream_checkout_stops_at_source_stage(tmp_path):
    checkout = _upstream_checkout(tmp_path / "checkout")
    code, report, stderr = _run(
        ["--seed-dir", str(checkout), "--upstream-commit", UPSTREAM_COMMIT],
        tmp_path / "work",
    )
    assert code == 0, stderr
    assert report["stopped_at"] == "source_stage"
    assert "upstream-checkout" in report["reason"]
    assert report["source_stage"]["release"] == "tb21-" + UPSTREAM_COMMIT[:8]
    assert report["seed_stage"]["seed_row_count"] == 89
    assert "derive_stage" not in report


def test_upstream_commit_pin_is_enforced(tmp_path):
    checkout = _upstream_checkout(tmp_path / "checkout")
    code, report, stderr = _run(
        ["--seed-dir", str(checkout), "--upstream-commit", "deadbeef"],
        tmp_path / "work",
    )
    assert code == 2
    assert "40-hex" in stderr
