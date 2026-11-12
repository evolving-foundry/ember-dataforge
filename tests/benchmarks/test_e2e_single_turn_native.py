"""HF single-turn native e2e full-chain tests (M5): offline fixture-scale serial run + D7 judge requirement.

Aligned with the HF single-turn onboarding implementation plan §2 M5 acceptance: the new ``scripts/rsi_e2e_single_turn_native.py``
follows the six-stage terminal structure through seed ingestion → per-family source manifest → re-expression →
taskset/suite freezing → two-arm real evaluation → private readiness. When no judge is configured, only deterministic
families (reasoning_qa / code_gen) run, and families containing llm_rubric benchmarks
(knowledge_qa / health_qa) enforce an explicit requirement—the build audit must fail with ``judge_unconfigured``
rather than fabricate scores; once a judge is configured (local stub HTTP endpoint), all 4 families run to completion and
judge_fingerprint is recorded.
"""

import json
import os
import subprocess
import sys
import threading
from http.server import HTTPServer
from pathlib import Path

from tests.benchmarks.test_single_turn_source import _StubJudgeHandler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = _PROJECT_ROOT / "scripts" / "e2e_single_turn_native.py"


def _run(args: list, work_dir: Path) -> tuple[int, dict, str]:
    existing = os.environ.get("PYTHONPATH")
    pythonpath = os.pathsep.join(filter(None, [str(_PROJECT_ROOT), existing]))
    env = dict(os.environ, PYTHONPATH=pythonpath)
    process = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--work-dir", str(work_dir)],
        capture_output=True, text=True, env=env, cwd=str(_PROJECT_ROOT),
    )
    report: dict = {}
    marker = process.stdout.find('"seed_stage"')
    if marker >= 0:
        start = process.stdout.rfind("{", 0, marker)
        report = json.loads(process.stdout[start:])
    return process.returncode, report, process.stderr


def _assert_dual_arm_and_readiness(report: dict, families: list) -> None:
    eval_stage = report["eval_stage"]
    for family in families:
        assert eval_stage["reference_pass_rate"][family] == 1.0
        assert eval_stage["baseline_pass_rate"][family] == 0.0
    assert eval_stage["comparable"] is True
    readiness = report["readiness_stage"]
    for family in families:
        assert readiness["scope"][family] == "owned_native"
    assert readiness["selected_evidence_complete"] is True
    assert readiness["selected_distribution"] == "private_evaluation_only"


def test_full_chain_offline_without_judge(tmp_path):
    code, report, stderr = _run([], tmp_path / "work")
    assert code == 0, stderr
    assert report["seed_stage"]["seed_source"] == "synthetic-fixture"
    assert report["seed_stage"]["seed_row_count"] >= 1
    assert report["source_stage"]["dataset_id"] == "hf_single_turn_fixture"
    derive_stage = report["derive_stage"]
    assert derive_stage["contamination_screen"] == "pass"
    assert set(derive_stage["derived_count"]) == {"reasoning_qa", "code_gen"}
    assert all(count >= 1 for count in derive_stage["derived_count"].values())

    judge_stage = report["judge_stage"]
    assert judge_stage["configured"] is False
    assert judge_stage["run_families"] == ["reasoning_qa", "code_gen"]
    redlined = {row["family"]: row for row in judge_stage["redline"]["families"]}
    assert set(redlined) == {"knowledge_qa", "health_qa"}
    assert all(
        row["enforced"] is True and row["category"] == "judge_unconfigured"
        for row in redlined.values()
    )

    taskset_stage = report["taskset_stage"]
    for family in ("reasoning_qa", "code_gen"):
        assert taskset_stage["tasksets"][family]["item_count"] == (
            derive_stage["derived_count"][family]
        )
        assert taskset_stage["tasksets"][family]["taskset_sha256"]
        assert taskset_stage["suites"][family]["sha256"]
    _assert_dual_arm_and_readiness(report, ["reasoning_qa", "code_gen"])
    eval_stage = report["eval_stage"]
    assert all(fp is None for fp in eval_stage["judge_fingerprints"].values())

    work = Path(report["work_dir"])
    assert (work / "trusted" / "cases").is_dir()
    assert (work / "source-reasoning_qa.json").is_file()
    assert (work / "augment-code_gen.json").is_file()


def test_judge_args_are_all_or_nothing(tmp_path):
    code, report, stderr = _run(
        ["--judge-base-url", "http://127.0.0.1:9/v1"], tmp_path / "work"
    )
    assert code == 2
    assert "must be provided together" in stderr


def test_full_chain_with_stub_judge(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _StubJudgeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code, report, stderr = _run(
            [
                "--judge-base-url", f"http://127.0.0.1:{server.server_port}/v1",
                "--judge-api-key", "stub-key",
                "--judge-model", "stub-judge",
            ],
            tmp_path / "work",
        )
    finally:
        server.shutdown()
    assert code == 0, stderr
    families = ["reasoning_qa", "code_gen", "knowledge_qa", "health_qa"]
    judge_stage = report["judge_stage"]
    assert judge_stage["configured"] is True
    assert judge_stage["run_families"] == families
    assert judge_stage["redline"]["enforced"] is False
    derive_stage = report["derive_stage"]
    assert set(derive_stage["derived_count"]) == set(families)
    assert all(count >= 1 for count in derive_stage["derived_count"].values())

    _assert_dual_arm_and_readiness(report, families)
    fingerprints = report["eval_stage"]["judge_fingerprints"]
    assert fingerprints["reasoning_qa"] is None
    assert fingerprints["code_gen"] is None
    # llm_rubric family scoring goes through the judge → fingerprint is non-empty and from the same source (same stub config).
    assert fingerprints["knowledge_qa"]
    assert fingerprints["knowledge_qa"] == fingerprints["health_qa"]
