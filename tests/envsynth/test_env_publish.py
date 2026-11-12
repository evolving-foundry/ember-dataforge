"""Gym-native publisher tests: one loadable package, no handoff transform."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

from data_designer.pipelines.envsynth import artifacts, package
from data_designer.pipelines.envsynth.profile_compilers import get_profile_compiler
from tests.envsynth.env_fixtures import run_template_build


PUBLISH_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gym_publish_env.py"

TICKETS = [
    ("GD-2026-0001", "线路检修", "已审批"),
    ("GD-2026-0002", "停送电操作", "审核中"),
    ("GD-2026-0003", "缺陷处理", "草稿"),
]


def _payload(compile_result_path, *, name="office-power"):
    return {
        "compileResultPath": str(compile_result_path),
        "name": name,
    }


def _gym_root(tmp_path: Path) -> Path:
    root = tmp_path / "gym"
    (root / "overlay").mkdir(parents=True)
    return root


def _run_publish(payload, gym_root, *, register=False):
    args = [
        sys.executable,
        str(PUBLISH_SCRIPT),
        "--gym-root",
        str(gym_root),
    ]
    if register:
        args.append("--register")
    return subprocess.run(
        args,
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
    )


def _assert_no_private_answers(value):
    if isinstance(value, dict):
        forbidden = {"gold", "gold_solution", "negative_solution"}
        assert not (set(value) & forbidden)
        for item in value.values():
            _assert_no_private_answers(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_private_answers(item)


def _snapshot_bytes(root: Path):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _asset_zip(bench_dir: Path) -> Path:
    assets = sorted((bench_dir / "assets").glob("gym-env-package-v*.zip"))
    assert len(assets) == 1
    return assets[0]


def test_publish_copies_one_gym_package_and_is_byte_idempotent(tmp_path):
    _, snapshot, compile_result_path = run_template_build(
        tmp_path, TICKETS, asset_name="gym-assets"
    )
    gym_root = _gym_root(tmp_path)
    result = _run_publish(_payload(compile_result_path), gym_root)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["status"] == "published"
    assert output["runnable"] is True
    assert output["benchmark"] == "synth-office-power"
    assert output["item_count"] == snapshot["taskset"]["itemCount"]

    bench_dir = gym_root / "overlay" / "benchmarks" / "synth-office-power"
    assert (bench_dir / "config.yaml").is_file()
    assert (bench_dir / "manifest.yaml").is_file()
    assert (bench_dir / "prepare.py").is_file()
    assert (bench_dir / "publish.json").is_file()
    assert (gym_root / "overlay/benchmarks/prompts/synth-office-power/agent.yaml").is_file()
    prompt = yaml.safe_load(
        (gym_root / "overlay/benchmarks/prompts/synth-office-power/agent.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert prompt == get_profile_compiler("office").agent_prompt("synth-office-power")
    assert not (bench_dir / "env").exists()
    assert not (bench_dir / "handoff.json").exists()

    asset_zip = _asset_zip(bench_dir)
    assert asset_zip.name == "gym-env-package-v{}-{}.zip".format(
        snapshot["envRelease"]["version"], snapshot["envRelease"]["fingerprint"]
    )
    with zipfile.ZipFile(asset_zip) as archive:
        names = set(archive.namelist())
        assert {
            "manifest.json",
            "env_spec.json",
            "tasks.jsonl",
            "fs/snapshot.jsonl",
            "rules.yaml",
            "events.yaml",
            "npcs.yaml",
            "state/seed.jsonl",
            "grounding/index.jsonl",
            "llm_sim.yaml",
        } <= names
        tasks = [
            json.loads(line)
            for line in archive.read("tasks.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]
    assert len(tasks) == snapshot["taskset"]["itemCount"]
    public_fields = {
        "task_id",
        "env_release",
        "goal",
        "success_check",
        "max_steps",
        "seed",
    }
    for task in tasks:
        assert set(task) == public_fields
        assert task["env_release"] == {
            "ref_id": snapshot["envRelease"]["refId"],
            "version": snapshot["envRelease"]["version"],
            "fingerprint": snapshot["envRelease"]["fingerprint"],
        }
        verifier = task["success_check"]["verifier_spec"]
        assert set(verifier) == {"kind", "payload"}
        assert verifier["kind"] == "state_assert"
        assert verifier["payload"]["checks"]
        _assert_no_private_answers(task)

    config = yaml.safe_load((bench_dir / "config.yaml").read_text(encoding="utf-8"))
    agent_key = "synth_office_power_agent"
    dataset = config[agent_key]["responses_api_agents"]["synth_env_agent"]["datasets"][0]
    assert dataset["type"] == "benchmark"
    assert dataset["jsonl_fpath"] == "benchmarks/synth-office-power/data/synth-office-power.jsonl"
    assert dataset["prepare_script"] == "benchmarks/synth-office-power/prepare.py"
    assert dataset["prompt_config"] == "benchmarks/prompts/synth-office-power/agent.yaml"

    manifest = yaml.safe_load((bench_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["kind"] == "benchmark"
    assert manifest["resources_server"] == "synth_env"
    assert manifest["agent_server"] == "synth_env_agent"
    assert manifest["datasets"][0]["jsonl_fpath"] == dataset["jsonl_fpath"]

    with zipfile.ZipFile(asset_zip) as archive:
        archive_tasks_bytes = archive.read("tasks.jsonl")
    prepared = subprocess.run(
        [sys.executable, str(bench_dir / "prepare.py")],
        capture_output=True,
        text=True,
    )
    assert prepared.returncode == 0, prepared.stderr
    data_path = bench_dir / "data" / "synth-office-power.jsonl"
    assert data_path.is_file()
    assert data_path.read_bytes() == archive_tasks_bytes

    before = _snapshot_bytes(gym_root)
    again = _run_publish(_payload(compile_result_path), gym_root)
    assert again.returncode == 0, again.stderr
    assert _snapshot_bytes(gym_root) == before


def test_publish_register_updates_overlay_and_package_paths(tmp_path):
    _, snapshot, compile_result_path = run_template_build(
        tmp_path, TICKETS, asset_name="register-assets"
    )
    gym_root = _gym_root(tmp_path)
    overlay = gym_root / "overlay"
    (overlay / "overlay-manifest.yaml").write_text(
        yaml.safe_dump(
            {"schema_version": 2, "components": {}}, sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    config_path = overlay / "resources_servers/synth_env/configs/synth_env.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {"synth_env": {"resources_servers": {"synth_env": {}}}},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    tool = gym_root / "scripts/refactor/update_overlay_manifest.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("raise SystemExit(0)\n", encoding="utf-8")

    result = _run_publish(_payload(compile_result_path), gym_root, register=True)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["registration"]["status"] == "registered"
    assert output["registration"]["component"] == "benchmarks/synth-office-power"

    overlay_manifest = yaml.safe_load(
        (overlay / "overlay-manifest.yaml").read_text(encoding="utf-8")
    )
    component = overlay_manifest["components"]["benchmarks/synth-office-power"]
    assert component["kind"] == "component"
    assert component["complete"] is True

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["synth_env"]["resources_servers"]["synth_env"]["package_paths"]
    assert paths == [output["registration"]["packagePath"]]
    assert paths[0].startswith("benchmarks/synth-office-power/assets/gym-env-package-v")
    assert snapshot["envRelease"]["fingerprint"] in paths[0]


def test_publisher_rejects_release_fingerprint_mismatch(tmp_path):
    _, snapshot, compile_result_path = run_template_build(
        tmp_path, TICKETS, asset_name="mismatch-assets"
    )
    tampered = json.loads(compile_result_path.read_text(encoding="utf-8"))
    tampered["envRelease"]["fingerprint"] = "0" * 16
    tampered_path = tmp_path / "tampered-compile-result.json"
    tampered_path.write_text(
        json.dumps(tampered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = _run_publish(_payload(tampered_path), _gym_root(tmp_path))
    assert result.returncode == 2
    assert "fingerprint" in result.stderr


def test_publisher_rejects_tampered_taskset_file(tmp_path):
    _, snapshot, compile_result_path = run_template_build(
        tmp_path, TICKETS, asset_name="taskset-tamper-assets"
    )
    release_dir = Path(snapshot["envRelease"]["assetDir"]).parent
    taskset_path = release_dir / artifacts.ENV_TASKS_FILE
    taskset_payload = json.loads(taskset_path.read_text(encoding="utf-8"))
    taskset_payload["tasks"] = taskset_payload["tasks"][:-1]
    taskset_payload["itemCount"] = len(taskset_payload["tasks"])
    taskset_path.write_text(
        json.dumps(taskset_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = _run_publish(_payload(compile_result_path), _gym_root(tmp_path))
    assert result.returncode == 2
    assert "taskset" in result.stderr


def test_publisher_rejects_taskset_that_diverges_from_package(tmp_path):
    _, snapshot, compile_result_path = run_template_build(
        tmp_path, TICKETS, asset_name="taskset-divergence-assets"
    )
    release_dir = Path(snapshot["envRelease"]["assetDir"]).parent
    taskset_path = release_dir / artifacts.ENV_TASKS_FILE
    taskset_payload = json.loads(taskset_path.read_text(encoding="utf-8"))
    taskset_payload["tasks"][0]["goal"] = "tampered goal that still validates"
    taskset_path.write_text(
        json.dumps(taskset_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = _run_publish(_payload(compile_result_path), _gym_root(tmp_path))
    assert result.returncode == 2
    assert "does not match" in result.stderr


def test_publish_rejects_unknown_payload_fields(tmp_path):
    _, snapshot, compile_result_path = run_template_build(
        tmp_path, TICKETS, asset_name="unknown-field-assets"
    )
    payload = _payload(compile_result_path)
    payload["resourcesServer"] = {"type": "resources_servers", "name": "placeholder"}
    result = _run_publish(payload, _gym_root(tmp_path))
    assert result.returncode == 2
    assert "unknown fields" in result.stderr


def test_publish_rejects_missing_overlay_checkout(tmp_path):
    _, snapshot, compile_result_path = run_template_build(
        tmp_path, TICKETS, asset_name="missing-overlay-assets"
    )
    gym_root = tmp_path / "no-overlay"
    gym_root.mkdir()
    result = _run_publish(_payload(compile_result_path), gym_root)
    assert result.returncode == 2
    assert "overlay" in result.stderr
