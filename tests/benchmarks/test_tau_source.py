"""Synthetic tau text fixture; no official task or simulator data is bundled."""

from __future__ import annotations
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.pipelines.benchmarks.families.tau import import_tau_snapshot


import json
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import BenchmarkError, file_sha256
from data_designer.pipelines.benchmarks.runners.tau_offline import VERSION
from data_designer.pipelines.benchmarks.families.tau import (
    assert_tau_offline_build,
    derive_tau_offline_spec,
    import_tau_snapshot,
    verify_tau_source_manifest,
)
from data_designer.pipelines.benchmarks.suite import create_suite
from tests.benchmarks.benchmark_review import approve_fixture_suite

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    _PROJECT_ROOT / "data_designer" / "pipelines" / "benchmarks" / "runners" / "tau_offline.py"
)
DELIVERABLE = "tau_offline_artifact.json"
REQUIRED_MESSAGE = "order update confirmed"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _task(reward_basis: list[str], communicate_info: list[str] | None) -> dict:
    criteria: dict = {
        "actions": [
            {"name": "update_order", "arguments": {"value": "PRIVATE TARGET"}}
        ],
        "reward_basis": reward_basis,
    }
    if communicate_info is not None:
        criteria["communicate_info"] = communicate_info
    return {
        "id": "synthetic-1",
        "user_scenario": {
            "persona": "PRIVATE PERSONA",
            "instructions": "PRIVATE SIMULATOR REQUEST: change the order",
        },
        "initial_state": {"initialization_data": {"order": "old"}},
        "evaluation_criteria": criteria,
    }


def _fixture(
    tmp_path: Path,
    reward_basis: list[str] | None = None,
    communicate_info: list[str] | None = None,
) -> dict:
    if reward_basis is None:
        reward_basis = ["DB", "COMMUNICATE"]
    if communicate_info is None:
        communicate_info = [REQUIRED_MESSAGE]
    task = _task(reward_basis, communicate_info)
    tasks = tmp_path / "tasks.json"
    splits = tmp_path / "split_tasks.json"
    policy = tmp_path / "policy.md"
    db = tmp_path / "db.json"
    _write(tasks, [task])
    _write(splits, {"train": [task["id"]], "test": [], "base": [task["id"]]})
    policy.write_text("PRIVATE POLICY: verify the caller", encoding="utf-8")
    _write(db, {"order": "old", "account": "PRIVATE ACCOUNT"})
    environment = tmp_path / "environment.txt"
    environment.write_text("fixture simulator", encoding="utf-8")
    verifier_pin = {
        "path": str(RUNNER_PATH),
        "sha256": file_sha256(RUNNER_PATH),
        "version": VERSION,
        "command": [sys.executable, str(RUNNER_PATH)],
    }
    registry = tmp_path / "runtime.json"
    _write(
        registry,
        {
            "schema_version": "benchmark-runtime-registry-v1",
            "cases": {
                task["id"]: {
                    "environment": {
                        "path": str(environment),
                        "sha256": file_sha256(environment),
                    },
                    "verifier": dict(verifier_pin),
                    "assets": [],
                }
            },
        },
    )
    return {
        "tasksPath": str(tasks),
        "tasksSha256": file_sha256(tasks),
        "splitPath": str(splits),
        "splitSha256": file_sha256(splits),
        "policyPath": str(policy),
        "policySha256": file_sha256(policy),
        "dbPath": str(db),
        "dbSha256": file_sha256(db),
        "domain": "retail",
        "release": "tau-fixture-1",
        "split": "development",
        "upstreamSplit": "train",
        "license": {"id": "fixture-only", "derive_allowed": True},
        "runtimeRegistryPath": str(registry),
        "runtimeRegistrySha256": file_sha256(registry),
    }


def _import_source(request: dict, tmp_path: Path) -> dict:
    input_path = tmp_path / "import-input.json"
    _write(input_path, {"schemaVersion": 1, "request": request})
    source_path = tmp_path / "source.json"
    request = json.loads(Path(str(input_path)).read_text(encoding="utf-8"))["request"]
    source = import_tau_snapshot(
        request["tasksPath"], request["tasksSha256"],
        split_path=request["splitPath"],
        split_sha256=request["splitSha256"],
        policy_path=request["policyPath"],
        policy_sha256=request["policySha256"],
        db_path=request["dbPath"],
        db_sha256=request["dbSha256"],
        user_db_path=request.get("userDbPath"),
        user_db_sha256=request.get("userDbSha256"),
        domain=request["domain"],
        release=request["release"],
        split=request["split"],
        upstream_split=request["upstreamSplit"],
        license_record=request["license"],
        runtime_registry_path=request.get("runtimeRegistryPath"),
        runtime_registry_sha256=request.get("runtimeRegistrySha256"),
    )
    Path(str(source_path)).write_text(json.dumps(source), encoding="utf-8")
    return json.loads(source_path.read_text(encoding="utf-8"))


def _expected_deliverable(db_order: str) -> dict:
    return {
        "db_state": {"order": db_order, "account": "PRIVATE ACCOUNT"},
        "messages": [{"role": "agent", "content": REQUIRED_MESSAGE}],
    }


def _write_deliverable(root: Path, db_order: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / DELIVERABLE).write_text(
        json.dumps(_expected_deliverable(db_order), ensure_ascii=False),
        encoding="utf-8",
    )




def test_tau_offline_admission_marks_excluded_tasks(tmp_path):
    request = _fixture(tmp_path, reward_basis=["ENV"], communicate_info=None)
    source = _import_source(request, tmp_path)
    assert source["cases"][0]["tau"]["offline"] == {
        "admission": "excluded",
        "reason": "simulator_required",
    }
    with pytest.raises(BenchmarkError, match="not admitted for offline derivation"):
        derive_tau_offline_spec(
            source, source["cases"][0], {"gold": {"order": "exchanged"}}
        )
    with pytest.raises(BenchmarkError, match="excluded from offline derivation"):
        assert_tau_offline_build({"verifier": {"version": VERSION}}, {}, source["cases"][0])


def test_tau_offline_unassertable_communicate_info_is_excluded(tmp_path):
    request = _fixture(tmp_path, communicate_info=[])
    source = _import_source(request, tmp_path)
    assert source["cases"][0]["tau"]["offline"] == {
        "admission": "excluded",
        "reason": "communicate_info_not_assertable",
    }


def test_tau_offline_directive_validation(tmp_path):
    request = _fixture(tmp_path)
    source = _import_source(request, tmp_path)
    source_case = source["cases"][0]
    directive = {
        "artifact": {"path": DELIVERABLE},
        "expected_state": {
            "patch": [{"op": "replace", "path": ["order"], "value": "exchanged"}]
        },
        "messages": [REQUIRED_MESSAGE],
    }
    derived = derive_tau_offline_spec(source, source_case, directive)
    spec = derived["verifier_spec"]
    assert spec["kind"] == "unit_test"
    assert spec["payload"]["artifact"]["path"] == DELIVERABLE
    assert "db_state.json" in spec["payload"]["test_script"]
    assert derived["initial_state_sha256"] != source_case["initial_state_sha256"]
    assert derived["expected_state_sha256"] != source_case["expected_state_sha256"]

    unknown = dict(directive)
    unknown["unexpected"] = 1
    with pytest.raises(BenchmarkError, match="accepts only"):
        derive_tau_offline_spec(source, source_case, unknown)
    both = {**directive, "gold": {"order": "exchanged"}}
    with pytest.raises(BenchmarkError, match="exactly one of expected_state or gold"):
        derive_tau_offline_spec(source, source_case, both)
    neither = {"artifact": {"path": DELIVERABLE}, "messages": [REQUIRED_MESSAGE]}
    with pytest.raises(BenchmarkError, match="exactly one of expected_state or gold"):
        derive_tau_offline_spec(source, source_case, neither)
    no_messages = {"expected_state": directive["expected_state"]}
    with pytest.raises(BenchmarkError, match="assertable required message strings"):
        derive_tau_offline_spec(source, source_case, no_messages)
    with pytest.raises(BenchmarkError, match="artifact_compare cannot assert messages"):
        derive_tau_offline_spec(
            source, source_case,
            {"gold": {"order": "exchanged"}, "messages": [REQUIRED_MESSAGE]},
        )
    with pytest.raises(BenchmarkError, match="clean relative path"):
        derive_tau_offline_spec(
            source, source_case,
            {**directive, "artifact": {"path": "../escape.json"}},
        )

    runtime = {
        "verifier": {
            "path": str(RUNNER_PATH), "sha256": file_sha256(RUNNER_PATH),
            "version": "legacy-action-replay",
            "command": [sys.executable, str(RUNNER_PATH)],
        }
    }
    with pytest.raises(BenchmarkError, match="must pin the offline runner"):
        assert_tau_offline_build(runtime, derived, source_case)
    with pytest.raises(BenchmarkError, match="lacks a unit_test/artifact_compare"):
        assert_tau_offline_build(
            {
                "verifier": {
                    "path": str(RUNNER_PATH), "sha256": file_sha256(RUNNER_PATH),
                    "version": VERSION,
                    "command": [sys.executable, str(RUNNER_PATH)],
                }
            },
            {}, source_case,
        )


def test_tau_offline_messages_need_communicate_basis(tmp_path):
    request = _fixture(tmp_path, reward_basis=["DB"], communicate_info=None)
    source = _import_source(request, tmp_path)
    assert source["cases"][0]["tau"]["offline"] == {
        "admission": "expressible",
        "mode": "db",
    }
    with pytest.raises(BenchmarkError, match="need a COMMUNICATE reward basis"):
        derive_tau_offline_spec(
            source, source["cases"][0],
            {
                "expected_state": {
                    "patch": [{"op": "replace", "path": ["order"], "value": "x"}]
                },
                "messages": [REQUIRED_MESSAGE],
            },
        )
    derived = derive_tau_offline_spec(
        source, source["cases"][0],
        {
            "expected_state": {
                "patch": [{"op": "replace", "path": ["order"], "value": "x"}]
            }
        },
    )
    assert derived["verifier_spec"]["kind"] == "unit_test"


def test_tau_split_and_voice_boundary(tmp_path):
    request = _fixture(tmp_path)
    args = dict(
        split_path=request["splitPath"], split_sha256=request["splitSha256"],
        policy_path=request["policyPath"], policy_sha256=request["policySha256"],
        db_path=request["dbPath"], db_sha256=request["dbSha256"],
        domain=request["domain"], release=request["release"],
        split="development", upstream_split="base",
        license_record=request["license"],
    )
    with pytest.raises(BenchmarkError, match="test/base cannot seed"):
        import_tau_snapshot(request["tasksPath"], request["tasksSha256"], **args)
