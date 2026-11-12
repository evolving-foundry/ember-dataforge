"""C1 probes: CLI `benchmark show` / `benchmark verify` (design document: evaluation-
extension capability completion, section 3.7).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.build import build_benchmark_taskset
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.sources.hf_single_turn_seed import BENCHMARK_FAMILIES
from tests.benchmarks.benchmark_review import approve_fixture_suite
from tests.benchmarks.test_single_turn_source import (
    build_config,
    derive_rows,
    prepare_family,
)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cli-show-verify")
    prepared = prepare_family(tmp_path, "reasoning_qa")
    rows = derive_rows(
        prepared,
        tuple(name for name, fam in BENCHMARK_FAMILIES.items()
              if fam == "reasoning_qa"),
    )
    assert rows
    config, config_path = build_config(tmp_path, prepared, rows)
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(prepared["source"], ensure_ascii=False), encoding="utf-8"
    )
    store = ArtifactCatalog(str(tmp_path / "catalog"))
    snapshot = build_benchmark_taskset(
        store, str(source_path), config_path, str(tmp_path / "trusted"),
        "cli-fixture",
        expected_family="reasoning_qa",
        expected_release=prepared["source"]["release"],
        expected_split="development",
    )
    yield {"store": store, "snapshot": snapshot, "tmp_path": tmp_path}
    store.close()


def _cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "data_designer.cli.main", *argv],
        capture_output=True, text=True,
    )


def _suite_ref(suite):
    return f"{suite['ref_id']}@{suite['version']}"


def test_show_candidate_matches_suite_payload(frozen):
    store, snapshot = frozen["store"], frozen["snapshot"]
    from data_designer.pipelines.benchmarks.suite import create_suite

    suite = create_suite(store, "cli-show", [snapshot["taskset"]])
    frozen["suite"] = suite

    result = _cli("benchmark", "show", "--catalog", store.uri,
                  "--suite", _suite_ref(suite))
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["suite"]["ref_id"] == suite["ref_id"]
    assert body["suite"]["suite_sha256"] == suite["suite_sha256"]
    assert body["suite"]["split"] == "development"
    pin = suite["tasksets"][0]
    taskset = store.get_artifact("taskset", pin["ref_id"], pin["version"])["payload"]
    items = taskset["taskset"]
    entry = body["tasksets"][0]
    assert entry["family"] == pin["family"]
    assert entry["fingerprint"] == pin["fingerprint"]
    assert entry["task_count"] == len(items)
    distribution = {}
    for item in items:
        distribution[item["difficulty"]] = distribution.get(item["difficulty"], 0) + 1
    assert entry["difficulty_distribution"] == distribution
    assert body["publication"] == {"status": "candidate", "reason": "awaiting review"}
    assert body["review_evidence"] is None


def test_show_reflects_approval_and_evidence_pin(frozen, tmp_path):
    store, snapshot = frozen["store"], frozen["snapshot"]
    from data_designer.pipelines.benchmarks.suite import create_suite

    suite = create_suite(store, "cli-show-approved", [snapshot["taskset"]])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    pin = approve_fixture_suite(store, suite, evidence)

    result = _cli("benchmark", "show", "--catalog", store.uri,
                  "--suite", _suite_ref(suite))
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["publication"]["status"] == "approved"
    assert body["review_evidence"] == pin


def test_verify_suite_passes_then_fails_on_tampered_catalog(frozen, tmp_path):
    store, snapshot = frozen["store"], frozen["snapshot"]
    from data_designer.pipelines.benchmarks.suite import create_suite

    suite = create_suite(store, "cli-verify-ok", [snapshot["taskset"]])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    approve_fixture_suite(store, suite, evidence)

    result = _cli("benchmark", "verify", "--catalog", store.uri,
                  "--suite", _suite_ref(suite))
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["verified"] is True
    assert body["publication"]["status"] == "approved"

    tampered = tmp_path / "tampered-catalog"
    shutil.copytree(store.uri, tampered)
    for path in tampered.glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry["kind"] == "taskset" and entry["ref_id"] == suite["tasksets"][0]["ref_id"]:
            entry["payload"]["family"] = "tampered"
            path.write_text(json.dumps(entry), encoding="utf-8")
            break
    else:
        pytest.fail("taskset catalog entry not found")
    result = _cli("benchmark", "verify", "--catalog", str(tampered),
                  "--suite", _suite_ref(suite))
    assert result.returncode == 1
    assert result.stderr.strip()


def test_verify_quarantined_is_a_legal_terminal_state(frozen, tmp_path):
    store, snapshot = frozen["store"], frozen["snapshot"]
    from data_designer.pipelines.benchmarks.suite import (
        create_suite,
        set_publication_status,
    )

    suite = create_suite(store, "cli-verify-quarantine", [snapshot["taskset"]])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    approve_fixture_suite(store, suite, evidence)
    set_publication_status(store, suite["ref_id"], suite["version"],
                           "quarantined", "cli verify drill")
    result = _cli("benchmark", "verify", "--catalog", store.uri,
                  "--suite", _suite_ref(suite))
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["verified"] is True
    assert body["publication"]["status"] == "quarantined"


def test_show_approved_only_rejects_candidate(frozen):
    store, snapshot = frozen["store"], frozen["snapshot"]
    from data_designer.pipelines.benchmarks.suite import create_suite

    candidate = create_suite(store, "cli-show-candidate", [snapshot["taskset"]])
    result = _cli("benchmark", "show", "--catalog", store.uri,
                  "--suite", _suite_ref(candidate), "--approved-only")
    assert result.returncode == 1
    assert "not approved" in result.stderr


def test_verify_package_passes_and_fails_on_missing_resource(tmp_path, monkeypatch):
    import data_designer.pipelines.benchmarks.build as builder
    monkeypatch.setattr(builder, "assess_taskset", lambda items: {"quality_overall_score": 1.0})
    from data_designer.pipelines.benchmarks.runners import workbuddy
    from data_designer.pipelines.benchmarks.contracts import file_sha256
    runner = Path(workbuddy.__file__).resolve()
    environment = tmp_path / "environment.json"
    environment.write_text("{}")
    reference, negative = tmp_path / "reference", tmp_path / "negative"
    for root, value in ((reference, {"total": 42}), (negative, {"total": 0})):
        (root / "output").mkdir(parents=True)
        (root / "output/result.json").write_text(json.dumps(value))
    runtime = {"environment": {"path": str(environment), "sha256": file_sha256(environment)},
               "verifier": {"path": str(runner), "sha256": file_sha256(runner),
                            "command": [sys.executable, str(runner)], "version": workbuddy.VERSION}, "assets": []}
    case = {"source_task_id": "seed", "task_text": "Original source accounting exercise.",
            "initial_state_sha256": "a" * 64, "expected_state_sha256": "b" * 64}
    source = {"schema_version": "benchmark-source-v1", "family": "workbuddy", "release": "fixture",
              "split": "development", "license": {"id": "owned-fixture", "derive_allowed": True},
              "cases": [case], **runtime}
    variant = {"source_task_id": "seed", "variant_id": "v1", "operation": "mutation",
               "public": {"task_text": "Compute the total from the supplied fictional accounting entries.",
                          "success_criteria": "Write output/result.json with the computed total.",
                          "tools_available": ["files"], "difficulty": "easy"},
               "private": {"initial_state_sha256": "c" * 64, "expected_state_sha256": "d" * 64,
                           "verifier_spec": {"kind": "artifact_compare", "generator_ref": "owned-fixture@1",
                                             "payload": {"artifact": {"path": "output/result.json"}, "gold": {"total": 42}}}},
               "audit": {"reviewed": True, "reference_root": str(reference), "negative_root": str(negative)}}
    config = {"schema_version": "benchmark-augment-v1", "family": "workbuddy", "release": "fixture",
              "split": "development", "variants": [variant]}
    source_path, config_path = tmp_path / "source.json", tmp_path / "augment.json"
    source_path.write_text(json.dumps(source))
    config_path.write_text(json.dumps(config))
    from data_designer.pipelines.benchmarks.package import freeze_package
    catalog = ArtifactCatalog()
    build = build_benchmark_taskset(catalog, str(source_path), str(config_path),
                                    str(tmp_path / "trusted"), "cli-package")
    payload = catalog.get_artifact("taskset", build["taskset"]["ref_id"],
                                   build["taskset"]["version"])["payload"]
    package = freeze_package(payload, tmp_path / "package")
    package_dir = Path(package["manifest_uri"]).parent

    result = _cli("benchmark", "verify", "--package", str(package_dir))
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["verified"] is True
    assert body["digest"] == package["digest"]

    environment.unlink()
    result = _cli("benchmark", "verify", "--package", str(package_dir))
    assert result.returncode == 1
    assert result.stderr.strip()


def test_cli_argument_discipline(frozen):
    store, suite = frozen["store"], frozen["suite"]
    result = _cli("benchmark", "verify", "--catalog", store.uri)
    assert result.returncode == 1
    result = _cli("benchmark", "verify", "--catalog", store.uri,
                  "--suite", _suite_ref(suite), "--package", "somewhere")
    assert result.returncode == 1
    result = _cli("benchmark", "show", "--catalog", store.uri, "--suite", "no-version")
    assert result.returncode == 1
    result = _cli("benchmark", "show", "--catalog", store.uri,
                  "--suite", "missing-suite@3")
    assert result.returncode == 1
