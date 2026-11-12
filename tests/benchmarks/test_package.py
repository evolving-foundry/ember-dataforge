from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.build import build_benchmark_taskset
from data_designer.pipelines.benchmarks.contracts import file_sha256
from data_designer.pipelines.benchmarks.package import freeze_package
from data_designer.pipelines.benchmarks.reader import enumerate_public, read_package
from data_designer.pipelines.benchmarks.verification import verify_task
from data_designer.storage.catalog import ArtifactCatalog


@pytest.fixture
def benchmark_package(tmp_path, monkeypatch):
    import data_designer.pipelines.benchmarks.build as builder
    monkeypatch.setattr(builder, "assess_taskset", lambda items: {"quality_overall_score": 1.0})
    from data_designer.pipelines.benchmarks.runners import workbuddy
    runner = Path(workbuddy.__file__).resolve()
    environment = tmp_path / "environment.json"
    environment.write_text('{}')
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
    source_path.write_text(json.dumps(source)); config_path.write_text(json.dumps(config))
    catalog = ArtifactCatalog()
    build = build_benchmark_taskset(catalog, str(source_path), str(config_path), str(tmp_path / "trusted"), "fixture")
    pin = build["taskset"]
    payload = catalog.get_artifact("taskset", pin["ref_id"], pin["version"])["payload"]
    return freeze_package(payload, tmp_path / "package")


def test_freeze_read_verify_and_private_boundary(benchmark_package, tmp_path):
    package = benchmark_package
    public = enumerate_public(package)[0]
    assert 'gold' not in json.dumps(public)
    assert 'private' not in json.dumps(public)
    candidate = tmp_path / "candidate"
    (candidate / "output").mkdir(parents=True)
    (candidate / "output/result.json").write_text('{"total":42}')
    verdict = verify_task(package["manifest_uri"], package["digest"], public["task_id"], candidate)
    assert verdict["passed"] is True and verdict["score"] == 1
    (candidate / "output/result.json").write_text('{"total":0}')
    verdict = verify_task(package["manifest_uri"], package["digest"], public["task_id"], candidate)
    assert verdict["passed"] is False and verdict["score"] == 0
    with pytest.raises(ValueError, match="overlaps private"):
        verify_task(package["manifest_uri"], package["digest"], public["task_id"], Path(package["manifest_uri"]).parent)
    with pytest.raises(ValueError, match="digest mismatch"):
        read_package(package["manifest_uri"], expected_digest="sha256:" + "0" * 64)
    moved = tmp_path / "relocated"
    shutil.move(Path(package["manifest_uri"]).parent, moved)
    assert read_package(moved, verify_resources=True)["digest"] == package["digest"]


def test_evaluator_bridge_and_explicit_unsupported_requirements(benchmark_package, tmp_path):
    from data_designer.integrations.evaluator.bridge import PROTOCOL, handle
    from data_designer.integrations.evaluator.export import export_evaluator
    request = {"protocol_version": PROTOCOL, "manifest": benchmark_package["manifest_uri"], "digest": benchmark_package["digest"]}
    source = handle("resolve-source", request)
    assert source["revision"] == benchmark_package["digest"]
    items = handle("enumerate", request)["items"]
    assert items[0]["deliverables"] == ["output/result.json"]
    assert "gold" not in json.dumps(items)
    for requirements in ({"attempts": 2}, {"family_weights": {"workbuddy": 1}}, {"seed": 42}, {"aggregation": "macro"}):
        with pytest.raises(ValueError):
            export_evaluator(request["manifest"], tmp_path / "unsupported", requirements=requirements)
    assert not (tmp_path / "unsupported").exists()
