"""Offline, synthetic SWE-bench-format source import tests (swe follow-up M2).

Covers the full offline derivation pipeline: JSONL import + offline admission markers (expressible / excluded
build_required), the gold expected tree materialized into trusted verifier_spec, two-arm real build evaluation, suite freezing,
and eval worker scoring with the real ``swe_offline`` runner (gold 1.0 / corrupted 0.0 /
missing invalid_candidate). Fully offline, depends only on git.
"""

from __future__ import annotations
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.pipelines.benchmarks.families.swe import import_swe_jsonl


import json
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import BenchmarkError, file_sha256
from data_designer.pipelines.benchmarks.runners.swe_offline import VERSION
from data_designer.pipelines.benchmarks.families.swe import verify_swe_source_manifest
from data_designer.pipelines.benchmarks.suite import create_suite
from tests.benchmarks.benchmark_review import approve_fixture_suite
from tests.benchmarks.swe_offline_fixture import (
    EXCLUDED_INSTANCE_ID,
    INSTANCE_ID,
    expected_file_shas,
    prepare_area,
    swe_variant,
    write_audit_roots,
)

RELEASE = "fixture-release-1"


def _source_request(pins: dict) -> dict:
    return {
        "schemaVersion": 1,
        "request": {
            "snapshotPath": pins["snapshot_path"],
            "snapshotSha256": pins["snapshot_sha256"],
            "release": RELEASE,
            "split": "development",
            "upstreamSplit": "dev",
            "license": {"id": "fixture-only", "derive_allowed": True},
            "runtimeRegistryPath": pins["runtime_registry_path"],
            "runtimeRegistrySha256": pins["runtime_registry_sha256"],
            "offlineBaseRegistryPath": pins["offline_registry_path"],
            "offlineBaseRegistrySha256": pins["offline_registry_sha256"],
        },
    }


def _write_input(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _import_source(tmp_path: Path) -> tuple[dict, Path, dict]:
    pins = prepare_area(tmp_path)
    input_path = _write_input(tmp_path / "input.json", _source_request(pins))
    output_path = tmp_path / "source.json"
    request = json.loads(Path(input_path).read_text(encoding="utf-8"))["request"]
    source = import_swe_jsonl(
        request["snapshotPath"],
        request["snapshotSha256"],
        release=request["release"],
        split=request["split"],
        upstream_split=request["upstreamSplit"],
        license_record=request["license"],
        runtime_registry_path=request.get("runtimeRegistryPath"),
        runtime_registry_sha256=request.get("runtimeRegistrySha256"),
        offline_base_registry_path=request.get("offlineBaseRegistryPath"),
        offline_base_registry_sha256=request.get("offlineBaseRegistrySha256"),
    )
    Path(str(output_path)).write_text(json.dumps(source), encoding="utf-8")
    source = json.loads(output_path.read_text(encoding="utf-8"))
    assert source["family"] == "swe"
    assert source["source_snapshot"]["sha256"] == pins["snapshot_sha256"]
    assert source["offline_base_registry_sha256"] == pins["offline_registry_sha256"]
    expressible = source["cases"][0]
    assert expressible["swe"]["offline"] == {
        "admission": "expressible",
        "base_snapshot": {
            "path": pins["base_zip_path"],
            "sha256": pins["base_zip_sha256"],
        },
    }
    expressible_two = source["cases"][1]
    assert expressible_two["swe"]["offline"] == {
        "admission": "expressible",
        "base_snapshot": {
            "path": pins["base_zip_two_path"],
            "sha256": pins["base_zip_two_sha256"],
        },
    }
    excluded = source["cases"][2]
    assert excluded["swe"]["offline"] == {
        "admission": "excluded",
        "reason": "build_required",
    }
    assert expressible["runtime"]["verifier"]["version"] == VERSION
    verify_swe_source_manifest(source)
    expressible["swe"]["offline"]["base_snapshot"]["sha256"] = "0" * 64
    with pytest.raises(BenchmarkError, match="drift"):
        verify_swe_source_manifest(source)
    expressible["swe"]["offline"]["base_snapshot"]["sha256"] = pins["base_zip_sha256"]
    output_text = output_path.read_text(encoding="utf-8")
    assert "PRIVATE TARGET TEST" not in output_text
    assert "-value = 1" not in output_text
    return source, output_path, pins


def test_swe_import_marks_offline_admission_and_keeps_gold_in_snapshot(tmp_path):
    _import_source(tmp_path)






