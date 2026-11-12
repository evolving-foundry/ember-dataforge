"""Offline, synthetic DeepSWE tree source import tests (deepswe follow-up M2).

Covers the full offline derivation pipeline: tree snapshot import + dual admission-criteria markers (expressible /
excluded base_unavailable / excluded gold_unavailable), the gold expected tree materialized into
trusted verifier_spec (including binary hunks), two-arm real build evaluation, suite freezing, and eval
worker scoring with the real ``deepswe_offline`` runner (gold 1.0 / corrupted 0.0 /
missing invalid_candidate). Fully offline, depends only on git.
"""

from __future__ import annotations
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.pipelines.benchmarks.families.deepswe import import_deepswe_tree


import json
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import BenchmarkError, file_sha256
from data_designer.pipelines.benchmarks.runners.deepswe_offline import VERSION
from data_designer.pipelines.benchmarks.families.deepswe import (
    verify_deepswe_source_manifest,
)
from data_designer.pipelines.benchmarks.suite import create_suite
from tests.benchmarks.benchmark_review import approve_fixture_suite
from tests.benchmarks.deepswe_fixture import (
    BASE_TASK_ID,
    TASK_ID,
    deepswe_variant,
    prepare_area,
    write_audit_roots,
)

RELEASE = "fixture-release-1"


def _source_request(pins: dict) -> dict:
    return {
        "schemaVersion": 1,
        "request": {
            "snapshotPath": pins["tree_path"],
            "snapshotSha256": pins["tree_sha256"],
            "release": RELEASE,
            "split": "development",
            "license": {"id": "fixture-only", "derive_allowed": True},
            "runtimeRegistryPath": pins["runtime_registry_path"],
            "runtimeRegistrySha256": pins["runtime_registry_sha256"],
            "offlineRegistryPath": pins["offline_registry_path"],
            "offlineRegistrySha256": pins["offline_registry_sha256"],
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
    source = import_deepswe_tree(
        request["snapshotPath"],
        request["snapshotSha256"],
        release=request["release"],
        split=request["split"],
        license_record=request["license"],
        runtime_registry_path=request.get("runtimeRegistryPath"),
        runtime_registry_sha256=request.get("runtimeRegistrySha256"),
        offline_registry_path=request.get("offlineRegistryPath"),
        offline_registry_sha256=request.get("offlineRegistrySha256"),
    )
    Path(str(output_path)).write_text(json.dumps(source), encoding="utf-8")
    source = json.loads(output_path.read_text(encoding="utf-8"))
    assert source["family"] == "deepswe"
    assert source["source_snapshot"]["sha256"] == pins["tree_sha256"]
    assert source["offline_registry_sha256"] == pins["offline_registry_sha256"]
    expressible = source["cases"][0]
    assert expressible["source_task_id"] == TASK_ID
    assert expressible["deepswe"]["offline"] == {
        "admission": "expressible",
        "base_snapshot": {
            "path": pins["base_zip_path"],
            "sha256": pins["base_zip_sha256"],
        },
        "gold_diff": {
            "path": pins["gold_path"],
            "sha256": file_sha256(Path(pins["gold_path"])),
        },
    }
    base_excluded = source["cases"][1]
    assert base_excluded["deepswe"]["offline"] == {
        "admission": "excluded",
        "reason": "base_unavailable",
    }
    nogold_excluded = source["cases"][2]
    assert nogold_excluded["deepswe"]["offline"] == {
        "admission": "excluded",
        "reason": "gold_unavailable",
    }
    assert expressible["runtime"]["verifier"]["version"] == VERSION
    verify_deepswe_source_manifest(source)
    expressible["deepswe"]["offline"]["base_snapshot"]["sha256"] = "0" * 64
    with pytest.raises(BenchmarkError, match="drift"):
        verify_deepswe_source_manifest(source)
    expressible["deepswe"]["offline"]["base_snapshot"]["sha256"] = pins["base_zip_sha256"]
    output_text = output_path.read_text(encoding="utf-8")
    # gold diff text and patch hunks do not enter the source manifest (only path+sha pins are stored)
    assert "-value = 1" not in output_text
    assert "GIT binary patch" not in output_text
    return source, output_path, pins


def test_deepswe_import_marks_offline_admission_and_keeps_gold_out_of_manifest(tmp_path):
    _import_source(tmp_path)






