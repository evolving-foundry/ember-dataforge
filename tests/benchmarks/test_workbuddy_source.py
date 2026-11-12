"""workbuddy source import and build tests (M2): seed→source manifest, D2/D3 pins, contamination defenses, and full-chain build.

Aligned with implementation plan §2 M2 acceptance: fixture-derived tasks (≥1) complete ``benchmark_taskset_worker``,
producing taskset artifacts, with the item key set exactly ``PUBLIC_KEYS`` and ``benchmark.family == "workbuddy"``.
Seed materials use the repository's desensitized synthetic fixture (real subsets are not committed).
"""

import json
import tarfile
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import (
    PUBLIC_KEYS,
    file_sha256,
    normalized_text_sha256,
)
from data_designer.pipelines.benchmarks.families.workbuddy import (
    build_workbuddy_augment_config,
    derive_release,
    import_workbuddy_source,
    screen_workbuddy_contamination,
    verify_workbuddy_source_manifest,
)
from data_designer.pipelines.benchmarks.build import build_benchmark_taskset
from data_designer.pipelines.benchmarks.runners.workbuddy import VERSION as RUNNER_VERSION
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.sources.workbuddy_seed import WBBenchSeedAdapter

_BASE = Path(__file__).resolve().parents[2]
FIXTURE_DIR = (
    _BASE / "data_designer" / "sources" / "fixtures" / "wbbench_office_fixture"
)
RUNNER_PATH = _BASE / "data_designer" / "pipelines" / "benchmarks" / "runners" / "workbuddy.py"

LICENSE = {
    "id": "local-synthetic-fixture",
    "derive_allowed": True,
    "redistribute_allowed": False,
    "access_conditions": "synthetic fixture, local evaluation only",
}


def _seed_rows():
    return WBBenchSeedAdapter(FIXTURE_DIR).load_seed_tasks()


def _archive(tmp_path: Path):
    archive = tmp_path / "wb-bench-office-fixture-v1.0.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(FIXTURE_DIR, arcname="wb-bench-office-fixture-v1.0")
    return str(archive), file_sha256(archive)


def _import(tmp_path: Path, **overrides):
    rows = _seed_rows()
    archive_path, archive_sha = _archive(tmp_path)
    kwargs = dict(
        release=derive_release("synthetic-fixture", rows, archive_sha),
        split="development",
        license_record=dict(LICENSE),
        seed_source="synthetic-fixture",
        dataset_id="wb-bench-office-fixture-v1.0",
        archive_path=archive_path,
        archive_sha256=archive_sha,
    )
    kwargs.update(overrides)
    return import_workbuddy_source(rows, **kwargs), rows


def test_import_source_pins_release_license_and_exclusion(tmp_path):
    source, rows = _import(tmp_path)
    assert source["schema_version"] == "benchmark-source-v1"
    assert source["family"] == "workbuddy"
    assert source["release"].startswith("synthetic-fixture:")
    assert len(source["release"]) == len("synthetic-fixture:") + 8
    assert source["license"]["derive_allowed"] is True
    assert source["exclusion_text_sha256"] == sorted(
        {normalized_text_sha256(row["task_text"]) for row in rows}
    )
    assert len(source["cases"]) == len(rows)
    case = source["cases"][0]
    assert set(case) >= {
        "source_task_id", "task_text", "tools_available", "success_criteria",
        "difficulty", "initial_state_sha256", "expected_state_sha256",
        "workbuddy",
    }
    assert case["workbuddy"]["category"] in {"data-file-ops", "doc-ops"}


def test_release_semantics_follow_seed_source(tmp_path):
    rows = _seed_rows()
    assert derive_release("synthetic-fixture", rows).startswith(
        "synthetic-fixture:"
    )
    with pytest.raises(Exception, match="archive SHA-256"):
        derive_release("real-hf-subset", rows)
    real = derive_release("real-hf-subset", rows, "f" * 64)
    assert real == "wb-bench-" + "f" * 8


def test_real_subset_cannot_authorize_derivation(tmp_path):
    rows = _seed_rows()
    with pytest.raises(Exception, match="D3"):
        import_workbuddy_source(
            rows,
            release=derive_release("real-hf-subset", rows, "f" * 64),
            split="development",
            license_record={**LICENSE, "derive_allowed": True},
            seed_source="real-hf-subset",
            dataset_id="wb-bench-office-v1.0",
        )


def test_verify_source_manifest_rejects_drift(tmp_path):
    source, _ = _import(tmp_path)
    verify_workbuddy_source_manifest(source)
    drifted = json.loads(json.dumps(source))
    drifted["cases"][0]["task_text"] = "rewritten instruction"
    with pytest.raises(Exception, match="drift"):
        verify_workbuddy_source_manifest(drifted)


def test_contamination_screen_rejects_seed_text(tmp_path):
    rows = _seed_rows()
    screen_workbuddy_contamination(
        [{"task_text": "a genuinely different derived instruction"}], rows
    )
    with pytest.raises(Exception, match="matches a seed text"):
        screen_workbuddy_contamination(
            [{"task_text": rows[0]["task_text"]}], rows
        )


def _derived_row(seed_row):
    return {
        "source_task_id": seed_row["seed_id"],
        "task_text": (
            "The regional operations lead refreshed the Q1 ledger. Total the "
            "approved revenue rows by region from the workbook in the "
            "workspace and record which region leads, keeping the listed "
            "exclusions out of every aggregate."
        ),
        "tools_available": ["workdir"],
        "success_criteria": (
            "Write output/summary.json with exactly the keys total and "
            "top_region matching the frozen gold values."
        ),
        "difficulty": "medium",
        "item_fingerprint": "a" * 16,
        "verifier_spec": {
            "kind": "artifact_compare",
            "generator_ref": "workbuddy_mutation@fixture",
            "payload": {
                "artifact": {"path": "output/summary.json", "format": "json"},
                "gold": {"total": 1500.0, "top_region": "华东"},
                "fields": {
                    "total": {"type": "number", "tolerance": 0.01},
                    "top_region": {"type": "string"},
                },
            },
        },
    }


def _write(path: Path, value) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(path)


def _build_chain(tmp_path: Path, row):
    source, _ = _import(tmp_path)
    environment = tmp_path / "workspace-snapshot.bin"
    environment.write_bytes(b"frozen initial workspace snapshot")
    reference = tmp_path / "reference"
    negative = tmp_path / "negative"
    reference.mkdir()
    negative.mkdir()
    deliverable = reference / "output" / "summary.json"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text(
        json.dumps({"total": 1500.0, "top_region": "华东"}), encoding="utf-8"
    )
    bad = negative / "output" / "summary.json"
    bad.parent.mkdir(parents=True)
    bad.write_text(
        json.dumps({"total": 1.0, "top_region": "华南"}), encoding="utf-8"
    )
    config = build_workbuddy_augment_config(
        [row],
        release=source["release"],
        split=source["split"],
        strategy="mutation",
        environment={
            "path": str(environment),
            "sha256": file_sha256(environment),
        },
        verifier_path=str(RUNNER_PATH),
        audit_roots={
            "reviewed": True,
            "reference_root": str(reference),
            "negative_root": str(negative),
        },
    )
    source_path = _write(tmp_path / "source.json", source)
    config_path = _write(tmp_path / "config.json", config)
    snapshot = build_benchmark_taskset(
        ArtifactCatalog(str(tmp_path / "catalog")),
        source_path, config_path, str(tmp_path / "trusted"), "workbuddy-fixture",
        expected_family="workbuddy", expected_release=source["release"],
        expected_split="development",
    )
    return snapshot, config


def test_augment_config_builds_workbuddy_taskset(tmp_path):
    rows = _seed_rows()
    snapshot, config = _build_chain(tmp_path, _derived_row(rows[0]))
    assert snapshot["item_count"] == 1
    assert snapshot["taskset"]["ref_id"] == "workbuddy-fixture"
    store = ArtifactCatalog(str(tmp_path / "catalog"))
    try:
        pin = snapshot["taskset"]
        payload = store.get_artifact("taskset", pin["ref_id"], pin["version"])[
            "payload"
        ]
        item = payload["taskset"][0]
        assert set(item) == PUBLIC_KEYS
        assert item["benchmark"]["family"] == "workbuddy"
        assert item["task_id"].startswith("workbuddy:")
        assert item["benchmark"]["source_release"].startswith("synthetic-fixture:")
        assert "verifier_spec" not in json.dumps(item)
        assert "gold" not in json.dumps(item["benchmark"])
        runtime = payload["runtime_by_task"][item["task_id"]]
        assert runtime["verifier"]["version"] == RUNNER_VERSION
        assert str(RUNNER_PATH) in runtime["verifier"]["command"]
        manifest_path = (
            Path(payload["trusted_manifest_dir"])
            / f"{item['benchmark']['case_manifest_sha256']}.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        spec = manifest["private"]["verifier_spec"]
        assert spec["kind"] == "artifact_compare"
        assert spec["payload"]["gold"] == {"total": 1500.0, "top_region": "华东"}
        assert manifest["private"]["deliverable_path"] == "output/summary.json"
        audit = payload["build_report"]["audit"][0]
        assert audit["reference"]["passed"] is True
        assert audit["negative"]["passed"] is False
    finally:
        store.close()


def test_augment_config_rejects_escaping_deliverable(tmp_path):
    rows = _seed_rows()
    row = _derived_row(rows[0])
    row["verifier_spec"]["payload"]["artifact"]["path"] = "../escape.json"
    environment = tmp_path / "env.bin"
    environment.write_bytes(b"workspace")
    with pytest.raises(Exception, match="clean relative path"):
        build_workbuddy_augment_config(
            [row],
            release="synthetic-fixture:abcd1234",
            split="development",
            strategy="mutation",
            environment={"path": str(environment), "sha256": file_sha256(environment)},
            verifier_path=str(RUNNER_PATH),
            audit_roots={"reviewed": True, "reference_root": str(tmp_path),
                         "negative_root": str(tmp_path)},
        )


def test_augment_config_rejects_unknown_operation(tmp_path):
    rows = _seed_rows()
    environment = tmp_path / "env.bin"
    environment.write_bytes(b"workspace")
    with pytest.raises(Exception, match="unsupported workbuddy operation"):
        build_workbuddy_augment_config(
            [_derived_row(rows[0])],
            release="synthetic-fixture:abcd1234",
            split="development",
            strategy="cross_app",
            environment={"path": str(environment), "sha256": file_sha256(environment)},
            verifier_path=str(RUNNER_PATH),
            audit_roots={"reviewed": True, "reference_root": str(tmp_path),
                         "negative_root": str(tmp_path)},
        )
