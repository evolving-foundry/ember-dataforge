"""C2 probes: job-level splits orchestration (design document: evaluation-extension
capability completion, section 4.5).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.api import build_benchmark
from data_designer.pipelines.benchmarks.build import build_benchmark_taskset
from data_designer.pipelines.benchmarks.contracts import BenchmarkError
from data_designer.pipelines.benchmarks.package import freeze_package
from data_designer.pipelines.benchmarks.contracts import file_sha256
from data_designer.storage.catalog import ArtifactCatalog


@pytest.fixture(autouse=True)
def quality_gate(monkeypatch):
    import data_designer.pipelines.benchmarks.build as builder
    monkeypatch.setattr(builder, "assess_taskset", lambda items: {"quality_overall_score": 1.0})


def _write_split_fixture(tmp_path, split):
    """Workbuddy single-case fixture pinned to one split (test_package pattern). Task
    text differs per split: builds into a shared catalog reject any derived text that
    duplicates an already-published taskset.
    """
    from data_designer.pipelines.benchmarks.runners import workbuddy

    exercise = {
        "development": "Original source accounting exercise.",
        "regression": "Original source ledger reconciliation exercise.",
    }[split]
    prompt = {
        "development": "Compute the total from the supplied fictional accounting entries.",
        "regression": "Compute the net balance from the supplied fictional ledger entries.",
    }[split]
    runner = Path(workbuddy.__file__).resolve()
    environment = tmp_path / f"environment-{split}.json"
    environment.write_text("{}")
    reference, negative = tmp_path / f"reference-{split}", tmp_path / f"negative-{split}"
    for root, value in ((reference, {"total": 42}), (negative, {"total": 0})):
        (root / "output").mkdir(parents=True, exist_ok=True)
        (root / "output/result.json").write_text(json.dumps(value))
    runtime = {"environment": {"path": str(environment), "sha256": file_sha256(environment)},
               "verifier": {"path": str(runner), "sha256": file_sha256(runner),
                            "command": [sys.executable, str(runner)], "version": workbuddy.VERSION}, "assets": []}
    case = {"source_task_id": f"seed-{split}", "task_text": exercise,
            "initial_state_sha256": "a" * 64, "expected_state_sha256": "b" * 64}
    source = {"schema_version": "benchmark-source-v1", "family": "workbuddy", "release": "fixture",
              "split": split, "license": {"id": "owned-fixture", "derive_allowed": True},
              "cases": [case], **runtime}
    variant = {"source_task_id": f"seed-{split}", "variant_id": "v1", "operation": "mutation",
               "public": {"task_text": prompt,
                          "success_criteria": "Write output/result.json with the computed total.",
                          "tools_available": ["files"], "difficulty": "easy"},
               "private": {"initial_state_sha256": "c" * 64, "expected_state_sha256": "d" * 64,
                           "verifier_spec": {"kind": "artifact_compare", "generator_ref": "owned-fixture@1",
                                             "payload": {"artifact": {"path": "output/result.json"}, "gold": {"total": 42}}}},
               "audit": {"reviewed": True, "reference_root": str(reference), "negative_root": str(negative)}}
    config = {"schema_version": "benchmark-augment-v1", "family": "workbuddy", "release": "fixture",
              "split": split, "variants": [variant]}
    source_path = tmp_path / f"source-{split}.json"
    config_path = tmp_path / f"augment-{split}.json"
    source_path.write_text(json.dumps(source))
    config_path.write_text(json.dumps(config))
    return source_path, config_path


def _splits_job(tmp_path, catalog_dir, *, regression_augment=None):
    dev_source, dev_augment = _write_split_fixture(tmp_path, "development")
    reg_source, reg_augment = _write_split_fixture(tmp_path, "regression")
    if regression_augment is not None:
        reg_augment = regression_augment
    return {
        "id": "swe-eval-expand-001",
        "catalog": str(catalog_dir),
        "output": str(tmp_path / "out" / "tasksets"),
        "quality_threshold": 0.6,
        "splits": [
            {"split": "development", "source": str(dev_source),
             "augmentation": str(dev_augment),
             "ref_id": "swe-expand-development",
             "suite_id": "swe-expand-suite-development"},
            {"split": "regression", "source": str(reg_source),
             "augmentation": str(reg_augment),
             "ref_id": "swe-expand-regression",
             "suite_id": "swe-expand-suite-regression"},
        ],
    }


def test_two_split_job_builds_tasksets_suites_and_outputs(tmp_path):
    catalog_dir = tmp_path / "catalog"
    job = _splits_job(tmp_path, catalog_dir)
    # suite_id is optional per split (design §4.3): a job without suites must
    # still orchestrate cleanly.
    suiteless = _splits_job(tmp_path, tmp_path / "catalog-suiteless")
    for entry in suiteless["splits"]:
        entry.pop("suite_id")
    suiteless["output"] = str(tmp_path / "out-suiteless" / "tasksets")
    suiteless_result = build_benchmark(suiteless, base_directory=tmp_path)
    assert set(suiteless_result) == {"development", "regression"}
    suiteless_store = ArtifactCatalog(str(tmp_path / "catalog-suiteless"))
    assert suiteless_store.list_artifacts("benchmark_suite") == []

    result = build_benchmark(job, base_directory=tmp_path)

    assert set(result) == {"development", "regression"}
    output = tmp_path / "out" / "tasksets"
    assert (output / "development" / "manifest.json").is_file()
    assert (output / "regression" / "manifest.json").is_file()

    store = ArtifactCatalog(str(catalog_dir))
    tasksets = {entry["ref_id"]: entry for entry in store.list_artifacts("taskset")}
    assert set(tasksets) == {"swe-expand-development", "swe-expand-regression"}
    suites = {entry["ref_id"]: entry for entry in store.list_artifacts("benchmark_suite")}
    assert set(suites) == {"swe-expand-suite-development", "swe-expand-suite-regression"}
    for split, ref_id, suite_id in (
        ("development", "swe-expand-development", "swe-expand-suite-development"),
        ("regression", "swe-expand-regression", "swe-expand-suite-regression"),
    ):
        payload = store.get_artifact("taskset", ref_id)["payload"]
        assert payload["split"] == split
        suite = suites[suite_id]["payload"]
        assert suite["split"] == split
        assert suite["tasksets"][0]["ref_id"] == ref_id
        assert suite["tasksets"][0]["fingerprint"] == payload["taskset_fingerprint"]
        assert result[split]["digest"]


def test_second_split_build_failure_leaves_no_output_or_suite(tmp_path):
    catalog_dir = tmp_path / "catalog"
    bad_augment = tmp_path / "augment-bad.json"
    bad_augment.write_text(json.dumps({"schema_version": "benchmark-augment-v0"}))
    job = _splits_job(tmp_path, catalog_dir, regression_augment=str(bad_augment))
    output = tmp_path / "out"

    with pytest.raises(BenchmarkError, match="wrong source or augmentation schema"):
        build_benchmark(job, base_directory=tmp_path)

    assert not (tmp_path / "out" / "tasksets").exists()
    store = ArtifactCatalog(str(catalog_dir))
    assert store.list_artifacts("benchmark_suite") == []
    dev = store.get_artifact("taskset", "swe-expand-development")
    assert dev is not None
    payload = dev["payload"]
    from data_designer.contracts.provenance import content_digest as fingerprint_of
    assert fingerprint_of(payload["taskset"]) == payload["taskset_fingerprint"]


def test_cli_splits_job_end_to_end(tmp_path):
    catalog_dir = tmp_path / "catalog"
    job = _splits_job(tmp_path, catalog_dir)
    config_path = tmp_path / "job.json"
    config_path.write_text(json.dumps(job))
    # The real quality gate needs the Ray/Curator stack; stub it to test the
    # CLI wiring and output mapping.
    bootstrap = (
        "import sys\n"
        f"sys.argv = ['dataforge', 'benchmark', 'build', '--config', {str(config_path)!r}]\n"
        "import data_designer.pipelines.benchmarks.build as builder\n"
        "builder.assess_taskset = lambda items: {'quality_overall_score': 1.0}\n"
        "from data_designer.cli.main import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    result = subprocess.run([sys.executable, "-c", bootstrap],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert set(body["splits"]) == {"development", "regression"}
    assert not set(body) - {"splits"}
    for package in body["splits"].values():
        assert set(package) == {"manifest_uri", "digest"}


def test_schema_rejects_holdout_top_level_conflicts_and_duplicate_refs(tmp_path):
    catalog_dir = tmp_path / "catalog"
    job = _splits_job(tmp_path, catalog_dir)
    job["splits"][0]["split"] = "holdout"
    with pytest.raises(ValueError, match="splits"):
        build_benchmark(job, base_directory=tmp_path)

    job = _splits_job(tmp_path, catalog_dir)
    job["source"] = str(tmp_path / "source-development.json")
    with pytest.raises(ValueError, match="source/augmentation"):
        build_benchmark(job, base_directory=tmp_path)

    job = _splits_job(tmp_path, catalog_dir)
    job["splits"][1]["ref_id"] = "swe-expand-development"
    with pytest.raises(ValueError, match="duplicate ref_id"):
        build_benchmark(job, base_directory=tmp_path)

    job = _splits_job(tmp_path, catalog_dir)
    job["splits"][1]["suite_id"] = "swe-expand-suite-development"
    with pytest.raises(ValueError, match="duplicate suite_id"):
        build_benchmark(job, base_directory=tmp_path)


def test_single_split_job_unchanged_and_deterministic(tmp_path):
    source, augment = _write_split_fixture(tmp_path, "development")
    base_job = {
        "id": "legacy-single",
        "quality_threshold": 0.6,
        "source": str(source),
        "augmentation": str(augment),
    }

    first = build_benchmark(
        {**base_job, "catalog": str(tmp_path / "catalog-one"),
         "output": str(tmp_path / "out-one")}, base_directory=tmp_path
    )
    second = build_benchmark(
        {**base_job, "catalog": str(tmp_path / "catalog-two"),
         "output": str(tmp_path / "out-two")}, base_directory=tmp_path
    )
    assert first["digest"] == second["digest"]
    assert first["manifest_uri"].endswith("out-one/manifest.json")

    store = ArtifactCatalog(str(tmp_path / "catalog-one"))
    pin = store.list_artifacts("taskset")[0]["payload"]
    # Same fingerprint as an equivalent direct build call (current-behavior comparison).
    direct = build_benchmark_taskset(
        ArtifactCatalog(), str(source), str(augment), str(tmp_path / "trusted"),
        "legacy-single", quality_threshold=0.6,
    )
    assert direct["taskset"]["fingerprint"] == pin["taskset_fingerprint"]

    with pytest.raises(FileExistsError):
        build_benchmark(
            {**base_job, "catalog": str(tmp_path / "catalog-one"),
             "output": str(tmp_path / "out-one")}, base_directory=tmp_path
        )
