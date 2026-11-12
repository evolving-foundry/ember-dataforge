"""C4 probes: Python API handle layer (design document: evaluation-extension capability
completion, section 6.5).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from data_designer.pipelines.benchmarks.api import (
    BenchmarkBuildResult,
    SuiteHandle,
    build,
    compare,
    load_suite,
)
from data_designer.pipelines.benchmarks.contracts import BenchmarkError
from data_designer.pipelines.benchmarks.suite import create_suite
from tests.benchmarks.benchmark_review import approve_fixture_suite
from tests.benchmarks.test_report import build_workbuddy_suite, build_single_taskset


def test_build_matches_cli_output(tmp_path, monkeypatch):
    import data_designer.pipelines.benchmarks.build as builder
    monkeypatch.setattr(builder, "assess_taskset",
                        lambda items: {"quality_overall_score": 1.0})
    source, augment = build_single_taskset(tmp_path)

    result = build({
        "id": "api-single", "catalog": str(tmp_path / "catalog"),
        "output": str(tmp_path / "out-api"), "source": str(source),
        "augmentation": str(augment),
    }, base_directory=tmp_path)
    assert isinstance(result, BenchmarkBuildResult)
    assert result.manifest_uri.endswith("out-api/manifest.json")

    # Same bytes through the CLI entry point (stubbed quality gate).
    config_path = tmp_path / "job.json"
    config_path.write_text(json.dumps({
        "id": "api-cli", "catalog": str(tmp_path / "catalog-cli"),
        "output": str(tmp_path / "out-cli"), "source": str(source),
        "augmentation": str(augment),
    }))
    bootstrap = (
        "import sys\n"
        f"sys.argv = ['dataforge', 'benchmark', 'build', '--config', {str(config_path)!r}]\n"
        "import data_designer.pipelines.benchmarks.build as builder\n"
        "builder.assess_taskset = lambda items: {'quality_overall_score': 1.0}\n"
        "from data_designer.cli.main import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    cli = subprocess.run([sys.executable, "-c", bootstrap],
                         capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["digest"] == result.digest


def test_build_describe_matches_taskset_payload(tmp_path, monkeypatch):
    store, suite = build_workbuddy_suite(tmp_path, monkeypatch)
    import data_designer.pipelines.benchmarks.api as bench_api

    pin = suite["tasksets"][0]
    payload = store.get_artifact("taskset", pin["ref_id"], pin["version"])["payload"]
    result = bench_api.build({
        "id": "api-describe", "catalog": str(tmp_path / "catalog-2"),
        "output": str(tmp_path / "out-describe"),
        "source": str(tmp_path / "source.json"),
        "augmentation": str(tmp_path / "augment.json"),
    }, base_directory=tmp_path)
    stats = result.describe()
    assert stats["family"] == payload["family"]
    assert stats["release"] == payload["source_release"]
    assert stats["split"] == payload["split"]
    assert stats["task_count"] == payload["item_count"]
    assert stats["quality"] == payload["build_report"]["quality_overall_score"]
    assert stats["audit"] == {"completed": payload["item_count"],
                              "passed": payload["item_count"],
                              "negative": payload["item_count"]}


def test_load_suite_approval_gate_and_sha_drift(tmp_path, monkeypatch):
    store, suite = build_workbuddy_suite(tmp_path, monkeypatch)
    catalog_uri = store.uri

    with pytest.raises(BenchmarkError, match="not approved"):
        load_suite(catalog_uri, suite["ref_id"], suite["version"])

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    approve_fixture_suite(store, suite, evidence)
    handle = load_suite(catalog_uri, suite["ref_id"], suite["version"])
    assert isinstance(handle, SuiteHandle)
    assert handle.fingerprint == suite["suite_sha256"]

    with pytest.raises(BenchmarkError, match="drift"):
        load_suite(catalog_uri, suite["ref_id"], suite["version"],
                   sha256="0" * 64)


def test_task_items_frozen_order_and_isolation(tmp_path, monkeypatch):
    store, suite = build_workbuddy_suite(tmp_path, monkeypatch)
    handle = load_suite(store.uri, suite["ref_id"], suite["version"],
                        require_approved=False)

    items = handle.task_items()
    expected = []
    for pin in suite["tasksets"]:
        payload = store.get_artifact("taskset", pin["ref_id"], pin["version"])["payload"]
        expected.extend(item["task_id"] for item in payload["taskset"])
    assert [item["task_id"] for item in items] == expected

    items[0]["task_text"] = "tampered"
    items.append({"rogue": True})
    again = handle.task_items()
    assert again[0]["task_text"] != "tampered"
    assert len(again) == len(expected)


def test_compare_delegates_and_rejects_incomparable_records(tmp_path, monkeypatch):
    from tests.benchmarks.test_report import _record

    baseline = _record(["t1", "t2"], ["failed", "failed"])
    candidate = _record(["t1", "t2"], ["passed", "failed"])
    report = compare(baseline, candidate)
    assert report.macro_delta == pytest.approx(0.5)

    with pytest.raises(BenchmarkError, match="different suites or tasks"):
        compare(baseline, _record(["t1", "t2"], ["passed", "failed"],
                                  sha="1" * 64))
