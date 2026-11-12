"""Synthetic AppWorld directory fixture; no protected bundle is included."""

from __future__ import annotations
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.pipelines.benchmarks.families.appworld import import_appworld_snapshot


import json
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import BenchmarkError, file_sha256
from data_designer.pipelines.benchmarks.families.appworld import (
    import_appworld_snapshot,
    verify_appworld_source_manifest,
)
from data_designer.pipelines.benchmarks.suite import create_suite
from tests.benchmarks.benchmark_review import approve_fixture_suite


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")




def test_appworld_test_split_cannot_seed_development(tmp_path):
    root = tmp_path / "appworld-data"
    dataset = root / "datasets" / "test_normal.txt"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("synthetic_1\n", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="test cannot seed"):
        import_appworld_snapshot(
            str(root), file_sha256(dataset),
            upstream_split="test_normal", release="fixture",
            split="development",
            license_record={
                "id": "fixture", "derive_allowed": True,
                "redistribute_allowed": False,
                "access_conditions": "fixture",
            },
        )
