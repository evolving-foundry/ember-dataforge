"""Synthetic GAIA metadata fixture; it contains no gated benchmark content."""

from __future__ import annotations
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.pipelines.benchmarks.families.gaia import (
    import_gaia_snapshot,
    in_generation_checkout,
)


import json
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import BenchmarkError, file_sha256
from data_designer.pipelines.benchmarks.families.gaia import (
    import_gaia_snapshot,
    in_generation_checkout,
    verify_gaia_source_manifest,
)
from data_designer.pipelines.benchmarks.suite import create_suite
from tests.benchmarks.benchmark_review import approve_fixture_suite




def test_gaia_parquet_snapshot_keeps_answer_private(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    snapshot = tmp_path / "metadata.parquet"
    parquet.write_table(
        pyarrow.Table.from_pylist(
            [
                {
                    "task_id": "synthetic-parquet-1",
                    "Question": "Which value follows the synthetic event?",
                    "Level": 1,
                    "Final answer": "PRIVATE PARQUET ANSWER",
                    "file_name": "",
                    "file_path": "",
                }
            ]
        ),
        snapshot,
    )
    source = import_gaia_snapshot(
        str(snapshot),
        file_sha256(snapshot),
        release="GAIA-parquet-fixture-1",
        split="development",
        upstream_split="dev",
        license_record={
            "id": "fixture-only",
            "derive_allowed": True,
            "redistribute_allowed": False,
            "access_conditions": "private synthetic fixture",
        },
    )
    assert source["source_snapshot"]["format"] == "gaia-parquet-v1"
    assert source["cases"][0]["source_row"]["row_index"] == 0
    assert "PRIVATE PARQUET ANSWER" not in json.dumps(source)
    verify_gaia_source_manifest(source)
    with snapshot.open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(BenchmarkError, match="snapshot drift"):
        verify_gaia_source_manifest(source)
    snapshot.write_bytes(b"not a parquet file")
    with pytest.raises(BenchmarkError, match="cannot read GAIA Parquet"):
        import_gaia_snapshot(
            str(snapshot),
            file_sha256(snapshot),
            release="GAIA-parquet-fixture-1",
            split="development",
            upstream_split="dev",
            license_record=source["license"],
        )
