"""Synthetic WebArena-Verified fixture without official benchmark data."""

from __future__ import annotations
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.pipelines.benchmarks.families.webarena import (
    import_webarena_snapshot,
)


import json
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import BenchmarkError, file_sha256
from data_designer.pipelines.benchmarks.families.webarena import (
    import_webarena_snapshot,
    verify_webarena_source_manifest,
)
from data_designer.pipelines.benchmarks.suite import create_suite
from tests.benchmarks.benchmark_review import approve_fixture_suite


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")




def test_webarena_derivation_requires_selection(tmp_path):
    snapshot = tmp_path / "tasks.json"
    _write(
        snapshot,
        [
            {
                "task_id": 1,
                "intent_template_id": 1,
                "revision": 1,
                "sites": ["gitlab"],
                "start_urls": ["__GITLAB__"],
                "intent": "Find the synthetic issue.",
                "eval": [{"evaluator": "AgentResponseEvaluator"}],
            }
        ],
    )
    with pytest.raises(BenchmarkError, match="pinned selection"):
        import_webarena_snapshot(
            str(snapshot), file_sha256(snapshot),
            release="fixture", split="development",
            license_record={"id": "fixture", "derive_allowed": True},
        )
