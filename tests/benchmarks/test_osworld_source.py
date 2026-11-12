"""Synthetic OSWorld V2 class fixture; no gated tasks or assets are bundled."""

from __future__ import annotations
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.pipelines.benchmarks.families.osworld import import_osworld_snapshot


import json
import sys
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import BenchmarkError, file_sha256
from data_designer.pipelines.benchmarks.families.osworld import (
    import_osworld_snapshot,
    verify_osworld_source_manifest,
)
from data_designer.pipelines.benchmarks.runners.osworld_offline import VERSION
from data_designer.pipelines.benchmarks.suite import create_suite
from tests.benchmarks.benchmark_review import approve_fixture_suite

RUNNER_PATH = Path(__file__).resolve().parents[2] / (
    "data_designer/pipelines/benchmarks/runners/osworld_offline.py"
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, admission: str = "expressible") -> dict:
    task_dir = tmp_path / "task_class"
    task_dir.mkdir()
    task_class = task_dir / "task_001.py"
    task_class.write_text(
        """class Task001:
    id = '001'
    snapshot = 'fixture-desktop'
    instruction = 'Edit the synthetic document and save the changed file.'
    related_apps = ['writer']
    def setup(self, controller):
        pass
    def evaluate(self, env):
        return 1.0  # PRIVATE FIXTURE ORACLE
TASK_CLASS = Task001
""",
        encoding="utf-8",
    )
    hashes = tmp_path / "task_hashes.json"
    _write(
        hashes,
        {"files": {"task_001.py": {
            "sha256": file_sha256(task_class), "size": task_class.stat().st_size,
        }}},
    )
    release = tmp_path / "release.json"
    _write(
        release,
        {
            "schema_version": 1,
            "release": "osworld-fixture-1",
            "status": "active",
            "osworld_code": {"tag": "fixture"},
            "website_code": {"tag": "fixture"},
            "tasks": {"tag": "fixture"},
            "assets": {"tag": "fixture"},
            "provider_images": {"docker": {"image": "fixture"}},
            "task_hash_manifest": {
                "sha256": "sha256:" + file_sha256(hashes),
                "task_count": 1,
            },
        },
    )
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    asset = asset_root / "expected.dat"
    asset.write_text("PRIVATE FIXTURE ASSET", encoding="utf-8")
    inventory = tmp_path / "asset-inventory.json"
    _write(
        inventory,
        {
            "schema_version": "osworld-asset-inventory-v1",
            "files": {
                "expected.dat": {
                    "sha256": file_sha256(asset),
                    "size": asset.stat().st_size,
                }
            },
        },
    )
    selection = tmp_path / "selection.json"
    _write(selection, {"tasks": ["001"]})
    initial_root = tmp_path / "initial"
    initial_file = initial_root / "note.txt"
    initial_file.parent.mkdir(parents=True, exist_ok=True)
    initial_file.write_text("original note\n", encoding="utf-8")
    offline_registry = tmp_path / "offline-state-registry.json"
    if admission == "expressible":
        entry = {
            "admission": "expressible",
            "initial_files": {
                "docs/note.txt": {
                    "path": str(initial_file),
                    "sha256": file_sha256(initial_file),
                }
            },
        }
    else:
        entry = {"admission": "excluded", "reason": "runtime_state"}
    _write(
        offline_registry,
        {"schema_version": "osworld-offline-state-registry-v1", "cases": {"001": entry}},
    )
    environment = tmp_path / "vm-image.txt"
    environment.write_text("fixture VM", encoding="utf-8")
    registry = tmp_path / "runtime.json"
    _write(
        registry,
        {
            "schema_version": "benchmark-runtime-registry-v1",
            "cases": {
                "001": {
                    "environment": {
                        "path": str(environment),
                        "sha256": file_sha256(environment),
                    },
                    "verifier": {
                        "path": str(RUNNER_PATH),
                        "sha256": file_sha256(RUNNER_PATH),
                        "version": VERSION,
                        "command": [sys.executable, str(RUNNER_PATH)],
                    },
                    "assets": [],
                }
            },
        },
    )
    return {
        "releasePath": str(release),
        "releaseSha256": file_sha256(release),
        "taskHashesPath": str(hashes),
        "taskHashesSha256": file_sha256(hashes),
        "taskClassDir": str(task_dir),
        "assetInventoryPath": str(inventory),
        "assetInventorySha256": file_sha256(inventory),
        "assetRoot": str(asset_root),
        "selectionPath": str(selection),
        "selectionSha256": file_sha256(selection),
        "selectionUse": "derivation",
        "split": "development",
        "license": {
            "id": "fixture-only",
            "derive_allowed": True,
            "redistribute_allowed": False,
            "access_conditions": "private synthetic fixture",
        },
        "runtimeRegistryPath": str(registry),
        "runtimeRegistrySha256": file_sha256(registry),
        "offlineStateRegistryPath": str(offline_registry),
        "offlineStateRegistrySha256": file_sha256(offline_registry),
    }




def test_osworld_derivation_requires_selection(tmp_path):
    request = _fixture(tmp_path)
    with pytest.raises(BenchmarkError, match="pinned selection"):
        import_osworld_snapshot(
            request["releasePath"], request["releaseSha256"],
            task_hashes_path=request["taskHashesPath"],
            task_hashes_sha256=request["taskHashesSha256"],
            task_class_dir=request["taskClassDir"],
            asset_inventory_path=request["assetInventoryPath"],
            asset_inventory_sha256=request["assetInventorySha256"],
            asset_root=request["assetRoot"],
            split="development", license_record=request["license"],
        )


