"""Shared swe-offline M2 fixture: mini repo JSONL + pinned base snapshot.

Builds the trusted-side inputs the offline derivation chain needs: a two-row
SWE-bench-format JSONL (one gold patch applies to the base snapshot, one does
not), a zip of the base workspace, a runtime registry pinning the real
``runners/swe_offline.py``, and an offline base registry admitting only the
applicable instance.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from data_designer.pipelines.benchmarks.contracts import file_sha256
from data_designer.pipelines.benchmarks.runners.swe_offline import VERSION

RUNNER_PATH = Path(__file__).resolve().parents[2] / (
    "data_designer/pipelines/benchmarks/runners/swe_offline.py"
)

INSTANCE_ID = "fixture__repo-1"
INSTANCE_TWO_ID = "fixture__repo-2"
EXCLUDED_INSTANCE_ID = "fixture__repo-3"
BASE_FILES = {
    "src/counter.py": "value = 1\n",
    "README.md": "# fixture repo\n",
}
BASE_TWO_FILES = {
    "src/counter.py": "value = 10\n",
    "README.md": "# fixture repo\n",
}
GOLD_NEW = {"src/counter.py": "value = 2\n"}
GOLD_NEW_TWO = {"src/counter.py": "value = 11\n"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unified(old: str, new: str, relative: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )


def gold_patch() -> str:
    return "".join(
        _unified(BASE_FILES[rel], new, rel) for rel, new in GOLD_NEW.items()
    )


def corrupted_patch() -> str:
    # tamper the removed line so git apply must reject the patch
    return gold_patch().replace("-value = 1\n", "-value = 1 tampered\n")


def gold_patch_two() -> str:
    return "".join(
        _unified(BASE_TWO_FILES[rel], new, rel) for rel, new in GOLD_NEW_TWO.items()
    )


def corrupted_patch_two() -> str:
    # tamper the removed line so git apply must reject the patch
    return gold_patch_two().replace("-value = 10\n", "-value = 10 tampered\n")


def inapplicable_patch() -> str:
    # a valid diff whose removal line never existed in the base snapshot
    return _unified("value = 999\n", "value = 1000\n", "src/counter.py")


def expected_file_shas(new_files: dict | None = None) -> dict:
    return {
        rel: sha256_bytes(new.encode("utf-8"))
        for rel, new in (new_files if new_files is not None else GOLD_NEW).items()
    }


def _row(instance_id: str, base_commit: str, statement: str, patch: str) -> dict:
    return {
        "instance_id": instance_id,
        "repo": "fixture/repo",
        "base_commit": base_commit,
        "problem_statement": statement,
        "patch": patch,
        "test_patch": "PRIVATE TARGET TEST",
        "FAIL_TO_PASS": '["tests/test_counter.py::test_concurrent"]',
        "PASS_TO_PASS": ["tests/test_counter.py::test_single"],
        "created_at": "2026-01-02T00:00:00Z",
        "version": "1.0",
    }


def prepare_area(area: Path) -> dict:
    """Write the JSONL snapshot, base zip, environment, and both registries."""
    area.mkdir(parents=True, exist_ok=True)
    snapshot = area / "swe.jsonl"
    rows = [
        _row(
            INSTANCE_ID,
            "a" * 40,
            "Resolve the counter fault before the next release.",
            gold_patch(),
        ),
        _row(
            INSTANCE_TWO_ID,
            "c" * 40,
            "Resolve the second counter fault before the next release.",
            gold_patch_two(),
        ),
        _row(
            EXCLUDED_INSTANCE_ID,
            "b" * 40,
            "Resolve the third counter fault before the next release.",
            inapplicable_patch(),
        ),
    ]
    snapshot.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    base_zip = area / "base.zip"
    with zipfile.ZipFile(base_zip, "w") as archive:
        for relative, content in BASE_FILES.items():
            archive.writestr(relative, content)
    base_zip_two = area / "base-two.zip"
    with zipfile.ZipFile(base_zip_two, "w") as archive:
        for relative, content in BASE_TWO_FILES.items():
            archive.writestr(relative, content)
    environment = area / "environment.txt"
    environment.write_text("frozen fixture environment", encoding="utf-8")
    verifier_pin = {
        "path": str(RUNNER_PATH),
        "sha256": file_sha256(RUNNER_PATH),
        "version": VERSION,
        "command": [sys.executable, str(RUNNER_PATH)],
    }
    runtime_registry = area / "registry.json"
    runtime_registry.write_text(
        json.dumps(
            {
                "schema_version": "benchmark-runtime-registry-v1",
                "cases": {
                    INSTANCE_ID: {
                        "environment": {
                            "path": str(environment),
                            "sha256": file_sha256(environment),
                        },
                        "verifier": dict(verifier_pin),
                        "assets": [],
                    },
                    INSTANCE_TWO_ID: {
                        "environment": {
                            "path": str(environment),
                            "sha256": file_sha256(environment),
                        },
                        "verifier": dict(verifier_pin),
                        "assets": [],
                    },
                    EXCLUDED_INSTANCE_ID: {
                        "environment": {
                            "path": str(environment),
                            "sha256": file_sha256(environment),
                        },
                        "verifier": dict(verifier_pin),
                        "assets": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    offline_registry = area / "offline-bases.json"
    offline_registry.write_text(
        json.dumps(
            {
                "schema_version": "swe-offline-base-registry-v1",
                "cases": {
                    INSTANCE_ID: {
                        "path": str(base_zip),
                        "sha256": sha256_bytes(base_zip.read_bytes()),
                    },
                    INSTANCE_TWO_ID: {
                        "path": str(base_zip_two),
                        "sha256": sha256_bytes(base_zip_two.read_bytes()),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "snapshot_path": str(snapshot),
        "snapshot_sha256": file_sha256(snapshot),
        "base_zip_path": str(base_zip),
        "base_zip_sha256": sha256_bytes(base_zip.read_bytes()),
        "base_zip_two_path": str(base_zip_two),
        "base_zip_two_sha256": sha256_bytes(base_zip_two.read_bytes()),
        "environment_path": str(environment),
        "runtime_registry_path": str(runtime_registry),
        "runtime_registry_sha256": file_sha256(runtime_registry),
        "offline_registry_path": str(offline_registry),
        "offline_registry_sha256": file_sha256(offline_registry),
        "gold_patch": gold_patch(),
        "gold_patch_two": gold_patch_two(),
        "corrupted_patch": corrupted_patch(),
        "corrupted_patch_two": corrupted_patch_two(),
    }


def write_audit_roots(area: Path, pins: dict) -> tuple[str, str]:
    reference = area / "reference"
    negative = area / "negative"
    reference.mkdir(parents=True, exist_ok=True)
    negative.mkdir(parents=True, exist_ok=True)
    (reference / "patch.diff").write_text(pins["gold_patch"], encoding="utf-8")
    (negative / "patch.diff").write_text(pins["corrupted_patch"], encoding="utf-8")
    return str(reference), str(negative)


def swe_variant(
    instance_id: str,
    *,
    variant_id: str = "v1",
    reference_root: str,
    negative_root: str,
    task_text: str = "Resolve the counter fault.",
    success_criteria: str = "Concurrent updates keep the count.",
) -> dict:
    return {
        "source_task_id": instance_id,
        "variant_id": variant_id,
        "operation": "regression",
        "public": {
            "task_text": task_text,
            "tools_available": ["terminal"],
            "success_criteria": success_criteria,
            "difficulty": "medium",
        },
        "swe_offline": {"deliverable": "patch.diff", "timeout_s": 30},
        "audit": {
            "reviewed": True,
            "reference_root": reference_root,
            "negative_root": negative_root,
        },
    }
