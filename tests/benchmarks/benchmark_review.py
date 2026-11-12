"""Explicit synthetic review attestation for fixture suites only."""

from __future__ import annotations

import json
from pathlib import Path

from data_designer.pipelines.benchmarks.contracts import file_sha256
from data_designer.pipelines.benchmarks.suite import set_publication_status
from data_designer.storage.catalog import ArtifactCatalog


def write_fixture_review(suite: dict, directory: Path) -> dict[str, str]:
    path = directory / f"fixture-review-{suite['suite_sha256']}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "benchmark-suite-review-v1",
                "scope": "synthetic_fixture",
                "suite_sha256": suite["suite_sha256"],
                "tasksets": suite["tasksets"],
                "reviewer": "synthetic-fixture-test",
                "evidence_refs": ["synthetic fixture reference/negative runs"],
                "native_positive_negative_verified": True,
                "private_boundary_verified": True,
                "license_checked": True,
                "environment_reset_verified": True,
            }
        ),
        encoding="utf-8",
    )
    return {"path": str(path), "sha256": file_sha256(path)}


def approve_fixture_suite(
    store: ArtifactCatalog, suite: dict, directory: Path
) -> dict[str, str]:
    pin = write_fixture_review(suite, directory)
    set_publication_status(
        store,
        suite["ref_id"],
        suite["version"],
        "approved",
        "synthetic fixture protocol review",
        review_evidence=pin,
    )
    return pin
