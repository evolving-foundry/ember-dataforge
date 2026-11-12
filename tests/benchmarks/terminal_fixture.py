"""Shared terminal intake M2 fixture: source import, re-expression, audit roots.

Builds the trusted-side inputs the terminal derivation chain needs from the
in-repo sanitized ``tb21_terminal_fixture``: a pinned seed archive, the frozen
``benchmark-source-v1`` manifest, re-expressed verifier_specs with gold and
negative deliverables, and the audit/candidate roots the native build and eval
layers consume. Mirrors the archived ``tests/rsi/swe_offline_fixture.py`` fixture.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from data_designer.pipelines.benchmarks.contracts import file_sha256
from data_designer.pipelines.benchmarks.runners.terminal import VERSION
from data_designer.pipelines.benchmarks.families.terminal import (
    build_terminal_augment_config,
    derive_release,
    derive_terminal_spec,
    import_terminal_source,
    screen_terminal_contamination,
    verify_terminal_source_manifest,
)
from data_designer.sources.terminal_seed import TB21SeedAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = (
    PROJECT_ROOT / "data_designer" / "sources" / "fixtures"
    / "tb21_terminal_fixture"
)
RUNNER_PATH = (
    PROJECT_ROOT / "data_designer" / "pipelines" / "benchmarks" / "runners" / "terminal.py"
)

LICENSE = {"id": "internal-use", "derive_allowed": True}
EXPRESSIBLE_IDS = ("tb-fixture-001-fix-script", "tb-fixture-002-rollup")
EXCLUDED_ID = "tb-fixture-003-image-state"

# Per-case re-expression directives (candidate deliverable declaration).
DIRECTIVES = {
    "tb-fixture-001-fix-script": {
        "kind": "unit_test", "path": "deliverable/files.json", "timeout_s": 15,
    },
    "tb-fixture-002-rollup": {
        "kind": "artifact_compare", "path": "deliverable/summary.json",
        "timeout_s": 15,
    },
}

GOLD_UNIT_TEST_FILES = {"scripts/sum.py": "def total(a, b):\n    return a + b\n"}
GOLD_ARTIFACT = {"total_revenue_cny": 1500.0, "row_count": 3}


def tar_fixture(tmp_path: Path) -> tuple[str, str]:
    archive = tmp_path / "terminal-seed.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(FIXTURE_DIR, arcname=FIXTURE_DIR.name)
    return str(archive), file_sha256(archive)


def _derived_task_text(case: dict, deliverable: str) -> str:
    return (
        f"{case['task_text']}\n\nCandidate protocol: write the deliverable "
        f"JSON described above to the declared path '{deliverable}' under "
        f"your candidate root."
    )


def derived_rows(tmp_path: Path, source: dict) -> tuple[list[dict], dict]:
    """Re-express every expressible case into an augment row + spec bundle."""
    rows: list[dict] = []
    derived: dict[str, dict] = {}
    for case in source["cases"]:
        if case["terminal"]["expressibility"] != "expressible":
            continue
        seed_id = case["source_task_id"]
        directive = DIRECTIVES[seed_id]
        bundle = derive_terminal_spec(source, case, directive)
        deliverable = bundle["verifier_spec"]["payload"]["artifact"]["path"]
        rows.append(
            {
                "source_task_id": seed_id,
                "task_text": _derived_task_text(case, deliverable),
                "tools_available": case["tools_available"],
                "success_criteria": case["success_criteria"],
                "difficulty": case["difficulty"],
                "expected_state_sha256": bundle["expected_state_sha256"],
                "item_fingerprint": seed_id.replace("tb-fixture-", "v") + "-a",
                "verifier_spec": bundle["verifier_spec"],
            }
        )
        derived[seed_id] = bundle
    screen_terminal_contamination(rows, source["cases"])
    return rows, derived


def prepare_source(tmp_path: Path) -> dict:
    """Tar the fixture, import the source manifest, verify every pin."""
    adapter = TB21SeedAdapter(FIXTURE_DIR)
    seed_rows = adapter.load_seed_tasks()
    archive_path, archive_sha = tar_fixture(tmp_path)
    release = derive_release("synthetic-fixture", seed_rows)
    source = import_terminal_source(
        seed_rows,
        release=release,
        split="development",
        license_record=dict(LICENSE),
        seed_source="synthetic-fixture",
        dataset_id=adapter.dataset_id,
        archive_path=archive_path,
        archive_sha256=archive_sha,
    )
    verify_terminal_source_manifest(source)
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(source, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return {
        "adapter": adapter,
        "seed_rows": seed_rows,
        "archive_path": archive_path,
        "archive_sha256": archive_sha,
        "release": release,
        "source": source,
        "source_path": str(source_path),
    }


def _write_json_file(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def write_roots(root: Path, derived: dict, field: str) -> str:
    """Write one deliverable per derived case (gold or negative) under root."""
    for bundle in derived.values():
        deliverable = bundle["verifier_spec"]["payload"]["artifact"]["path"]
        _write_json_file(root / deliverable, bundle[field])
    return str(root)


def build_config(
    tmp_path: Path, source: dict, rows: list[dict], derived: dict,
) -> tuple[dict, str]:
    """Augment config with per-case setup-snapshot environment pins."""
    adapter = TB21SeedAdapter(FIXTURE_DIR)
    environments = []
    for row in rows:
        archive = tmp_path / f"env-{row['source_task_id']}.tar.gz"
        sha = adapter.export_setup_archive(archive, row["source_task_id"])
        environments.append({"path": str(archive), "sha256": sha})
    reference_root = write_roots(
        tmp_path / "audit-reference", derived, "gold_deliverable"
    )
    negative_root = write_roots(
        tmp_path / "audit-negative", derived, "negative_deliverable"
    )
    config = build_terminal_augment_config(
        rows,
        release=source["release"],
        split="development",
        strategy="setup_fault",
        environment=environments,
        verifier_path=str(RUNNER_PATH),
        audit_roots={
            "reviewed": True,
            "reference_root": reference_root,
            "negative_root": negative_root,
        },
    )
    config_path = tmp_path / "augment.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return config, str(config_path)


RUNNER_VERSION = VERSION
