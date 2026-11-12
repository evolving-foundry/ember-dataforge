"""workbuddy real-subset native ingest e2e (M1): seed ingestion → native license freeze →
workspace public-surface materialization → per-task ingest report → wb_composite ingest rows → augment config.

Against ``e2e_workbuddy_native.py`` (fixture derivation path): this script follows the design
§5/§6.2/§9 native_ingest production line --

- the license grants only ``native_ingest_allowed=true`` (mutually exclusive with derive_allowed, D3),
  and carries the D2 review triple (reviewer/date/evidence+evidence_sha256);
- contamination screening is replaced by the per-task ingest report (task_text fingerprint == seed fingerprint + cross-release
  collision check); tasks with collisions are marked excluded, and this script fails explicitly, never silently passing them;
- ``materialize_public_inputs`` unpacks the workspace (excluding ``tests/**`` and
  PRIVATE_MARKERS) to produce the D5 public-surface manifest;
- ``build_native_ingest_rows`` produces the wb_composite verifier_spec (gold/grading
  pins exist only on the trusted side);
- ``build_workbuddy_augment_config`` pins the ingest report and public inputs,
  and ``verify_native_ingest_pins`` re-checks them on the spot.

The wb_composite scoring runner belongs to M2; this script stops explicitly at the augment config,
and the verifier pin in the config is a placeholder shim (raises NotImplementedError when run), so it must not
be used to run taskset build/evaluation.

Usage (scripts/__init__.py has a relative import beyond the top level, so it cannot be run as a module):
  PYTHONPATH=. python3 scripts/wbbench_native_ingest.py \
      --seed-archive <workbuddy-seed.tar.gz> --work-dir <directory outside the repo>
  # or --seed-dir pointing to an unpacked subset root (containing dataset.toml)

Exit code: 0 = full chain passes; 2 = precondition failure (explicit failure, never silent).
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_designer.pipelines.benchmarks.contracts import (
    canonical_sha256,
    file_sha256,
)
from data_designer.pipelines.benchmarks.families.workbuddy import (
    build_native_ingest_rows,
    build_workbuddy_augment_config,
    derive_release,
    import_workbuddy_source,
    verify_workbuddy_source_manifest,
)
from data_designer.pipelines.benchmarks.families.workbuddy_audit import (
    build_native_ingest_report,
    materialize_public_inputs,
)
from data_designer.sources.workbuddy_seed import WBBenchSeedAdapter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# D2 review evidence (design §12 D2: reviewer and date are frozen with the design document)
_D2_REVIEWER = "chadwweng"
_D2_REVIEW_DATE = "2026-10-09"
_DESIGN_DOC = (
    _PROJECT_ROOT / "docs" / "RSI_workbuddy真实子集原生收题设计_v0.1.md"
)

_PLACEHOLDER_RUNNER = '''"""Placeholder for the wb_composite scoring runner (production implementation lands in M2).

M1 native ingestion freezes only the ingestion chain. This shim is pinned into the verifier slot of the augment config solely to satisfy the config schema. It fails explicitly when run and must not be used for evaluation.
"""
raise NotImplementedError(
    "wb_composite judge runner lands in milestone M2; "
    "do not evaluate native-ingest tasksets built from this placeholder"
)
'''


def _write_json(path: Path, value: Dict[str, Any]) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(path)


def _tar_seed_dir(seed_dir: Path, work: Path) -> tuple:
    archive = work / "workbuddy-seed.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(seed_dir, arcname=seed_dir.name)
    return str(archive), file_sha256(archive)


def _adapter_and_archive(
    seed_dir: Optional[str], seed_archive: Optional[str], work: Path
) -> tuple:
    if bool(seed_dir) == bool(seed_archive):
        raise RuntimeError("Exactly one of --seed-dir or --seed-archive must be provided")
    if seed_archive:
        adapter = WBBenchSeedAdapter.from_archive(
            seed_archive, seed_source="real-hf-subset"
        )
        archive_path = str(seed_archive)
        archive_sha = adapter.archive_sha256
    else:
        seed_path = Path(seed_dir)
        if not (seed_path / "dataset.toml").is_file():
            raise RuntimeError(f"Seed directory is missing dataset.toml: {seed_path}")
        adapter = WBBenchSeedAdapter(seed_path, seed_source="real-hf-subset")
        archive_path, archive_sha = _tar_seed_dir(seed_path, work)
    return adapter, archive_path, archive_sha


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="WorkBuddy real-subset native ingestion e2e (M1: through augment config)"
    )
    ap.add_argument("--seed-dir", default=None,
                    help="Extracted workbuddy-bench subset root (contains dataset.toml)")
    ap.add_argument("--seed-archive", default=None,
                    help="workbuddy-seed.tar.gz (single top-level directory; mutually exclusive with --seed-dir)")
    ap.add_argument("--work-dir", default=None,
                    help="Working directory (source manifest, public inputs, report, config; must be outside the repository)")
    ap.add_argument("--reviewer", default=_D2_REVIEWER,
                    help="D2 native-ingest reviewer (defaults to the frozen design-document value)")
    ap.add_argument("--review-date", default=_D2_REVIEW_DATE,
                    help="D2 review date YYYY-MM-DD (defaults to the frozen design-document value)")
    args = ap.parse_args(argv)
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="wb-native-ingest-"))
    work.mkdir(parents=True, exist_ok=True)

    # ── 1. Seed ingestion (real-hf-subset) ──────────────────────────────
    adapter, archive_path, archive_sha = _adapter_and_archive(
        args.seed_dir, args.seed_archive, work
    )
    seed_rows = adapter.load_seed_tasks()
    release = derive_release("real-hf-subset", seed_rows, archive_sha)
    if not release.startswith("wb-bench-"):
        raise RuntimeError(f"native ingestion requires a wb-bench- release: {release}")

    # ── 2. Native license (D3 mutual exclusion + D2 review triple) ─────
    if not _DESIGN_DOC.is_file():
        raise RuntimeError(f"Design document is missing; the D2 evidence chain is broken: {_DESIGN_DOC}")
    evidence = [
        f"design_doc:{_DESIGN_DOC.name}#sha256:{file_sha256(_DESIGN_DOC)}",
        f"seed_archive:{archive_path}#sha256:{archive_sha}",
        "scope:verbatim_ingest_only;redistribute=false;private_evaluation_only",
    ]
    license_record = {
        "id": f"wb-bench-native-ingest@{release}",
        "derive_allowed": False,
        "native_ingest_allowed": True,
        "redistribute_allowed": False,
        "access_conditions": "local private evaluation only; verbatim ingest",
        "native_ingest_review": {
            "reviewer": args.reviewer,
            "date": args.review_date,
            "evidence": evidence,
            "evidence_sha256": canonical_sha256(evidence),
        },
    }

    # ── 3. Source manifest freeze ──────────────────────────────────────
    source = import_workbuddy_source(
        seed_rows,
        release=release,
        split="development",
        license_record=license_record,
        seed_source="real-hf-subset",
        dataset_id=adapter.dataset_id,
        archive_path=archive_path,
        archive_sha256=archive_sha,
    )
    verify_workbuddy_source_manifest(source)
    source_path = _write_json(work / "source.json", source)

    # ── 4. Full per-task records + public-surface materialization + ingest report ──
    native_records = adapter.load_native_task_records()
    seed_ids = {row["seed_id"] for row in seed_rows}
    native_ids = {record["seed_id"] for record in native_records}
    if native_ids != seed_ids:
        raise RuntimeError(
            "native records and seed rows have inconsistent coverage: "
            f"native-only={sorted(native_ids - seed_ids)} "
            f"seed-only={sorted(seed_ids - native_ids)}"
        )
    public_inputs = materialize_public_inputs(
        native_records, str(work / "public_inputs")
    )
    ingest_report = build_native_ingest_report(
        native_records, release=release, public_inputs=public_inputs
    )
    excluded = [task for task in ingest_report["tasks"] if task["excluded"]]
    report_path = Path(_write_json(work / "ingest_report.json", ingest_report))
    if excluded:
        raise RuntimeError(
            "The ingest report contains excluded tasks (cross-release text collisions); refusing ingestion: "
            f"{[task['task_id'] for task in excluded]}"
        )

    # ── 5. wb_composite ingest rows + augment config (verifier placeholder, implemented in M2) ──
    rows = build_native_ingest_rows(native_records)
    placeholder = work / "wb_composite_runner_placeholder.py"
    placeholder.write_text(_PLACEHOLDER_RUNNER, encoding="utf-8")
    config = build_workbuddy_augment_config(
        rows,
        release=release,
        split="development",
        strategy="native_ingest",
        environment={"path": archive_path, "sha256": archive_sha},
        verifier_path=str(placeholder),
        verifier_version="wb-composite-placeholder-m1",
        audit_roots={
            "reviewed": True,
            "reviewer": args.reviewer,
            "review_date": args.review_date,
            "design_doc_sha256": file_sha256(_DESIGN_DOC),
        },
        ingest_report={"path": str(report_path), "sha256": file_sha256(report_path)},
        public_inputs=public_inputs,
    )
    config_path = _write_json(work / "config.json", config)

    print(json.dumps({
        "seed_stage": {
            "seed_source": "real-hf-subset",
            "dataset_id": adapter.dataset_id,
            "seed_row_count": len(seed_rows),
            "archive_path": archive_path,
            "archive_sha256": archive_sha,
        },
        "source_stage": {
            "release": release,
            "source_path": source_path,
            "source_sha256": file_sha256(source_path),
            "license": {
                "native_ingest_allowed": True,
                "derive_allowed": False,
                "native_ingest_review": license_record["native_ingest_review"],
            },
        },
        "ingest_stage": {
            "public_inputs_dir": public_inputs["dir"],
            "public_input_file_counts": {
                task_id: len(entries)
                for task_id, entries in public_inputs["by_task"].items()
            },
            "ingest_report_path": str(report_path),
            "ingest_report_sha256": file_sha256(report_path),
            "excluded_count": len(excluded),
            "contamination_screen": "per-task ingest report (design §5)",
        },
        "config_stage": {
            "strategy": "native_ingest",
            "row_count": len(rows),
            "verifier_kind": "wb_composite",
            "verifier_runner": "placeholder (production runner lands in M2)",
            "config_path": config_path,
            "config_sha256": file_sha256(config_path),
            "stopped_before_build": True,
        },
        "work_dir": str(work),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(json.dumps({"error": str(error), "exit_code": 2},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
