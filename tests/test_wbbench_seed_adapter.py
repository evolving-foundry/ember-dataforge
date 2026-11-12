"""WBBenchSeedAdapter unit tests (M0): fixture loading / field mapping / taxonomy / archiving.

Run (repo convention, pytest-compatible under tests/):
  PYTHONPATH=. python3 tests/test_wbbench_seed_adapter.py
  or pytest tests/test_wbbench_seed_adapter.py

Optional integration test (real HF subset; external directory specified by an environment variable): enabled when
WB_OFFICE_ARCHIVE points to wb-bench-office-v1.0.tar.gz; skipped otherwise
(real data is not committed, respecting the Tencent license requirement).
"""

import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

base_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_root))

from data_designer.sources.workbuddy_seed import (  # noqa: E402
    WBBenchSeedAdapter,
    WBBenchSeedAdapterError,
)

FIXTURE_DIR = (
    base_root / "data_designer" / "sources"
    / "fixtures" / "wbbench_office_fixture"
)


def test_fixture_load_seed_tasks():
    adapter = WBBenchSeedAdapter(FIXTURE_DIR)
    rows = adapter.load_seed_tasks()
    assert adapter.seed_source == "synthetic-fixture"
    assert adapter.subset == "office"
    assert len(rows) == 2

    by_id = {r["seed_id"]: r for r in rows}
    sales = by_id["demo-sales-rollup-L1-001"]
    assert sales["difficulty"] == "easy"
    assert sales["category"] == "data-file-ops"
    assert sales["verifier_form"] == "rule"  # llm_weight=0
    assert sales["deliverable_type"] == ["json"]
    assert sales["tools_available"] == []
    assert "sales_q1.csv" in sales["task_text"]
    assert "output/summary.json" in sales["success_criteria"]

    actions = by_id["demo-action-items-L2-002"]
    assert actions["difficulty"] == "medium"
    assert actions["category"] == "doc-ops"
    assert actions["verifier_form"] == "composite"  # 0.8/0.2
    assert actions["role"] == "unspecified"


def test_fixture_taxonomy_grid():
    grid = WBBenchSeedAdapter(FIXTURE_DIR).taxonomy_grid()
    assert grid == {
        "data-file-ops": {"easy": 1},
        "doc-ops": {"medium": 1},
    }


def test_fixture_manifest():
    manifest = WBBenchSeedAdapter(FIXTURE_DIR).build_manifest()
    assert manifest["seed_source"] == "synthetic-fixture"
    assert manifest["task_count"] == 2
    assert manifest["verifier_forms"] == {"rule": 1, "composite": 1}
    assert manifest["categories"] == {"data-file-ops": 1, "doc-ops": 1}


def test_from_archive():
    tmp = Path(tempfile.mkdtemp(prefix="wb-adapter-test-"))
    archive = tmp / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(FIXTURE_DIR, arcname="wb-bench-office-fixture-v1.0")
    adapter = WBBenchSeedAdapter.from_archive(archive)
    rows = adapter.load_seed_tasks()
    assert len(rows) == 2
    assert adapter.archive_sha256 and len(adapter.archive_sha256) == 64
    assert adapter.seed_source == "synthetic-fixture"


def test_missing_dataset_toml_raises():
    try:
        WBBenchSeedAdapter(Path(tempfile.mkdtemp(prefix="wb-empty-")))
    except WBBenchSeedAdapterError:
        return
    raise AssertionError("missing dataset.toml must fail explicitly")


def test_role_heuristics():
    from data_designer.sources.workbuddy_seed import (
        _role_from_keywords,
        _verifier_form_from_judge,
    )
    assert _role_from_keywords(["finance-quant"]) == "algo"
    assert _role_from_keywords(["governance"]) == "ops"
    assert _role_from_keywords(["state-diff-mock-app"]) == "developer"
    assert _role_from_keywords(["something-else"]) == "unspecified"
    assert _verifier_form_from_judge(None) == "rule"
    assert _verifier_form_from_judge({"score_merge": {"rule_weight": 1.0}}) == "rule"
    assert _verifier_form_from_judge({"score_merge": {"llm_weight": 1.0}}) == "llm_rubric"
    assert _verifier_form_from_judge(
        {"score_merge": {"rule_weight": 0.8, "llm_weight": 0.2}}) == "composite"


def test_task_count_mismatch_raises():
    meta = (FIXTURE_DIR / "dataset.toml").read_text(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp(prefix="wb-badcount-"))
    (tmp / "dataset.toml").write_text(
        meta.replace("task_count = 2", "task_count = 5"), encoding="utf-8")
    shutil.copytree(FIXTURE_DIR / "tasks", tmp / "tasks")
    try:
        WBBenchSeedAdapter(tmp).load_seed_tasks()
    except WBBenchSeedAdapterError as exc:
        assert "task_count mismatch" in str(exc)
        return
    raise AssertionError("task_count mismatch must fail explicitly")


def test_real_office_archive_optional():
    archive = os.environ.get("WB_OFFICE_ARCHIVE")
    if not archive or not Path(archive).is_file():
        print("skip: WB_OFFICE_ARCHIVE is not set (real data is not committed)")
        return
    adapter = WBBenchSeedAdapter.from_archive(archive)
    rows = adapter.load_seed_tasks()
    assert adapter.seed_source == "real-hf-subset"
    assert adapter.subset == "office"
    assert len(rows) == 50
    manifest = adapter.build_manifest()
    assert manifest["categories"].get("data-file-ops") == 24
    assert manifest["categories"].get("doc-ops") == 17
    assert manifest["categories"].get("automation-workdir") == 9
    # License requirement: real question text must not enter the repository—validate only row presence and structure
    assert all(r["task_text"] and r["difficulty"] in {"easy", "medium", "hard"}
               for r in rows)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
