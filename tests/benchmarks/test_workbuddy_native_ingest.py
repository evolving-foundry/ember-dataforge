"""workbuddy native task ingestion tests (M1): native license gate, public-surface materialization, ingest report,
and wb_composite task-ingest rows with the full-chain native build mode.

Aligned with design §5/§6.2/§9/§12: seed materials are a test-local synthetic real-layout fixture (real subsets are not committed);
the runner shim responds per the native.py protocol (reference passes, negative fails);
the wb_composite production runner belongs to M2.
"""

import io
import json
import tarfile
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import (
    PUBLIC_KEYS,
    canonical_sha256,
    file_sha256,
    normalized_text_sha256,
)
from data_designer.pipelines.benchmarks.families.workbuddy import (
    build_native_ingest_rows,
    build_workbuddy_augment_config,
    derive_release,
    import_workbuddy_source,
    verify_native_ingest_pins,
    verify_workbuddy_source_manifest,
)
from data_designer.pipelines.benchmarks.families.workbuddy_audit import (
    INGEST_REPORT_SCHEMA,
    build_native_ingest_report,
    materialize_public_inputs,
)
from data_designer.pipelines.benchmarks.build import build_benchmark_taskset
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.sources.workbuddy_seed import WBBenchSeedAdapter

SHIM_VERSION = "test-wb-composite-shim-v1"

SHIM_RUNNER = '''import json, sys
request = json.load(open(sys.argv[1], encoding="utf-8"))
passed = request["phase"] == "reference"
result = {
    "task_id": request["taskId"],
    "status": "completed",
    "passed": passed,
    "score": 1.0 if passed else 0.0,
    "native_metrics": {"composite": 1.0 if passed else 0.0},
    "verifier_version": %r,
    "evidence_refs": ["test-shim:" + request["phase"]],
    "runtime_seconds": 0.0,
}
json.dump(result, open(sys.argv[2], "w", encoding="utf-8"))
''' % SHIM_VERSION

DATASET_TOML = """\
[dataset]
id = "wb-bench-office-native-v1.0"
schema = "harbor"
task_count = 2
"""

TASK_TOML = """\
[task]
name = "office/{task_id}"
keywords = ["quarterly", "summary"]

[metadata]
difficulty = "medium"
category = "data-file-ops"

[environment]
mcp_servers = []

[verifier]
timeout_sec = 1800.0
"""

JUDGE_YAML = """\
artifacts:
  - id: summary
    path: output/summary.xlsx
    type: xlsx
    required: true
  - id: notes
    path: output/notes.md
    type: md
    required: false
llm_judge:
  method: rubric_binary_mean
  rubrics:
    - id: r1
      question: Does the summary include every required region?
    - id: r2
      question: Are the totals consistent with the workspace data?
score_merge:
  rule_weight: 0.8
  llm_weight: 0.2
"""


def _workspace_tar(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _make_seed(root: Path) -> Path:
    """Synthetic real-layout fixture: two tasks, with a workspace containing public inputs and private files."""
    seed = root / "wb-bench-office-native-v1.0"
    (seed / "tasks").mkdir(parents=True)
    (seed / "dataset.toml").write_text(DATASET_TOML, encoding="utf-8")
    for index, task_id in enumerate(("t01", "t02")):
        task_dir = seed / "tasks" / task_id
        (task_dir / "tests" / "gold").mkdir(parents=True)
        (task_dir / "tests" / "grading").mkdir(parents=True)
        (task_dir / "environment").mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            TASK_TOML.format(task_id=task_id), encoding="utf-8"
        )
        (task_dir / "instruction.md").write_text(
            f"Produce the quarter-{index + 1} regional summary workbook and a "
            f"one-page note from the workspace inputs.\n",
            encoding="utf-8",
        )
        (task_dir / "tests" / "judge.yaml").write_text(JUDGE_YAML, encoding="utf-8")
        (task_dir / "tests" / "gold" / "gold_answer.json").write_text(
            json.dumps({"summary": {"total": 1500.0 + index}}), encoding="utf-8"
        )
        (task_dir / "tests" / "grading" / "check.py").write_text(
            "# grading rubric reference\n", encoding="utf-8"
        )
        workspace = _workspace_tar(
            {
                f"input/data{index}.csv": f"k,v\n1,{index}\n".encode(),
                "input/notes.md": b"# raw meeting notes\n",
                "tests/judge.yaml": b"judge: internal\n",
                "gold/gold_answer.json": b'{"leaked": true}\n',
                "input/secret_key.txt": b"private material\n",
            }
        )
        (task_dir / "environment" / "workspace.tar.gz").write_bytes(workspace)
    return seed


def _native_license() -> dict:
    evidence = [
        "design_doc:RSI_workbuddy设计_v0.1.md#sha256:" + "a" * 64,
        "seed_archive:workbuddy-seed.tar.gz#sha256:" + "b" * 64,
        "scope:verbatim_ingest_only;redistribute=false;private_evaluation_only",
    ]
    return {
        "id": "wb-bench-native-ingest@wb-bench-test0001",
        "derive_allowed": False,
        "native_ingest_allowed": True,
        "redistribute_allowed": False,
        "access_conditions": "local private evaluation only; verbatim ingest",
        "native_ingest_review": {
            "reviewer": "chadwweng",
            "date": "2026-10-09",
            "evidence": evidence,
            "evidence_sha256": canonical_sha256(evidence),
        },
    }


def _import_native(tmp_path: Path, seed: Path, **overrides):
    archive = tmp_path / "workbuddy-seed.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(seed, arcname=seed.name)
    archive_sha = file_sha256(archive)
    adapter = WBBenchSeedAdapter(seed, seed_source="real-hf-subset")
    rows = adapter.load_seed_tasks()
    release = derive_release("real-hf-subset", rows, archive_sha)
    kwargs = dict(
        release=release,
        split="development",
        license_record=_native_license(),
        seed_source="real-hf-subset",
        dataset_id=adapter.dataset_id,
        archive_path=str(archive),
        archive_sha256=archive_sha,
    )
    kwargs.update(overrides)
    return import_workbuddy_source(rows, **kwargs), rows, adapter, release


def _ingest_chain(tmp_path: Path, seed: Path):
    """license → source → native records → public inputs → report → rows."""
    source, rows, adapter, release = _import_native(tmp_path, seed)
    verify_workbuddy_source_manifest(source)
    records = adapter.load_native_task_records()
    public_inputs = materialize_public_inputs(
        records, str(tmp_path / "public_inputs")
    )
    report = build_native_ingest_report(
        records, release=release, public_inputs=public_inputs
    )
    report_path = tmp_path / "ingest_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    rows_out = build_native_ingest_rows(records)
    return source, release, records, public_inputs, report, report_path, rows_out


    # ── native license gate (D3 mutual exclusion + D2 review triplet) ─────────────────────────


def test_native_license_freezes_into_source_manifest(tmp_path):
    seed = _make_seed(tmp_path)
    source, rows, _, _ = _import_native(tmp_path, seed)
    assert source["release"].startswith("wb-bench-")
    assert source["license"]["native_ingest_allowed"] is True
    assert source["license"]["derive_allowed"] is False
    review = source["license"]["native_ingest_review"]
    assert review["reviewer"] == "chadwweng"
    assert review["date"] == "2026-10-09"
    assert review["evidence"]
    assert review["evidence_sha256"]
    assert len(source["cases"]) == len(rows) == 2


def test_native_and_derivation_cannot_coexist(tmp_path):
    seed = _make_seed(tmp_path)
    rows = WBBenchSeedAdapter(seed, seed_source="real-hf-subset").load_seed_tasks()
    license_record = {**_native_license(), "derive_allowed": True}
    with pytest.raises(Exception, match="D3"):
        import_workbuddy_source(
            rows,
            release="wb-bench-test0001",
            split="development",
            license_record=license_record,
            seed_source="real-hf-subset",
            dataset_id="wb-bench-office-native-v1.0",
        )


def test_native_requires_wb_bench_release(tmp_path):
    seed = _make_seed(tmp_path)
    rows = WBBenchSeedAdapter(seed, seed_source="real-hf-subset").load_seed_tasks()
    with pytest.raises(Exception, match="wb-bench-"):
        import_workbuddy_source(
            rows,
            release="synthetic-fixture:abcd1234",
            split="development",
            license_record=_native_license(),
            seed_source="real-hf-subset",
            dataset_id="wb-bench-office-native-v1.0",
        )


@pytest.mark.parametrize("mutate,reason", [
    (lambda lic: lic.pop("native_ingest_review"), "native_ingest_review"),
    (lambda lic: lic["native_ingest_review"].update(evidence=[]), "evidence"),
    (lambda lic: lic["native_ingest_review"].update(evidence_sha256="bad"), "sha256"),
    (lambda lic: lic["native_ingest_review"].update(reviewer=""), "reviewer"),
])
def test_native_requires_review_triple(tmp_path, mutate, reason):
    seed = _make_seed(tmp_path)
    rows = WBBenchSeedAdapter(seed, seed_source="real-hf-subset").load_seed_tasks()
    license_record = _native_license()
    mutate(license_record)
    with pytest.raises(Exception, match=reason):
        import_workbuddy_source(
            rows,
            release="wb-bench-test0001",
            split="development",
            license_record=license_record,
            seed_source="real-hf-subset",
            dataset_id="wb-bench-office-native-v1.0",
        )


    # ── public-surface materialization (D5) and ingest report ──────────────────────────────────────────


def test_materialize_public_inputs_filters_private_material(tmp_path):
    seed = _make_seed(tmp_path)
    adapter = WBBenchSeedAdapter(seed, seed_source="real-hf-subset")
    records = adapter.load_native_task_records()
    out_root = tmp_path / "public"
    public_inputs = materialize_public_inputs(records, str(out_root))
    assert Path(public_inputs["dir"]) == out_root
    for record in records:
        entries = public_inputs["by_task"][record["seed_id"]]
        names = {entry["path"] for entry in entries}
        assert names <= {"input/notes.md", "input/data0.csv", "input/data1.csv"}
        for entry in entries:
            target = out_root / record["seed_id"] / entry["path"]
            assert file_sha256(target) == entry["sha256"]
    # Private surfaces never touch disk: tests/**, gold/**, and any PRIVATE_MARKERS hits are removed
    dumped = json.dumps(public_inputs)
    for forbidden in ("tests/", "gold/", "secret"):
        assert forbidden not in dumped


def test_materialize_public_inputs_requires_absolute_root(tmp_path):
    seed = _make_seed(tmp_path)
    adapter = WBBenchSeedAdapter(seed, seed_source="real-hf-subset")
    records = adapter.load_native_task_records()
    with pytest.raises(Exception, match="absolute"):
        materialize_public_inputs(records, "relative/root")


def test_ingest_report_collision_marks_excluded(tmp_path):
    seed = _make_seed(tmp_path)
    adapter = WBBenchSeedAdapter(seed, seed_source="real-hf-subset")
    records = adapter.load_native_task_records()
    public_inputs = materialize_public_inputs(records, str(tmp_path / "public"))
    report = build_native_ingest_report(
        records, release="wb-bench-v1", public_inputs=public_inputs
    )
    assert report["schema_version"] == INGEST_REPORT_SCHEMA
    assert all(not task["excluded"] for task in report["tasks"])
    assert all(task["native_evidence"]["seed_text_verbatim"] is True
               for task in report["tasks"])
    foreign = normalized_text_sha256(records[0]["task_text"])
    collision = build_native_ingest_report(
        records, release="wb-bench-v2", public_inputs=public_inputs,
        foreign_text_sha256=[foreign],
    )
    by_id = {task["task_id"]: task for task in collision["tasks"]}
    assert by_id[records[0]["seed_id"]]["excluded"] is True
    assert by_id[records[0]["seed_id"]]["native_evidence"]["foreign_release_collision"]
    assert by_id[records[1]["seed_id"]]["excluded"] is False


    # ── wb_composite task-ingest rows ────────────────────────────────────────────────────


def test_build_native_ingest_rows_composite_payload(tmp_path):
    seed = _make_seed(tmp_path)
    adapter = WBBenchSeedAdapter(seed, seed_source="real-hf-subset")
    records = adapter.load_native_task_records()
    rows = build_native_ingest_rows(records)
    assert [row["source_task_id"] for row in rows] == ["t01", "t02"]
    for row, record in zip(rows, records):
        spec = row["verifier_spec"]
        assert spec["kind"] == "wb_composite"
        assert spec["generator_ref"] == (
            f"wb_bench_native_ingest@{row['source_task_id']}"
        )
        payload = spec["payload"]
        assert payload["artifacts"][0]["path"] == "output/summary.xlsx"
        assert payload["score_merge"] == {"rule_weight": 0.8, "llm_weight": 0.2}
        assert payload["gold"] == record["gold"]
        assert payload["timeout_sec"] == 1800.0
        assert payload["grading_dir_sha256"]
        assert set(payload["env_pin"]) == {"path", "sha256"}
        assert len(row["item_fingerprint"]) == 16


def test_build_native_ingest_rows_rejects_bad_records(tmp_path):
    seed = _make_seed(tmp_path)
    adapter = WBBenchSeedAdapter(seed, seed_source="real-hf-subset")
    records = adapter.load_native_task_records()
    with pytest.raises(Exception, match="judge document"):
        build_native_ingest_rows([{**records[0], "judge": None}])
    duplicate = [records[0], {**records[1], "task_text": records[0]["task_text"]}]
    with pytest.raises(Exception, match="duplicate native task text"):
        build_native_ingest_rows(duplicate)


    # ── pin validation and full-chain native build mode (shim runner) ────────────────────────


def _write_deliverable(root: Path, artifact_path: str, value: dict) -> None:
    target = root / artifact_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value), encoding="utf-8")


def _native_build_chain(tmp_path: Path):
    seed = _make_seed(tmp_path)
    source, release, records, public_inputs, report, report_path, rows_out = (
        _ingest_chain(tmp_path, seed)
    )
    shim = tmp_path / "wb_composite_shim.py"
    shim.write_text(SHIM_RUNNER, encoding="utf-8")
    reference_root = tmp_path / "audit-reference"
    negative_root = tmp_path / "audit-negative"
    for record in records:
        _write_deliverable(reference_root, "output/summary.xlsx", {"total": 1500.0})
        _write_deliverable(negative_root, "output/summary.xlsx", {"total": 1.0})
    config = build_workbuddy_augment_config(
        rows_out,
        release=release,
        split="development",
        strategy="native_ingest",
        environment={
            "path": source["source_snapshot"]["path"],
            "sha256": source["source_snapshot"]["sha256"],
        },
        verifier_path=str(shim),
        verifier_version=SHIM_VERSION,
        audit_roots={
            "reviewed": True,
            "reference_root": str(reference_root),
            "negative_root": str(negative_root),
        },
        ingest_report={
            "path": str(report_path), "sha256": file_sha256(report_path),
        },
        public_inputs=public_inputs,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(source, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    request = {
        "schemaVersion": 1,
        "request": {
            "storePath": str(tmp_path / "catalog"),
            "sourceManifestPath": str(source_path),
            "augmentConfigPath": str(config_path),
            "sourceSha256": file_sha256(source_path),
            "augmentConfigSha256": file_sha256(config_path),
            "trustedDir": str(tmp_path / "trusted"),
            "refId": "workbuddy-native-ingest",
            "family": "workbuddy",
            "sourceRelease": release,
            "split": "development",
        },
    }
    request_path = tmp_path / "build-input.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    snapshot_path = tmp_path / "snapshot.json"
    build = request["request"]
    snapshot = build_benchmark_taskset(ArtifactCatalog(build["storePath"]),
        build["sourceManifestPath"], build["augmentConfigPath"], build["trustedDir"], build["refId"],
        expected_family=build["family"], expected_release=build["sourceRelease"], expected_split=build["split"])
    return snapshot, config, report_path


def test_verify_native_ingest_pins_rejects_drift_and_private_paths(tmp_path):
    seed = _make_seed(tmp_path)
    _, _, _, public_inputs, _, report_path, rows_out = _ingest_chain(
        tmp_path, seed
    )
    config = {
        "ingest_report": {"path": str(report_path),
                          "sha256": file_sha256(report_path)},
        "public_inputs": public_inputs,
    }
    pins = verify_native_ingest_pins(
        config, {row["source_task_id"] for row in rows_out}
    )
    assert set(pins) == {
        "ingest_report_sha256", "public_inputs_dir", "public_inputs_by_task"
    }
    # sha drift
    drifted = {
        "ingest_report": {"path": str(report_path), "sha256": "c" * 64},
        "public_inputs": public_inputs,
    }
    with pytest.raises(Exception, match="drift"):
        verify_native_ingest_pins(drifted, {"t01", "t02"})
    # Public surface references private material
    poisoned = {
        "ingest_report": {"path": str(report_path),
                          "sha256": file_sha256(report_path)},
        "public_inputs": {
            "dir": public_inputs["dir"],
            "by_task": {
                "t01": [{"path": "gold/gold_answer.json", "sha256": "d" * 64}],
                "t02": public_inputs["by_task"]["t02"],
            },
        },
    }
    with pytest.raises(Exception, match="private material"):
        verify_native_ingest_pins(poisoned, {"t01", "t02"})


def test_native_build_freezes_taskset_with_payload_pins(tmp_path):
    snapshot, config, _ = _native_build_chain(tmp_path)
    assert snapshot["item_count"] == 2
    store = ArtifactCatalog(str(tmp_path / "catalog"))
    try:
        pin = snapshot["taskset"]
        payload = store.get_artifact(
            "taskset", pin["ref_id"], pin["version"]
        )["payload"]
        assert payload["ingest_report_sha256"] == file_sha256(
            Path(config["ingest_report"]["path"])
        )
        assert set(payload["public_inputs_by_task"]) == {"t01", "t02"}
    # D5 keyset regression: public item key set is unchanged; verifier_spec/gold never leak
        for item in payload["taskset"]:
            assert set(item) == PUBLIC_KEYS
            assert item["benchmark"]["family"] == "workbuddy"
            assert item["benchmark"]["source_release"].startswith("wb-bench-")
            public_json = json.dumps(item)
            assert "verifier_spec" not in public_json
            assert "gold" not in json.dumps(item["benchmark"])
            assert "gold_answer" not in public_json
            runtime = payload["runtime_by_task"][item["task_id"]]
            assert runtime["verifier"]["version"] == SHIM_VERSION
            manifest_path = (
                Path(payload["trusted_manifest_dir"])
                / f"{item['benchmark']['case_manifest_sha256']}.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            spec = manifest["private"]["verifier_spec"]
            assert spec["kind"] == "wb_composite"
            assert spec["payload"]["gold"]["summary"]["total"] in (1500.0, 1501.0)
            assert manifest["private"]["deliverable_path"] == "output/summary.xlsx"
        audit = payload["build_report"]["audit"]
        assert len(audit) == 2
        for entry in audit:
            assert entry["reference"]["passed"] is True
            assert entry["negative"]["passed"] is False
    finally:
        store.close()
