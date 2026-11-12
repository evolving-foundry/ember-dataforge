"""terminal source import and build tests (M2): seed onboarding, T2/T3/T8 pins, re-expression, and full-chain build.

Aligned with terminal onboarding implementation plan §2 M2 acceptance: ``TB21SeedAdapter`` classifies the
expressible subset per T8 criteria; the ``benchmark-source-v1`` manifest freezes release/license/snapshot pins and can re-check drift;
the re-expression spec scores 1.0 on the gold artifact and 0.0 on the negative artifact; fixture-derived tasks (≥1)
complete ``benchmark_taskset_worker``, the item key set is exactly ``PUBLIC_KEYS``, and
``benchmark.family == "terminal"``. Seed materials use the repository's desensitized synthetic fixture.
"""

import json
import shutil
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import PUBLIC_KEYS, file_sha256
from data_designer.pipelines.benchmarks.families.terminal import (
    UPSTREAM_TASK_COUNT,
    assert_terminal_build,
    derive_release,
    derive_terminal_spec,
    import_terminal_source,
    screen_terminal_contamination,
    verify_terminal_source_manifest,
)
from data_designer.pipelines.benchmarks.build import build_benchmark_taskset
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.sources.terminal_seed import (
    EXCLUDED_REASON_STATE_IN_IMAGE,
    TB21SeedAdapter,
    TB21SeedAdapterError,
)
from data_designer.verifiers import run_verifier
from tests.benchmarks.terminal_fixture import (
    EXCLUDED_ID,
    EXPRESSIBLE_IDS,
    FIXTURE_DIR,
    LICENSE,
    RUNNER_VERSION,
    build_config,
    derived_rows,
    prepare_source,
)

UPSTREAM_COMMIT = "7131e4375048a0e408a8fb404b5f499d726b695b"


def _adapter():
    return TB21SeedAdapter(FIXTURE_DIR)


def _import(tmp_path: Path, **overrides):
    prepared = prepare_source(tmp_path)
    return prepared


def test_seed_adapter_classifies_expressibility():
    rows = _adapter().load_seed_tasks()
    assert {row["seed_id"] for row in rows} == set(EXPRESSIBLE_IDS) | {EXCLUDED_ID}
    by_id = {row["seed_id"]: row for row in rows}
    for seed_id in EXPRESSIBLE_IDS:
        assert by_id[seed_id]["expressibility"] == "expressible"
        assert by_id[seed_id]["exclusion_reason"] is None
    assert by_id[EXCLUDED_ID]["expressibility"] == "excluded"
    assert by_id[EXCLUDED_ID]["exclusion_reason"] == EXCLUDED_REASON_STATE_IN_IMAGE
    assert "scripts/sum.py" in by_id[EXPRESSIBLE_IDS[0]]["setup_files"]
    assert by_id[EXCLUDED_ID]["docker_image"] == "internal-registry/cfg-base:2.1"
    assert by_id[EXCLUDED_ID]["setup_files"] == {}


def test_seed_manifest_counts_and_provenance():
    manifest = _adapter().build_manifest()
    assert manifest["seed_source"] == "synthetic-fixture"
    assert manifest["dataset_id"] == "terminal-bench-2-1-fixture-v1.0"
    assert manifest["task_count"] == 3
    assert manifest["expressible_count"] == 2
    assert manifest["excluded"] == {EXCLUDED_REASON_STATE_IN_IMAGE: 1}


def test_release_semantics_fixture_and_upstream():
    rows = _adapter().load_seed_tasks()
    release = derive_release("synthetic-fixture", rows)
    assert len(release) == len("synthetic-fixture:") + 8
    assert release.startswith("synthetic-fixture:")
    dummy = [{"seed_id": f"up-{index:03d}"} for index in range(UPSTREAM_TASK_COUNT)]
    assert derive_release(
        "upstream-checkout", dummy, upstream_commit=UPSTREAM_COMMIT
    ) == "tb21-" + UPSTREAM_COMMIT[:8]
    with pytest.raises(Exception, match="unknown terminal seed_source"):
        derive_release("real-hf-subset", rows)


def test_upstream_release_requires_commit_and_task_count():
    dummy = [{"seed_id": f"up-{index:03d}"} for index in range(UPSTREAM_TASK_COUNT)]
    with pytest.raises(Exception, match="40-hex"):
        derive_release("upstream-checkout", dummy, upstream_commit="deadbeef")
    short = dummy[:-1]
    with pytest.raises(Exception, match=f"exactly {UPSTREAM_TASK_COUNT}"):
        derive_release(
            "upstream-checkout", short, upstream_commit=UPSTREAM_COMMIT
        )


def test_import_rejects_release_license_and_expressible_pins(tmp_path):
    prepared = prepare_source(tmp_path)
    rows = prepared["seed_rows"]
    base = dict(
        split="development",
        license_record=dict(LICENSE),
        seed_source="synthetic-fixture",
        dataset_id=prepared["adapter"].dataset_id,
        archive_path=prepared["archive_path"],
        archive_sha256=prepared["archive_sha256"],
    )
    with pytest.raises(Exception, match="release pin mismatch"):
        import_terminal_source(
            rows, release="synthetic-fixture:00000000", **base
        )
    with pytest.raises(Exception, match="license decision required"):
        import_terminal_source(
            rows,
            release=prepared["release"],
            license_record={"id": "internal-use"},
            **{k: v for k, v in base.items() if k != "license_record"},
        )
    all_excluded = [
        dict(row, expressibility="excluded", exclusion_reason="runtime_state")
        for row in rows
    ]
    with pytest.raises(Exception, match="no expressible cases"):
        import_terminal_source(
            all_excluded,
            release=derive_release("synthetic-fixture", all_excluded),
            **base,
        )


def test_verify_source_manifest_rejects_drift(tmp_path):
    prepared = prepare_source(tmp_path)
    source = prepared["source"]
    verify_terminal_source_manifest(source)
    case_drift = json.loads(json.dumps(source))
    case_drift["cases"][0]["task_text"] = "rewritten instruction"
    with pytest.raises(Exception, match="drift"):
        verify_terminal_source_manifest(case_drift)
    release_drift = json.loads(json.dumps(source))
    release_drift["release"] = "synthetic-fixture:00000000"
    with pytest.raises(Exception, match="release pin mismatch"):
        verify_terminal_source_manifest(release_drift)


def test_runtime_state_heuristic_and_metadata_override(tmp_path):
    root = tmp_path / "mini-dataset"
    tasks = root / "tasks"
    for name, instruction, override in (
        (
            "t1",
            "Start the service and confirm it is listening on port 8080.",
            "",
        ),
        (
            "t2",
            "Check the ssh service logs and write a summary file.",
            'expressibility = "expressible"',
        ),
    ):
        task_dir = tasks / name
        task_dir.mkdir(parents=True)
        meta = f'[metadata]\ndifficulty = "easy"\n{override}'.strip()
        (task_dir / "task.toml").write_text(
            f'[task]\nname = "terminal/{name}"\n\n{meta}\n\n[environment]\n',
            encoding="utf-8",
        )
        (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
    (root / "dataset.toml").write_text(
        '[dataset]\nid = "mini-fixture"\ntask_count = 2\n', encoding="utf-8"
    )
    rows = {row["seed_id"]: row for row in TB21SeedAdapter(root).load_seed_tasks()}
    assert rows["t1"]["exclusion_reason"] == "runtime_state"
    assert rows["t2"]["expressibility"] == "expressible"
    (tasks / "t3").mkdir()
    (tasks / "t3" / "task.toml").write_text(
        '[task]\nname = "terminal/t3"\n\n[metadata]\ndifficulty = "easy"\n'
        'expressibility = "bogus"\n\n[environment]\n',
        encoding="utf-8",
    )
    (tasks / "t3" / "instruction.md").write_text(
        "Sum two numbers.", encoding="utf-8"
    )
    (root / "dataset.toml").write_text(
        '[dataset]\nid = "mini-fixture"\ntask_count = 3\n', encoding="utf-8"
    )
    with pytest.raises(TB21SeedAdapterError, match="expressibility"):
        TB21SeedAdapter(root).load_seed_tasks()


def test_derivation_rejects_excluded_case(tmp_path):
    prepared = prepare_source(tmp_path)
    excluded = next(
        case for case in prepared["source"]["cases"]
        if case["terminal"]["expressibility"] == "excluded"
    )
    with pytest.raises(Exception, match="excluded from derivation"):
        derive_terminal_spec(
            prepared["source"], excluded,
            {"kind": "unit_test", "path": "deliverable/files.json"},
        )


def test_derived_spec_scores_gold_and_negative(tmp_path):
    prepared = prepare_source(tmp_path)
    _, derived = derived_rows(tmp_path, prepared["source"])
    assert set(derived) == set(EXPRESSIBLE_IDS)
    for bundle in derived.values():
        positive = run_verifier(
            bundle["verifier_spec"], bundle["gold_deliverable"]
        )
        assert positive["passed"] is True
        assert positive["score"] == 1.0
        negative = run_verifier(
            bundle["verifier_spec"], bundle["negative_deliverable"]
        )
        assert negative["passed"] is False
        assert negative["score"] == 0.0


def test_contamination_screen_rejects_seed_text(tmp_path):
    prepared = prepare_source(tmp_path)
    rows, _ = derived_rows(tmp_path, prepared["source"])
    screen_terminal_contamination(rows, prepared["source"]["cases"])
    with pytest.raises(Exception, match="matches a seed text"):
        screen_terminal_contamination(
            [{"task_text": prepared["seed_rows"][0]["task_text"]}],
            prepared["source"]["cases"],
        )


def _write(path: Path, value) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(path)


def test_full_build_chain_produces_terminal_taskset(tmp_path):
    prepared = prepare_source(tmp_path)
    rows, derived = derived_rows(tmp_path, prepared["source"])
    config, config_path = build_config(
        tmp_path, prepared["source"], rows, derived
    )
    request = {
        "schemaVersion": 1,
        "request": {
            "storePath": str(tmp_path / "catalog"),
            "sourceManifestPath": prepared["source_path"],
            "augmentConfigPath": config_path,
            "sourceSha256": file_sha256(prepared["source_path"]),
            "augmentConfigSha256": file_sha256(config_path),
            "trustedDir": str(tmp_path / "trusted"),
            "refId": "terminal-fixture",
            "family": "terminal",
            "sourceRelease": prepared["source"]["release"],
            "split": "development",
        },
    }
    request_path = _write(tmp_path / "build-input.json", request)
    snapshot_path = tmp_path / "snapshot.json"
    build = request["request"]
    snapshot = build_benchmark_taskset(ArtifactCatalog(build["storePath"]),
        build["sourceManifestPath"], build["augmentConfigPath"], build["trustedDir"], build["refId"],
        expected_family=build["family"], expected_release=build["sourceRelease"], expected_split=build["split"])
    assert snapshot["item_count"] == len(EXPRESSIBLE_IDS)
    assert snapshot["taskset"]["ref_id"] == "terminal-fixture"
    store = ArtifactCatalog(str(tmp_path / "catalog"))
    try:
        pin = snapshot["taskset"]
        payload = store.get_artifact("taskset", pin["ref_id"], pin["version"])[
            "payload"
        ]
        items = payload["taskset"]
        assert len(items) == len(EXPRESSIBLE_IDS)
        for item in items:
            assert set(item) == PUBLIC_KEYS
            assert item["benchmark"]["family"] == "terminal"
            assert item["task_id"].startswith("terminal:")
            assert item["benchmark"]["source_release"] == prepared["source"][
                "release"
            ]
            assert "verifier_spec" not in json.dumps(item)
            assert "solution" not in json.dumps(item["benchmark"])
            runtime = payload["runtime_by_task"][item["task_id"]]
            assert runtime["verifier"]["version"] == RUNNER_VERSION
            assert "runners/terminal.py" in " ".join(
                runtime["verifier"]["command"]
            )
            manifest_path = (
                Path(payload["trusted_manifest_dir"])
                / f"{item['benchmark']['case_manifest_sha256']}.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            spec = manifest["private"]["verifier_spec"]
            assert spec["kind"] in {"unit_test", "artifact_compare"}
            assert manifest["private"]["deliverable_path"].startswith(
                "deliverable/"
            )
        audit = payload["build_report"]["audit"]
        assert len(audit) == len(EXPRESSIBLE_IDS)
        for entry in audit:
            assert entry["reference"]["passed"] is True
            assert entry["reference"]["score"] == 1.0
            assert entry["negative"]["passed"] is False
            assert entry["negative"]["score"] == 0.0
    finally:
        store.close()


def test_assert_terminal_build_rejections(tmp_path):
    prepared = prepare_source(tmp_path)
    source = prepared["source"]
    expressible = source["cases"][0]
    excluded = next(
        case for case in source["cases"]
        if case["terminal"]["expressibility"] == "excluded"
    )
    runtime = {"verifier": {"version": RUNNER_VERSION}}
    private = {"verifier_spec": {"kind": "unit_test"}}
    with pytest.raises(Exception, match="excluded from derivation"):
        assert_terminal_build(runtime, private, excluded)
    with pytest.raises(Exception, match="lacks family metadata"):
        assert_terminal_build(runtime, private, {"source_task_id": "x"})
    with pytest.raises(Exception, match="offline runner shim"):
        assert_terminal_build(
            {"verifier": {"version": "other-verifier"}}, private, expressible
        )
    with pytest.raises(Exception, match="verifier_spec"):
        assert_terminal_build(
            runtime, {"verifier_spec": {"kind": "llm_rubric"}}, expressible
        )


def test_seed_task_count_mismatch(tmp_path):
    root = tmp_path / "fixture-copy"
    shutil.copytree(FIXTURE_DIR, root)
    (root / "dataset.toml").write_text(
        '[dataset]\nid = "terminal-bench-2-1-fixture-v1.0"\ntask_count = 89\n',
        encoding="utf-8",
    )
    with pytest.raises(TB21SeedAdapterError, match="task_count"):
        TB21SeedAdapter(root).load_seed_tasks()
