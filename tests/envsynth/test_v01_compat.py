"""v0.1 command, artifact store and publisher compatibility tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from data_designer.storage.catalog import ArtifactCatalog
from tests.envsynth.env_fixtures import build_seed, worker_request, write_input
from tests.envsynth.test_env_doc_import import _require_document_runtime, _write_docx


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_SCRIPT = REPO_ROOT / "scripts" / "gym_publish_env.py"
TICKETS = [
    ("GD-2026-0001", "线路检修", "已审批"),
    ("GD-2026-0002", "停送电操作", "审核中"),
]


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def test_v01_source_worker_command_path(tmp_path):
    _require_document_runtime()
    source = tmp_path / "seed.docx"
    _write_docx(source)
    input_path = tmp_path / "source-input.json"
    output_dir = tmp_path / "seed-output"
    input_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "request": {
                    "sourcePaths": [str(source)],
                    "family": "power-ticket",
                    "anchorHoldoutRatio": 0.3,
                    "maxQuestions": 20,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = _run_module(
        "data_designer.pipelines.envsynth.cli.source_import",
        str(input_path),
        str(output_dir),
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert Path(summary["seedManifestPath"]).is_file()


def test_v01_synthesis_worker_freezes_store_and_publisher_reads_it(tmp_path):
    _, seed_dir = build_seed(tmp_path, TICKETS, seed_name="v01-seed")
    request = worker_request(tmp_path, seed_dir, asset_name="v01-assets")
    store_path = tmp_path / "artifact-store"
    request["request"]["storePath"] = str(store_path)
    input_path = write_input(tmp_path, request, name="v01-input.json")
    snapshot_path = tmp_path / "v01-snapshot.json"

    result = _run_module(
        "data_designer.pipelines.envsynth.cli.synthesis_worker",
        input_path,
        str(snapshot_path),
    )
    assert result.returncode == 0, result.stderr
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    snapshot_again_path = tmp_path / "v01-snapshot-again.json"
    repeated = _run_module(
        "data_designer.pipelines.envsynth.cli.synthesis_worker",
        input_path,
        str(snapshot_again_path),
    )
    assert repeated.returncode == 0, repeated.stderr
    snapshot_again = json.loads(snapshot_again_path.read_text(encoding="utf-8"))
    assert snapshot_again["envRelease"]["version"] == snapshot["envRelease"]["version"]

    store = ArtifactCatalog(str(store_path))
    try:
        release = store.get_artifact(
            "env_release", request["request"]["refId"], snapshot["envRelease"]["version"]
        )
        taskset = store.get_artifact(
            "taskset", snapshot["taskset"]["refId"], snapshot["taskset"]["version"]
        )
        releases = [
            artifact
            for artifact in store.list_artifacts("env_release")
            if artifact["ref_id"] == request["request"]["refId"]
        ]
        tasksets = [
            artifact
            for artifact in store.list_artifacts("taskset")
            if artifact["ref_id"] == snapshot["taskset"]["refId"]
        ]
    finally:
        store.close()
    assert len(releases) == 1
    assert len(tasksets) == 1
    assert release is not None
    assert taskset is not None
    assert release["fingerprint"] == snapshot["envRelease"]["fingerprint"]
    assert release["payload"]["assetDir"] == snapshot["envRelease"]["assetDir"]
    assert release["payload"]["buildFingerprint"]
    assert Path(snapshot["envRelease"]["assetDir"]).is_dir()
    assert taskset["fingerprint"] == snapshot["taskset"]["fingerprint"]

    gym_root = tmp_path / "gym"
    (gym_root / "overlay").mkdir(parents=True)
    published = subprocess.run(
        [
            sys.executable,
            str(PUBLISH_SCRIPT),
            "--gym-root",
            str(gym_root),
        ],
        cwd=str(REPO_ROOT),
        input=json.dumps(
            {
                "storePath": str(store_path),
                "refId": request["request"]["refId"],
                "version": snapshot["envRelease"]["version"],
                "name": "office-power",
            },
            ensure_ascii=False,
        ),
        capture_output=True,
        text=True,
    )
    assert published.returncode == 0, published.stderr
    output = json.loads(published.stdout)
    assert output["status"] == "published"
    assert output["runnable"] is True
    assert output["benchmark"] == "synth-office-power"
    asset_path = (
        gym_root
        / "overlay"
        / output["assets"]["packageZip"]
    )
    assert asset_path.is_file()
    assert snapshot["envRelease"]["fingerprint"] in asset_path.name
    config = gym_root / "overlay" / output["assets"]["config"]
    assert config.is_file()
