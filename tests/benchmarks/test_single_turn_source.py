"""single-turn M2 tests: seed→source manifest, release/snapshot SHA/row-count pins, R2 materialization and rejection,
contamination fingerprint rejection, explicit D9 exclusion rejection, and full augment build chain (≥1 task/family through benchmark_taskset_worker).
"""

from __future__ import annotations

import json
import random
import shutil
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import (
    PUBLIC_KEYS,
    file_sha256,
)
from data_designer.pipelines.benchmarks.families import hf_single_turn as src
from data_designer.pipelines.benchmarks.families.hf_single_turn import (
    assert_hf_single_turn_build,
    build_hf_single_turn_augment_config,
    derive_hf_single_turn_spec,
    derive_release,
    import_hf_single_turn_source,
    screen_hf_single_turn_contamination,
    verify_hf_single_turn_source_manifest,
)
from data_designer.pipelines.benchmarks.build import build_benchmark_taskset
from data_designer.storage.catalog import ArtifactCatalog
from data_designer.sources.hf_single_turn_seed import (
    BENCHMARK_FAMILIES,
    HfSingleTurnSeedAdapter,
    HfSingleTurnSeedAdapterError,
)
from data_designer.verifiers import llm_judge, run_verifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = (
    PROJECT_ROOT / "data_designer" / "sources" / "fixtures"
    / "hf_single_turn_fixture"
)
RUNNER_PATHS = {
    family: PROJECT_ROOT / "data_designer" / "pipelines" / "benchmarks" / "runners"
    / f"{family}.py"
    for family in src.SINGLE_TURN_FAMILIES
}
LICENSE = {"id": "internal-use", "derive_allowed": True}
_STRATEGY = {
    "reasoning_qa": "language_transfer",
    "knowledge_qa": "paraphrase",
    "code_gen": "docstring_perturbation",
    "health_qa": "scenario_variation",
}


    # ── Shared fixture assembly ────────────────────────────────────────────────────────


def tar_fixture(tmp_path: Path) -> tuple[str, str]:
    archive = tmp_path / "hf-single-turn-snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(FIXTURE_DIR, arcname=FIXTURE_DIR.name)
    return str(archive), file_sha256(archive)


def prepare_family(tmp_path: Path, family: str) -> dict:
    adapter = HfSingleTurnSeedAdapter(FIXTURE_DIR)
    seed_rows = [
        row for row in adapter.load_seed_tasks() if row["family"] == family
    ]
    archive_path, archive_sha = tar_fixture(tmp_path)
    release = derive_release("synthetic-fixture", seed_rows)
    source = import_hf_single_turn_source(
        seed_rows,
        family=family,
        release=release,
        split="development",
        license_record=dict(LICENSE),
        seed_source="synthetic-fixture",
        dataset_id=adapter.dataset_id,
        archive_path=archive_path,
        archive_sha256=archive_sha,
    )
    verify_hf_single_turn_source_manifest(source)
    return {
        "adapter": adapter,
        "seed_rows": seed_rows,
        "archive_path": archive_path,
        "archive_sha256": archive_sha,
        "release": release,
        "source": source,
    }


def _candidate_protocol(case: dict) -> str:
    # D6 per-task paths are deterministically derived from source_task_id (the public item carries this field).
    slug = case["source_task_id"].replace(":", "-")
    return (
        f"\n\nCandidate protocol: write your final answer to "
        f"deliverables/{slug}/"
        f"{'solution.py' if case['hf_single_turn']['benchmark'] == 'humaneval' else 'response.txt'} "
        "under your candidate root."
    )


def derive_rows(prepared: dict, benchmarks: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    for case in prepared["source"]["cases"]:
        metadata = case["hf_single_turn"]
        if metadata["admission"] != "expressible":
            continue
        if metadata["benchmark"] not in benchmarks:
            continue
        bundle = derive_hf_single_turn_spec(
            prepared["source"], case, {"timeout_s": 20.0, "pass_threshold": 0.5}
        )
        rows.append(
            {
                "source_task_id": case["source_task_id"],
                "task_text": case["task_text"] + _candidate_protocol(case),
                "tools_available": case["tools_available"],
                "success_criteria": case["success_criteria"],
                "difficulty": case["difficulty"],
                "item_fingerprint": case["source_task_id"].replace(":", "-") + "-a",
                "verifier_spec": bundle["verifier_spec"],
                "gold_deliverable": bundle["gold_deliverable"],
                "negative_deliverable": bundle["negative_deliverable"],
            }
        )
    screen_hf_single_turn_contamination(rows, prepared["seed_rows"])
    return rows


def write_deliverables(root: Path, rows: list[dict], field: str) -> str:
    # Deliverable keys are per-task relative paths (deliverables/<source_task_id>/…),
    # so a shared root directory does not collide.
    for row in rows:
        for relative, text in row[field].items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    return str(root)


def build_config(tmp_path: Path, prepared: dict, rows: list[dict]) -> tuple[dict, str]:
    family = prepared["source"]["family"]
    archive = tmp_path / "prompts.tar.gz"
    sha = prepared["adapter"].export_prompt_archive(archive)
    config = build_hf_single_turn_augment_config(
        rows,
        family=family,
        release=prepared["source"]["release"],
        split="development",
        strategy=_STRATEGY[family],
        prompt_archive={"path": str(archive), "sha256": sha},
        verifier_path=str(RUNNER_PATHS[family]),
        audit_roots={
            "reviewed": True,
            "reference_root": write_deliverables(
                tmp_path / "audit-reference", rows, "gold_deliverable"
            ),
            "negative_root": write_deliverables(
                tmp_path / "audit-negative", rows, "negative_deliverable"
            ),
        },
    )
    config_path = tmp_path / "augment.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return config, str(config_path)


class _StubJudgeHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server protocol method name
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        content = json.dumps(
            {"passed": True, "score": 5, "reason": "stub judge"}
        )
        body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def stub_judge(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _StubJudgeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(
        "DATA_FORGE_JUDGE_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
    )
    monkeypatch.setenv("DATA_FORGE_JUDGE_API_KEY", "stub-key")
    monkeypatch.setenv("DATA_FORGE_JUDGE_MODEL", "stub-judge")
    monkeypatch.setenv("DATA_FORGE_JUDGE_TEMPERATURE", "0.0")
    yield server
    server.shutdown()


@pytest.fixture(autouse=True)
def _reset_judge_config():
    yield
    llm_judge._JUDGE_CONFIG = None


    # ── seed adapters and release ────────────────────────────────────────────────────


def test_seed_adapter_fixture_rows_and_exclusion():
    adapter = HfSingleTurnSeedAdapter(FIXTURE_DIR)
    rows = adapter.load_seed_tasks()
    assert len(rows) == 24
    by_benchmark = {name: 0 for name in BENCHMARK_FAMILIES}
    for row in rows:
        by_benchmark[row["benchmark"]] += 1
        assert row["family"] == BENCHMARK_FAMILIES[row["benchmark"]]
    assert by_benchmark == {
        "gsm8k": 3, "math500": 2, "mgsm": 2, "drop": 2, "mmlu": 2,
        "mmlu_pro": 2, "gpqa": 2, "triviaqa": 2, "simpleqa": 3,
        "humaneval": 2, "healthbench": 2,
    }
    excluded = [
        row for row in rows if row["admission"] == "excluded"
    ]
    assert len(excluded) == 1
    assert excluded[0]["seed_id"] == "simpleqa:row:row-2"
    assert excluded[0]["exclusion_reason"].startswith("answer_not_recomputable")


def test_release_semantics_fixture_and_upstream():
    adapter = HfSingleTurnSeedAdapter(FIXTURE_DIR)
    rows = adapter.load_seed_tasks()
    release = derive_release("synthetic-fixture", rows)
    assert release.startswith("synthetic-fixture:")
    assert len(release.split(":")[1]) == 8
    upstream = derive_release(
        "upstream-checkout", rows, "OpenAI/gsm8k", "a" * 64
    )
    assert upstream == "hf-openai-gsm8k-" + "a" * 8
    with pytest.raises(Exception, match="dataset_id"):
        derive_release("upstream-checkout", rows, "", "a" * 64)
    with pytest.raises(Exception, match="seed_source"):
        derive_release("other", rows)


def test_row_count_pin_rejects_drift(tmp_path):
    root = tmp_path / "fixture-copy"
    shutil.copytree(FIXTURE_DIR, root)
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    meta["row_counts"]["gsm8k"] = 99
    (root / "dataset.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(HfSingleTurnSeedAdapterError, match="row count"):
        HfSingleTurnSeedAdapter(root).load_seed_tasks()


def test_snapshot_schema_pin_rejects_unknown_schema(tmp_path):
    root = tmp_path / "fixture-copy"
    shutil.copytree(FIXTURE_DIR, root)
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    meta["schema"] = "hf-single-turn-snapshot-v2"
    (root / "dataset.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(HfSingleTurnSeedAdapterError, match="schema"):
        HfSingleTurnSeedAdapter(root)


    # ── source manifest: import / re-check / drift rejection ─────────────────────────────────────


@pytest.mark.parametrize("family", sorted(src.SINGLE_TURN_FAMILIES))
def test_import_and_verify_source_manifest(family, tmp_path):
    prepared = prepare_family(tmp_path, family)
    source = prepared["source"]
    assert source["family"] == family
    assert source["split"] == "development"
    assert source["release"] == prepared["release"]
    assert source["source_snapshot"]["format"] == "hf-single-turn-snapshot-v1"
    benchmarks = {
        case["hf_single_turn"]["benchmark"] for case in source["cases"]
    }
    assert benchmarks == {
        name for name, fam in BENCHMARK_FAMILIES.items() if fam == family
    }
    if family == "knowledge_qa":
        excluded = next(
            case for case in source["cases"]
            if case["hf_single_turn"]["admission"] == "excluded"
        )
        assert excluded["hf_single_turn"]["exclusion_reason"].startswith(
            "answer_not_recomputable"
        )
    # D9: excluded rows retain task_text (the question itself); only gold that cannot be recomputed is rejected.
        assert excluded["task_text"].strip()


@pytest.mark.parametrize("family", sorted(src.SINGLE_TURN_FAMILIES))
def test_verify_source_manifest_rejects_drift(family, tmp_path):
    prepared = prepare_family(tmp_path, family)
    source = prepared["source"]
    drifted = json.loads(json.dumps(source))
    drifted["cases"][0]["task_text"] += " tampered"
    with pytest.raises(Exception, match="drift"):
        verify_hf_single_turn_source_manifest(drifted)
    drifted_release = dict(source)
    drifted_release["release"] = "synthetic-fixture:00000000"
    with pytest.raises(Exception, match="release pin mismatch"):
        verify_hf_single_turn_source_manifest(drifted_release)


def test_snapshot_sha_pin_rejects_tampered_archive(tmp_path):
    prepared = prepare_family(tmp_path, "reasoning_qa")
    archive = tmp_path / "tampered.tar.gz"
    shutil.copy(prepared["archive_path"], archive)
    archive.write_bytes(archive.read_bytes()[:-1] + b"x")
    source = json.loads(json.dumps(prepared["source"]))
    source["source_snapshot"]["path"] = str(archive)
    with pytest.raises(Exception, match="SHA-256"):
        verify_hf_single_turn_source_manifest(source)


    # ── R2 randomized decision materialization ─────────────────────────────────────────


def test_r2_gpqa_option_shuffle_materialized():
    adapter = HfSingleTurnSeedAdapter(FIXTURE_DIR)
    gold = adapter.gold_fields("gpqa:row:row-0")
    expected_order = list(range(4))
    random.Random(42).shuffle(expected_order)
    raw_options = ["Electron", "Proton", "Neutron", "Positron"]
    assert gold["option_order"] == expected_order
    assert gold["options"] == [raw_options[i] for i in expected_order]
    assert gold["answer_letter"] == "ABCD"[expected_order.index(0)]
    # Materialization records only the mode and seed, without leaking answer positions.
    row = next(
        r for r in adapter.load_seed_tasks() if r["seed_id"] == "gpqa:row:row-0"
    )
    assert row["materialization"] == {"mode": "option_shuffle", "seed": 42}


def test_r2_drop_few_shot_selection_frozen():
    adapter = HfSingleTurnSeedAdapter(FIXTURE_DIR)
    rows = adapter.load_seed_tasks()
    pool = [
        json.loads(line)
        for line in (FIXTURE_DIR / "rows" / "drop_fewshot.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["benchmark"] != "drop":
            continue
        index = int(row["seed_id"].rsplit("-", 1)[-1])
        expected = sorted(random.Random(42 + index).sample(range(len(pool)), 3))
        assert row["materialization"]["selected_train_rows"] == expected
    # Exactly 3 examples, frozen into the prompt in selection order.
        parts = row["task_text"].split("\n\n")
        assert len(parts) == 4
        assert "Passage: " + pool[expected[0]]["passage"] in parts[0]
        assert "Passage: " + pool[expected[2]]["passage"] in parts[2]


    # ── D9: explicit rejection of derivation requests for excluded variants ─────────────────────────────────────


def test_derivation_rejects_excluded_case(tmp_path):
    prepared = prepare_family(tmp_path, "knowledge_qa")
    excluded = next(
        case for case in prepared["source"]["cases"]
        if case["hf_single_turn"]["admission"] == "excluded"
    )
    with pytest.raises(Exception, match="excluded from derivation"):
        derive_hf_single_turn_spec(
            prepared["source"], excluded, {"timeout_s": 20.0}
        )


def test_derive_rejects_unknown_directive_keys(tmp_path):
    prepared = prepare_family(tmp_path, "reasoning_qa")
    case = prepared["source"]["cases"][0]
    with pytest.raises(Exception, match="directive"):
        derive_hf_single_turn_spec(
            prepared["source"], case, {"timeout_s": 20.0, "kind": "unit_test"}
        )


    # ── Scoring re-expression: gold passes / negative fails (deterministic benches covered one by one) ──────────────


def _verdict_for(prepared: dict, seed_id: str, *, judge=None):
    case = next(
        case for case in prepared["source"]["cases"]
        if case["source_task_id"] == seed_id
    )
    bundle = derive_hf_single_turn_spec(
        prepared["source"], case, {"timeout_s": 20.0, "pass_threshold": 0.5}
    )
    spec = bundle["verifier_spec"]
    relative = spec["payload"]["artifact"]["path"]
    # unit_test accepts a file dictionary; llm_rubric accepts deliverable text (consistent with native runner
    # semantics—empty-string short-circuit scoring 0 depends on text, not dict serialization).
    def artifact(which):
        content = bundle[f"{which}_deliverable"][relative]
        return {relative: content} if spec["kind"] == "unit_test" else content
    return (
        run_verifier(spec, artifact("gold"), judge=judge),
        run_verifier(spec, artifact("negative"), judge=judge),
    )


@pytest.mark.parametrize("family,seed_id", [
    ("reasoning_qa", "gsm8k:row:row-0"),
    ("reasoning_qa", "math500:row:row-0"),
    ("reasoning_qa", "mgsm:en:row-0"),
    ("reasoning_qa", "mgsm:zh:row-1"),
    ("reasoning_qa", "drop:row:row-0"),
    ("reasoning_qa", "drop:row:row-1"),
    ("knowledge_qa", "mmlu:row:row-0"),
    ("knowledge_qa", "mmlu_pro:row:row-0"),
    ("knowledge_qa", "gpqa:row:row-0"),
    ("knowledge_qa", "triviaqa:row:row-0"),
    ("code_gen", "humaneval:row:row-0"),
    ("code_gen", "humaneval:row:row-1"),
])
def test_unit_test_reexpression_gold_passes_negative_fails(
    family, seed_id, tmp_path
):
    prepared = prepare_family(tmp_path, family)
    gold, negative = _verdict_for(prepared, seed_id)
    assert gold["passed"] is True, gold["detail"]
    assert gold["score"] == 1.0
    assert negative["passed"] is False
    assert negative["score"] == 0.0


def _fake_judge(content: str) -> llm_judge.LlmJudge:
    class FakeClient:
        def chat(self, *args, **kwargs):
            response = type(
                "R", (),
                {"choices": [type(
                    "C", (),
                    {"message": type("M", (), {"content": content})()},
                )]},
            )
            return response()

    return llm_judge.LlmJudge(FakeClient(), model="judge-model")


@pytest.mark.parametrize("family,seed_id", [
    ("knowledge_qa", "simpleqa:row:row-0"),
    ("health_qa", "healthbench:row:row-0"),
])
def test_llm_rubric_reexpression_gold_passes_negative_fails(
    family, seed_id, tmp_path
):
    judge = _fake_judge('{"passed": true, "score": 5, "reason": "ok"}')
    prepared = prepare_family(tmp_path, family)
    gold, negative = _verdict_for(prepared, seed_id, judge=judge)
    assert gold["passed"] is True
    assert gold["score"] == 1.0  # Linear 1–5 scaling to 0–1
    assert negative["passed"] is False
    assert negative["score"] == 0.0  # Empty responses are explicitly scored 0 without calling the judge


    # ── Contamination fingerprints ────────────────────────────────────────────────


def test_contamination_screen_rejects_seed_text(tmp_path):
    prepared = prepare_family(tmp_path, "reasoning_qa")
    seed_text = prepared["seed_rows"][0]["task_text"]
    with pytest.raises(Exception, match="seed text"):
        screen_hf_single_turn_contamination(
            [{"task_text": seed_text}], prepared["seed_rows"]
        )


    # ── Build assertion gate ──────────────────────────────────────────────────────────────


def test_assert_build_rejections(tmp_path):
    # Excluded rows in knowledge_qa (simpleqa empty answer); kind consistency uses health_qa.
    knowledge = prepare_family(tmp_path, "knowledge_qa")
    excluded = next(
        case for case in knowledge["source"]["cases"]
        if case["hf_single_turn"]["admission"] == "excluded"
    )
    prepared = prepare_family(tmp_path, "health_qa")
    source = prepared["source"]
    expressible = next(
        case for case in source["cases"]
        if case["hf_single_turn"]["admission"] == "expressible"
    )
    runtime = {"verifier": {"version": "single-turn-verifier-shim-v1"}}
    spec = {"kind": "llm_rubric"}
    with pytest.raises(Exception, match="excluded from derivation"):
        assert_hf_single_turn_build(runtime, {"verifier_spec": spec}, excluded)
    with pytest.raises(Exception, match="lacks family metadata"):
        assert_hf_single_turn_build(
            runtime, {"verifier_spec": spec}, {"source_task_id": "x"}
        )
    with pytest.raises(Exception, match="offline runner shim"):
        assert_hf_single_turn_build(
            {"verifier": {"version": "other"}}, {"verifier_spec": spec},
            expressible,
        )
    with pytest.raises(Exception, match="must freeze as llm_rubric"):
        assert_hf_single_turn_build(
            runtime, {"verifier_spec": {"kind": "unit_test"}}, expressible
        )


    # ── full-chain build: ≥1 task per family through benchmark_taskset_worker ────────────────────


def _write(path: Path, value) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(path)


@pytest.mark.parametrize("family", sorted(src.SINGLE_TURN_FAMILIES))
def test_full_build_chain_produces_single_turn_taskset(family, tmp_path, stub_judge):
    prepared = prepare_family(tmp_path, family)
    rows = derive_rows(prepared, tuple(
        name for name, fam in BENCHMARK_FAMILIES.items() if fam == family
    ))
    assert rows
    config, config_path = build_config(tmp_path, prepared, rows)
    source_path = _write(
        tmp_path / "source.json",
        json.loads(json.dumps(prepared["source"])),
    )
    request = {
        "schemaVersion": 1,
        "request": {
            "storePath": str(tmp_path / "catalog"),
            "sourceManifestPath": source_path,
            "augmentConfigPath": config_path,
            "sourceSha256": file_sha256(source_path),
            "augmentConfigSha256": file_sha256(config_path),
            "trustedDir": str(tmp_path / "trusted"),
            "refId": f"{family}-fixture",
            "family": family,
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
    assert snapshot["item_count"] == len(rows)
    store = ArtifactCatalog(str(tmp_path / "catalog"))
    try:
        pin = snapshot["taskset"]
        payload = store.get_artifact(
            "taskset", pin["ref_id"], pin["version"]
        )["payload"]
        items = payload["taskset"]
        assert len(items) == len(rows)
        for item in items:
            assert set(item) == PUBLIC_KEYS
            assert item["benchmark"]["family"] == family
            assert item["task_id"].startswith(f"{family}:")
            serialized = json.dumps(item, ensure_ascii=False)
    # Trusted-structure key-name leak scan; the bare word "gold" would false-positive on question text (e.g. gpqa gold element).
            for marker in ('"gold_deliverable"', '"negative_deliverable"',
                           '"verifier_spec"', '"answer_letter"',
                           '"canonical_solution"', '"reference_response"',
                           '"aliases"', '"rubric"', '"option_order"'):
                assert marker not in serialized
        audit = payload["build_report"]["audit"]
        assert len(audit) == len(rows)
        for entry in audit:
            assert entry["reference"]["passed"] is True
            assert entry["reference"]["score"] == 1.0
            assert entry["negative"]["passed"] is False
            assert entry["negative"]["score"] == 0.0
    finally:
        store.close()
