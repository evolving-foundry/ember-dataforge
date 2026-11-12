"""Hardening regressions for the environment-synthesis contract."""

from __future__ import annotations

import copy
import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import pytest

from data_designer.pipelines.envsynth import (
    artifacts,
    curator_contract,
    gates,
    grounding,
    gym_contract,
    interview,
    package,
    seed_import,
    synthesizer,
)
from data_designer.pipelines.envsynth.cli import env_compile, synthesis_worker
from data_designer.pipelines.envsynth.domain_adapters import PowerTicketDomainAdapter
from data_designer.pipelines.envsynth.profile_compilers import CompileContext
from data_designer.pipelines.envsynth.profile_compilers.office import OfficeProfileCompiler
from data_designer.pipelines.envsynth.redaction import redact_text, sanitize_draft
from data_designer.verifiers import run_verifier
from data_designer.pipelines.envsynth.domains import user_examples_seed_adapter as adapter
from tests.envsynth.env_fixtures import (
    build_seed,
    parsed_row,
    worker_request,
    write_input,
)


TICKETS = [
    ("GD-2026-0001", "线路检修", "已审批"),
    ("GD-2026-0002", "停送电操作", "审核中"),
    ("GD-2026-0003", "缺陷处理", "草稿"),
]


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages):
        return self.replies.pop(0)


class RaisingClient:
    def complete(self, messages):
        raise RuntimeError("upstream unavailable")


def _runner(domain, client, *, max_rounds=2):
    draft = synthesizer.template_draft(domain, seed=42)
    return synthesizer.AgentRunner(
        domain=domain,
        draft=copy.deepcopy(draft),
        client=client,
        max_rounds=max_rounds,
    )


def test_agent_model_exception_fails_closed(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="agent-model-failure")
    result = _runner(bundle.domain, RaisingClient(), max_rounds=1).run()
    assert result.status == "synthesis_failed"
    assert "max_rounds_reached_without_final" in result.failed_checks
    assert "model_call_failed" in result.failed_checks


def test_agent_invalid_json_then_final_still_fails_closed(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="agent-invalid-json")
    client = ScriptedClient(
        ["not json", json.dumps({"action": "final", "draft": {}})]
    )
    result = _runner(bundle.domain, client, max_rounds=2).run()
    assert result.status == "synthesis_failed"
    assert "invalid_model_json" in result.failed_checks


def test_worker_does_not_emit_snapshot_for_failed_agent_synthesis(
    tmp_path, monkeypatch
):
    _, seed_dir = build_seed(tmp_path, TICKETS, seed_name="agent-worker-failure")
    request = worker_request(tmp_path, seed_dir, asset_name="agent-failure-assets")
    request["request"]["synthesis"] = {
        "mode": "agent",
        "maxRounds": 1,
        "model": {
            "baseUrl": "http://127.0.0.1:1",
            "apiKeyEnv": "UNUSED_TEST_KEY",
            "model": "fake-model",
        },
        "grounding": {"embedModel": "char-ngram-v1", "topK": 4},
    }
    input_path = write_input(tmp_path, request, name="agent-failure-input.json")
    snapshot = tmp_path / "agent-failure.json"
    monkeypatch.setattr(
        env_compile, "OpenAICompatibleChatClient", lambda **kwargs: RaisingClient()
    )

    assert env_compile.run_worker([input_path, str(snapshot)]) == env_compile.EXIT_FAILED
    assert not snapshot.exists()
    diagnostics = json.loads(
        (tmp_path / "agent-failure.diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["stage"] == "synthesis"
    assert "model_call_failed" in diagnostics["synthesis"]["failedChecks"]


def test_synthesis_worker_parses_input_exactly_once(tmp_path, monkeypatch):
    _, seed_dir = build_seed(tmp_path, TICKETS, seed_name="parse-once-seed")
    request = worker_request(tmp_path, seed_dir, asset_name="parse-once-assets")
    request["request"]["storePath"] = str(tmp_path / "artifact-store")
    input_path = write_input(tmp_path, request, name="parse-once-input.json")
    snapshot = tmp_path / "parse-once-snapshot.json"

    calls = []
    original = env_compile.parse_input

    def counting_parse(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(env_compile, "parse_input", counting_parse)
    assert synthesis_worker.run_worker([input_path, str(snapshot)]) == 0
    assert calls == [input_path]
    assert snapshot.is_file()


def test_curation_questions_are_isolated_by_context():
    rows = parsed_row(
        [
            ("GD-2026-0101", "线路检修", "神秘状态"),
            ("GD-2026-0102", "缺陷处理", "神秘状态"),
        ]
    )
    domain = adapter.map_parsed_rows([rows], anchor_holdout_ratio=0.3)
    questions = interview.blocking_questions(domain)
    assert len(questions) == 2
    assert len({question["id"] for question in questions}) == 2
    assert len({tuple(question["contextRefs"]) for question in questions}) == 2

    resolved, unresolved, applied = interview.apply_answers(
        domain, {questions[0]["id"]: "approved"}
    )
    statuses = {ticket["ticket_id"]: ticket["status"] for ticket in resolved["tickets"]}
    assert statuses["GD-2026-0101"] == "approved"
    assert statuses["GD-2026-0102"] is None
    assert unresolved == [questions[1]]
    assert applied == [questions[0]["id"]]


class FixedBackend(grounding.EmbeddingBackend):
    def __init__(self, version, vectors):
        self.version = version
        self._vectors = vectors

    def embed(self, texts):
        return [list(self._vectors[index % len(self._vectors)]) for index, _ in enumerate(texts)]


def test_grounding_rejects_inconsistent_backend_dimensions():
    backend = FixedBackend("broken-v1", [[1.0, 0.0], [1.0]])
    with pytest.raises(ValueError, match="dimension"):
        grounding.build_index(
            [{"chunk_id": "a", "text": "alpha"}, {"chunk_id": "b", "text": "beta"}],
            backend=backend,
        )


def test_grounding_rejects_wrong_backend_version_and_query_dimension():
    chunks = [
        {"chunk_id": "a", "text": "线路检修", "role": "seed"},
        {"chunk_id": "b", "text": "审批规则", "role": "regulation"},
    ]
    index = grounding.build_index(chunks)
    other = FixedBackend("other-v1", [[0.0] * index["dim"]])
    with pytest.raises(ValueError, match="embeddingModel"):
        grounding.retrieve(index, "检修", backend=other)

    short = FixedBackend("char-ngram-v1", [[1.0]])
    with pytest.raises(ValueError, match="dimension"):
        grounding.retrieve(index, "检修", backend=short)


def test_grounding_rejects_malformed_chunk_vectors():
    index = grounding.build_index([{"chunk_id": "a", "text": "检修"}])
    malformed = copy.deepcopy(index)
    malformed["chunks"][0]["vector"] = [1.0]
    with pytest.raises(ValueError, match="dimension"):
        grounding.retrieve(malformed, "检修")

    missing_model = copy.deepcopy(index)
    missing_model.pop("embeddingModel")
    with pytest.raises(ValueError, match="embeddingModel"):
        grounding.retrieve(missing_model, "检修")


def _gym_task():
    return {
        "task_id": "env-task-0001",
        "ticket_id": "GD-2026-0002",
        "instruction": "处理工单",
        "initial_state": {
            "ticket": {"ticket_id": "GD-2026-0002", "status": "submitted"}
        },
        "verifier_spec": {
            "kind": "artifact_compare",
            "payload": {"gold": {"status": "approved"}},
            "generator_ref": "envsynth@test#seed=42",
        },
        "gold_solution": {
            "ticket_id": "GD-2026-0002",
            "action": "approve",
            "next_status": "reviewing",
        },
        "negative_solution": {
            "ticket_id": "GD-2026-0002",
            "action": "approve",
            "next_status": "approved",
        },
    }


def _projected_task():
    model = PowerTicketDomainAdapter().extract(
        {"family": "power-ticket", "terms": []},
        {"tasks": [_gym_task()]},
    )
    context = CompileContext(ref_id="office-power", version=1, seed=42)
    return OfficeProfileCompiler().build_tasks(
        model,
        context,
        fingerprint="a" * 16,
    )[0]


def test_gym_projection_contains_only_public_fields():
    row = _projected_task()
    assert set(row) == {
        "task_id",
        "env_release",
        "goal",
        "success_check",
        "max_steps",
        "seed",
    }
    assert "approve_reviewer" in row["goal"]
    assert row["success_check"] == {
        "verifier_spec": {
            "kind": "state_assert",
            "payload": {
                "checks": [
                    {
                        "collection": "tickets",
                        "match": {"ticket_id": "GD-2026-0002"},
                        "expect": {"status": "reviewing"},
                    }
                ]
            },
        }
    }
    text = json.dumps(row, ensure_ascii=False)
    assert "gold" not in text
    assert "negative_solution" not in text
    assert "generator_ref" not in text


def test_gym_validation_rejects_nested_answer_leakage_and_bad_ids():
    row = _projected_task()
    leaked = copy.deepcopy(row)
    leaked["success_check"]["gold"] = {"status": "approved"}
    with pytest.raises(gym_contract.GymContractError, match="leaks"):
        gym_contract.validate_tasks([leaked])

    bad_id = copy.deepcopy(row)
    bad_id["task_id"] = []
    with pytest.raises(gym_contract.GymContractError):
        gym_contract.validate_tasks([bad_id])


def test_generated_gym_prepare_rejects_non_contract_rows(tmp_path):
    import zipfile

    benchmark_dir = tmp_path / "benchmark"
    assets_dir = benchmark_dir / "assets"
    assets_dir.mkdir(parents=True)
    row = _projected_task()
    row["gold"] = {"status": "approved"}
    zip_path = assets_dir / "demo.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("tasks.jsonl", json.dumps(row, ensure_ascii=False) + "\n")
    prepare_path = benchmark_dir / "prepare.py"
    prepare_path.write_bytes(
        gym_contract.prepare_script("demo", zip_name="demo.zip", data_name="demo.jsonl")
    )

    prepared = subprocess.run(
        [sys.executable, str(prepare_path)], capture_output=True, text=True
    )
    assert prepared.returncode != 0
    assert "leaks private answer fields" in prepared.stderr


def test_redaction_removes_nested_source_paths():
    value = {
        "assetRoot": "/tmp/private",
        "source": {
            "doc_id": "pd-123",
            "path_ref": "/tmp/private/seed.docx",
            "relativeRef": "seed.docx",
        },
        "provenance": {
            "source": {
                "doc_id": "pd-123",
                "pathRef": "/tmp/private/seed.docx",
                "sha256": "a" * 64,
            }
        },
        "tasks": [{"task_id": "t", "instruction": "x", "source": {"pathRef": "/tmp/x"}}],
    }
    clean = sanitize_draft(value)
    text = json.dumps(clean, ensure_ascii=False)
    assert "/tmp/private" not in text
    assert "/tmp/x" not in text
    assert clean["source"]["doc_id"] == "pd-123"
    assert clean["provenance"]["source"]["sha256"] == "a" * 64


def test_redact_text_covers_host_spellings_and_preserves_virtual_paths():
    unc = chr(92) * 2 + "server" + chr(92) + "share" + chr(92) + "seed.docx"
    text = (
        "linux=/home/alice/private/seed.docx "
        "windows=C:\\Users\\alice\\seed.docx "
        "unc={} virtual=/report.md allowed=/tmp/work/out.txt".format(unc)
    )
    clean = redact_text(text, allowed_paths=("/report.md", "/tmp/work/out.txt"))
    assert "/home/alice" not in clean
    assert "C:\\Users" not in clean
    assert "server" not in clean
    assert "virtual=/report.md" in clean
    assert "allowed=/tmp/work/out.txt" in clean


def test_package_replay_and_ticket_paths_are_redacted(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="replay-redaction")
    result = synthesizer.synthesize(bundle.domain, mode="template", seed=42)
    domain = copy.deepcopy(bundle.domain)
    domain["tickets"][0]["pathRef"] = "/tmp/private/ticket.json"
    domain["tickets"][0]["title"] = "review /home/alice/private/seed.txt"
    domain["replay"] = {
        "pathRef": "/tmp/private/replay.json",
        "trie": {"source": {"path_ref": "/tmp/private/source.docx"}, "safe": True},
    }
    env_package = package.build_env_package(
        domain,
        result.draft,
        ref_id="office-power",
        version=1,
        seed=42,
        anchor_ratio=0.3,
    )
    replay = env_package.files["audit/replay/trie.json"].decode("utf-8")
    assert "/tmp/private" not in replay
    assert '"safe": true' in replay
    env_spec = env_package.files["env_spec.json"].decode("utf-8")
    assert "/tmp/private" not in env_spec
    assert "/home/alice" not in env_spec
    state_seed = env_package.files["state/seed.jsonl"].decode("utf-8")
    assert "/home/alice" not in state_seed
    audit_env_spec = env_package.files["audit/env_spec.yaml"].decode("utf-8")
    assert "/tmp/private" not in audit_env_spec


def test_seed_import_artifacts_do_not_embed_absolute_paths(tmp_path):
    _, seed_dir = build_seed(
        tmp_path, TICKETS, seed_name="path-free-seed", include_path_ref=True
    )
    marker = str(tmp_path).encode("utf-8")
    for path in seed_dir.rglob("*"):
        if path.is_file():
            assert marker not in path.read_bytes(), path


def test_source_discovery_order_is_path_independent(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "z.docx").write_bytes(b"docx-content")
    (first_root / "a.pdf").write_bytes(b"pdf-content")
    (second_root / "y.pdf").write_bytes(b"pdf-content")
    (second_root / "b.docx").write_bytes(b"docx-content")

    def inventory(root):
        return [
            (
                item["sha256"],
                item["role"],
                item["mediaType"],
                item["byteSize"],
            )
            for item in seed_import.discover_source_files([str(root)])
        ]

    assert inventory(first_root) == inventory(second_root)


def test_parsed_rows_are_normalized_by_input_index():
    first = parsed_row([("GD-2026-0201", "线路检修", "已审批")])
    second = parsed_row([("GD-2026-0202", "缺陷处理", "草稿")])
    first["input_index"] = 1
    second["input_index"] = 0
    rows = curator_contract.validate_parsed_rows([first, second])
    assert [row["input_index"] for row in rows] == [0, 1]


def test_release_metadata_uses_env_release_as_commit_marker(tmp_path, monkeypatch):
    calls = []
    original = artifacts._atomic_write

    def capture(path, data):
        calls.append(Path(path).name)
        return original(path, data)

    monkeypatch.setattr(artifacts, "_atomic_write", capture)
    release_dir = tmp_path / "release"
    artifacts.write_release_metadata(
        release_dir,
        {"schemaVersion": artifacts.ENV_RELEASE_SCHEMA},
        {"schemaVersion": artifacts.ENV_TASKSET_SCHEMA, "tasks": []},
    )
    assert calls == [artifacts.ENV_TASKS_FILE, artifacts.ENV_RELEASE_FILE]

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / artifacts.ENV_TASKS_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(artifacts.ArtifactConflictError):
        artifacts.load_release_metadata(incomplete)


def _frozen_bundle(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="frozen-seed")
    result = synthesizer.synthesize(bundle.domain, mode="template", seed=42)
    assert result.status == "synthesized"
    env_package = package.build_env_package(
        bundle.domain,
        result.draft,
        ref_id="office-power",
        version=1,
        seed=42,
        anchor_ratio=0.3,
        source_files=bundle.files,
        seed_manifest_sha256=bundle.manifest_sha256,
        expert_review={"mode": "waived", "sampleRate": 0.0},
    )
    report = gates.run_gates(env_package, quality_threshold=0.5)
    assert report["passed"], report
    return package.finalize_bundle(env_package, report)


def test_release_rejects_same_version_with_different_bytes(tmp_path):
    frozen = _frozen_bundle(tmp_path)
    release_dir = tmp_path / "release"
    package.write_release(str(release_dir), frozen)
    package.write_release(str(release_dir), frozen)

    tampered = copy.deepcopy(frozen)
    tampered.files["rules.yaml"] += b"# changed\n"
    with pytest.raises(package.EnvPackageError, match="different content"):
        package.write_release(str(release_dir), tampered)


def test_validate_draft_rejects_structural_errors(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="draft-validation")
    base = synthesizer.template_draft(bundle.domain, seed=42)

    def invalid(mutator):
        draft = copy.deepcopy(base)
        mutator(draft)
        with pytest.raises(synthesizer.SynthesisError):
            synthesizer.validate_draft(draft)

    invalid(lambda draft: draft["tasks"][0].__setitem__("instruction", " "))
    invalid(lambda draft: draft["tasks"][0].pop("anchor_id"))
    invalid(
        lambda draft: draft["tasks"].append(copy.deepcopy(draft["tasks"][0]))
    )
    invalid(
        lambda draft: draft["rules"][0]["transitions"][0].__setitem__(
            "from", "unknown-state"
        )
    )
    invalid(lambda draft: draft["npc_cards"].__setitem__("../escape", {}))
    invalid(lambda draft: draft["npc_cards"].__setitem__("unsafe-value", "text"))


def test_package_rejects_unsafe_file_components(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="unsafe-package")
    result = synthesizer.synthesize(bundle.domain, mode="template", seed=42)
    draft = copy.deepcopy(result.draft)
    draft["npc_cards"]["../escape"] = {"display_name": "bad"}

    with pytest.raises(package.EnvPackageError, match="safe file-path component"):
        package.build_env_package(
            bundle.domain,
            draft,
            ref_id="office-power",
            version=1,
            seed=42,
            anchor_ratio=0.3,
        )

    with pytest.raises(package.EnvPackageError, match="ref_id"):
        package.build_env_package(
            bundle.domain,
            result.draft,
            ref_id="../escape",
            version=1,
            seed=42,
            anchor_ratio=0.3,
        )


def test_gym_validation_rejects_invalid_scalar_and_release_types():
    row = _projected_task()
    for field, value in (
        ("max_steps", "40"),
        ("max_steps", True),
        ("seed", "42"),
        ("seed", True),
    ):
        bad = copy.deepcopy(row)
        bad[field] = value
        with pytest.raises(gym_contract.GymContractError):
            gym_contract.validate_tasks([bad])

    bad_release = copy.deepcopy(row)
    bad_release["env_release"]["version"] = "1"
    with pytest.raises(gym_contract.GymContractError):
        gym_contract.validate_tasks([bad_release])

    bad_payload = copy.deepcopy(row)
    bad_payload["success_check"]["verifier_spec"]["payload"] = {"checks": []}
    with pytest.raises(gym_contract.GymContractError):
        gym_contract.validate_tasks([bad_payload])

    bad_generator = copy.deepcopy(row)
    bad_generator["success_check"]["verifier_spec"]["generator_ref"] = " "
    with pytest.raises(gym_contract.GymContractError):
        gym_contract.validate_tasks([bad_generator])

    bad_ref = copy.deepcopy(row)
    bad_ref["env_release"]["ref_id"] = " "
    with pytest.raises(gym_contract.GymContractError):
        gym_contract.validate_tasks([bad_ref])


def test_distribution_check_verifier_is_registered():
    spec = {
        "kind": "distribution_check",
        "payload": {
            "reference": [3, 1],
            "candidate": [30, 10],
            "threshold": 0.99,
        },
        "generator_ref": "envsynth@test#seed=42",
    }
    result = run_verifier(spec, None)
    assert result["passed"] is True
    assert result["score"] == 1.0

    bad = copy.deepcopy(spec)
    bad["payload"]["candidate"] = [1, -1]
    with pytest.raises(ValueError):
        run_verifier(bad, None)


def _allocate_release_version(asset_root):
    with artifacts.release_lock(asset_root, "office-power"):
        version = artifacts.next_release_version(asset_root, "office-power")
        artifacts.release_dir(asset_root, "office-power", version).mkdir(
            parents=True
        )
        time.sleep(0.05)


def test_release_lock_serializes_version_allocation(tmp_path):
    asset_root = str(tmp_path / "assets")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_allocate_release_version, args=(asset_root,))
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    versions = sorted(
        path.name
        for path in (Path(asset_root) / "office-power").iterdir()
        if path.is_dir()
    )
    assert versions == ["v1", "v2", "v3", "v4"]


def test_curator_contract_rejects_semantic_violations(tmp_path):
    row = parsed_row([("GD-2026-0301", "线路检修", "已审批")])
    error_row = {
        "input_index": 0,
        "doc_id": None,
        "parsed_doc": None,
        "error": {
            "code": "UNREADABLE_DOCUMENT",
            "stage": "read",
            "message": "cannot read source",
            "retryable": False,
        },
    }
    assert curator_contract.validate_parsed_rows([error_row]) == [error_row]

    bad_doc_id = copy.deepcopy(row)
    bad_doc_id["parsed_doc"]["doc_id"] = "pd-" + "0" * 16
    bad_doc_id["doc_id"] = bad_doc_id["parsed_doc"]["doc_id"]
    with pytest.raises(curator_contract.CuratorContractError, match="doc_id"):
        curator_contract.validate_parsed_rows([bad_doc_id])

    row_mismatch = copy.deepcopy(row)
    row_mismatch["doc_id"] = "pd-" + "0" * 16
    with pytest.raises(curator_contract.CuratorContractError, match="doc_id"):
        curator_contract.validate_parsed_rows([row_mismatch])

    bad_blocks = copy.deepcopy(row)
    bad_blocks["parsed_doc"]["blocks"][0]["block_id"] = "b-000002"
    with pytest.raises(curator_contract.CuratorContractError, match="block_id"):
        curator_contract.validate_parsed_rows([bad_blocks])

    bad_page_count = copy.deepcopy(row)
    bad_page_count["parsed_doc"]["source"]["page_count"] = 1
    with pytest.raises(curator_contract.CuratorContractError, match="page_count"):
        curator_contract.validate_parsed_rows([bad_page_count])

    sparse = copy.deepcopy(row)
    sparse["input_index"] = 2
    with pytest.raises(curator_contract.CuratorContractError, match="contiguously"):
        curator_contract.validate_parsed_rows([row, sparse])

    with pytest.raises(curator_contract.CuratorContractError, match="output_key"):
        curator_contract.parse_documents(
            [str(tmp_path / "seed.docx")],
            parse_options={"output_key": "document"},
        )