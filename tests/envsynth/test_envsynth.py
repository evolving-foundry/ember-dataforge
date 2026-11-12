"""Environment synthesis tests: C-IR-2 mapping, package, gates, worker, agent loop."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from data_designer.pipelines.envsynth import (
    artifacts,
    curator_contract,
    gates,
    gym_contract,
    interview,
    package,
)
from data_designer.pipelines.envsynth import synthesizer
from data_designer.pipelines.envsynth.cli import env_compile as worker
from data_designer.pipelines.envsynth.domains import user_examples_seed_adapter as adapter
from tests.envsynth.env_fixtures import (
    build_seed,
    parsed_row,
    run_template_build,
    worker_request,
    write_input,
)


TICKETS = [
    ("GD-2026-0001", "线路检修", "已审批"),
    ("GD-2026-0002", "停送电操作", "审核中"),
    ("GD-2026-0003", "缺陷处理", "草稿"),
]


def _statuses(domain):
    return {ticket["ticket_id"]: ticket["status"] for ticket in domain["tickets"]}


def test_c_ir_2_row_envelope_is_validated():
    row = parsed_row(TICKETS)
    assert curator_contract.validate_parsed_rows([row]) == [row]

    bad_version = copy.deepcopy(row)
    bad_version["parsed_doc"]["schema_version"] = "parsed-doc-v1"
    with pytest.raises(curator_contract.CuratorContractError):
        curator_contract.validate_parsed_rows([bad_version])

    bad_index = copy.deepcopy(row)
    bad_index["input_index"] = "0"
    with pytest.raises(curator_contract.CuratorContractError):
        curator_contract.validate_parsed_rows([bad_index])

    duplicate = copy.deepcopy(row)
    with pytest.raises(curator_contract.CuratorContractError):
        curator_contract.validate_parsed_rows([row, duplicate])


def test_table_mapping_and_unknown_status_question_roundtrip():
    domain = adapter.map_parsed_rows([parsed_row(TICKETS)], anchor_holdout_ratio=0.3)
    assert domain["schemaVersion"] == adapter.DOMAIN_SCHEMA_VERSION
    assert _statuses(domain) == {
        "GD-2026-0001": "approved",
        "GD-2026-0002": "reviewing",
        "GD-2026-0003": "draft",
    }
    assert domain["anchorSplit"]["overlap"] is False

    unknown = parsed_row(TICKETS[:2] + [("GD-2026-0004", "待定工单", "神秘状态")])
    bad_domain = adapter.map_parsed_rows([unknown], anchor_holdout_ratio=0.3)
    questions = interview.blocking_questions(bad_domain)
    assert [question["code"] for question in questions] == ["unknown_status"]
    question = questions[0]
    assert question["options"] == list(adapter.CANONICAL_STATUSES)
    assert question["contextRefs"]

    resolved, unresolved, applied = interview.apply_answers(
        bad_domain, {question["id"]: "已审批"}
    )
    assert unresolved == []
    assert applied == [question["id"]]
    assert _statuses(resolved)["GD-2026-0004"] == "approved"


def test_missing_status_is_never_silently_filled():
    domain = adapter.map_parsed_rows(
        [parsed_row(TICKETS[:2] + [("GD-2026-0005", "缺少状态", "")])],
        anchor_holdout_ratio=0.3,
    )
    questions = interview.blocking_questions(domain)
    assert [question["code"] for question in questions] == ["missing_status"]
    assert _statuses(domain)["GD-2026-0005"] is None


def test_normalize_status_is_a_public_export():
    assert "normalize_status" in adapter.__all__
    assert adapter.normalize_status("已审批") == "approved"
    assert adapter.normalize_status("未知状态") is None


def test_template_chain_freezes_a_verifiable_package(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, include_path_ref=True)
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
    assert report["anchorReplay"]["passed"]

    final = package.finalize_bundle(env_package, report)
    release = package.write_release(str(tmp_path / "release"), final)
    manifest, files = package.load_env_package(str(tmp_path / "release" / "package"))
    assert manifest["fingerprint"] == final.content_fingerprint
    assert release["assetSha256"] == final.asset_sha256
    assert manifest["ref_id"] == "office-power"
    assert {
        "env_spec.json",
        "tasks.jsonl",
        "fs/snapshot.jsonl",
        "state/seed.jsonl",
        "rules.yaml",
        "events.yaml",
        "npcs.yaml",
        "grounding/index.jsonl",
        "llm_sim.yaml",
        "audit/npc_cards/applicant.yaml",
        "audit/env_spec.yaml",
        "audit/gym_projection.json",
    } <= set(files)
    projection = json.loads(files["audit/gym_projection.json"])
    assert "fsm_timeouts" in projection["auditOnly"]
    assert projection["projected"]["verifier"] == "gold action -> state_assert"
    blob = b"".join(files.values())
    assert b"source.xlsx" not in blob
    assert str(tmp_path).encode("utf-8") not in blob


def test_worker_template_build_is_idempotent_and_param_versioned(tmp_path):
    asset_root, first, _ = run_template_build(
        tmp_path, TICKETS, snapshot_name="snapshot-1.json"
    )
    input_path = tmp_path / "seed-input.json"
    second_path = tmp_path / "snapshot-2.json"
    assert worker.run_worker([str(input_path), str(second_path)]) == worker.EXIT_OK
    second = json.loads(second_path.read_text(encoding="utf-8"))

    assert second["envRelease"]["version"] == first["envRelease"]["version"]
    assert second["envRelease"]["fingerprint"] == first["envRelease"]["fingerprint"]
    assert second["envRelease"]["assetSha256"] == first["envRelease"]["assetSha256"]
    assert second["taskset"] == first["taskset"]

    release_dirs = sorted((Path(asset_root) / "office-power").glob("v*"))
    assert len(release_dirs) == 1
    release_meta, taskset_payload = artifacts.load_release_metadata(release_dirs[0])
    assert release_meta["schemaVersion"] == artifacts.ENV_RELEASE_SCHEMA
    assert release_meta["taskset"] == first["taskset"]
    assert taskset_payload["schemaVersion"] == artifacts.ENV_TASKSET_SCHEMA
    assert artifacts.fingerprint_of(taskset_payload) == first["taskset"]["fingerprint"]
    assert taskset_payload["gymContract"]["taskSchema"] == gym_contract.GYM_TASK_SCHEMA
    assert taskset_payload["itemCount"] == len(taskset_payload["tasks"])
    for task in taskset_payload["tasks"]:
        for key in (
            "task_id",
            "env_release",
            "goal",
            "success_check",
            "max_steps",
            "seed",
        ):
            assert key in task
        assert task["env_release"]["ref_id"] == "office-power"
        assert task["env_release"]["fingerprint"] == first["envRelease"]["fingerprint"]
        verifier = task["success_check"]["verifier_spec"]
        assert verifier["kind"] == "state_assert"
        assert verifier["payload"]["checks"]
        assert task["max_steps"] == gym_contract.DEFAULT_MAX_STEPS
        assert task["seed"] == 42
    assert str(tmp_path) not in json.dumps(taskset_payload, ensure_ascii=False)
    assert (release_dirs[0] / artifacts.ENV_RELEASE_FILE).is_file()
    assert (release_dirs[0] / artifacts.ENV_TASKS_FILE).is_file()

    asset_dir = Path(first["envRelease"]["assetDir"])
    assert (asset_dir.parent / package.ZIP_NAME).is_file()
    for path in asset_dir.rglob("*"):
        if path.is_file():
            assert str(tmp_path).encode("utf-8") not in path.read_bytes()

    request = json.loads(input_path.read_text(encoding="utf-8"))
    request["request"]["qualityThreshold"] = 0.45
    changed_input = write_input(tmp_path, request, name="seed-input-v2.json")
    changed_snapshot = tmp_path / "snapshot-3.json"
    assert worker.run_worker([changed_input, str(changed_snapshot)]) == worker.EXIT_OK
    third = json.loads(changed_snapshot.read_text(encoding="utf-8"))
    assert third["envRelease"]["version"] == first["envRelease"]["version"] + 1
    assert third["taskset"]["version"] == first["taskset"]["version"] + 1
    assert third["envRelease"]["fingerprint"] != first["envRelease"]["fingerprint"]
    assert third["envRelease"]["assetSha256"] != first["envRelease"]["assetSha256"]
    assert len(sorted((Path(asset_root) / "office-power").glob("v*"))) == 2


def test_worker_curation_failure_then_answers_unblock(tmp_path):
    tickets = TICKETS[:2] + [("GD-2026-0004", "待定工单", "神秘状态")]
    _, seed_dir = build_seed(tmp_path, tickets, seed_name="seed-curation")
    request = worker_request(
        tmp_path,
        seed_dir,
        asset_name="curation-assets",
    )
    first_input = write_input(tmp_path, request, name="curation-1.json")
    first_snapshot = tmp_path / "curation-1.json.snapshot"
    assert worker.run_worker([first_input, str(first_snapshot)]) == worker.EXIT_FAILED
    assert not first_snapshot.exists()

    diagnostics = json.loads(
        (tmp_path / "curation-1.json.diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["stage"] == "curation"
    curation = json.loads(
        (tmp_path / "curation-1.json.curation_questions.json").read_text(
            encoding="utf-8"
        )
    )
    question = [item for item in curation["questions"] if item["code"] == "unknown_status"][0]
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(
        json.dumps(
            {"schemaVersion": 1, "answers": {question["id"]: "approved"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    request["request"]["curationAnswersPath"] = str(answers_path)
    second_input = write_input(tmp_path, request, name="curation-2.json")
    second_snapshot = tmp_path / "curation-2.json.snapshot"
    assert worker.run_worker([second_input, str(second_snapshot)]) == worker.EXIT_OK
    snapshot = json.loads(second_snapshot.read_text(encoding="utf-8"))

    _, files = package.load_env_package(snapshot["envRelease"]["assetDir"])
    env_spec = yaml.safe_load(files["audit/env_spec.yaml"].decode("utf-8"))
    ticket = [
        item
        for item in env_spec["initialState"]["tickets"]
        if item["ticket_id"] == "GD-2026-0004"
    ][0]
    assert ticket["status"] == "approved"


def test_worker_seed_sha_mismatch_is_build_failure(tmp_path):
    _, seed_dir = build_seed(tmp_path, TICKETS, seed_name="seed-sha")
    request = worker_request(tmp_path, seed_dir)
    request["request"]["seedManifestSha256"] = "0" * 64
    input_path = write_input(tmp_path, request, name="sha-input.json")
    snapshot_path = tmp_path / "sha-snapshot.json"
    assert worker.run_worker([input_path, str(snapshot_path)]) == worker.EXIT_FAILED
    assert not snapshot_path.exists()
    diagnostics = json.loads(
        (tmp_path / "sha-snapshot.diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["stage"] == "seed_import"


def test_worker_gate_failure_writes_diagnostics(tmp_path, monkeypatch):
    _, seed_dir = build_seed(tmp_path, TICKETS, seed_name="seed-gate")
    request = worker_request(tmp_path, seed_dir)
    input_path = write_input(tmp_path, request, name="gate-input.json")
    snapshot_path = tmp_path / "gate-snapshot.json"

    def fake_gates(bundle, **kwargs):
        return {
            "schemaVersion": "env-gate-report-v1",
            "passed": False,
            "gates": {
                "anchor_replay": "failed",
                "distribution": "passed",
                "expert_review": "waived",
            },
            "failedChecks": ["anchor_gate:injected"],
            "anchorReplay": {"passed": False, "failedChecks": ["anchor_gate:injected"]},
            "distribution": {"passed": True, "failedChecks": []},
            "expertReview": {"passed": True, "status": "waived"},
        }

    monkeypatch.setattr(worker, "run_gates", fake_gates)
    assert worker.run_worker([input_path, str(snapshot_path)]) == worker.EXIT_FAILED
    assert not snapshot_path.exists()
    diagnostics = json.loads(
        (tmp_path / "gate-snapshot.diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["stage"] == "gates"
    assert diagnostics["gateReport"]["failedChecks"] == ["anchor_gate:injected"]


def test_worker_rejects_relative_or_existing_snapshot(tmp_path):
    _, seed_dir = build_seed(tmp_path, TICKETS, seed_name="seed-path")
    request = worker_request(tmp_path, seed_dir)
    input_path = write_input(tmp_path, request, name="path-input.json")
    assert worker.run_worker([input_path, "relative.json"]) == worker.EXIT_USAGE
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    assert worker.run_worker([input_path, str(existing)]) == worker.EXIT_USAGE


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages):
        return self.replies.pop(0)


def _runner(domain, replies, max_rounds=2):
    draft = synthesizer.template_draft(domain, seed=42)
    return synthesizer.AgentRunner(
        domain=domain,
        draft=copy.deepcopy(draft),
        client=ScriptedClient(replies),
        max_rounds=max_rounds,
    )


def test_agent_loop_enforces_max_rounds(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="seed-agent")
    reply = json.dumps({"action": "read_seed", "query": "ticket"})
    result = _runner(bundle.domain, [reply, reply], max_rounds=2).run()
    assert len(result.rounds) == 2
    assert "max_rounds_reached_without_final" in result.warnings
    assert result.status == "synthesis_failed"


def test_agent_loop_rejects_tools_outside_whitelist(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="seed-agent-tool")
    replies = [
        json.dumps({"action": "rm_rf", "path": "/"}),
        json.dumps({"action": "final", "draft": {}}),
    ]
    result = _runner(bundle.domain, replies, max_rounds=2).run()
    assert "tool_not_whitelisted" in result.failed_checks
    assert result.tool_calls[0]["ok"] is False
    assert result.status == "synthesis_failed"


def test_agent_loop_ask_user_returns_needs_curation(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="seed-agent-ask")
    reply = json.dumps(
        {
            "action": "ask_user",
            "question": "审批时限是多少？",
            "field": "due_at",
            "blocking": True,
        }
    )
    result = _runner(bundle.domain, [reply], max_rounds=1).run()
    assert result.status == "needs_curation"
    assert any(item.get("code") == "agent_question" for item in result.questions)


def test_worker_rejects_more_than_three_rounds(tmp_path):
    _, seed_dir = build_seed(tmp_path, TICKETS, seed_name="seed-round-limit")
    request = worker_request(tmp_path, seed_dir)
    request["request"]["synthesis"]["maxRounds"] = 4
    input_path = write_input(tmp_path, request, name="round-limit-input.json")
    snapshot_path = tmp_path / "round-limit-snapshot.json"
    assert worker.run_worker([input_path, str(snapshot_path)]) == worker.EXIT_USAGE
    assert not snapshot_path.exists()


def test_agent_runner_rejects_more_than_three_rounds(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="seed-agent-round-limit")
    draft = synthesizer.template_draft(bundle.domain, seed=42)
    with pytest.raises(synthesizer.SynthesisError):
        synthesizer.AgentRunner(
            domain=bundle.domain,
            draft=draft,
            client=ScriptedClient([]),
            max_rounds=4,
        )


def test_self_check_runs_a_real_gate(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="seed-self-check")
    replies = [
        json.dumps({"action": "self_check", "gate": "anchor_replay"}),
        json.dumps({"action": "final", "draft": {}}),
    ]
    result = _runner(bundle.domain, replies, max_rounds=2).run()
    check = result.tool_calls[0]["result"]
    assert check["ok"] is True
    assert check["gate"] == "anchor_replay"
    assert check["detail"]["checks"]
    assert result.status == "synthesized"


def test_agent_cannot_replace_anchors_or_verifier_contracts(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="seed-agent-immutable")
    original = synthesizer.template_draft(bundle.domain, seed=42)
    bad_anchor = copy.deepcopy(original)
    bad_anchor["anchors"] = []
    result = _runner(
        bundle.domain,
        [json.dumps({"action": "final", "draft": {"anchors": []}})],
        max_rounds=1,
    ).run()
    assert "invalid_final_draft" in result.failed_checks
    assert result.draft["anchors"] == original["anchors"]

    bad_task = copy.deepcopy(original["tasks"])
    bad_task[0]["verifier_spec"] = {"kind": "dom_assert", "payload": {}}
    result = _runner(
        bundle.domain,
        [json.dumps({"action": "final", "draft": {"tasks": bad_task}})],
        max_rounds=1,
    ).run()
    assert "invalid_final_draft" in result.failed_checks
    assert result.draft["tasks"][0]["verifier_spec"] == original["tasks"][0]["verifier_spec"]
