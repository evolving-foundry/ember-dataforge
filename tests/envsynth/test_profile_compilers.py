"""Tests for the DomainAdapter -> EnvironmentModel -> ProfileCompiler stack."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from data_designer.pipelines.envsynth import gates, package, synthesizer
from data_designer.pipelines.envsynth.domain_adapters import (
    DomainAdapterError,
    adapt_environment,
    register_domain_adapter,
    resolve_domain_adapter,
    unregister_domain_adapter,
)
from data_designer.pipelines.envsynth.environment_model import (
    EnvironmentModel,
    TaskDefinition,
)
from data_designer.pipelines.envsynth.profile_compilers import (
    CompileContext,
    ProfileCompileError,
    UnsupportedProfileError,
    get_profile_compiler,
    register_profile_compiler,
    supported_profiles,
    unregister_profile_compiler,
)
from tests.envsynth.env_fixtures import build_seed


TICKETS = [
    ("GD-2026-3001", "线路检修", "已审批"),
    ("GD-2026-3002", "缺陷处理", "审核中"),
]
TERMINAL_COMMAND = "cat README.md > report.md"
TERMINAL_EXPECTED = {"/report.md": "release checklist\n"}


def _office_case(tmp_path):
    bundle, _ = build_seed(tmp_path, TICKETS, seed_name="layers-seed")
    result = synthesizer.synthesize(bundle.domain, mode="template", seed=42)
    assert result.status == "synthesized"
    return bundle, result


def _terminal_domain():
    return {
        "family": "terminal",
        "schemaVersion": "domain-terminal-v1",
        "glossary": ["report"],
        "chunks": [
            {
                "chunk_id": "chunk-terminal-1",
                "text": "Create the report file from README with cat.",
                "role": "seed",
            }
        ],
        "terminal": {
            "workspace": {"/README.md": "release checklist\n"},
            "replay": {
                "entries": [
                    {
                        "cwd": "/",
                        "cmd": TERMINAL_COMMAND,
                        "stdout": "",
                        "fs_writes": dict(TERMINAL_EXPECTED),
                        "done": True,
                    }
                ]
            },
            "danger_patterns": ["rm -rf /"],
            "command_max_length": 2000,
            "dynamics": "replay",
        },
    }


def _terminal_draft():
    verifier = {
        "kind": "artifact_compare",
        "payload": {
            "artifact": {"path": "/report.md", "format": "text"},
            "gold": {"report": TERMINAL_EXPECTED["/report.md"]},
        },
        "generator_ref": "envsynth@test#seed=42",
    }
    task = {
        "task_id": "terminal-task-1",
        "anchor_id": "terminal-anchor-1",
        "instruction": "Create /report.md from /README.md using cat.",
        "initial_state": {"env": {"cwd": "/"}},
        "gold_solution": {"report": TERMINAL_EXPECTED["/report.md"]},
        "negative_solution": {"report": "wrong content\n"},
        "verifier_spec": verifier,
        "expected_files": dict(TERMINAL_EXPECTED),
    }
    anchor = {
        "anchor_id": "terminal-anchor-1",
        "text": "Create the report file from README with cat.",
        "gold_solution": dict(task["gold_solution"]),
        "negative_solution": dict(task["negative_solution"]),
    }
    return {"tasks": [task], "anchors": [anchor], "rules": []}


def _gym_root():
    env_root = os.environ.get("DATA_FORGE_GYM")
    candidates = [Path(env_root).resolve()] if env_root else []
    candidates.append((Path(__file__).resolve().parents[2].parent / "AscendAgent-Gym"))
    for candidate in candidates:
        if (candidate / "overlay/resources_servers/synth_env").is_dir():
            return candidate
    return None


def test_builtin_profile_registry_is_complete():
    assert {"office", "terminal"} <= set(supported_profiles())
    assert get_profile_compiler("office").profile == "office"
    assert get_profile_compiler("terminal").profile == "terminal"


def test_builtin_profiles_own_distinct_agent_prompts():
    office = get_profile_compiler("office").agent_prompt("synth-office")
    terminal = get_profile_compiler("terminal").agent_prompt("synth-terminal")
    assert "office operator" in office["system"].lower()
    assert "terminal operator" in terminal["system"].lower()
    assert "synth-office" in office["system"]
    assert "synth-terminal" in terminal["system"]
    assert office != terminal


def test_unknown_profile_is_rejected_before_compilation(tmp_path):
    bundle, result = _office_case(tmp_path)
    with pytest.raises(package.EnvPackageError, match="not registered"):
        package.build_env_package(
            bundle.domain,
            result.draft,
            ref_id="office-power",
            version=1,
            profile="future-profile",
        )


def test_power_ticket_adapter_extracts_generic_model(tmp_path):
    bundle, result = _office_case(tmp_path)
    model = adapt_environment(bundle.domain, result.draft)
    assert model.family == "power-ticket"
    assert len(model.state_collections) == 1
    collection = model.collection()
    assert collection.name == "tickets"
    assert collection.key == "ticket_id"
    assert collection.label == "ticket"
    assert collection.state_field == "status"
    assert {action.name for action in model.actions} >= {"approve", "reject", "submit"}
    assert len(model.tasks) == len(result.draft["tasks"])


def test_terminal_adapter_requires_explicit_family_or_schema():
    with pytest.raises(DomainAdapterError, match="no domain adapter supports"):
        resolve_domain_adapter({"terminal": {"workspace": {}}}, {})


def test_office_compiler_preserves_gym_projection_contract(tmp_path):
    bundle, result = _office_case(tmp_path)
    compiled = package.build_env_package(
        bundle.domain,
        result.draft,
        ref_id="office-power",
        version=1,
        seed=42,
    )
    env_spec = json.loads(compiled.files["env_spec.json"])
    actions = {item["name"]: item for item in env_spec["actions"]}
    assert actions["approve_reviewer"]["description"] == "Execute ticket action approve (reviewer)"
    assert actions["approve_reviewer"]["parameters"]["required"] == ["ticket_id"]
    state_fields = env_spec["state_schema"]["tickets"]["fields"]
    assert state_fields["ticket_id"]["required"] is True
    assert state_fields["status"]["required"] is True

    task = compiled.gym_tasks[0]
    assert set(task) == {
        "task_id",
        "env_release",
        "goal",
        "success_check",
        "max_steps",
        "seed",
    }
    assert task["success_check"]["verifier_spec"]["kind"] == "state_assert"
    audit_env_spec = yaml.safe_load(compiled.files["audit/env_spec.yaml"])
    assert audit_env_spec["stateSchema"]["ticket"]["status"]["enum"]
    assert audit_env_spec["initialState"]["tickets"]


def test_terminal_compiler_emits_real_gym_terminal_files(tmp_path):
    compiled = package.build_env_package(
        _terminal_domain(),
        _terminal_draft(),
        ref_id="terminal-example",
        version=1,
        profile="terminal",
        seed=42,
    )
    assert "replay/replay.jsonl" in compiled.files
    assert "fs/snapshot.jsonl" in compiled.files
    assert "state/seed.jsonl" not in compiled.files

    env_spec = json.loads(compiled.files["env_spec.json"])
    assert env_spec["action_schema"]["name"] == "bash"
    assert env_spec["danger_patterns"] == ["rm -rf /"]
    snapshot = [
        json.loads(line)
        for line in compiled.files["fs/snapshot.jsonl"].decode("utf-8").splitlines()
        if line
    ]
    assert snapshot == [{"path": "/README.md", "content": "release checklist\n"}]
    replay = [
        json.loads(line)
        for line in compiled.files["replay/replay.jsonl"].decode("utf-8").splitlines()
        if line
    ]
    assert replay[0]["cmd"] == TERMINAL_COMMAND
    assert replay[0]["fs_writes"] == TERMINAL_EXPECTED

    task = compiled.gym_tasks[0]
    assert task["success_check"]["verifier_spec"] == {
        "kind": "artifact_compare",
        "payload": {"expect_files": TERMINAL_EXPECTED},
    }


def test_terminal_goal_redacts_host_paths_but_keeps_virtual_workspace():
    draft = _terminal_draft()
    draft["tasks"][0]["instruction"] = (
        "Read /home/alice/private/seed.txt and /README.md, then write /report.md."
    )
    domain = _terminal_domain()
    domain["terminal"]["replay"]["entries"][0]["stdout"] = (
        "loaded /home/alice/private/seed.txt\n"
    )
    compiled = package.build_env_package(
        domain,
        draft,
        ref_id="terminal-redaction",
        version=1,
        profile="terminal",
    )
    goal = compiled.gym_tasks[0]["goal"]
    assert "/home/alice" not in goal
    assert "<redacted-path>" in goal
    assert "/README.md" in goal
    assert "/report.md" in goal
    assert "/home/alice" not in compiled.files["replay/replay.jsonl"].decode("utf-8")
    audit = compiled.files["audit/tasks.jsonl"].decode("utf-8")
    assert "/home/alice" not in audit


def test_terminal_package_freezes_and_loads_through_gym(tmp_path):
    compiled = package.build_env_package(
        _terminal_domain(),
        _terminal_draft(),
        ref_id="terminal-example",
        version=1,
        profile="terminal",
        seed=42,
    )
    report = gates.run_gates(compiled, quality_threshold=0.5)
    assert report["passed"], report
    frozen = package.finalize_bundle(compiled, report)
    release = package.write_release(str(tmp_path / "release"), frozen)
    manifest, files = package.load_env_package(str(tmp_path / "release" / "package"))
    assert manifest["profile"] == "terminal"
    assert release["assetSha256"] == frozen.asset_sha256
    assert manifest["fingerprint"] == frozen.content_fingerprint

    gym_root = _gym_root()
    if gym_root is not None:
        overlay = str(gym_root / "overlay")
        if overlay not in sys.path:
            sys.path.insert(0, overlay)
        from resources_servers.synth_env.dynamics.replay import ReplayEngine
        from resources_servers.synth_env.package_loader import load_package_zip
        from resources_servers.synth_env.profiles.terminal import TerminalProfile

        loaded = load_package_zip(frozen.zip_bytes)
        task = next(iter(loaded.tasks.values()))
        profile = TerminalProfile(loaded.env_spec, loaded.fs_snapshot)
        fs = profile.snapshot()
        replay = ReplayEngine(loaded.replay)
        replay.execute(fs, "/", TERMINAL_COMMAND)
        assert profile.check_success(fs, task.success_check.verifier_spec) is True
    else:
        assert files["replay/replay.jsonl"]


def test_terminal_compiler_fails_closed_without_replay_or_expected_files():
    domain = _terminal_domain()
    domain["terminal"].pop("replay")
    with pytest.raises(package.EnvPackageError, match="replay"):
        package.build_env_package(
            domain,
            _terminal_draft(),
            ref_id="terminal-no-replay",
            version=1,
            profile="terminal",
        )

    draft = _terminal_draft()
    draft["tasks"][0].pop("expected_files")
    with pytest.raises(package.EnvPackageError, match="expected_files"):
        package.build_env_package(
            _terminal_domain(),
            draft,
            ref_id="terminal-no-expected",
            version=1,
            profile="terminal",
        )


def test_terminal_compiler_rejects_unsafe_replay_paths():
    domain = _terminal_domain()
    domain["terminal"]["replay"]["entries"][0]["fs_writes"] = {
        "relative/report.md": "x"
    }
    with pytest.raises(package.EnvPackageError, match="absolute"):
        package.build_env_package(
            domain,
            _terminal_draft(),
            ref_id="terminal-relative",
            version=1,
            profile="terminal",
        )


class _DemoAdapter:
    family = "demo-family"

    def supports(self, domain, draft):
        del draft
        return domain.get("family") == self.family

    def extract(self, domain, draft, *, grounding_chunks=()):
        del domain, grounding_chunks
        return EnvironmentModel(
            family=self.family,
            tasks=(
                TaskDefinition(task_id="demo-task-1", instruction="do it"),
            ),
        )


class _DemoCompiler:
    profile = "demo-profile"

    def agent_prompt(self, name):
        return {"system": "demo {}".format(name), "user": "act"}

    def build_files(self, model, context):
        del model, context
        return {
            "env_spec.json": b'{"action_schema":{"name":"demo"}}',
            "fs/snapshot.jsonl": b"",
        }

    def build_tasks(self, model, context, *, fingerprint):
        task = model.tasks[0]
        return [
            {
                "task_id": task.task_id,
                "env_release": {
                    "ref_id": context.ref_id,
                    "version": context.version,
                    "fingerprint": fingerprint,
                },
                "goal": task.instruction,
                "success_check": {
                    "verifier_spec": {
                        "kind": "artifact_compare",
                        "payload": {"expect_files": {"/demo.txt": "ok"}},
                    }
                },
                "max_steps": context.max_steps,
                "seed": context.seed,
            }
        ]

    def projection_report(self, model):
        del model
        return {"profile": self.profile, "projected": ["demo"]}


class _EscapingCompiler(_DemoCompiler):
    profile = "demo-escape"

    def build_files(self, model, context):
        files = super().build_files(model, context)
        files["../escape.txt"] = b"escape"
        return files


class _InvalidTaskCompiler(_DemoCompiler):
    profile = "demo-invalid-task"

    def build_tasks(self, model, context, *, fingerprint):
        row = super().build_tasks(model, context, fingerprint=fingerprint)[0]
        row["gold_solution"] = {"status": "approved"}
        return [row]


def _demo_draft():
    return {
        "tasks": [{"task_id": "demo-task-1", "instruction": "do it"}],
        "anchors": [{"anchor_id": "demo-anchor-1"}],
    }


def _build_demo(profile="demo-profile"):
    return package.build_env_package(
        {"family": "demo-family"},
        _demo_draft(),
        ref_id="demo-env",
        version=1,
        profile=profile,
    )


def test_custom_adapter_and_compiler_can_extend_without_editing_builtins():
    register_domain_adapter(_DemoAdapter())
    register_profile_compiler(_DemoCompiler())
    try:
        with pytest.raises(DomainAdapterError, match="already registered"):
            register_domain_adapter(_DemoAdapter())
        with pytest.raises(UnsupportedProfileError, match="already registered"):
            register_profile_compiler(_DemoCompiler())

        compiled = _build_demo()
        assert compiled.model is not None
        assert compiled.compiler.profile == "demo-profile"
        assert compiled.gym_tasks[0]["goal"] == "do it"
        assert json.loads(compiled.files["audit/gym_projection.json"])["profile"] == (
            "demo-profile"
        )
    finally:
        unregister_domain_adapter("demo-family")
        unregister_profile_compiler("demo-profile")
    assert resolve_domain_adapter(
        {"family": "power-ticket"}, {}
    ).family == "power-ticket"
    assert "demo-profile" not in supported_profiles()


def test_custom_compiler_cannot_escape_package_root():
    register_domain_adapter(_DemoAdapter())
    register_profile_compiler(_EscapingCompiler())
    try:
        with pytest.raises(package.EnvPackageError, match="escapes its root"):
            _build_demo("demo-escape")
    finally:
        unregister_domain_adapter("demo-family")
        unregister_profile_compiler("demo-escape")


def test_invalid_compiler_task_rows_are_rejected_before_writing():
    register_domain_adapter(_DemoAdapter())
    register_profile_compiler(_InvalidTaskCompiler())
    try:
        with pytest.raises(package.EnvPackageError, match="Gym task row validation failed"):
            _build_demo("demo-invalid-task")
    finally:
        unregister_domain_adapter("demo-family")
        unregister_profile_compiler("demo-invalid-task")


def test_empty_expected_files_falls_back_to_verifier_payload():
    draft = _terminal_draft()
    draft["tasks"][0]["expected_files"] = {}
    draft["tasks"][0]["verifier_spec"]["payload"]["expect_files"] = dict(
        TERMINAL_EXPECTED
    )
    compiled = package.build_env_package(
        _terminal_domain(),
        draft,
        ref_id="terminal-fallback",
        version=1,
        profile="terminal",
    )
    assert compiled.gym_tasks[0]["success_check"]["verifier_spec"]["payload"] == {
        "expect_files": TERMINAL_EXPECTED
    }


def test_adapter_fails_closed_on_malformed_domain_collections(tmp_path):
    bundle, result = _office_case(tmp_path)
    mutations = [
        ("npc_cards", ["not-an-object"]),
        ("rules", [["not-an-object"]]),
        ("event_model", {"events": ["not-an-object"]}),
    ]
    for field, value in mutations:
        draft = copy.deepcopy(result.draft)
        draft[field] = value
        with pytest.raises(package.EnvPackageError, match="Domain semantic extraction failed"):
            package.build_env_package(
                bundle.domain,
                draft,
                ref_id="office-invalid-" + field,
                version=1,
            )


def test_terminal_replay_rejects_duplicate_and_relative_keys():
    duplicate = _terminal_domain()
    first = duplicate["terminal"]["replay"]["entries"][0]
    second = copy.deepcopy(first)
    second["cmd"] = "  " + first["cmd"].replace(" ", "   ") + "  "
    duplicate["terminal"]["replay"]["entries"].append(second)
    with pytest.raises(package.EnvPackageError, match="duplicates"):
        package.build_env_package(
            duplicate,
            _terminal_draft(),
            ref_id="terminal-duplicate",
            version=1,
            profile="terminal",
        )

    relative = _terminal_domain()
    relative["terminal"]["replay"]["entries"][0]["cwd"] = "work"
    with pytest.raises(package.EnvPackageError, match="absolute"):
        package.build_env_package(
            relative,
            _terminal_draft(),
            ref_id="terminal-relative-cwd",
            version=1,
            profile="terminal",
        )
