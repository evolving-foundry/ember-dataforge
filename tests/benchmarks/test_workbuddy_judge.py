"""wb_composite LLM judge runtime tests (M4, design §8.2).

Covers:
- no backend → ``skipped_no_backend`` (offline CI does not access the network; native results are unchanged);
- judge_ok → weighted_sum fusion; judge_error → pure rule score + llm_missing (explicit);
- write_back_policy: critical rule failures (output_contract, etc.) are never compensated by judge scores;
- four-tuple cache hits (canned responses keyed by rubric_id+artifact sha+model+prompt_version);
- under bounded concurrency, one rubric failure → whole task judge_error (errors recorded explicitly, not silently);
- evidence loading: missing manifest fails explicitly; referenced artifacts that fail extraction are visible as missing evidence;
- runner integration: evaluation phase injects the backend via env → sidecar+judge+fusion;
  the reference phase remains deterministic with audit_exempt even when a backend is configured.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import validate_native_result
from data_designer.pipelines.benchmarks.runners import (
    workbuddy_composite as runner,
)
from data_designer.pipelines.benchmarks.runners import workbuddy_judge as judge

    # ── Unit: backend config and fusion semantics ────────────────────────────────────────────


def test_backend_config_absent_env(monkeypatch):
    for key in ("WB_JUDGE_BASE_URL", "WB_JUDGE_API_KEY", "WB_JUDGE_MODEL"):
        monkeypatch.delenv(key, raising=False)
    assert judge.backend_config() is None


def test_backend_config_env_injection(monkeypatch):
    monkeypatch.setenv("WB_JUDGE_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("WB_JUDGE_API_KEY", "token-xyz")
    monkeypatch.setenv("WB_JUDGE_MODEL", "glm-5.3-flash")
    config = judge.backend_config()
    assert config is not None
    assert config["model"] == "glm-5.3-flash"
    assert config["temperature"] == 0.0
    assert config["thinking_level"] == "low"
    assert config["max_concurrency"] >= 1
    fingerprint = judge.judge_fingerprint(config)
    assert len(fingerprint) == 64
    # fingerprint intentionally excludes key/base_url
    assert "token-xyz" not in fingerprint


def test_merge_scores_rule_only_when_judge_missing():
    merge = {"method": "weighted_sum", "rule_weight": 0.8,
             "llm_weight": 0.2, "pass_threshold": 0.8}
    result = judge.merge_scores(0.9, None, merge, {})
    assert result["score"] == 0.9
    assert result["passed"] is True
    assert result["llm_missing"] is True
    assert result["merge_source"] == "rule_only"


def test_merge_scores_weighted_fusion():
    merge = {"method": "weighted_sum", "rule_weight": 0.8,
             "llm_weight": 0.2, "pass_threshold": 0.8}
    result = judge.merge_scores(
        0.9, {"status": "judge_ok", "llm_score": 0.5}, merge, {}
    )
    assert result["score"] == pytest.approx(0.82)
    assert result["passed"] is True
    assert result["llm_missing"] is False


def test_merge_scores_judge_error_degrades_to_rule():
    merge = {"method": "weighted_sum", "rule_weight": 0.8,
             "llm_weight": 0.2, "pass_threshold": 0.8}
    result = judge.merge_scores(
        0.9, {"status": "judge_error", "llm_score": None}, merge, {}
    )
    assert result["score"] == 0.9
    assert result["llm_missing"] is True
    assert result["merge_source"] == "rule_only"


def test_write_back_policy_judge_never_compensates_critical_failure():
    merge = {"method": "weighted_sum", "rule_weight": 0.8,
             "llm_weight": 0.2, "pass_threshold": 0.8}
    diagnostics = {
        "failed_checks": [
            {"name": "report_valid", "group": "output_contract", "detail": "x"}
        ]
    }
    result = judge.merge_scores(
        0.9, {"status": "judge_ok", "llm_score": 1.0}, merge, diagnostics
    )
    # fused score 0.92 >= 0.8, but critical rule failures cannot be compensated by judge scores
    assert result["score"] == pytest.approx(0.92)
    assert result["passed"] is False
    assert result["critical_failure"] == "output_contract/report_valid"


def test_merge_scores_rejects_unknown_method():
    with pytest.raises(ValueError, match="unsupported score_merge"):
        judge.merge_scores(1.0, None, {"method": "max"}, {})


def test_critical_rule_failure_keyword_surface():
    assert judge.critical_rule_failure(
        {"failed_checks": [{"name": "citation_path_ok", "group": "content"}]}
    )
    assert judge.critical_rule_failure({"failed_checks": []}) is None
    assert judge.critical_rule_failure(
        {"failed_checks": [{"name": "value_matches_gold", "group": "content"}]}
    ) is None

    # ── Unit: cache keys and canned responses ──────────────────────────────────────────────────


def _rubric(rid: str, refs: list) -> dict:
    return {"id": rid, "question": "ok?", "artifact_refs": refs}


def test_cache_key_is_four_tuple_stable():
    config = {"model": "glm-5.3-flash"}
    key_a = judge.rubric_cache_key(
        _rubric("r1", ["a"]), {"a": "sha-1"}, config
    )
    key_b = judge.rubric_cache_key(
        _rubric("r1", ["a"]), {"a": "sha-1"}, config
    )
    assert key_a == key_b
    assert key_a != judge.rubric_cache_key(
        _rubric("r1", ["a"]), {"a": "sha-2"}, config
    )
    assert key_a != judge.rubric_cache_key(
        _rubric("r2", ["a"]), {"a": "sha-1"}, config
    )


def test_extract_response_openai_shape():
    from types import SimpleNamespace

    resp = SimpleNamespace(
        model="glm-5.3-flash",
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"passed": true, "reason": "ok"}'))],
        usage=SimpleNamespace(prompt_tokens=57, completion_tokens=30,
                              total_tokens=87),
    )
    content, tokens = judge._extract_response(resp)
    assert json.loads(content)["passed"] is True
    assert tokens == {"prompt_tokens": 57, "completion_tokens": 30,
                      "total_tokens": 87}


def test_extract_response_anthropic_shape_volcano_apigw():
    from types import SimpleNamespace

    # Volcengine apigw /compatible observed shape: request uses OpenAI form, response uses anthropic message form
    resp = SimpleNamespace(
        id="msg_x", type="message", role="assistant",
        model="glm-5.3-flash",
        content=[
            {"type": "thinking", "thinking": "推理……"},
            {"type": "text", "text": '{"passed": false,'},
            {"type": "text", "text": ' "reason": "缺少标题行"}'},
        ],
        usage={"input_tokens": 57, "output_tokens": 139,
               "cache_read_input_tokens": 0},
        stop_reason="end_turn",
    )
    content, tokens = judge._extract_response(resp)
    assert json.loads(content) == {"passed": False, "reason": "缺少标题行"}
    assert tokens == {"prompt_tokens": 57, "completion_tokens": 139}


def test_extract_response_unrecognized_rejected():
    import pytest
    from types import SimpleNamespace

    with pytest.raises(judge.JudgeCallError):
        judge._extract_response(SimpleNamespace(model="m", choices=[],
                                                content="plain string"))


def test_judge_rubric_binary_retries_transient(monkeypatch):
    from types import SimpleNamespace

    calls = []

    class _FakeClient:
        def __init__(self, _config):
            pass

        def chat(self, **_kwargs):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError(
                    "LLM API request failed: Remote end closed connection "
                    "without response")
            return SimpleNamespace(
                model="glm-5.3-flash",
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content='{"passed": true, "reason": "ok"}'))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                      total_tokens=15),
            )

    monkeypatch.setattr(judge, "WB_JUDGE_MAX_CONCURRENCY", None,
                        raising=False)
    monkeypatch.setattr(
        "curator.utils.llm_api_config.OpenAICompatibleChatClient",
        _FakeClient,
    )
    monkeypatch.setattr(judge.time, "sleep", lambda _s: None)
    result = judge.judge_rubric_binary(
        {"id": "r1", "question": "q"}, {"a": "evidence"},
        {"model": "glm-5.3-flash", "base_url": "https://gw.example",
         "api_key": "k"},
    )
    assert result["binary"] == 1.0
    assert len(calls) == 3


def test_judge_rubric_binary_no_retry_on_auth_error(monkeypatch):
    calls = []

    class _FakeClient:
        def __init__(self, _config):
            pass

        def chat(self, **_kwargs):
            calls.append(1)
            raise RuntimeError("LLM API HTTP 401: bad key")

    monkeypatch.setattr(
        "curator.utils.llm_api_config.OpenAICompatibleChatClient",
        _FakeClient,
    )
    monkeypatch.setattr(judge.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="401"):
        judge.judge_rubric_binary(
            {"id": "r1", "question": "q"}, {"a": "evidence"},
            {"model": "glm-5.3-flash", "base_url": "https://gw.example",
             "api_key": "k"},
        )
    assert len(calls) == 1


def test_judge_task_rubrics_canned_cache_hit(tmp_path, monkeypatch):
    config = {"model": "glm-5.3-flash", "max_concurrency": 2}
    rubrics = [_rubric("r1", ["a"]), _rubric("r2", ["a"])]
    shas = {"a": "sha-1"}

    def _fail(_rubric, _evidence, _config, **_kwargs):
        raise AssertionError("judge must not be called on a cache hit")

    monkeypatch.setattr(judge, "judge_rubric_binary", _fail)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Preloaded canned judge responses (keyed by artifact sha)
    for rid in ("r1", "r2"):
        key = judge.rubric_cache_key(_rubric(rid, ["a"]), shas, config)
        (cache_dir / f"{key}.json").write_text(
            json.dumps({"binary": 1.0, "reason": "canned",
                        "model": "glm-5.3-flash",
                        "prompt_version": judge.JUDGE_PROMPT_VERSION}),
            encoding="utf-8",
        )
    result = judge.judge_task_rubrics(
        rubrics, {"a": "evidence"}, shas, config, cache_dir=str(cache_dir)
    )
    assert result["status"] == "judge_ok"
    assert result["llm_score"] == 1.0
    assert result["cached_count"] == 2
    assert result["call_count"] == 0
    assert all(item["reason"] == "canned"
               for item in result["rubric_results"])


def test_judge_task_rubrics_bounded_call_and_error(tmp_path, monkeypatch):
    config = {"model": "glm-5.3-flash", "max_concurrency": 2}
    rubrics = [_rubric("r1", ["a"]), _rubric("r2", ["a"]),
               _rubric("r3", ["a"])]
    calls = []

    def _fake(rubric, evidence, cfg, **kwargs):
        calls.append(rubric["id"])
        if rubric["id"] == "r2":
            raise judge.JudgeCallError("unparseable judge output")
        return {"binary": 1.0, "reason": "ok", "latency_seconds": 0.01,
                "tokens": {"prompt_tokens": 10, "completion_tokens": 1}}

    monkeypatch.setattr(judge, "judge_rubric_binary", _fake)
    result = judge.judge_task_rubrics(
        rubrics, {"a": "evidence"}, {"a": "sha-1"}, config,
        cache_dir=str(tmp_path / "cache"),
    )
    assert result["status"] == "judge_error"
    assert result["llm_score"] is None
    assert len(result["errors"]) == 1
    assert "r2" in result["errors"][0]
    # Successful rubric results are still recorded explicitly
    assert [item["rubric_id"] for item in result["rubric_results"]] == [
        "r1", "r3"
    ]


def test_judge_task_rubrics_empty_rubrics_rejected():
    with pytest.raises(judge.JudgeCallError, match="no rubrics"):
        judge.judge_task_rubrics([], {}, {}, {"model": "m"})

    # ── Unit: evidence loading ──────────────────────────────────────────────────────────


def _write_logs(logs: Path, records: list, texts: dict) -> None:
    (logs / "artifact_text").mkdir(parents=True, exist_ok=True)
    for name, content in texts.items():
        (logs / "artifact_text" / name).write_text(content, encoding="utf-8")
    (logs / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": records}), encoding="utf-8"
    )


def test_load_judge_evidence_reads_manifest(tmp_path):
    logs = tmp_path / "verifier"
    _write_logs(
        logs,
        [{"id": "a", "text_path": "artifact_text/a.md", "sha256": "sha-1",
          "extract_status": "ok"}],
        {"a.md": "evidence body"},
    )
    evidence, shas = judge.load_judge_evidence(
        str(logs), [{"id": "b", "path": "output/x.json", "type": "json"}]
    )
    assert evidence["a"] == "evidence body"
    assert shas["a"] == "sha-1"
    # Declared artifacts not extracted by the sidecar are marked explicitly, not silently
    assert "evidence missing" in evidence["b"]


def test_load_judge_evidence_failed_extract_visible(tmp_path):
    logs = tmp_path / "verifier"
    _write_logs(
        logs,
        [{"id": "a", "extract_status": "error", "error": "artifact not found"}],
        {},
    )
    evidence, _shas = judge.load_judge_evidence(str(logs), [])
    assert "artifact not found" in evidence["a"]


def test_load_judge_evidence_missing_manifest(tmp_path):
    with pytest.raises(judge.JudgeCallError, match="manifest missing"):
        judge.load_judge_evidence(str(tmp_path), [])

    # ── Integration: runner evaluation phase wiring ────────────────────────────────────────

TASK_ID = "t01"
GOLD = {"expected_value": 42}
ARTIFACTS = [
    {"id": "report", "path": "output/report.json", "type": "json",
     "required": True}
]
RUBRICS = [{"id": "r1", "question": "complete?",
            "artifact_refs": ["report"]}]
SCORE_MERGE = {"method": "weighted_sum", "rule_weight": 0.8,
               "llm_weight": 0.2, "pass_threshold": 0.8}

EVAL_CORE = '''import json
from pathlib import Path


def evaluate(output_dir, gold_path=None):
    p = Path(output_dir) / "report.json"
    value = json.loads(p.read_text()).get("value") if p.is_file() else None
    checks = [
        {"name": "report_exists", "passed": p.is_file(),
         "group": "output_contract"},
        {"name": "value_matches_gold",
         "passed": value == 42, "group": "content"},
    ]
    return {"checks": checks, "total_count": 2,
            "passed_count": sum(1 for c in checks if c["passed"] is True)}
'''

TEST_VERIFY = '''import json, os, sys
from pathlib import Path
import pytest

GRADING_DIR = Path(__file__).resolve().parent
GOLD_PATH = Path(os.environ.get("WB_BENCH_GOLD_PATH", "/tests/gold/gold_answer.json"))
OUTPUT_PATH = os.environ.get("WB_BENCH_OUTPUT_PATH", "/workspace/output")
VERIFIER_LOGS = Path(os.environ.get("WB_BENCH_VERIFIER_LOGS", "/logs/verifier"))
sys.path.insert(0, str(GRADING_DIR))


def _load_result():
    try:
        from eval_core import evaluate
        return evaluate(OUTPUT_PATH, str(GOLD_PATH))
    except Exception as exc:
        return {"checks": [{"name": "eval_failed", "passed": False,
                            "detail": repr(exc)}]}


_RESULT = _load_result()
VERIFIER_LOGS.mkdir(parents=True, exist_ok=True)
(VERIFIER_LOGS / "eval_result.json").write_text(
    json.dumps(_RESULT), encoding="utf-8")
_CHECKS = _RESULT.get("checks") or [{"name": "no_checks", "passed": False}]
_IDS = [f"{i:03d}_{c.get('name', 'x')}" for i, c in enumerate(_CHECKS)]


@pytest.mark.parametrize("idx", range(len(_CHECKS)), ids=_IDS)
def test_atomic_check(idx):
    check = _CHECKS[idx]
    assert check.get("passed") is True, check.get("detail", "")
'''


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _build_case(tmp: Path):
    task_files = {
        "instruction.md": b"Produce output/report.json with the expected value.",
        "task.toml": b"[task]\nname = 't01'\n",
        "tests/judge.yaml": (
            "version: 1\nartifacts:\n- id: report\n  path: output/report.json\n"
            "  type: json\n  required: true\nllm_judge:\n"
            "  method: rubric_binary_mean\n"
            "  rubrics:\n  - id: r1\n    question: complete?\nscore_merge:\n"
            "  method: weighted_sum\n  rule_weight: 0.8\n  llm_weight: 0.2\n"
            "  pass_threshold: 0.8\n"
        ).encode(),
        "tests/gold/gold_answer.json": json.dumps(GOLD).encode(),
        "tests/grading/eval_core.py": EVAL_CORE.encode(),
        "tests/grading/test_verify.py": TEST_VERIFY.encode(),
        "tests/verifier.toml": b'[run]\ncwd = "/workspace"\n',
    }
    archive = tmp / "seed-archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in task_files.items():
            _add(tar, f"wb-m4-fixture/tasks/{TASK_ID}/{name}", data)
    manifest = {
        "schema_version": "benchmark-case-v1",
        "task_id": TASK_ID,
        "family": "workbuddy",
        "private": {"verifier_spec": {
            "kind": "wb_composite",
            "generator_ref": f"wb_bench_native_ingest@{TASK_ID}",
            "payload": {
                "artifacts": ARTIFACTS,
                "rubrics": RUBRICS,
                "llm_judge_method": "rubric_binary_mean",
                "score_merge": SCORE_MERGE,
                "grading_dir_sha256": "unused",
                "grading_source": {
                    "path": str(archive), "sha256": "unused",
                    "member_prefix": f"tasks/{TASK_ID}/tests",
                },
                "env_pin": {"path": str(archive), "sha256": "x"},
                "gold": GOLD,
                "timeout_sec": 600,
            },
        }},
    }
    manifest_path = tmp / "case_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return archive, manifest_path


def _candidate(tmp: Path, value: int) -> Path:
    root = tmp / "candidate"
    (root / "output").mkdir(parents=True, exist_ok=True)
    (root / "output" / "report.json").write_text(json.dumps({"value": value}))
    return root


@pytest.fixture
def runner_env(monkeypatch, tmp_path):
    """Skip the real sandbox and sidecar, inject a fake judge, and isolate the cache directory."""
    monkeypatch.setenv("WB_JUDGE_CACHE_DIR", str(tmp_path / "cache"))
    calls = {"sidecar": 0, "judge": 0, "load": 0}

    monkeypatch.setattr(runner, "_build_sandbox",
                        lambda: tmp_path / "fake-root")
    # The runner re-checks that the unpacked gold equals the frozen gold: place a consistent copy in the fake root.
    fake_root = tmp_path / "fake-root"
    gold_dir = fake_root / "tests" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    (gold_dir / "gold_answer.json").write_text(
        json.dumps(GOLD), encoding="utf-8")
    (fake_root / "tests" / "grading").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runner, "_teardown_sandbox", lambda root: None)
    monkeypatch.setattr(runner, "_extract_workspace", lambda pin, dest: None)
    monkeypatch.setattr(runner, "_extract_grading", lambda pin, dest: None)
    monkeypatch.setattr(runner, "_dir_content_sha256", lambda path: "unused")

    def _fake_pytest(root, timeout):
        logs = Path(root) / "logs" / "verifier"
        logs.mkdir(parents=True, exist_ok=True)
        checks = [
            {"name": "report_exists", "passed": True,
             "group": "output_contract"},
            {"name": "value_matches_gold", "passed": True,
             "group": "content"},
        ]
        (logs / "eval_result.json").write_text(
            json.dumps({"checks": checks, "total_count": 2,
                        "passed_count": 2}), encoding="utf-8")
        return 0, False

    monkeypatch.setattr(runner, "_run_pytest", _fake_pytest)

    def _fake_sidecar(root, timeout):
        calls["sidecar"] += 1

    monkeypatch.setattr(runner, "_judge_sidecar_invocation", _fake_sidecar)

    def _fake_load(logs_dir, artifacts):
        calls["load"] += 1
        return {"report": "evidence"}, {"report": "sha-1"}

    monkeypatch.setattr(judge, "load_judge_evidence", _fake_load)
    return calls, monkeypatch


def _request(manifest: Path, candidate: Path, phase: str) -> dict:
    return {
        "schemaVersion": 1,
        "taskId": TASK_ID,
        "caseManifestPath": str(manifest),
        "candidateRoot": str(candidate),
        "phase": phase,
        "publicItem": {"task_id": TASK_ID},
    }


def test_evaluation_no_backend_skipped(tmp_path, runner_env):
    calls, monkeypatch = runner_env
    for key in ("WB_JUDGE_BASE_URL", "WB_JUDGE_API_KEY", "WB_JUDGE_MODEL"):
        monkeypatch.delenv(key, raising=False)
    archive, manifest = _build_case(tmp_path)
    result = runner.score_request(
        _request(manifest, _candidate(tmp_path, 42), "evaluation")
    )
    validate_native_result(result, TASK_ID)
    assert result["llm_judge_status"] == "skipped_no_backend"
    assert result["score"] == 1.0
    diag = result["rule_diagnostics"]["judge"]
    assert diag["judge_status"] is None
    assert diag["llm_missing"] is True
    assert diag["merge_source"] == "rule_only"
    assert calls["sidecar"] == 0 and calls["judge"] == 0


def test_evaluation_judge_ok_fusion(tmp_path, runner_env):
    calls, monkeypatch = runner_env
    monkeypatch.setenv("WB_JUDGE_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("WB_JUDGE_API_KEY", "token")
    monkeypatch.setenv("WB_JUDGE_MODEL", "glm-5.3-flash")

    def _fake_judge(rubrics, evidence, shas, cfg, **kwargs):
        calls["judge"] += 1
        assert cfg["model"] == "glm-5.3-flash"
        assert evidence == {"report": "evidence"}
        return {
            "status": "judge_ok", "llm_score": 0.5,
            "fingerprint": "f" * 64, "rubric_results": [
                {"rubric_id": "r1", "binary": 0.0, "cached": False,
                 "reason": "incomplete", "latency_seconds": 0.1,
                 "tokens": {}}
            ],
            "call_count": 1, "cached_count": 0,
            "total_latency_seconds": 0.1,
            "prompt_tokens": 10, "completion_tokens": 2, "errors": [],
        }

    monkeypatch.setattr(judge, "judge_task_rubrics", _fake_judge)
    archive, manifest = _build_case(tmp_path)
    result = runner.score_request(
        _request(manifest, _candidate(tmp_path, 42), "evaluation")
    )
    validate_native_result(result, TASK_ID)
    assert result["llm_judge_status"] == "judge_ok"
    assert result["score"] == pytest.approx(0.9)  # 0.8*1.0 + 0.2*0.5
    assert result["passed"] is True
    diag = result["rule_diagnostics"]["judge"]
    assert diag["merge_source"] == "weighted_sum"
    assert diag["llm_missing"] is False
    assert diag["judge_call_count"] == 1
    assert diag["judge_prompt_tokens"] == 10
    assert calls["sidecar"] == 1 and calls["load"] == 1 and calls["judge"] == 1


def test_evaluation_judge_error_degrades(tmp_path, runner_env):
    calls, monkeypatch = runner_env
    monkeypatch.setenv("WB_JUDGE_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("WB_JUDGE_API_KEY", "token")
    monkeypatch.setenv("WB_JUDGE_MODEL", "glm-5.3-flash")

    def _fake_judge(rubrics, evidence, shas, cfg, **kwargs):
        return {"status": "judge_error", "llm_score": None,
                "errors": ["r1: gateway unreachable"],
                "fingerprint": "f" * 64, "rubric_results": [],
                "call_count": 0, "cached_count": 0,
                "total_latency_seconds": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0}

    monkeypatch.setattr(judge, "judge_task_rubrics", _fake_judge)
    archive, manifest = _build_case(tmp_path)
    result = runner.score_request(
        _request(manifest, _candidate(tmp_path, 42), "evaluation")
    )
    validate_native_result(result, TASK_ID)
    assert result["llm_judge_status"] == "judge_error"
    # Pure rule score + llm_missing flag (explicit degradation, not silent)
    assert result["score"] == 1.0
    diag = result["rule_diagnostics"]["judge"]
    assert diag["llm_missing"] is True
    assert diag["merge_source"] == "rule_only"
    assert "gateway unreachable" in diag["judge_errors"][0]


def test_evaluation_manifest_missing_becomes_judge_error(tmp_path, runner_env):
    calls, monkeypatch = runner_env
    monkeypatch.setenv("WB_JUDGE_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("WB_JUDGE_API_KEY", "token")
    monkeypatch.setenv("WB_JUDGE_MODEL", "glm-5.3-flash")

    def _fail_load(logs_dir, artifacts):
        raise judge.JudgeCallError("judge evidence manifest missing: x")

    monkeypatch.setattr(judge, "load_judge_evidence", _fail_load)
    archive, manifest = _build_case(tmp_path)
    result = runner.score_request(
        _request(manifest, _candidate(tmp_path, 42), "evaluation")
    )
    validate_native_result(result, TASK_ID)
    assert result["llm_judge_status"] == "judge_error"
    assert result["score"] == 1.0
    assert "manifest missing" in (
        result["rule_diagnostics"]["judge"]["judge_errors"][0]
    )


def test_reference_phase_stays_audit_exempt_with_backend(tmp_path, runner_env):
    calls, monkeypatch = runner_env
    monkeypatch.setenv("WB_JUDGE_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("WB_JUDGE_API_KEY", "token")
    monkeypatch.setenv("WB_JUDGE_MODEL", "glm-5.3-flash")
    archive, manifest = _build_case(tmp_path)
    result = runner.score_request(
        _request(manifest, _candidate(tmp_path, 42), "reference")
    )
    validate_native_result(result, TASK_ID)
    assert result["llm_judge_status"] == "audit_exempt"
    assert result["score"] == 1.0
    assert calls["sidecar"] == 0 and calls["judge"] == 0
    assert result["rule_diagnostics"]["judge"] == {}
