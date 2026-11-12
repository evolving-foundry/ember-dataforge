"""Contract and hard-requirement tests for the two trusted_worker modes and the evaluator metric-surface plugin.

Covers the parts of the evaluator onboarding design doc (v0.1) acceptance criteria §8 that can be verified in-repo:
  * op mode protocol (§4.3): schemaVersion / unknown keys / exit codes 0/1/2 / empty stdout requirement
  * metric mapping (§4.4): the metric key names of both ops are the contract; non-finite scores must not become 0 (§4.5)
  * Q2 CandidateTextOverlapScore (§5): bleu/cider/ngram + perItem details
  * evaluation mode (RSI §3.9.2 v0.2.1): score == metrics[objective.metric], and failures do not write result files
  * op mode registry is a superset of the evaluation registry (§4.6)
  * TS plugin: validate rejects unknown keys, empty input short-circuits without spawning, and end-to-end compute (§8-1/5)
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

# Stub torch BEFORE importing any Curator module (same pattern as test_rsi_contract.py).
if "torch" not in sys.modules:  # pragma: no cover
    sys.modules["torch"] = types.ModuleType("torch")

try:
    import ray
    HAS_RAY = True
except ImportError:  # pragma: no cover
    ray = None  # type: ignore[assignment]
    HAS_RAY = False

if not HAS_RAY:  # pragma: no cover
    pytest.skip("Ray is not installed", allow_module_level=True)

from curator.trusted_worker import OP_REGISTRY, run as evaluation_run
from curator.trusted_worker.op import main as op_mode_main
from curator.trusted_worker.op.protocol import handle as op_mode_handle
from curator.trusted_worker.op.registry import OP_MODE_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_PATH = REPO_ROOT / "integrations" / "curator" / "evaluator" / "curator-quality.mjs"
NODE_TIMEOUT_S = int(os.environ.get("CURATOR_PLUGIN_TEST_TIMEOUT", "300"))


def _ray_env() -> dict[str, str]:
    env = dict(os.environ)
    if ray.is_initialized():
        env["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
    return env


def _write_jsonl(path: Path, rows: list) -> str:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return str(path)


def _run_op_mode(request: dict, tmp_path: Path) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = op_mode_handle(json.dumps(request), stdout, stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def _op_request(tmp_path: Path, rows: list, op: str = "CandidateLogDeduplicate",
                params: dict | None = None, **overrides) -> dict:
    request = {
        "schemaVersion": 1,
        "op": op,
        "params": params or {},
        "dataPath": _write_jsonl(tmp_path / "data.jsonl", rows),
        "outputDir": str(tmp_path / "out"),
    }
    request.update(overrides)
    return request


def _evaluation_envelope(tmp_path: Path, op: str, data_rows: list,
                         objective_metric: str = "quality_score") -> dict:
    data_key = "trajectories_path" if op == "CandidateBatchQualityAssess" else "raw_logs_path"
    return {
        "schemaVersion": 2,
        "request": {
            "evaluationId": "eval-0001",
            "candidate": "cand-1",
            "plan": {
                "id": "plan-1",
                "objective": {"metric": objective_metric, "direction": "maximize"},
                "config": {
                    "op": op,
                    data_key: _write_jsonl(tmp_path / "data.jsonl", data_rows),
                },
            },
        },
        "target": {"profile": {"id": "p", "harness": "dsh", "version": "0.1", "config": {}}, "content": None},
        "output": {"id": "scope-1", "rootDir": str(tmp_path / "out")},
    }


# ---------------------------------------------------------------------------
# Registry (§4.6: the op mode registry is a superset of the evaluation registry)
# ---------------------------------------------------------------------------


class TestRegistries:
    def test_op_mode_registry_is_superset_of_evaluation_registry(self):
        assert set(OP_REGISTRY).issubset(set(OP_MODE_REGISTRY))

    def test_q1_and_q2_ops_registered(self):
        assert {"CandidateLogDeduplicate", "CandidateBatchQualityAssess",
                "CandidateTextOverlapScore"} <= set(OP_MODE_REGISTRY)


# ---------------------------------------------------------------------------
# op mode protocol (§4.3 / §4.5) — exit codes and hard requirements
# ---------------------------------------------------------------------------


class TestOpModeProtocol:
    def test_invalid_json_exit_2_no_stdout(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        code = op_mode_main(io.StringIO("{not json"), stdout, stderr)
        assert code == 2
        assert stdout.getvalue() == ""

    def test_wrong_schema_version_exit_2(self, tmp_path):
        request = _op_request(tmp_path, [{"taskId": "t1", "text": "a"}])
        request["schemaVersion"] = 2
        code, out, _ = _run_op_mode(request, tmp_path)
        assert code == 2
        assert out == ""

    def test_unknown_request_key_exit_2(self, tmp_path):
        request = _op_request(tmp_path, [{"taskId": "t1", "text": "a"}], extraKey=1)
        code, out, _ = _run_op_mode(request, tmp_path)
        assert code == 2
        assert out == ""

    def test_unknown_op_exit_1_with_structured_diagnostic(self, tmp_path):
        request = _op_request(tmp_path, [{"taskId": "t1", "text": "a"}], op="NoSuchOp")
        code, out, err = _run_op_mode(request, tmp_path)
        assert code == 1
        assert out == ""
        diagnostic = json.loads(err.strip().splitlines()[-1])
        assert diagnostic["code"] == "OP_NOT_FOUND"

    def test_unknown_params_key_exit_2(self, tmp_path):
        request = _op_request(tmp_path, [{"taskId": "t1", "text": "a"}], params={"nope": 1})
        code, out, _ = _run_op_mode(request, tmp_path)
        assert code == 2
        assert out == ""

    def test_row_missing_required_field_exit_2(self, tmp_path):
        request = _op_request(tmp_path, [{"taskId": "t1"}])
        code, out, _ = _run_op_mode(request, tmp_path)
        assert code == 2
        assert out == ""

    def test_row_wrong_field_type_exit_2(self, tmp_path):
        request = _op_request(tmp_path, [{"taskId": "t1", "text": 42}])
        code, out, _ = _run_op_mode(request, tmp_path)
        assert code == 2

    def test_empty_data_file_exit_2(self, tmp_path):
        request = _op_request(tmp_path, [])
        code, out, _ = _run_op_mode(request, tmp_path)
        assert code == 2
        assert out == ""

    def test_unreadable_data_path_exit_1(self, tmp_path):
        request = _op_request(tmp_path, [{"taskId": "t1", "text": "a"}])
        request["dataPath"] = str(tmp_path / "missing.jsonl")
        code, out, err = _run_op_mode(request, tmp_path)
        assert code == 1
        assert out == ""
        assert json.loads(err.strip().splitlines()[-1])["code"] == "DATA_UNREADABLE"


# ---------------------------------------------------------------------------
# Q1 metric op (§4.4 metric mapping + determinism)
# ---------------------------------------------------------------------------


class TestCandidateLogDeduplicate:
    ROWS = [
        {"taskId": f"t{i}", "text": text}
        for i, text in enumerate(["alpha", "beta", "alpha", "gamma", "beta"], start=1)
    ]

    def test_metrics_contract_and_dedup_ratio(self, tmp_path):
        code, out, _ = _run_op_mode(_op_request(tmp_path, self.ROWS), tmp_path)
        assert code == 0
        response = json.loads(out)
        assert response["schemaVersion"] == 1
        assert response["op"] == "CandidateLogDeduplicate"
        metrics = response["metrics"]
        assert set(metrics) == {"input_count", "kept", "removed", "dedup_ratio"}
        assert metrics["input_count"] == 5.0
        assert metrics["kept"] == 3.0
        assert metrics["removed"] == 2.0
        assert metrics["dedup_ratio"] == 0.4

    def test_artifact_lists_first_occurrence_per_unique_text(self, tmp_path):
        code, out, _ = _run_op_mode(_op_request(tmp_path, self.ROWS), tmp_path)
        response = json.loads(out)
        assert response["artifacts"] == [{"relativePath": "deduped-logs.json"}]
        detail = json.loads((tmp_path / "out" / "deduped-logs.json").read_text(encoding="utf-8"))
        assert detail == [
            {"taskId": "t1", "text": "alpha"},
            {"taskId": "t2", "text": "beta"},
            {"taskId": "t4", "text": "gamma"},
        ]

    def test_deterministic_for_same_input(self, tmp_path):
        runs = []
        for attempt in range(2):
            scoped = tmp_path / f"run-{attempt}"
            scoped.mkdir()
            request = _op_request(scoped, self.ROWS)
            request["outputDir"] = None
            code, out, _ = _run_op_mode(request, scoped)
            assert code == 0
            runs.append(json.dumps(json.loads(out)["metrics"], sort_keys=True))
        assert runs[0] == runs[1]


class TestCandidateBatchQualityAssess:
    ROWS = [
        {"taskId": "t1", "text": "Solve problem variant one."},
        {"taskId": "t2", "text": "Solve problem variant two."},
        {"taskId": "t3", "text": "Solve problem variant three."},
    ]

    def test_metrics_key_contract(self, tmp_path):
        code, out, _ = _run_op_mode(
            _op_request(tmp_path, self.ROWS, op="CandidateBatchQualityAssess"), tmp_path
        )
        assert code == 0
        metrics = json.loads(out)["metrics"]
        assert set(metrics) == {
            "quality_score", "completeness", "consistency",
            "uniqueness", "text_quality", "issues_count",
        }
        assert 0.0 <= metrics["quality_score"] <= 1.0

    def test_disabled_dimension_is_null_not_zero(self, tmp_path):
        params = {"quality_config": {"enabled_dimensions": ["completeness"]}}
        code, out, _ = _run_op_mode(
            _op_request(tmp_path, self.ROWS, op="CandidateBatchQualityAssess", params=params),
            tmp_path,
        )
        assert code == 0
        metrics = json.loads(out)["metrics"]
        assert metrics["completeness"] is not None
        assert metrics["uniqueness"] is None
        assert metrics["text_quality"] is None

    def test_unknown_quality_config_key_exit_2(self, tmp_path):
        params = {"quality_config": {"nonsense": 1}}
        code, out, _ = _run_op_mode(
            _op_request(tmp_path, self.ROWS, op="CandidateBatchQualityAssess", params=params),
            tmp_path,
        )
        assert code == 2

    def test_artifact_quality_report_written(self, tmp_path):
        code, out, _ = _run_op_mode(
            _op_request(tmp_path, self.ROWS, op="CandidateBatchQualityAssess"), tmp_path
        )
        response = json.loads(out)
        assert response["artifacts"] == [{"relativePath": "quality-report.json"}]
        report = json.loads((tmp_path / "out" / "quality-report.json").read_text(encoding="utf-8"))
        assert report["operation_name"] == "candidate_batch_quality_assess"


# ---------------------------------------------------------------------------
# Q2 CandidateTextOverlapScore (§5)
# ---------------------------------------------------------------------------


class TestCandidateTextOverlapScore:
    ROWS = [
        {"taskId": "t1", "candidate": "the cat sat on the mat",
         "reference": "the cat sat on a mat"},
        {"taskId": "t2", "candidate": "completely different words entirely here",
         "reference": "the cat sat on a mat"},
    ]

    @pytest.mark.parametrize("metric", ["bleu", "cider", "ngram"])
    def test_metrics_contract_and_score_range(self, tmp_path, metric):
        request = _op_request(tmp_path, self.ROWS, op="CandidateTextOverlapScore",
                              params={"metric": metric})
        code, out, _ = _run_op_mode(request, tmp_path)
        assert code == 0
        response = json.loads(out)
        assert response["op"] == "CandidateTextOverlapScore"
        metrics = response["metrics"]
        assert set(metrics) == {"mean_score", "count"}
        assert 0.0 <= metrics["mean_score"] <= 1.0
        assert metrics["count"] == 2.0

    def test_identical_pair_scores_higher_than_disjoint(self, tmp_path):
        request = _op_request(tmp_path, self.ROWS, op="CandidateTextOverlapScore",
                              params={"metric": "bleu"})
        code, out, _ = _run_op_mode(request, tmp_path)
        per_item = {
            json.loads(line)["taskId"]: json.loads(line)["score"]
            for line in (tmp_path / "out" / "overlap-scores.jsonl").read_text(encoding="utf-8").splitlines()
        }
        assert per_item["t1"] > per_item["t2"]
        assert per_item["t2"] == 0.0

    def test_per_item_path_null_without_output_dir(self, tmp_path):
        request = _op_request(tmp_path, self.ROWS, op="CandidateTextOverlapScore")
        request["outputDir"] = None
        code, out, _ = _run_op_mode(request, tmp_path)
        assert code == 0
        assert json.loads(out)["perItemPath"] is None

    def test_invalid_metric_exit_2(self, tmp_path):
        request = _op_request(tmp_path, self.ROWS, op="CandidateTextOverlapScore",
                              params={"metric": "rouge"})
        code, out, _ = _run_op_mode(request, tmp_path)
        assert code == 2


# ---------------------------------------------------------------------------
# op mode subprocess end-to-end (§4.3 invocation protocol)
# ---------------------------------------------------------------------------


class TestOpModeSubprocess:
    def test_end_to_end_dedup_via_stdin_stdout(self, tmp_path):
        data = _write_jsonl(tmp_path / "texts.jsonl", [
            {"taskId": "t1", "text": "alpha"},
            {"taskId": "t2", "text": "alpha"},
        ])
        request = json.dumps({
            "schemaVersion": 1, "op": "CandidateLogDeduplicate", "params": {},
            "dataPath": data, "outputDir": str(tmp_path / "out"),
        })
        env = {**_ray_env(), "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.run(
            [sys.executable, "-m", "curator.trusted_worker.op"],
            input=request, capture_output=True, text=True, env=env, timeout=NODE_TIMEOUT_S,
        )
        assert proc.returncode == 0, proc.stderr
        response = json.loads(proc.stdout)
        assert response["metrics"]["input_count"] == 2.0
        assert response["metrics"]["kept"] == 1.0
        assert response["metrics"]["removed"] == 1.0

    def test_bad_schema_version_exits_2_without_stdout(self, tmp_path):
        env = {**_ray_env(), "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.run(
            [sys.executable, "-m", "curator.trusted_worker.op"],
            input='{"schemaVersion": 99}', capture_output=True, text=True, env=env,
            timeout=NODE_TIMEOUT_S,
        )
        assert proc.returncode == 2
        assert proc.stdout == ""


# ---------------------------------------------------------------------------
# evaluation mode (RSI §3.9.2 v0.2.1 production package contract)
# ---------------------------------------------------------------------------


class TestEvaluationMode:
    def test_quality_success_score_equals_objective_metric(self, tmp_path):
        envelope = _evaluation_envelope(tmp_path, "CandidateBatchQualityAssess",
                                        TestCandidateBatchQualityAssess.ROWS)
        input_path, result_path = tmp_path / "in.json", tmp_path / "result.json"
        input_path.write_text(json.dumps(envelope), encoding="utf-8")
        assert evaluation_run(str(input_path), str(result_path)) == 0
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["score"] == result["metrics"]["quality_score"]
        assert result["evaluationId"] == "eval-0001"
        assert result["planId"] == "plan-1"
        assert (tmp_path / "out" / "quality-report.json").exists()

    def test_real_rsi_command_envelope_with_target_config_id(self, tmp_path):
        """The RSI evaluator/command envelope (assertPlan requires targetConfigId) must be accepted."""
        envelope = _evaluation_envelope(tmp_path, "CandidateBatchQualityAssess",
                                        TestCandidateBatchQualityAssess.ROWS)
        envelope["request"]["plan"]["targetConfigId"] = "profile-dsh-1"
        input_path, result_path = tmp_path / "in.json", tmp_path / "result.json"
        input_path.write_text(json.dumps(envelope), encoding="utf-8")
        assert evaluation_run(str(input_path), str(result_path)) == 0
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["score"] == result["metrics"]["quality_score"]

    def test_plan_target_config_id_wrong_type_exit_2(self, tmp_path):
        envelope = _evaluation_envelope(tmp_path, "CandidateBatchQualityAssess",
                                        TestCandidateBatchQualityAssess.ROWS)
        envelope["request"]["plan"]["targetConfigId"] = 123
        input_path, result_path = tmp_path / "in.json", tmp_path / "result.json"
        input_path.write_text(json.dumps(envelope), encoding="utf-8")
        assert evaluation_run(str(input_path), str(result_path)) == 2
        assert not result_path.exists()

    def test_dedup_success_with_dedup_ratio(self, tmp_path):
        envelope = _evaluation_envelope(tmp_path, "CandidateLogDeduplicate",
                                        TestCandidateLogDeduplicate.ROWS,
                                        objective_metric="dedup_pass_rate")
        input_path, result_path = tmp_path / "in.json", tmp_path / "result.json"
        input_path.write_text(json.dumps(envelope), encoding="utf-8")
        assert evaluation_run(str(input_path), str(result_path)) == 0
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["score"] == result["metrics"]["dedup_pass_rate"]
        assert result["metrics"]["input_count"] == 5.0
        assert (tmp_path / "out" / "deduped-logs.json").exists()

    def test_unknown_op_exit_1_no_result_file(self, tmp_path):
        envelope = _evaluation_envelope(tmp_path, "CandidateLogDeduplicate",
                                        TestCandidateLogDeduplicate.ROWS)
        envelope["request"]["plan"]["config"]["op"] = "NoSuchOp"
        input_path, result_path = tmp_path / "in.json", tmp_path / "result.json"
        input_path.write_text(json.dumps(envelope), encoding="utf-8")
        assert evaluation_run(str(input_path), str(result_path)) == 1
        assert not result_path.exists()

    def test_wrong_schema_version_exit_2_no_result_file(self, tmp_path):
        envelope = _evaluation_envelope(tmp_path, "CandidateLogDeduplicate",
                                        TestCandidateLogDeduplicate.ROWS)
        envelope["schemaVersion"] = 1
        input_path, result_path = tmp_path / "in.json", tmp_path / "result.json"
        input_path.write_text(json.dumps(envelope), encoding="utf-8")
        assert evaluation_run(str(input_path), str(result_path)) == 2
        assert not result_path.exists()

    def test_malformed_json_exit_2_no_result_file(self, tmp_path):
        input_path, result_path = tmp_path / "in.json", tmp_path / "result.json"
        input_path.write_text("{not valid json", encoding="utf-8")
        assert evaluation_run(str(input_path), str(result_path)) == 2
        assert not result_path.exists()


# ---------------------------------------------------------------------------
# TS thin plugin (node e2e; skipped when node is unavailable)
# ---------------------------------------------------------------------------


def _node_available() -> bool:
    return shutil.which("node") is not None


@pytest.mark.skipif(not _node_available(), reason="node is not installed")
class TestCuratorQualityPlugin:
    PLUGIN_SNIPPET = r"""
    const m = await import(process.env.PLUGIN_PATH);
    const registrations = [];
    const ctx = { evaluator: { register(role, name, validate, factory) {
      registrations.push({ role, name, validate, factory });
    } } };
    m.default.apply(ctx);
    const { validate, factory } = registrations[0];
    const mode = process.env.NODE_TEST_MODE;
    const runtime = { resolvePath: (p) => p };
    if (mode === "validate-bad") {
      try { validate({ op: "CandidateLogDeduplicate", bogus: 1 }); console.log("FAIL no reject"); }
      catch (e) { console.log(JSON.stringify({ rejected: true, message: String(e.message) })); }
    } else if (mode === "compute" || mode === "empty") {
      const config = JSON.parse(process.env.PLUGIN_CONFIG);
      const instance = factory(validate(config), runtime);
      const snapshot = mode === "empty" ? { records: [] } : {
        records: [{
          key: { runId: "r", taskId: "t1", attemptId: "a1" },
          status: "completed",
          solve: { finalText: "alpha", usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } },
          verdict: null, errors: [], collection: null, environments: {},
          startedAt: "", finishedAt: "", durationMs: 0, phaseDurationMs: {},
        }, {
          key: { runId: "r", taskId: "t2", attemptId: "a2" },
          status: "timeout",
          solve: null, verdict: null, errors: [], collection: null, environments: {},
          startedAt: "", finishedAt: "", durationMs: 0, phaseDurationMs: {},
        }],
      };
      const value = await instance.compute(snapshot);
      console.log(JSON.stringify(value));
    }
    """

    def _run_node(self, tmp_path: Path, mode: str, config: dict | None = None) -> dict:
        env = {
            **_ray_env(),
            "PLUGIN_PATH": str(PLUGIN_PATH),
            "NODE_TEST_MODE": mode,
            "NODE_OPTIONS": "--input-type=module",
        }
        if config is not None:
            env["PLUGIN_CONFIG"] = json.dumps(config)
        proc = subprocess.run(
            ["node", "-e", self.PLUGIN_SNIPPET],
            capture_output=True, text=True, env=env, timeout=NODE_TIMEOUT_S, cwd=str(tmp_path),
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_registers_metrics_plugin(self, tmp_path):
        env = {**_ray_env(), "PLUGIN_PATH": str(PLUGIN_PATH), "NODE_TEST_MODE": "noop"}
        proc = subprocess.run(
            ["node", "-e", self.PLUGIN_SNIPPET],
            capture_output=True, text=True, env=env, timeout=NODE_TIMEOUT_S, cwd=str(tmp_path),
        )
        assert proc.returncode == 0, proc.stderr

    def test_validate_rejects_unknown_key(self, tmp_path):
        result = self._run_node(tmp_path, "validate-bad")
        assert result["rejected"] is True

    def test_empty_run_short_circuits_without_spawning(self, tmp_path):
        config = {
            "op": "CandidateLogDeduplicate",
            "pythonExecutable": "/nonexistent/python-binary",
        }
        result = self._run_node(tmp_path, "empty", config)
        assert result == {"eligible": 0}

    def test_compute_end_to_end_dedup(self, tmp_path):
        config = {
            "op": "CandidateLogDeduplicate",
            "pythonExecutable": sys.executable,
            "env": {
                "PYTHONPATH": str(REPO_ROOT),
                **({"RAY_ADDRESS": ray.get_runtime_context().gcs_address}
                   if ray.is_initialized() else {}),
            },
            "timeoutMs": NODE_TIMEOUT_S * 1000,
        }
        result = self._run_node(tmp_path, "compute", config)
        assert set(result) == {"input_count", "kept", "removed", "dedup_ratio"}
        assert result["input_count"] == 1.0
        assert result["kept"] == 1.0
        assert result["removed"] == 0.0
        assert result["dedup_ratio"] == 0.0
