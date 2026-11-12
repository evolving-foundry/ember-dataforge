"""wb_composite runner tests (M2): deps sandbox scoring against a real-layout fixture.

Covers design §8 deterministic phases:
- reference arm passes all (audit_exempt) / negative is rejected;
- evaluation missing deliverable → invalid_candidate (not silent);
- grading_source archive sha drift / grading directory content pin mismatch / gold mismatch → explicit rejection;
- legacy grading (does not write eval_result.json) → junit fallback path;
- build_workbuddy_augment_config injects the grading_source pin for composite rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import validate_native_result
from data_designer.pipelines.benchmarks.runners import workbuddy_composite as runner
from data_designer.pipelines.benchmarks.families.workbuddy import (
    build_native_ingest_rows,
    build_workbuddy_augment_config,
)
from data_designer.sources.workbuddy_seed import _dir_content_sha256

TASK_ID = "t01"
GOLD = {"expected_value": 42}
ARTIFACTS = [
    {"id": "report", "path": "output/report.json", "type": "json", "required": True}
]
JUDGE = {
    "artifacts": ARTIFACTS,
    "llm_judge": {
        "method": "rubric_binary_mean",
        "rubrics": [{"id": "r1", "question": "Is the report complete?"}],
    },
    "score_merge": {"rule_weight": 0.8, "llm_weight": 0.2},
}

EVAL_CORE = '''import json
from pathlib import Path


def evaluate(output_dir, gold_path=None):
    out = Path(output_dir)
    gold = json.loads(Path(gold_path).read_text()) if gold_path else {}
    checks = []
    p = out / "report.json"
    checks.append({"name": "report_exists", "passed": p.is_file(),
                   "group": "output_contract"})
    value = None
    if p.is_file():
        try:
            value = json.loads(p.read_text()).get("value")
            checks.append({"name": "report_valid", "passed": True,
                           "group": "output_contract"})
        except Exception as exc:
            checks.append({"name": "report_valid", "passed": False,
                           "detail": repr(exc), "group": "output_contract"})
    checks.append({"name": "value_matches_gold",
                   "passed": value == gold.get("expected_value"),
                   "group": "content"})
    return {"checks": checks, "total_count": len(checks),
            "passed_count": sum(1 for c in checks if c["passed"] is True)}
'''

TEST_VERIFY_NEW = '''import json, os, sys
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
    except Exception as exc:
        return {"checks": [{"name": "eval_core_import_failed", "passed": False,
                            "detail": repr(exc)}]}
    try:
        result = evaluate(OUTPUT_PATH, str(GOLD_PATH))
    except Exception as exc:
        return {"checks": [{"name": "evaluator_runtime_error", "passed": False,
                            "detail": repr(exc)}]}
    return result if isinstance(result, dict) else {"checks": []}


_RESULT = _load_result()
try:
    VERIFIER_LOGS.mkdir(parents=True, exist_ok=True)
    (VERIFIER_LOGS / "eval_result.json").write_text(
        json.dumps(_RESULT), encoding="utf-8")
except Exception:
    pass
_CHECKS = _RESULT.get("checks") or [{"name": "no_checks", "passed": False,
                                     "detail": "empty"}]
_IDS = [f"{i:03d}_{c.get('name', 'x')}" for i, c in enumerate(_CHECKS)]


@pytest.mark.parametrize("idx", range(len(_CHECKS)), ids=_IDS)
def test_atomic_check(idx):
    check = _CHECKS[idx]
    assert check.get("passed") is True, check.get("detail", "")
'''

_TEST_VERIFY_OLD_BLOCK = '''try:
    VERIFIER_LOGS.mkdir(parents=True, exist_ok=True)
    (VERIFIER_LOGS / "eval_result.json").write_text(
        json.dumps(_RESULT), encoding="utf-8")
except Exception:
    pass
'''

TEST_VERIFY_OLD = TEST_VERIFY_NEW.replace(
    _TEST_VERIFY_OLD_BLOCK,
    "# old-style grading: eval_result.json is never persisted\n",
)


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _workspace_tar(path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        _add(tar, "input/notes.csv", "k,v\n1,2\n".encode())
        _add(tar, "input/README.md", b"# fixture workspace\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


def _build_seed_archive(
    tmp: Path, *, persist_eval_result: bool = True
) -> Path:
    """Real-layout seed archive: <top>/tasks/<id>/{instruction,tests,environment}."""
    task_files = {
        "instruction.md": b"Produce output/report.json with the expected value.",
        "task.toml": b"[task]\nname = 't01'\n",
        "tests/judge.yaml": (
            "version: 1\nartifacts:\n- id: report\n  path: output/report.json\n"
            "  type: json\n  required: true\nllm_judge:\n"
            "  method: rubric_binary_mean\n"
            "  rubrics:\n  - id: r1\n    question: Is the report complete?\n"
            "score_merge:\n  method: weighted_sum\n  rule_weight: 0.8\n"
            "  llm_weight: 0.2\n"
        ).encode(),
        "tests/gold/gold_answer.json": json.dumps(GOLD).encode(),
        "tests/grading/eval_core.py": EVAL_CORE.encode(),
        "tests/grading/test_verify.py": (
            TEST_VERIFY_NEW if persist_eval_result else TEST_VERIFY_OLD
        ).encode(),
        "tests/verifier.toml": b'[run]\ncwd = "/workspace"\n',
        "tests/test.sh": b"#!/bin/bash\nexit 64\n",
    }
    archive = tmp / "seed-archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in task_files.items():
            _add(tar, f"wb-m2-fixture/tasks/{TASK_ID}/{name}", data)
        ws = tmp / "workspace.tar.gz"
        _workspace_tar(ws)
        data = ws.read_bytes()
        info = tarfile.TarInfo(
            name=(
                f"wb-m2-fixture/tasks/{TASK_ID}/environment/workspace.tar.gz"
            )
        )
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return archive


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(tmp: Path, archive: Path) -> dict:
    """Extract tests/grading from the archive to derive the dir content pin."""
    grading_dir = tmp / "_grading_extract" / TASK_ID
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            if (
                name.endswith("tests/grading/eval_core.py")
                or name.endswith("tests/grading/test_verify.py")
            ) and member.isfile():
                rel = Path(name).relative_to(
                    f"wb-m2-fixture/tasks/{TASK_ID}/tests/grading"
                )
                target = grading_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    ws = tmp / "workspace.tar.gz"
    return {
        "seed_id": TASK_ID,
        "task_text": "Produce output/report.json with the expected value.",
        "tools_available": ["file_io"],
        "success_criteria": "composite score >= 0.8",
        "difficulty": "medium",
        "deliverable_types": ["json"],
        "judge": JUDGE,
        "gold": GOLD,
        "grading_dir_sha256": _dir_content_sha256(grading_dir),
        "verifier_timeout_sec": 600,
        "workspace": {"path": str(ws), "sha256": _sha(ws)},
    }


def _spec(
    tmp: Path,
    archive: Path,
    record: dict,
    *,
    grading_sha: str | None = None,
) -> dict:
    return {
        "kind": "wb_composite",
        "generator_ref": f"wb_bench_native_ingest@{TASK_ID}",
        "payload": {
            "artifacts": ARTIFACTS,
            "rubrics": JUDGE["llm_judge"]["rubrics"],
            "llm_judge_method": "rubric_binary_mean",
            "score_merge": JUDGE["score_merge"],
            "grading_dir_sha256": grading_sha or record["grading_dir_sha256"],
            "env_pin": dict(record["workspace"]),
            "gold": record["gold"],
            "timeout_sec": record["verifier_timeout_sec"],
            "grading_source": {
                "path": str(archive),
                "sha256": _sha(archive),
                "member_prefix": f"tasks/{TASK_ID}/tests",
            },
        },
    }


def _write_manifest(tmp: Path, spec: dict) -> Path:
    manifest = {
        "schema_version": "benchmark-case-v1",
        "task_id": TASK_ID,
        "family": "workbuddy",
        "private": {"verifier_spec": spec},
    }
    path = tmp / "case_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _request(manifest: Path, candidate_root: Path, phase: str) -> dict:
    return {
        "schemaVersion": 1,
        "taskId": TASK_ID,
        "caseManifestPath": str(manifest),
        "candidateRoot": str(candidate_root),
        "phase": phase,
        "publicItem": {"task_id": TASK_ID},
    }


def _candidate(tmp: Path, name: str, value: int) -> Path:
    root = tmp / name
    (root / "output").mkdir(parents=True, exist_ok=True)
    (root / "output" / "report.json").write_text(json.dumps({"value": value}))
    return root


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wb-composite")
    archive = _build_seed_archive(tmp)
    record = _record(tmp, archive)
    return tmp, archive, record


def test_reference_phase_full_pass(seeded):
    tmp, archive, record = seeded
    manifest = _write_manifest(tmp, _spec(tmp, archive, record))
    result = runner.score_request(
        _request(manifest, _candidate(tmp, "reference", 42), "reference")
    )
    validate_native_result(result, TASK_ID)
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["llm_judge_status"] == "audit_exempt"
    assert result["native_metrics"]["rule_pass_rate"] == 1.0
    assert result["rule_diagnostics"]["rule_source"] == "eval_result"
    assert result["rule_diagnostics"]["rule_status"] == "full_pass"
    assert result["evidence_refs"]


def test_negative_phase_rejected(seeded):
    tmp, archive, record = seeded
    manifest = _write_manifest(tmp, _spec(tmp, archive, record))
    result = runner.score_request(
        _request(manifest, _candidate(tmp, "negative", 43), "negative")
    )
    validate_native_result(result, TASK_ID)
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert 0.0 <= result["score"] < 1.0
    assert result["llm_judge_status"] == "audit_exempt"
    assert result["rule_diagnostics"]["failed_checks"]


def test_evaluation_missing_required_artifact(seeded):
    tmp, archive, record = seeded
    manifest = _write_manifest(tmp, _spec(tmp, archive, record))
    root = tmp / "empty-candidate"
    root.mkdir()
    result = runner.score_request(_request(manifest, root, "evaluation"))
    validate_native_result(result, TASK_ID)
    assert result["status"] == "invalid_candidate"
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert "required deliverable missing" in (
        result["rule_diagnostics"]["invalid_reason"]
    )


def test_tampered_grading_archive_rejected(seeded):
    tmp, archive, record = seeded
    spec = _spec(tmp, archive, record)
    spec["payload"]["grading_source"]["sha256"] = "f" * 64
    manifest = _write_manifest(tmp, spec)
    with pytest.raises(ValueError, match="grading source SHA-256 drift"):
        runner.score_request(
            _request(manifest, _candidate(tmp, "ref-tampered", 42), "reference")
        )


def test_grading_dir_content_pin_enforced(tmp_path):
    archive = _build_seed_archive(tmp_path)
    record = _record(tmp_path, archive)
    poisoned = tmp_path / "poisoned.tar.gz"
    with tarfile.open(archive, "r:gz") as src, tarfile.open(poisoned, "w:gz") as dst:
        for member in src.getmembers():
            dst.addfile(member, src.extractfile(member))
        _add(
            dst,
            f"wb-m2-fixture/tasks/{TASK_ID}/tests/grading/extra_cheat.py",
            b"raise RuntimeError('grading tampered')\n",
        )
    spec = _spec(tmp_path, poisoned, record)
    manifest = _write_manifest(tmp_path, spec)
    with pytest.raises(ValueError, match="content pin"):
        runner.score_request(
            _request(manifest, _candidate(tmp_path, "ref-poisoned", 42), "reference")
        )


def test_gold_disagreement_rejected(tmp_path):
    archive = _build_seed_archive(tmp_path)
    record = _record(tmp_path, archive)
    spec = _spec(tmp_path, archive, record)
    spec["payload"]["gold"] = {"expected_value": 999}
    manifest = _write_manifest(tmp_path, spec)
    with pytest.raises(ValueError, match="frozen gold"):
        runner.score_request(
            _request(manifest, _candidate(tmp_path, "ref-gold", 42), "reference")
        )


def test_old_style_grading_junit_fallback(tmp_path):
    archive = _build_seed_archive(tmp_path, persist_eval_result=False)
    record = _record(tmp_path, archive)
    manifest = _write_manifest(tmp_path, _spec(tmp_path, archive, record))
    result = runner.score_request(
        _request(manifest, _candidate(tmp_path, "reference-old", 42), "reference")
    )
    validate_native_result(result, TASK_ID)
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["rule_diagnostics"]["rule_source"] == "junit"


def test_config_build_injects_grading_source(tmp_path):
    archive = _build_seed_archive(tmp_path)
    record = _record(tmp_path, archive)
    rows = build_native_ingest_rows([record])
    report = tmp_path / "ingest_report.json"
    report.write_text(
        json.dumps({"schema_version": "workbuddy-ingest-report-v1",
                    "release": "synthetic-fixture:wb-m2-test",
                    "tasks": [{"task_id": TASK_ID, "excluded": False}]}),
        encoding="utf-8",
    )
    public_root = tmp_path / "public_inputs"
    notes = public_root / TASK_ID / "input" / "notes.csv"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("k,v\n1,2\n", encoding="utf-8")
    runner_path = Path(runner.__file__).resolve()
    config = build_workbuddy_augment_config(
        rows,
        release="synthetic-fixture:wb-m2-test",
        split="holdout",
        strategy="native_ingest",
        environment={"path": str(archive), "sha256": _sha(archive)},
        verifier_path=str(runner_path),
        verifier_command=[sys.executable, str(runner_path)],
        audit_roots={
            "path": str(tmp_path / "audit"), "kind": "native_ingest_review"
        },
        ingest_report={"path": str(report), "sha256": _sha(report)},
        public_inputs={
            "dir": str(public_root),
            "by_task": {
                TASK_ID: [{"path": "input/notes.csv",
                           "sha256": _sha(notes)}]
            },
        },
    )
    payload = config["variants"][0]["private"]["verifier_spec"]["payload"]
    source = payload["grading_source"]
    assert source["path"] == str(archive)
    assert source["sha256"] == _sha(archive)
    assert source["member_prefix"] == f"tasks/{TASK_ID}/tests"


def test_runtime_pins_python_aliases_to_verifier_interpreter(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").write_bytes(b"host")
    interpreter = tmp_path / "python3.10"
    interpreter.write_bytes(b"verifier")

    runner._pin_runtime_interpreter(bin_dir, interpreter, "python3.10")

    assert (bin_dir / "python3.10").read_bytes() == b"verifier"
    assert (bin_dir / "python3").resolve() == (bin_dir / "python3.10").resolve()
    assert (bin_dir / "python").resolve() == (bin_dir / "python3.10").resolve()


def test_runtime_marker_rejects_missing_or_dangling_tools(tmp_path):
    key = runner._python_key()
    root = tmp_path / "runtime"
    bin_dir = root / "usr" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / key).write_bytes(b"")
    (bin_dir / "env").write_bytes(b"")
    (root / "marker.json").write_text(
        json.dumps(
            {
                "python": key,
                "layout": runner.RUNTIME_LAYOUT_VERSION,
                "interpreter": str(runner._runtime_interpreter()),
            }
        ),
        encoding="utf-8",
    )
    assert runner._runtime_marker(root, key)

    (bin_dir / "env").unlink()
    assert runner._runtime_marker(root, key) == {}
