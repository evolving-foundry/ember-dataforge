"""single-turn four-family native runner tests (M3).

Covers NeMo/HF single-turn evaluation onboarding implementation plan §2 M3 acceptance:
  - all branches for reference / negative / evaluation (missing artifact, wrong answer) / out-of-bounds paths;
  - semantic equivalence of deterministic matching for each pattern (answer-line / multichoice / numeric /
    fuzzy / trivia);
  - humaneval allowlist execution boundary (D8): candidate code only affects solution.py content,
    with a forced timeout;
  - judge threshold semantics (linear 1–5 scaling) and explicit failure when no judge is configured (D7).
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks import native
from data_designer.pipelines.benchmarks.contracts import NativeFailure
from data_designer.pipelines.benchmarks.families import hf_single_turn_matchers as m
from data_designer.verifiers import llm_judge

_RUNNERS = {
    family: Path(
        __import__(
            f"data_designer.pipelines.benchmarks.runners.{family}",
            fromlist=["VERSION"],
        ).__file__
    )
    for family in ("reasoning_qa", "knowledge_qa", "code_gen", "health_qa")
}

MATCHERS_SOURCE = Path(m.__file__).read_text(encoding="utf-8")


def script(body: str) -> str:
    """Compose a frozen unit_test script: matchers source + per-case body."""
    return MATCHERS_SOURCE + "\n\n" + textwrap.dedent(body)


def unit_spec(relative: str, test_body: str, timeout_s: float = 20.0) -> dict:
    return {
        "kind": "unit_test",
        "generator_ref": "hf-single-turn-fixture@synthetic",
        "payload": {
            "artifact": {"path": relative, "format": "text"},
            "test_script": script(test_body),
            "timeout_s": timeout_s,
        },
    }


def rubric_spec(relative: str, rubric: str, threshold: float) -> dict:
    return {
        "kind": "llm_rubric",
        "generator_ref": "hf-single-turn-fixture@synthetic",
        "payload": {
            "artifact": {"path": relative, "format": "text"},
            "rubric": rubric,
            "pass_threshold": threshold,
            "score_scale": {"min": 1, "max": 5},
        },
    }


def write_manifest(root: Path, family: str, task_id: str, spec: dict) -> Path:
    manifest = {
        "schema_version": "benchmark-case-v1",
        "task_id": task_id,
        "family": family,
        "private": {"verifier_spec": spec},
    }
    path = root / "manifests" / f"{task_id.replace(':', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def run(family: str, manifest: Path, candidate_root: Path, phase: str) -> dict:
    return native.run_case(
        [sys.executable, str(_RUNNERS[family])],
        task_id=json.loads(manifest.read_text(encoding="utf-8"))["task_id"],
        manifest_path=str(manifest),
        candidate_root=str(candidate_root),
        phase=phase,
        timeout=120.0,
        public_item={"placeholder": True},
    )


@pytest.fixture(autouse=True)
def _isolate_judge_env(monkeypatch):
    # The native runner subprocess is bootstrapped with the DATA_FORGE_JUDGE_* grading judge; tests never inherit the host environment.
    for key in ("DATA_FORGE_JUDGE_BASE_URL", "DATA_FORGE_JUDGE_API_KEY",
                "DATA_FORGE_JUDGE_MODEL", "DATA_FORGE_JUDGE_TEMPERATURE"):
        monkeypatch.delenv(key, raising=False)
    yield


def test_reference_phase_gold_answer_passes(tmp_path):
    manifest = write_manifest(
        tmp_path, "reasoning_qa", "reasoning_qa:gsm8k:row-0:v1",
        unit_spec(
            "response.txt",
            """
            text = open("response.txt", encoding="utf-8").read()
            assert numeric_answer_equal(
                extract_after_marker(text, "the answer is"), "42"
            ), "answer mismatch"
            """,
        ),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "response.txt").write_text(
        "Reasoning... The answer is 42.", encoding="utf-8"
    )
    result = run("reasoning_qa", manifest, candidate, "reference")
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_negative_phase_wrong_answer_fails(tmp_path):
    manifest = write_manifest(
        tmp_path, "reasoning_qa", "reasoning_qa:gsm8k:row-0:v1",
        unit_spec(
            "response.txt",
            """
            text = open("response.txt", encoding="utf-8").read()
            assert numeric_answer_equal(
                extract_after_marker(text, "the answer is"), "42"
            ), "answer mismatch"
            """,
        ),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "response.txt").write_text("The answer is 43.", encoding="utf-8")
    result = run("reasoning_qa", manifest, candidate, "negative")
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_evaluation_missing_deliverable_is_invalid_candidate(tmp_path):
    manifest = write_manifest(
        tmp_path, "knowledge_qa", "knowledge_qa:mmlu:row-0:v1",
        unit_spec(
            "response.txt",
            """
            text = open("response.txt", encoding="utf-8").read()
            assert multichoice_correct(text, "B", "ABCD"), "wrong option"
            """,
        ),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    result = run("knowledge_qa", manifest, candidate, "evaluation")
    assert result["status"] == "invalid_candidate"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_evaluation_wrong_answer_scores_zero_completed(tmp_path):
    manifest = write_manifest(
        tmp_path, "knowledge_qa", "knowledge_qa:triviaqa:row-0:v1",
        unit_spec(
            "response.txt",
            """
            text = open("response.txt", encoding="utf-8").read()
            assert trivia_answer_correct(text, ["Alexander Hamilton"]), "wrong"
            """,
        ),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "response.txt").write_text("Aaron Burr", encoding="utf-8")
    result = run("knowledge_qa", manifest, candidate, "evaluation")
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_symlink_escape_is_invalid_candidate_in_evaluation(tmp_path):
    manifest = write_manifest(
        tmp_path, "reasoning_qa", "reasoning_qa:math500:row-0:v1",
        unit_spec(
            "response.txt",
            """
            text = open("response.txt", encoding="utf-8").read()
            assert extract_boxed(text) == "\\\\frac{1}{2}", "wrong"
            """,
        ),
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("\\boxed{\\frac{1}{2}}", encoding="utf-8")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "response.txt").symlink_to(outside)
    result = run("reasoning_qa", manifest, candidate, "evaluation")
    assert result["status"] == "invalid_candidate"
    assert result["score"] == 0.0


def test_symlink_escape_fails_reference_audit_explicitly(tmp_path):
    manifest = write_manifest(
        tmp_path, "reasoning_qa", "reasoning_qa:math500:row-0:v1",
        unit_spec("response.txt", "assert True"),
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("gold", encoding="utf-8")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "response.txt").symlink_to(outside)
    with pytest.raises(NativeFailure):
        run("reasoning_qa", manifest, candidate, "reference")


def test_traversal_deliverable_path_is_rejected(tmp_path):
    manifest = write_manifest(
        tmp_path, "knowledge_qa", "knowledge_qa:gpqa:row-0:v1",
        unit_spec("../response.txt", "assert True"),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with pytest.raises(NativeFailure):
        run("knowledge_qa", manifest, candidate, "evaluation")


def test_wrong_family_manifest_is_rejected(tmp_path):
    manifest = write_manifest(
        tmp_path, "terminal", "reasoning_qa:mgsm:row-0:v1",
        unit_spec("response.txt", "assert True"),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "response.txt").write_text("70", encoding="utf-8")
    with pytest.raises(NativeFailure):
        run("reasoning_qa", manifest, candidate, "evaluation")


    # ── Deterministic matching semantic equivalence (Evaluator score.ts re-expression) ─────────────────────────


@pytest.mark.parametrize("prediction,gold,expected", [
    ("Work it out... The answer is 42.", "42", True),
    ("The answer is $1,234.50.", "1234.5", True),
    ("The answer is 42.", "43", False),
    ("no marker at all", "42", False),
])
def test_answer_line_semantics(prediction, gold, expected):
    extracted = m.extract_after_marker(prediction, "the answer is")
    assert m.numeric_answer_equal(extracted, gold) is expected


@pytest.mark.parametrize("prediction,gold,allowed,expected", [
    ("...so the answer is (B).", "B", "ABCD", True),
    ("Answer: C", "C", "ABCD", True),
    ("I choose (J) because...", "J", "ABCDEFGHIJ", True),
    ("The answer is (E).", "B", "ABCD", False),
    ("Option (E) seems right.", "E", "ABCDE", True),
    ("No letter here.", "B", "ABCD", False),
])
def test_multichoice_semantics(prediction, gold, allowed, expected):
    assert m.multichoice_correct(prediction, gold, allowed) is expected


@pytest.mark.parametrize("prediction,gold,expected", [
    ("The population is about 8,000,000 people.", "8000000", True),
    ("Result: 70.0", "70", True),
    ("It costs $3.14.", "3.14", True),
    ("Around 250.", "251", False),
    ("No numbers.", "1", False),
])
def test_numeric_semantics(prediction, gold, expected):
    assert m.numeric_answer_equal(prediction, gold) is expected


def test_boxed_semantics():
    assert m.extract_boxed("first \\boxed{7} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert m.extract_boxed("nested \\boxed{\\text{a}{b}}") == "\\text{a}{b}"
    assert m.extract_boxed("unbalanced \\boxed{9") == ""
    assert m.numeric_answer_equal("The answer is 70.", "70") is True


@pytest.mark.parametrize("prediction,spans,expected", [
    ("opened in the 1970s", ["the 1970s"], True),
    ("The 1970s.", ["the 1970s"], True),
    ("sometime last century", ["the 1970s"], False),
    ("January 3, 1972", ["January 3, 1972", "1972-01-03"], True),
])
def test_fuzzy_span_semantics(prediction, spans, expected):
    assert m.fuzzy_span_correct(prediction, spans) is expected


@pytest.mark.parametrize("prediction,aliases,expected", [
    ("Alexander Hamilton", ["Alexander Hamilton", "Hamilton"], True),
    ("the answer is alexander hamilton!", ["answer is alexander hamilton"], True),
    ("Aaron Burr", ["Hamilton"], False),
])
def test_trivia_alias_semantics(prediction, aliases, expected):
    assert m.trivia_answer_correct(prediction, aliases) is expected


    # ── humaneval allowlist execution boundary (D8) ────────────────────────────────────────────


HUMANEVAL_BODY = """
code = open("solution.py", encoding="utf-8").read()
namespace = {}
exec(compile(code, "solution.py", "exec"), namespace)
result = namespace["solve"](2, 3)
assert result == 5, "wrong result: %r" % (result,)
"""


def test_humaneval_candidate_solution_passes_and_fails(tmp_path):
    manifest = write_manifest(
        tmp_path, "code_gen", "code_gen:humaneval:row-0:v1",
        unit_spec("solution.py", HUMANEVAL_BODY),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "solution.py").write_text(
        "def solve(a, b):\n    return a + b\n", encoding="utf-8"
    )
    result = run("code_gen", manifest, candidate, "reference")
    assert result["passed"] is True and result["score"] == 1.0

    (candidate / "solution.py").write_text(
        "def solve(a, b):\n    return a * b\n", encoding="utf-8"
    )
    negative = run("code_gen", manifest, candidate, "negative")
    assert negative["passed"] is False and negative["score"] == 0.0


def test_humaneval_runaway_solution_hits_pinned_timeout(tmp_path):
    manifest = write_manifest(
        tmp_path, "code_gen", "code_gen:humaneval:row-0:v1",
        unit_spec(
            "solution.py",
            """
            code = open("solution.py", encoding="utf-8").read()
            namespace = {}
            exec(compile(code, "solution.py", "exec"), namespace)
            namespace["solve"]()
            """,
            timeout_s=2.0,
        ),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "solution.py").write_text(
        "def solve():\n    while True:\n        pass\n", encoding="utf-8"
    )
    result = run("code_gen", manifest, candidate, "negative")
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


    # ── judge threshold semantics and explicit failure when unconfigured (D7) ─────────────────────────────────────


def test_judge_unconfigured_fails_explicitly(tmp_path):
    manifest = write_manifest(
        tmp_path, "health_qa", "health_qa:healthbench:row-0:v1",
        rubric_spec("response.txt", "criterion one; criterion two", 0.5),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "response.txt").write_text("a thorough answer", encoding="utf-8")
    with pytest.raises(NativeFailure, match="judge"):
        run("health_qa", manifest, candidate, "evaluation")


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [
            type("C", (), {"message": type("M", (), {"content": content})()})()
        ]


def _fake_judge(content: str) -> llm_judge.LlmJudge:
    class FakeClient:
        def chat(self, *args, **kwargs):
            return _FakeResponse(content)

    return llm_judge.LlmJudge(FakeClient(), model="judge-model")


@pytest.mark.parametrize("raw,threshold,expected_passed,expected_score", [
    ('{"passed": true, "score": 4, "reason": "ok"}', 0.5, True, 0.75),
    ('{"passed": false, "score": 2, "reason": "weak"}', 0.5, False, 0.25),
    ('{"passed": true, "score": 3, "reason": "ok"}', 0.8, False, 0.5),
])
def test_judge_scale_folding_and_threshold(
    raw, threshold, expected_passed, expected_score
):
    verdict = llm_judge.judge_with_rubric(
        "the response text",
        {
            "rubric": "criterion one; criterion two",
            "pass_threshold": threshold,
            "score_scale": {"min": 1, "max": 5},
        },
        judge=_fake_judge(raw),
    )
    assert verdict["passed"] is expected_passed
    assert verdict["score"] == pytest.approx(expected_score)


def test_judge_out_of_scale_score_fails_explicitly():
    # An out-of-range judge score is abnormal grading-service behavior: report the execution failure explicitly, do not disguise it as a 0 score.
    verdict = llm_judge.judge_with_rubric(
        "text",
        {
            "rubric": "r",
            "pass_threshold": 0.5,
            "score_scale": {"min": 1, "max": 5},
        },
        judge=_fake_judge('{"score": 9}'),
    )
    assert verdict["status"] == "error" and "outside" in verdict["detail"]


def test_judge_fingerprint_pins_configuration():
    fingerprint_a = llm_judge.LlmJudge(
        object(), model="model-a", temperature=0.0
    ).fingerprint()
    assert fingerprint_a.startswith("sha256:") and len(fingerprint_a) == 71
    fingerprint_b = llm_judge.LlmJudge(
        object(), model="model-b", temperature=0.0
    ).fingerprint()
    assert fingerprint_b != fingerprint_a
