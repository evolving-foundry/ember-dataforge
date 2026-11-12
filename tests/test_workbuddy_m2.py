"""M2 unit tests: four verifier types + S2 taxonomy + S3 failure_probe + contamination detection."""

import pytest

from data_designer.generation.algorithms.builtin.workbuddy_contamination import (
    contamination_report,
    jaccard_similarity,
)
from data_designer.generation.algorithms.builtin.workbuddy_failure_probe import (
    WorkbuddyFailureProbeEngine,
)
from data_designer.generation.algorithms.builtin.workbuddy_taxonomy import (
    WorkbuddyTaxonomyEngine,
    taxonomy_gaps,
)
from data_designer.verifiers import (
    LlmJudgeConfigError,
    VerifierDispatchError,
    assert_dom,
    judge_with_rubric,
    run_unit_tests,
    run_verifier,
    validate_spec,
)


def _double_judge(task):
    """Dry-run dual judgment (§8.2): gold must pass, negative must fail."""
    gold = run_verifier(task.verifier_spec, task.gold_solution)
    neg = run_verifier(task.verifier_spec, task.negative_solution)
    assert gold["passed"], f"gold failed: {gold['detail']}"
    assert not neg["passed"], "negative case did not fail"


# ── dom_assert ───────────────────────────────────────────────────────────────


def test_dom_assert_pass_and_fail():
    html = ("<html><body><table><tr><td>a</td></tr><tr><td>b</td></tr></table>"
            "<a href='report.pdf'>dl</a><p>Q3 revenue rose</p></body></html>")
    payload = {"assertions": [
        {"type": "element_exists", "tag": "table"},
        {"type": "element_count", "tag": "tr", "op": ">=", "value": 2},
        {"type": "text_contains", "value": "Q3 revenue"},
        {"type": "attr_equals", "tag": "a", "attr": "href",
         "value": "report.pdf"},
    ]}
    ok = assert_dom(html, payload)
    assert ok["passed"] and ok["score"] == 1.0
    bad = assert_dom(html.replace("Q3 revenue", "Q4 revenue"), payload)
    assert not bad["passed"] and bad["score"] == 0.75


def test_dom_assert_config_errors():
    assert assert_dom("<p>x</p>", {})["passed"] is False
    assert assert_dom(123, {"assertions": [{"type": "element_exists",
                                            "tag": "p"}]})["passed"] is False


# ── unit_test ────────────────────────────────────────────────────────────────


def test_unit_test_pass_and_fail():
    payload = {
        "setup_files": {"input/data.csv": "region,revenue\neast,10\nwest,20\n"},
        "test_script": (
            "import json\n"
            "s = json.load(open('output/summary.json'))\n"
            "assert s['total'] == 30\n"),
        "timeout_s": 10,
    }
    ok = run_unit_tests({"output/summary.json": "{\"total\": 30}"}, payload)
    assert ok["passed"] and ok["score"] == 1.0
    bad = run_unit_tests({"output/summary.json": "{\"total\": 31}"}, payload)
    assert not bad["passed"] and "exited with code 1" in bad["detail"]
    missing = run_unit_tests({}, payload)
    assert not missing["passed"]


def test_unit_test_rejects_path_escape():
    payload = {"test_script": "pass", "timeout_s": 5}
    ok = run_unit_tests({"../evil.txt": "x"}, payload)
    assert not ok["passed"] and "Invalid file path" in ok["detail"]


# ── llm_rubric ───────────────────────────────────────────────────────────────


def test_llm_judge_requires_configuration():
    with pytest.raises(LlmJudgeConfigError):
        judge_with_rubric("report", {"rubric": "r"})


def test_llm_judge_scores_via_injected_client():
    from data_designer.verifiers import llm_judge

    class FakeResp:
        class choices:  # noqa: N801 - simulates the OpenAI response structure
            class message:  # noqa: N801
                content = '{"passed": true, "score": 0.9, "reason": "ok"}'
        choices = [choices]

    class FakeClient:
        def chat(self, messages, temperature=0.0):
            return FakeResp()

    judge = llm_judge.LlmJudge(FakeClient(), model="judge-model")
    verdict = judge_with_rubric("report text", {"rubric": "r1\nr2",
                                                "pass_threshold": 0.8},
                                judge=judge)
    assert verdict["passed"] and verdict["score"] == 0.9
    assert "judge-model" in verdict["detail"]
    assert "wb-judge-v1" in verdict["detail"]


def test_llm_judge_unparseable_output_fails_explicitly():
    from data_designer.verifiers import llm_judge

    class FakeResp:
        class choices:  # noqa: N801
            class message:  # noqa: N801
                content = "模型拒绝了评分请求"
        choices = [choices]

    class FakeClient:
        def chat(self, messages, temperature=0.0):
            return FakeResp()

    judge = llm_judge.LlmJudge(FakeClient(), model="m")
    verdict = judge_with_rubric("report", {"rubric": "r"}, judge=judge)
    assert verdict["status"] == "error" and "not parsable" in verdict["detail"]


def test_validate_spec_rejects_unknown_kind():
    with pytest.raises(VerifierDispatchError):
        validate_spec({"kind": "nope", "payload": {"a": 1}, "generator_ref": "g"})


# ── S2 taxonomy ──────────────────────────────────────────────────────────────

SEED_GRID = {
    "data-file-ops": {"easy": 24, "medium": 24, "hard": 0},
    "doc-ops": {"easy": 17, "medium": 17, "hard": 0},
    "automation-workdir": {"easy": 9, "medium": 9, "hard": 9},
}


def test_taxonomy_gaps_ordered_by_coverage():
    gaps = taxonomy_gaps(SEED_GRID)
    assert gaps[0] == ("doc-ops", "hard")  # 0-coverage priority
    assert ("data-file-ops", "medium") in gaps


def test_taxonomy_engine_deterministic_and_double_judged():
    e1 = WorkbuddyTaxonomyEngine(seed=7).generate(SEED_GRID, 6)
    e2 = WorkbuddyTaxonomyEngine(seed=7).generate(SEED_GRID, 6)
    assert [t.task_text for t in e1] == [t.task_text for t in e2]
    assert len(e1) == 6
    assert {t.family for t in e1} <= {"invoice_totals", "schedule_conflicts",
                                      "access_audit"}
    for t in e1:
        _double_judge(t)


def test_taxonomy_engine_excludes_previous_fingerprints():
    engine = WorkbuddyTaxonomyEngine(seed=7)
    first = engine.generate(SEED_GRID, 4)
    from data_designer.generation.algorithms.builtin.workbuddy_taxonomy import (
        _fingerprint_of,
    )
    exclude = [_fingerprint_of(t) for t in first]
    second = engine.generate(SEED_GRID, 4, exclude_fingerprints=exclude)
    assert {_fingerprint_of(t) for t in second}.isdisjoint(set(exclude))


# ── S3 failure_probe ─────────────────────────────────────────────────────────


def test_failure_probe_clusters_map_to_families():
    clusters = [
        {"cluster_id": "c-1", "error_category": "assertion"},
        {"cluster_id": "c-2", "error_category": "missing_data", "count": 2},
        {"cluster_id": "c-3", "error_category": "mystery"},  # Fallback family
    ]
    tasks = WorkbuddyFailureProbeEngine(seed=3).generate(clusters)
    assert len(tasks) == 4
    assert {t.family for t in tasks} == {
        "probe_boundary", "probe_missing_data", "probe_rounding"}
    for t in tasks:
        assert t.provenance_extra["strategy"] == "failure_probe"
        _double_judge(t)


def test_failure_probe_requires_cluster_id():
    with pytest.raises(ValueError, match="cluster_id"):
        WorkbuddyFailureProbeEngine(seed=1).generate([{"error_category": "x"}])


def test_failure_probe_deterministic():
    clusters = [{"cluster_id": "c-1", "error_category": "duplicate"}]
    a = WorkbuddyFailureProbeEngine(seed=5).generate(clusters)
    b = WorkbuddyFailureProbeEngine(seed=5).generate(clusters)
    assert [t.task_text for t in a] == [t.task_text for t in b]


# ── Contamination detection ─────────────────────────────────────────────────────────────────


def test_jaccard_similarity_bounds():
    assert jaccard_similarity("汇总各区域销售额", "汇总各区域销售额") == 1.0
    assert jaccard_similarity("", "abc") == 0.0


def test_contamination_report_flags_and_cleans():
    seeds = [{"seed_id": "s-1", "task_text": "汇总各区域销售额并写入 summary.json"}]
    near = {"task_id": "t-1", "task_text":
            "汇总各区域销售额并写入 summary.json（微调）"}
    clean = {"task_id": "t-2", "task_text":
             "审计权限矩阵并输出违规清单到 access_report.json"}
    report = contamination_report([near, clean], seeds, threshold=0.55)
    assert report["checked"] == 2
    assert report["flagged_count"] == 1
    assert report["flagged"][0]["task_id"] == "t-1"
    assert report["flagged"][0]["matched_seed_id"] == "s-1"


def main() -> int:  # Standalone run entry point (same style as test_wbbench_seed_adapter.py)
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if fn.__code__.co_argcount:  # Cases that need fixtures are left to pytest
                continue
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
