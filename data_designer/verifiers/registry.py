from importlib import import_module

IMPLEMENTATIONS = {
    "artifact_compare": "data_designer.verifiers.artifact_compare:compare_artifact",
    "dom_assert": "data_designer.verifiers.dom_assert:assert_dom",
    "unit_test": "data_designer.verifiers.unit_test:run_unit_tests",
    "llm_rubric": "data_designer.verifiers.llm_judge:judge_with_rubric",
    "rubric_critic": "data_designer.rubric.verifier_bridge:run_rubric_critic",
    "distribution_check": "data_designer.verifiers.distribution_check:check_distribution",
}


def load_verifier(kind):
    module, name = IMPLEMENTATIONS[kind].split(":")
    return getattr(import_module(module), name)
