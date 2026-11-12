"""Explicit verifier dispatch + lazy convenience exports.

Dispatch entry ``run_verifier(spec, artifact, judge=None)``: routes to the verifier implementation registered in the registry
according to ``spec["kind"]``. All five task verifiers are implemented:
``artifact_compare`` (Office structured artifacts), ``dom_assert`` (static HTML
assertions), ``unit_test`` (controlled-workspace assertion script), ``llm_rubric`` (LLM
judge; injected on the grading side), ``distribution_check`` (distribution gate).
Convenience symbols are resolved lazily through ``__getattr__``; importing this package does not load heavy dependencies.
"""

from __future__ import annotations

from importlib import import_module

from .dispatch import (
    SUPPORTED_VERIFIER_KINDS,
    VerifierDispatchError,
    run_verifier,
    validate_spec,
)

_EXPORTS = {
    "compare_artifact": ("artifact_compare", "compare_artifact"),
    "assert_dom": ("dom_assert", "assert_dom"),
    "run_unit_tests": ("unit_test", "run_unit_tests"),
    "check_distribution": ("distribution_check", "check_distribution"),
    "judge_with_rubric": ("llm_judge", "judge_with_rubric"),
    "LlmJudge": ("llm_judge", "LlmJudge"),
    "LlmJudgeConfigError": ("llm_judge", "LlmJudgeConfigError"),
    "JUDGE_PROMPT_VERSION": ("llm_judge", "JUDGE_PROMPT_VERSION"),
    "DEFAULT_PASS_THRESHOLD": ("llm_judge", "DEFAULT_PASS_THRESHOLD"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    return getattr(import_module(f"{__name__}.{module}"), symbol)
