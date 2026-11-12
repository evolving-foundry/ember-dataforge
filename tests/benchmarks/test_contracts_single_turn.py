"""Contract tests for the four single-turn families (M1): public item round-trip, family prefixes, and leak blocking.

Covers the NeMo/HF single-turn evaluation onboarding implementation plan §2 M1 acceptance:
  - ``reasoning_qa`` / ``knowledge_qa`` / ``code_gen`` / ``health_qa`` are registered as
    the ninth through twelfth families;
  - public items from all four families pass ``make_public_item`` / ``assert_public_item``;
  - task_ids without the matching prefix are rejected;
  - ``expected`` / ``aliases`` / ``rubric`` payloads are rejected as soon as they enter a public item (D5:
    expected answers/alias lists/full rubric text remain only in the trusted case manifest).
"""

import pytest

from data_designer.pipelines.benchmarks.contracts import (
    FAMILIES,
    BenchmarkError,
    assert_public_item,
    make_public_item,
    require_family,
)

SINGLE_TURN_FAMILIES = ("reasoning_qa", "knowledge_qa", "code_gen", "health_qa")


def _public_fields(family: str, benchmark: str) -> dict:
    return {
        "task_id": f"{family}:{benchmark}:row-0:v1",
        "task_text": "Answer the question with a single final response written "
                     "to response.txt.",
        "tools_available": [],
        "success_criteria": "The delivered response matches the frozen "
                            "verification semantics.",
        "difficulty": "medium",
        "family": family,
        "source_release": "hf-synthetic-fixture:abcd1234",
        "source_task_id": f"{benchmark}:row-0",
        "variant_id": "v1",
    }


def test_single_turn_families_are_registered_in_order():
    assert list(FAMILIES[-4:]) == list(SINGLE_TURN_FAMILIES)
    for family in SINGLE_TURN_FAMILIES:
        assert require_family(family) == family


@pytest.mark.parametrize("family,benchmark", [
    ("reasoning_qa", "gsm8k"),
    ("knowledge_qa", "mmlu"),
    ("code_gen", "humaneval"),
    ("health_qa", "healthbench"),
])
def test_single_turn_public_item_roundtrip(family: str, benchmark: str):
    item = make_public_item(
        _public_fields(family, benchmark), "a" * 64, "b" * 64, "c" * 64
    )
    assert set(item) == {
        "task_id", "task_text", "tools_available", "success_criteria",
        "difficulty", "item_fingerprint", "benchmark",
    }
    assert item["benchmark"]["family"] == family
    assert item["task_id"].startswith(f"{family}:")
    assert_public_item(item)


@pytest.mark.parametrize("family,benchmark", [
    ("reasoning_qa", "gsm8k"),
    ("knowledge_qa", "mmlu"),
    ("code_gen", "humaneval"),
    ("health_qa", "healthbench"),
])
def test_cross_family_task_id_prefix_is_rejected(family: str, benchmark: str):
    fields = _public_fields(family, benchmark)
    fields["task_id"] = f"terminal:{benchmark}:row-0:v1"
    with pytest.raises(BenchmarkError, match="family namespace"):
        make_public_item(fields, "a" * 64, "b" * 64, "c" * 64)


def test_expected_answer_payload_is_rejected():
    item = make_public_item(
        _public_fields("reasoning_qa", "gsm8k"), "a" * 64, "b" * 64, "c" * 64
    )
    # The exact PUBLIC_KEYS check rejects unknown keys first; "expected" in
    # PRIVATE_MARKERS keeps the leak class named at the contract layer.
    with pytest.raises(BenchmarkError, match="public case|private field"):
        assert_public_item({**item, "expected": "42"})


def test_alias_table_payload_is_rejected():
    item = make_public_item(
        _public_fields("knowledge_qa", "triviaqa"), "a" * 64, "b" * 64, "c" * 64
    )
    with pytest.raises(BenchmarkError, match="public case|private field"):
        assert_public_item(
            {**item, "aliases": ["answer one", "the answer is one"]}
        )


def test_rubric_payload_is_rejected():
    item = make_public_item(
        _public_fields("health_qa", "healthbench"), "a" * 64, "b" * 64, "c" * 64
    )
    with pytest.raises(BenchmarkError, match="public case|private field"):
        assert_public_item({**item, "rubric": "criterion one; criterion two"})


def test_hidden_test_code_for_code_gen_is_rejected():
    item = make_public_item(
        _public_fields("code_gen", "humaneval"), "a" * 64, "b" * 64, "c" * 64
    )
    with pytest.raises(BenchmarkError):
        assert_public_item({**item, "test_code": "assert candidate() == 1"})
