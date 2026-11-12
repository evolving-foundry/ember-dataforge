"""C3 probes: agent_task seed fail-fast validation (design document: evaluation-
extension capability completion, section 5.7).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from data_designer.pipelines.recipes.agent_task import (
    SEED_SIDECAR_PASSTHROUGH_KEY,
    SEED_SIDECAR_SOURCE_META_KEY,
    AgentTaskRecipeExecutor,
    SeedValidationError,
    validate_seed_rows,
)


def _legacy_fingerprint(seed_id, task_text, tools_available, success_criteria,
                        ground_truth, scoring_rubric, difficulty):
    # Byte-for-byte reimplementation of the pre-C3 fingerprint algorithm.
    payload = json.dumps(
        {
            "seed_id": seed_id,
            "task_text": task_text,
            "tools_available": tools_available,
            "success_criteria": success_criteria,
            "ground_truth_action_sequence": ground_truth,
            "scoring_rubric": scoring_rubric,
            "difficulty": difficulty,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


PUBLIC_KEYS = {
    "seed_id", "task_id", "task_text", "tools_available", "success_criteria",
    "ground_truth_action_sequence", "scoring_rubric", "difficulty",
    "item_fingerprint",
}


def test_minimal_row_passes_and_fingerprint_unchanged():
    items = validate_seed_rows([{"id": "seed-1", "task_text": "  Fix the report  ",
                                 "success_criteria": "report rebuilt"}])
    assert len(items) == 1
    item = items[0]
    assert item["seed_id"] == "seed-1"
    assert item["task_text"] == "Fix the report"
    expected = _legacy_fingerprint("seed-1", "Fix the report", [], "report rebuilt",
                                   [], "", "unspecified")
    assert item["item_fingerprint"] == expected
    assert item["task_id"] == f"ts-{expected}"


def test_full_row_fingerprint_unchanged():
    row = {
        "seed_id": "s-2",
        "task_text": "Do it",
        "tools_available": ["bash", "edit"],
        "success_criteria": "tests green",
        "ground_truth_action_sequence": ["run_tests"],
        "scoring_rubric": "graded on tests",
        "difficulty": "hard",
    }
    item = validate_seed_rows([row])[0]
    assert item["item_fingerprint"] == _legacy_fingerprint(
        "s-2", "Do it", ["bash", "edit"], "tests green", ["run_tests"],
        "graded on tests", "hard")


def test_empty_task_text_fails_with_line_number():
    with pytest.raises(SeedValidationError) as exc:
        validate_seed_rows([{"id": "a", "success_criteria": "c"}, {"task_text": "   "}])
    message = str(exc.value)
    assert "3 error(s)" in message
    assert "line 1: task_text is required" in message
    assert "line 2: task_text is required" in message
    assert "line 2: at least one of success_criteria/scoring_rubric/" \
           "ground_truth_action_sequence is required" in message
    assert exc.value.errors == [
        "line 1: task_text is required",
        "line 2: task_text is required",
        "line 2: at least one of success_criteria/scoring_rubric/"
        "ground_truth_action_sequence is required",
    ]


def test_judging_basis_required_and_dict_rubric_passes():
    with pytest.raises(SeedValidationError) as exc:
        validate_seed_rows([{"task_text": "no basis"}])
    assert "at least one of success_criteria/scoring_rubric/ground_truth_action_sequence is required" in str(exc.value)

    rubric = {"weights": {"correctness": 1.0}}
    item = validate_seed_rows([{"task_text": "rubric only", "scoring_rubric": rubric}])[0]
    assert item["scoring_rubric"] == rubric


def test_unknown_field_rejected_and_passthrough_goes_to_side_ledger():
    row = {"task_text": "t", "success_criteria": "c", "wb_keywords": ["alpha"]}
    with pytest.raises(SeedValidationError) as exc:
        validate_seed_rows([row])
    assert "line 1: unknown field(s) wb_keywords" in str(exc.value)

    item = validate_seed_rows([row], passthrough_columns=("wb_keywords",))[0]
    assert "wb_keywords" not in item
    assert item[SEED_SIDECAR_PASSTHROUGH_KEY] == {"wb_keywords": ["alpha"]}
    assert set(item) - PUBLIC_KEYS == {SEED_SIDECAR_PASSTHROUGH_KEY}


def test_type_validation_errors():
    rows = [
        {"task_text": "t", "success_criteria": "c", "tools_available": "bash"},
        {"task_text": "t", "success_criteria": "c", "tools_available": [1]},
        {"task_text": "t", "success_criteria": "c", "ground_truth_action_sequence": "run"},
        {"task_text": "t", "success_criteria": "c", "scoring_rubric": ""},
        {"task_text": "t", "success_criteria": "c", "difficulty": 3},
        {"task_text": "t", "success_criteria": "c", "source_meta": ["x"]},
    ]
    with pytest.raises(SeedValidationError) as exc:
        validate_seed_rows(rows)
    assert exc.value.errors == [
        "line 1: tools_available must be a list of strings",
        "line 2: tools_available must be a list of strings",
        "line 3: ground_truth_action_sequence must be a list",
        "line 4: scoring_rubric must be a dict or a non-empty string",
        "line 5: difficulty must be a string",
        "line 6: source_meta must be a dict",
    ]


def test_errors_aggregated_across_rows_and_capped_display():
    rows = [{"task_text": "t", "success_criteria": "c", "wb_bad": i} for i in range(25)]
    with pytest.raises(SeedValidationError) as exc:
        validate_seed_rows(rows)
    assert len(exc.value.errors) == 25
    assert "seed validation failed (25 error(s))" in str(exc.value)
    assert "(+5 more)" in str(exc.value)
    assert all(f"line {n}: unknown field(s) wb_bad" in err
               for n, err in enumerate(exc.value.errors, 1))
    assert exc.value.errors[:20] == [
        f"line {n}: unknown field(s) wb_bad" for n in range(1, 21)
    ]


def test_non_object_row_rejected():
    with pytest.raises(SeedValidationError) as exc:
        validate_seed_rows(["not a dict"])
    assert exc.value.errors == ["line 1: row must be an object"]


def test_source_meta_side_ledger_only_and_fingerprint_neutral():
    base = {"seed_id": "s-1", "task_text": "t", "success_criteria": "c"}
    with_meta = {**base, "source_meta": {"wb_subset": "office"}}
    plain = validate_seed_rows([base])[0]
    meta = validate_seed_rows([with_meta])[0]
    assert meta[SEED_SIDECAR_SOURCE_META_KEY] == {"wb_subset": "office"}
    assert "source_meta" not in meta
    assert meta["item_fingerprint"] == plain["item_fingerprint"]

    with pytest.raises(SeedValidationError) as exc:
        validate_seed_rows([{**base, "source_meta": "bad"}])
    assert "line 1: source_meta must be a dict" in str(exc.value)


def test_strict_false_restores_legacy_silent_drop():
    rows = [
        {"task_text": "", "wb_keywords": "k"},
        "not a dict",
        {"task_text": "keep me", "success_criteria": "c", "wb_keywords": "v"},
    ]
    items = validate_seed_rows(rows, strict=False, passthrough_columns=("wb_keywords",))
    assert len(items) == 1
    assert items[0]["task_text"] == "keep me"
    assert items[0][SEED_SIDECAR_PASSTHROUGH_KEY] == {"wb_keywords": "v"}


def _executor(config=None):
    from data_designer.integrations.curator.recipes import RecipeContext

    return AgentTaskRecipeExecutor(
        RecipeContext("agent_task", dict(config or {}), dict(config or {}))
    )


def _dataset(rows):
    import ray
    from base.dataset import MultimodalDataset

    return MultimodalDataset(ray.data.from_items(rows), modalities=["text"])


def test_executor_strict_default_raises():
    executor = _executor()
    with pytest.raises(SeedValidationError):
        executor.execute(_dataset([{"task_text": ""}]))


def test_executor_strict_false_drops_and_passthrough_sidecar_stays_off_public_face():
    executor = _executor({"strict": False, "passthrough_columns": ["wb_keywords"]})
    result = executor.execute(_dataset([
        {"id": "a", "task_text": "", "wb_keywords": "k"},
        {"id": "b", "task_text": "kept", "success_criteria": "c", "wb_keywords": "v"},
    ]))
    rows = result.dataset.take_all()
    assert len(rows) == 1
    row = next(r for r in rows if r.get("task_id"))
    public = row["agent_task"]
    assert public["task_text"] == "kept"
    assert "wb_keywords" not in public
    assert row[SEED_SIDECAR_PASSTHROUGH_KEY] == {"wb_keywords": "v"}
    assert row["sample_id"] == "b"
