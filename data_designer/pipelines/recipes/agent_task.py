"""agent_task recipe: deterministic evaluation-taskset item generator.

RSI E1 contracts require a fixed evaluation taskset (not synthesized in-loop);
this recipe wraps seed task specifications into the taskset item schema consumed
by ``rsi.evaluation.ExternalEvalHarness``.

Input row fields (validated; see ``validate_seed_rows``):
    seed_id              unique seed identifier (required; auto-hash if missing)
    id                   alias for seed_id
    task_text            task description (required, non-empty after strip)
    tools_available      list of tool names the agent may call (optional)
    success_criteria     acceptance description (optional judging basis)
    ground_truth_action_sequence
                         list of expected actions (optional judging basis)
    scoring_rubric       scoring notes: dict or non-empty string (optional judging basis)
    difficulty           free-form difficulty tag (optional)
    source_meta          dict of seed provenance metadata (optional; side ledger only)

At least one judging basis (success_criteria / scoring_rubric /
ground_truth_action_sequence) must be present per row. Unknown fields are
rejected unless declared via ``recipe_config.passthrough_columns``; their
values are kept in the side ledger (``seed_passthrough``), never in the
public item face. Validation is strict by default: all offending rows are
collected and raised once as ``SeedValidationError`` with line numbers.
``recipe_config.strict: false`` restores the legacy silent-drop behavior.

Output row fields (always written, deterministically):
    task_id              canonical id derived from seed_id fingerprint
    task_text
    tools_available
    success_criteria
    ground_truth_action_sequence
    scoring_rubric
    difficulty
    item_fingerprint     stable per-item hash for cross-render equality

This recipe is intentionally pure-Python (no VLM/LLM dependency). It does not
attempt to expand, paraphrase, or judge tasks; it just produces the taskset
items with provenance + audit fields wired by ``BaseSyntheticRecipeExecutor``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from base.dataset import MultimodalDataset
from data_designer.generation.recipes.context import BaseSyntheticRecipeExecutor


AGENT_TASK_SCHEMA_VERSION = "agent-task-v1"

_OUTPUT_FIELDS = (
    "task_id",
    "task_text",
    "tools_available",
    "success_criteria",
    "ground_truth_action_sequence",
    "scoring_rubric",
    "difficulty",
    "item_fingerprint",
)

SEED_ALLOWED_FIELDS = frozenset({
    "seed_id",
    "id",
    "task_text",
    "tools_available",
    "success_criteria",
    "ground_truth_action_sequence",
    "scoring_rubric",
    "difficulty",
    "source_meta",
})

SEED_SIDECAR_SOURCE_META_KEY = "seed_source_meta"
SEED_SIDECAR_PASSTHROUGH_KEY = "seed_passthrough"


class SeedValidationError(ValueError):
    """Aggregated seed-row validation failure (strict mode).

    Carries the full list of per-line error messages in ``errors``; the
    exception message shows the first 20 plus a total count.
    """

    def __init__(self, errors: List[str]):
        shown = errors[:20]
        detail = "; ".join(shown)
        if len(errors) > 20:
            detail += f" (+{len(errors) - 20} more)"
        super().__init__(f"seed validation failed ({len(errors)} error(s)): {detail}")
        self.errors = list(errors)


def _row_seed_id(row: Dict[str, Any]) -> str:
    if row.get("seed_id"):
        return str(row["seed_id"])
    if row.get("id"):
        return str(row["id"])
    return "seed-" + hashlib.sha256(
        json.dumps(row, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


def _canonical_item(
    row: Dict[str, Any],
    passthrough_columns: tuple = (),
) -> Dict[str, Any]:
    seed_id = _row_seed_id(row)
    task_text = str(row.get("task_text") or "").strip()
    tools_available = list(row.get("tools_available") or [])
    success_criteria = str(row.get("success_criteria") or "").strip()
    ground_truth = list(row.get("ground_truth_action_sequence") or [])
    rubric_raw = row.get("scoring_rubric")
    if isinstance(rubric_raw, dict):
        scoring_rubric: Any = rubric_raw
    else:
        scoring_rubric = str(rubric_raw or "").strip()
    difficulty = str(row.get("difficulty") or "unspecified").strip() or "unspecified"

    item_fingerprint_input = {
        "seed_id": seed_id,
        "task_text": task_text,
        "tools_available": tools_available,
        "success_criteria": success_criteria,
        "ground_truth_action_sequence": ground_truth,
        "scoring_rubric": scoring_rubric,
        "difficulty": difficulty,
    }
    item_fingerprint = hashlib.sha256(
        json.dumps(item_fingerprint_input, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()[:16]

    item = {
        "seed_id": seed_id,
        "task_id": f"ts-{item_fingerprint}",
        "task_text": task_text,
        "tools_available": tools_available,
        "success_criteria": success_criteria,
        "ground_truth_action_sequence": ground_truth,
        "scoring_rubric": scoring_rubric,
        "difficulty": difficulty,
        "item_fingerprint": item_fingerprint,
    }

    # Side ledger (D6/D7): never part of _OUTPUT_FIELDS or the public item face.
    source_meta = row.get("source_meta")
    if isinstance(source_meta, dict) and source_meta:
        item[SEED_SIDECAR_SOURCE_META_KEY] = dict(source_meta)
    passthrough = {
        str(col): row[col]
        for col in passthrough_columns
        if col in row and col not in SEED_ALLOWED_FIELDS
    }
    if passthrough:
        item[SEED_SIDECAR_PASSTHROUGH_KEY] = passthrough
    return item


def _judging_basis_present(row: Dict[str, Any]) -> bool:
    if str(row.get("success_criteria") or "").strip():
        return True
    rubric = row.get("scoring_rubric")
    if isinstance(rubric, dict):
        if rubric:
            return True
    elif str(rubric or "").strip():
        return True
    ground_truth = row.get("ground_truth_action_sequence")
    if isinstance(ground_truth, list) and ground_truth:
        return True
    return False


def _validate_strict_row(row: Dict[str, Any], line_no: int, allowed_fields: frozenset) -> List[str]:
    errors: List[str] = []
    unknown = sorted(set(row) - allowed_fields)
    if unknown:
        errors.append(f"line {line_no}: unknown field(s) {', '.join(unknown)}")
    if not str(row.get("task_text") or "").strip():
        errors.append(f"line {line_no}: task_text is required")
    if not _judging_basis_present(row):
        errors.append(
            "line {n}: at least one of success_criteria/scoring_rubric/"
            "ground_truth_action_sequence is required".format(n=line_no)
        )
    tools = row.get("tools_available")
    if tools is not None and (
        not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools)
    ):
        errors.append(f"line {line_no}: tools_available must be a list of strings")
    ground_truth = row.get("ground_truth_action_sequence")
    if ground_truth is not None and not isinstance(ground_truth, list):
        errors.append(f"line {line_no}: ground_truth_action_sequence must be a list")
    rubric = row.get("scoring_rubric")
    if rubric is not None and not (
        isinstance(rubric, dict) or (isinstance(rubric, str) and rubric.strip())
    ):
        errors.append(f"line {line_no}: scoring_rubric must be a dict or a non-empty string")
    difficulty = row.get("difficulty")
    if difficulty is not None and not isinstance(difficulty, str):
        errors.append(f"line {line_no}: difficulty must be a string")
    source_meta = row.get("source_meta")
    if source_meta is not None and not isinstance(source_meta, dict):
        errors.append(f"line {line_no}: source_meta must be a dict")
    return errors


def validate_seed_rows(
    rows: List[Any],
    *,
    strict: bool = True,
    passthrough_columns: tuple = (),
) -> List[Dict[str, Any]]:
    """Validate and canonicalize seed rows into taskset items.

    strict=True (default): every row is checked (object shape, field whitelist
    plus ``passthrough_columns``, required task_text, at least one judging
    basis, field types). All errors across all rows are collected and raised
    once as ``SeedValidationError`` with 1-based line numbers.

    strict=False: legacy behavior — non-dict rows and rows without task_text
    are silently dropped.
    """
    passthrough_columns = tuple(passthrough_columns or ())
    if not strict:
        items: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = _canonical_item(dict(row), passthrough_columns)
            if not item["task_text"]:
                continue
            items.append(item)
        return items

    allowed_fields = SEED_ALLOWED_FIELDS | set(passthrough_columns)
    errors: List[str] = []
    items = []
    for line_no, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            errors.append(f"line {line_no}: row must be an object")
            continue
        errors.extend(_validate_strict_row(row, line_no, allowed_fields))
        items.append(_canonical_item(dict(row), passthrough_columns))
    if errors:
        raise SeedValidationError(errors)
    return items


def build_agent_task_dataset(
    rows: List[Dict[str, Any]],
    *,
    strict: bool = True,
    passthrough_columns: tuple = (),
) -> MultimodalDataset:
    """Construct a MultimodalDataset of pre-canonicalized taskset items.

    Useful when callers want to skip the Ray ``map`` step (tests, deterministic
    pipelines). Input rows are validated and canonicalized through
    ``validate_seed_rows`` (strict aggregation by default).
    """
    try:
        import ray
    except ImportError as e:
        raise RuntimeError(
            "agent_task recipe requires ray to construct MultimodalDataset"
        ) from e
    items = validate_seed_rows(
        rows,
        strict=strict,
        passthrough_columns=passthrough_columns,
    )
    return MultimodalDataset(
        ray.data.from_items(items),
        modalities=["text"],
    )


class AgentTaskRecipeExecutor(BaseSyntheticRecipeExecutor):
    """agent_task recipe executor.

    No LLM/VLM call: seed task rows are validated (strict fail-fast
    aggregation by default; ``recipe_config.strict: false`` restores legacy
    silent-drop) and canonicalized into taskset items, then annotated with
    provenance/quality/audit fields by ``BaseSyntheticRecipeExecutor.annotate_dataset``.
    Extra metadata columns must be declared via ``recipe_config.passthrough_columns``
    and are kept in the side ledger only.
    """

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        if dataset is None:
            raise ValueError("AgentTaskRecipeExecutor requires an input dataset")

        config = self.recipe_config or {}
        items = validate_seed_rows(
            list(dataset.dataset.take_all()),
            strict=bool(config.get("strict", True)),
            passthrough_columns=tuple(config.get("passthrough_columns") or ()),
        )
        if not items:
            return dataset

        keep_rows: List[Dict[str, Any]] = []
        for item in items:
            item["sample_id"] = item["seed_id"]
            keep_rows.append(item)

        import ray
        regenerated_ds = MultimodalDataset(
            ray.data.from_items(keep_rows),
            modalities=["text"],
        )

        def attach(row: Dict[str, Any]) -> Dict[str, Any]:
            row = dict(row)
            row["agent_task"] = {
                "task_id": row.get("task_id"),
                "task_text": row.get("task_text"),
                "tools_available": row.get("tools_available"),
                "success_criteria": row.get("success_criteria"),
                "ground_truth_action_sequence": row.get("ground_truth_action_sequence"),
                "scoring_rubric": row.get("scoring_rubric"),
                "difficulty": row.get("difficulty"),
                "item_fingerprint": row.get("item_fingerprint"),
                "schema_version": AGENT_TASK_SCHEMA_VERSION,
            }
            return row

        annotated_inputs = regenerated_ds.map(attach)
        return self.annotate_dataset(
            annotated_inputs,
            output_keys=list(_OUTPUT_FIELDS),
            modality=["text"],
            generator_model=self.resolve_generator_model(self.recipe_config or {}),
        )