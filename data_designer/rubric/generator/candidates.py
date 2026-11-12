"""Candidate generation primitives (migrated from RuACT refinement/worker.py, decoupled
from asyncio/global state). Source policies retained: - Generator receives only query
+ one parent (`generator_input_mode` fixed to `query_and_parent_only`), and the
prompt explicitly states that response text is unavailable. - Structural validation:
candidate count, non-empty fields, canonical deduplication (title+completion_point).
- Child description is its atomic completion point (the registry enforces
consistency). - Discriminative filtering replaces the source `RUACT_CRITIC_NO_FILTER`
environment variable: filter bounds are passed explicitly by the caller
(:func:`data_designer.rubric.orchestrate.group.discriminative_filter_range`, default
[2,6] for n=8); `no_filter=True` is an explicit escape hatch recorded in the audit.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..protocol.schemas import PositiveRefinementProtocolError
from ..protocol.transaction import (
    BLEND_LAMBDA,
    REWARD_MODE,
    SCHEMA_VERSION,
    seed_bundle_hash,
)

GENERATOR_CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "completion_point"],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "completion_point": {"type": "string", "minLength": 1},
    },
}

GENERATOR_FEW_SHOT = 'Few-shot demonstrations (format and specificity only; do not copy their medical content):\n\nExample 1\nQuery: A patient has acute chest pain. What is the initial evaluation?\nParent rubric: Identifies time-sensitive evaluation for life-threatening cardiac causes.\nValid output:\n{"candidates":[\n  {"title":"Serial ischemia testing","completion_point":"States that serial ECG and troponin assessment are needed when acute coronary syndrome remains a concern."},\n  {"title":"Immediate instability check","completion_point":"Checks for hemodynamic instability before proceeding with routine evaluation."}\n]}\n\nExample 2\nQuery: How should suspected bacterial meningitis be managed initially?\nParent rubric: Recommends prompt empiric treatment for suspected bacterial meningitis.\nValid output:\n{"candidates":[\n  {"title":"Do not delay treatment","completion_point":"States that empiric antibiotics should not be delayed when lumbar puncture or imaging would postpone treatment."},\n  {"title":"Blood cultures first when feasible","completion_point":"Requests blood cultures before antibiotics when this can be done without delaying treatment."}\n]}\n\nExample 3\nQuery: What discharge advice is appropriate after a new medication is started?\nParent rubric: Provides safe, actionable follow-up advice.\nValid output:\n{"candidates":[\n  {"title":"Specific warning signs","completion_point":"Names concrete symptoms that require urgent reassessment rather than giving only generic follow-up advice."},\n  {"title":"Defined follow-up timing","completion_point":"Specifies when and with whom the patient should follow up."}\n]}\n\n'

REFINEMENT_JOB_RESPONSE_COUNT = 8


def generator_schema(min_count: int, max_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": min_count,
                "maxItems": max_count,
                "items": GENERATOR_CANDIDATE_SCHEMA,
            }
        },
    }


def generator_prompt(
    job: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    demonstrations: Any | None = None,
    rejection_feedback: Sequence[str] | None = None,
    polarity_direction: str = "positive",
    partial_weight_split: bool = False,
) -> str:
    """Build the decomposition prompt: demonstrations (data, not instructions) + parent
    + polarity/rejection feedback block. `partial_weight_split` replaces the source
    single_pass_worker.py prompt-string surgery: when there are fewer candidates than
    the original budget, the closing wording changes to 'fixed share' rather than
    'split the full budget evenly'.
    """
    if demonstrations is not None:
        examples = demonstrations.prompt(job["query"])
    else:
        examples = GENERATOR_FEW_SHOT
    min_candidate_count = int(job["min_candidate_count"])
    max_candidate_count = int(job["max_candidate_count"])
    feedback_block = ""
    if rejection_feedback:
        points = "\n".join((f"- {point}" for point in rejection_feedback))
        feedback_block = f"Previously rejected completion points (do NOT repeat these or any semantic equivalent; generate completion points on a DIFFERENT refinement operator or dimension):\n{points}\n\n"
    polarity_block = ""
    if polarity_direction == "convert_to_positive":
        polarity_block = '\nParent polarity: this rubric penalizes answers for MISSING one positive behavior (a negation shell such as "Does not mention / Does not state / Without"). Flip the shell and decompose positively: every candidate must be a POSITIVE completion point that specifies what a strong answer should state or do to cover the missing behavior. A fixed Judge will score each candidate as satisfied(1) or not(0) from the query and one answer.\n'
    elif polarity_direction == "keep_negative":
        polarity_block = '\nParent polarity: this rubric describes an ERRONEOUS action or claim that an answer commits (for example "Recommends wrong / Incorrectly states / Suggests a contraindicated"). Decompose the error negatively: every candidate must name ONE concrete sub-error or wrong variant of the parent mistake, described in the same observable direction (the error itself, not a prohibition, and never a "Does not mention" shell). A fixed Judge will score each candidate as committed(1) or not(0) from the query and one answer.\n'
    prompt = f"You are refining one rubric for a RaR training example.\n\n{examples}Now solve the target example. Follow the demonstrated JSON shape, but derive every completion point only from the target query and parent rubric.\n\nUser query:\n{job['query']}\n\nParent rubric:\nTitle: {parent['title']}\nDescription: {parent['description']}\n{polarity_block}\n{feedback_block}Generate between {min_candidate_count} and {max_candidate_count} atomic completion-point candidates. Do not use, request, infer, or mention\nany model responses or rollout statistics. The response texts are deliberately unavailable.\n\nFor every candidate:\n- Stay inside the exact objective and direction of the parent rubric.\n- Express exactly one objective, factually defensible, query-relevant detail that a stronger answer\n  should include to complete the parent requirement.\n- The point may operationalize or add detail to the parent, but must not introduce a different\n  evaluation objective.\n- Prefer distinct refinement operators when appropriate, such as mechanism, evidence, boundary\n  condition, consequence, management implication, operationalization, or a concrete example.\n- Keep the requirement atomic and binary-judgeable from the query and one answer.\n- Do not add vague demands such as being more comprehensive, detailed, clear, or accurate.\n- Do not introduce unrelated topics, style preferences, response IDs, weights, polarity labels,\n  scoring rules, or references to the parent.\n- Make the candidates semantically distinct, not paraphrases of one another.\n- Match the language of the parent rubric.\n\nReturn only a short standalone title and the single completion_point for each candidate. Accepted\ncompletion points will be used together as a fixed checklist alongside the retained parent. "
    closing = (
        "The selected checklist shares a fixed portion of the parent's original weight budget."
        if partial_weight_split
        else "The parent\nand the entire checklist split the parent's original weight budget equally."
    )
    return prompt + closing + "\n"


def compose_child_description(parent_description: str, completion_point: str) -> str:
    del parent_description
    return completion_point.strip()


def canonical_completion_point(candidate: Mapping[str, Any]) -> str:
    return " ".join(str(candidate.get("completion_point") or "").casefold().split())


def validate_generated(
    value: Mapping[str, Any],
    *,
    min_candidate_count: int,
    max_candidate_count: int,
    parent: Mapping[str, Any],
    polarity: str = "positive",
) -> list[dict[str, Any]]:
    candidates = value.get("candidates")
    if (
        not isinstance(candidates, list)
        or not min_candidate_count <= len(candidates) <= max_candidate_count
    ):
        raise PositiveRefinementProtocolError(
            f"generator must return between the requested minimum and maximum candidate count; expected={min_candidate_count}..{max_candidate_count}, observed={(len(candidates) if isinstance(candidates, list) else type(candidates).__name__)}"
        )
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise PositiveRefinementProtocolError(
                "generator candidate must be an object"
            )
        title = str(raw.get("title") or "").strip()
        completion_point = str(raw.get("completion_point") or "").strip()
        if not title or not completion_point:
            raise PositiveRefinementProtocolError(
                "generator title and completion point must be non-empty"
            )
        key = (
            " ".join(title.casefold().split()),
            " ".join(completion_point.casefold().split()),
        )
        if key in seen:
            raise PositiveRefinementProtocolError(
                "generator returned duplicate refinement candidates"
            )
        seen.add(key)
        validated.append(
            {
                "candidate_index": index,
                "title": title,
                "completion_point": completion_point,
                "description": compose_child_description(
                    str(parent["description"]), completion_point
                ),
                "polarity": polarity,
                "candidate_source": "online_generation",
                "source_candidate_index": index,
            }
        )
    return validated


def validate_seed_candidates(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(parent.get("seed_candidates") or ()):
        title = str(raw.get("title") or "").strip()
        completion_point = str(raw.get("completion_point") or "").strip()
        source_job_id = str(raw.get("source_job_id") or "").strip()
        if not title or not completion_point or (not source_job_id):
            raise PositiveRefinementProtocolError(
                "historical seed candidate is incomplete"
            )
        raw_polarity = raw.get("polarity")
        seed_polarity = (
            str(raw_polarity)
            if raw_polarity in {"positive", "negative"}
            else "positive"
        )
        candidates.append(
            {
                "candidate_index": index,
                "title": title,
                "completion_point": completion_point,
                "description": compose_child_description(
                    str(parent["description"]), completion_point
                ),
                "polarity": seed_polarity,
                "candidate_source": "historical_seed",
                "source_candidate_index": index,
                "source_job_id": source_job_id,
                "source_step": int(raw.get("source_step", 0)),
                "source_pass_count": int(raw.get("source_pass_count", -1)),
            }
        )
    return candidates


def reindex_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {**dict(candidate), "candidate_index": index}
        for index, candidate in enumerate(candidates)
    ]


def attach_judge_scores(
    candidates: Sequence[Mapping[str, Any]],
    grid: Sequence[Sequence[int]],
    *,
    response_ids: Sequence[str],
    max_children: int,
    filter_low: int = 2,
    filter_high: int = 6,
    no_filter: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Backfill candidate Judge grids by column and apply discriminative filtering
    (default [2,6] for n=8).
    """
    if len(grid) != len(response_ids) or any(
        (len(row) != len(candidates) for row in grid)
    ):
        raise PositiveRefinementProtocolError("candidate Judge grid is misaligned")
    if len(candidates) > max_children:
        raise PositiveRefinementProtocolError(
            "fixed child set exceeds the configured cap"
        )
    group_size = len(response_ids)
    if not 0 <= filter_low <= filter_high <= group_size:
        raise PositiveRefinementProtocolError(
            "discriminative filter bounds are invalid for the group size"
        )
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for column, raw_candidate in enumerate(candidates):
        candidate = dict(raw_candidate)
        pattern = [int(grid[row][column]) for row in range(group_size)]
        if any((value not in (0, 1) for value in pattern)):
            raise PositiveRefinementProtocolError(
                "candidate Judge returned a non-binary score"
            )
        pass_count = sum(pattern)
        constant = pass_count in {0, group_size}
        weak = not constant and (pass_count < filter_low or pass_count > filter_high)
        if no_filter:
            rejection_reason = None
        elif constant:
            rejection_reason = (
                "zero_pass_pattern" if pass_count == 0 else "full_pass_pattern"
            )
        elif weak:
            rejection_reason = "weak_imbalanced_pattern"
        else:
            rejection_reason = None
        audit.append(
            {
                "candidate_index": int(candidate["candidate_index"]),
                "pass_count": pass_count,
                "pass_rate": pass_count / group_size,
                "is_constant": constant,
                "is_weak": weak,
                "retained": no_filter or (not constant and (not weak)),
                "rejection_reason": rejection_reason,
            }
        )
        candidate["pass_count"] = pass_count
        candidate["judge_scores"] = [
            {"response_id": response_id, "score": pattern[index]}
            for index, response_id in enumerate(response_ids)
        ]
        if no_filter or (not constant and (not weak)):
            selected.append(candidate)
    return (selected, audit)


def fixed_meta_valid_children(
    candidates: Sequence[Mapping[str, Any]], *, max_children: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    seen_completion_points: set[str] = set()
    rejected_duplicates: list[dict[str, Any]] = []
    for raw in sorted(candidates, key=lambda item: int(item["candidate_index"])):
        candidate = dict(raw)
        completion_point = canonical_completion_point(candidate)
        if not completion_point:
            raise PositiveRefinementProtocolError(
                "Meta-valid child has an empty completion point"
            )
        if completion_point in seen_completion_points:
            rejected_duplicates.append(
                {
                    "candidate_index": int(candidate["candidate_index"]),
                    "candidate_source": str(candidate["candidate_source"]),
                    "reason": "canonical_completion_point_duplicate",
                }
            )
            continue
        seen_completion_points.add(completion_point)
        selected.append(candidate)
        if len(selected) == max_children:
            break
    return (selected, rejected_duplicates)


def validate_refinement_job(
    job: Mapping[str, Any], *, selection_mode: str = "saturated"
) -> None:
    """Validate evolution job structure (port of source worker.validate_refinement_job).
    `selection_mode` replaces the source `RUACT_CRITIC_SELECTION` environment
    variable: saturated selected parents must have pass_count 8, unreachable must
    have 0.
    """
    if job.get("schema_version") != SCHEMA_VERSION:
        raise PositiveRefinementProtocolError("unsupported refinement job schema")
    for field in (
        "job_id",
        "transaction_id",
        "query_id",
        "query",
        "group_id",
        "response_set_hash",
    ):
        if not isinstance(job.get(field), str) or not str(job[field]).strip():
            raise PositiveRefinementProtocolError(f"job.{field} must be non-empty")
    if job["job_id"] != job["transaction_id"]:
        raise PositiveRefinementProtocolError(
            "job id and lightweight transaction id must match"
        )
    include_negative = bool(job.get("include_negative_weight"))
    if job.get("gate_mode") not in {
        "saturated_positive_weight_fraction",
        "saturated_signed_weight_fraction",
    }:
        raise PositiveRefinementProtocolError("unsupported refinement gate")
    if job.get("generator_input_mode") != "query_and_parent_only":
        raise PositiveRefinementProtocolError(
            "generator must receive only the query and one parent"
        )
    if job.get("reward_mode") != REWARD_MODE:
        raise PositiveRefinementProtocolError(
            "unsupported completion decomposition reward mode"
        )
    if job.get("child_activation_mode") != "all_meta_valid_fixed_weights":
        raise PositiveRefinementProtocolError(
            "children must be activated independently of rollout pass rates"
        )
    if job.get("rollout_pass_rate_filtering") is not False:
        raise PositiveRefinementProtocolError(
            "rollout pass-rate filtering must be disabled"
        )
    members = job.get("group_members")
    if not isinstance(members, list) or len(members) != REFINEMENT_JOB_RESPONSE_COUNT:
        raise PositiveRefinementProtocolError(
            "refinement job must contain exactly eight responses"
        )
    rollout_indices = [int(item["rollout_index"]) for item in members]
    response_ids = [str(item["id"]) for item in members]
    if (
        rollout_indices != list(range(REFINEMENT_JOB_RESPONSE_COUNT))
        or len(set(response_ids)) != REFINEMENT_JOB_RESPONSE_COUNT
    ):
        raise PositiveRefinementProtocolError(
            "refinement responses must be unique and in rollout order"
        )
    expected_pass_count = 0 if selection_mode == "unreachable" else REFINEMENT_JOB_RESPONSE_COUNT
    if selection_mode not in {"saturated", "unreachable"}:
        raise PositiveRefinementProtocolError(
            "selection mode must be saturated or unreachable"
        )
    parents = job.get("selected_parents")
    if not isinstance(parents, list) or not 1 <= len(parents) <= 3:
        raise PositiveRefinementProtocolError(
            "refinement job must contain one to three parents"
        )
    for parent in parents:
        parent_weight = float(parent.get("weight", 0.0))
        valid_weight = parent_weight > 0.0 or (include_negative and parent_weight < 0.0)
        if (
            not isinstance(parent, Mapping)
            or not valid_weight
            or int(parent.get("pass_count", -1)) != expected_pass_count
        ):
            raise PositiveRefinementProtocolError(
                "every selected parent must match the configured selection and weight polarity"
            )
        for field in ("rubric_id", "title", "description", "parent_fingerprint"):
            if not isinstance(parent.get(field), str) or not str(parent[field]).strip():
                raise PositiveRefinementProtocolError(
                    f"parent.{field} must be non-empty"
                )
        seeds = parent.get("seed_candidates")
        if not isinstance(seeds, list):
            raise PositiveRefinementProtocolError(
                "parent.seed_candidates must be a list"
            )
        seen_seed_points: set[str] = set()
        for seed in seeds:
            if not isinstance(seed, Mapping):
                raise PositiveRefinementProtocolError(
                    "seed candidate must be an object"
                )
            for field in ("title", "completion_point", "source_job_id"):
                if not isinstance(seed.get(field), str) or not str(seed[field]).strip():
                    raise PositiveRefinementProtocolError(
                        f"seed.{field} must be non-empty"
                    )
            canonical = " ".join(str(seed["completion_point"]).casefold().split())
            if canonical in seen_seed_points:
                raise PositiveRefinementProtocolError(
                    "parent contains duplicate seed completion points"
                )
            seen_seed_points.add(canonical)
    if job.get("seed_bundle_hash") != seed_bundle_hash(parents):
        raise PositiveRefinementProtocolError("refinement seed bundle hash is invalid")
    min_candidate_count = int(job.get("min_candidate_count", 0))
    max_candidate_count = int(job.get("max_candidate_count", 0))
    if not 1 <= min_candidate_count <= max_candidate_count:
        raise PositiveRefinementProtocolError(
            "refinement candidate bounds must satisfy 1 <= min <= max"
        )
    child_cap = int(job.get("max_children_per_parent", 0))
    child_min = int(job.get("min_children_per_parent", 0))
    if not 1 <= child_min <= child_cap <= 3:
        raise PositiveRefinementProtocolError(
            "child bounds must satisfy 1 <= min <= max <= 3"
        )
    blend_lambda = float(job.get("blend_lambda", 0.0))
    if blend_lambda != BLEND_LAMBDA:
        raise PositiveRefinementProtocolError(
            f"parent-preserved refinement requires blend_lambda={BLEND_LAMBDA:g}"
        )
    group_std = float(job.get("group_reward_std", -1.0))
    epsilon = float(job.get("zero_std_epsilon", -1.0))
    if group_std < 0.0 or epsilon < 0.0:
        raise PositiveRefinementProtocolError(
            "zero-standard-deviation gate metadata is invalid"
        )
    if bool(job.get("zero_std_gate")) != (group_std <= epsilon):
        raise PositiveRefinementProtocolError(
            "zero-standard-deviation gate metadata is inconsistent"
        )
