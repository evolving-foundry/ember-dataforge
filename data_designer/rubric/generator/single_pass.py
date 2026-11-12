"""Single-pass evolution job orchestration (migrated from RuACT
refinement/single_pass_worker.py, decoupled from asyncio). Full lifecycle of one job:
structural validation -> per-parent candidate preparation (polarity triage + seed
pool + Generator top-up, `budget.prepare_parent` budgeted retries) -> one Judge grid
for all candidates (8xN, `judge.recovery.judge_candidates` chunked recovery) -> per-
parent discriminative filtering ([2,6], child cap and below_min_children semantics)
-> assemble decomposition and audit. Source policies retained: - Generator receives
only query + parent text and never touches responses or rollout statistics
(`generator_received_rollouts: False`). - Judge receives only query + answer +
candidate description columns (`descriptions_only`). - Candidate budget
CANDIDATE_BUDGET=5, minimum effective candidates MIN_EFFECTIVE_CANDIDATES=5. - Parent
budget exhaustion = the whole job fails explicitly (`RefinementBudgetExhausted`);
after filtering, fewer than min_children_per_parent = skip that parent (never
fabricate success). Deviations from the source (intentional on the inference side): -
asyncio.gather -> ThreadPoolExecutor (parent-level parallelism, width
`parent_workers`). - `RUACT_CRITIC_NO_FILTER` env -> explicit
`keep_constant_children` parameter (same name/meaning as the evolve/registry
constructor parameter), recorded in `child_pass_count_range` audit. -
`HEALTHBENCH_JUDGE_*` env audit read -> explicit `judge_route` parameter (default
None; env values are not recorded; credentials/endpoints never enter the audit).
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from ..judge.recovery import judge_candidates
from ..protocol.schemas import PositiveRefinementProtocolError
from ..protocol.transaction import SCHEMA_VERSION
from .budget import prepare_parent
from .candidates import (
    attach_judge_scores,
    canonical_completion_point,
    generator_prompt,
    generator_schema,
    validate_generated,
    validate_refinement_job,
    validate_seed_candidates,
    reindex_candidates,
    fixed_meta_valid_children,
)
from .client import WorkerConfig, generate_json
from .polarity import triage_polarity

EXECUTION_MODE = "single_pass_batched"
CANDIDATE_BUDGET = 5
MIN_EFFECTIVE_CANDIDATES = 5
PROMPT_NAME = "refinement"


def prepare_parent_candidates(
    job: Mapping[str, Any],
    parent: Mapping[str, Any],
    config: WorkerConfig,
    generator_call,
    *,
    demonstrations: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prepare candidates for one parent (migrated from
    single_pass_worker._prepare_parent). The seed pool is filtered by polarity and
    truncated to CANDIDATE_BUDGET; when it is below the minimum, Generator tops it up
    (the prompt states existing completion points and asks for distinct new ones).
    """
    started = time.monotonic()
    direction, triage = ("positive", None)
    if float(parent["weight"]) < 0:
        direction, triage = triage_polarity(job, parent, config=config)
    polarity = "negative" if direction == "keep_negative" else "positive"
    seeds = [
        candidate
        for candidate in validate_seed_candidates(parent)
        if candidate["polarity"] == polarity
    ]
    pool = seeds[:CANDIDATE_BUDGET]
    generated: list[dict[str, Any]] = []
    generator_meta: dict[str, Any] | None = None
    needed = CANDIDATE_BUDGET - len(pool)
    minimum_needed = max(0, MIN_EFFECTIVE_CANDIDATES - len(pool))
    if minimum_needed:
        prompt_job = {
            **job,
            "min_candidate_count": minimum_needed,
            "max_candidate_count": needed,
        }
        prompt = generator_prompt(
            prompt_job,
            parent,
            demonstrations=demonstrations,
            polarity_direction=direction,
            partial_weight_split=True,
        )
        if pool:
            prompt += (
                "\nExisting completion points; generate distinct additional points:\n"
                + "\n".join(("- " + c["completion_point"] for c in pool))
            )
        generated_value, generator_meta = generator_call(
            prompt,
            generator_schema(minimum_needed, needed),
            config=config,
        )
        generator_meta = {
            **generator_meta,
            "prompt_name": PROMPT_NAME,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_character_count": len(prompt),
            "prompt_utf8_bytes": len(prompt.encode()),
        }
        generated = validate_generated(
            generated_value,
            min_candidate_count=minimum_needed,
            max_candidate_count=needed,
            parent=parent,
            polarity=polarity,
        )
        pool = pool + generated
    pool = [
        {
            **candidate,
            "meta_verification": {
                "decision": "not_performed",
                "validation_scope": "format_and_deduplication",
                "necessity": "optional",
                "semantic_family": canonical_completion_point(candidate),
                "source": (
                    "preverified_library"
                    if candidate["candidate_source"] == "historical_seed"
                    else "online_generation"
                ),
            },
        }
        for candidate in reindex_candidates(pool)
    ]
    pool, duplicates = fixed_meta_valid_children(pool, max_children=CANDIDATE_BUDGET)
    call_audits = [m for m in (generator_meta, triage) if m is not None]
    audit = {
        "parent_rubric_id": str(parent["rubric_id"]),
        "generator": generator_meta,
        "generation_skipped": generator_meta is None,
        "generated_candidate_count": len(generated),
        "seed_candidate_count": len(seeds),
        "preverified_reused_count": sum(
            (c["candidate_source"] == "historical_seed" for c in pool)
        ),
        "seed_meta": None,
        "joint_meta": None,
        "semantic_meta_check_performed": False,
        "polarity_direction": direction,
        "polarity_triage": triage,
        "duplicate_rejections": duplicates,
        "fixed_child_count": len(pool),
        "candidate_budget": CANDIDATE_BUDGET,
        "topup_rounds": 0,
        "generator_call_count": len(call_audits),
        "generator_prompt_character_count": sum(
            (
                int(
                    m.get("prompt_character_count")
                    or m.get("triage_prompt_character_count")
                    or 0
                )
                for m in call_audits
            )
        ),
        "generator_prompt_utf8_bytes": sum(
            (
                int(m.get("prompt_utf8_bytes") or m.get("triage_prompt_utf8_bytes") or 0)
                for m in call_audits
            )
        ),
        "preparation_seconds": time.monotonic() - started,
    }
    return (pool, audit)


def decomposition_of(
    job: Mapping[str, Any], parent: Mapping[str, Any], children: list[dict[str, Any]]
) -> dict[str, Any]:
    """Assemble a decomposition in the registry submission format (migrated from
    single_pass_worker._decomposition).
    """
    return {
        "parent_rubric_id": str(parent["rubric_id"]),
        "parent_fingerprint": str(parent["parent_fingerprint"]),
        "parent_title": str(parent["title"]),
        "parent_description": str(parent["description"]),
        "parent_weight": float(parent["weight"]),
        "blend_lambda": float(job["blend_lambda"]),
        "children": [
            {
                "title": child["title"],
                "description": child["description"],
                "completion_point": child["completion_point"],
                "necessity": child["meta_verification"]["necessity"],
                "semantic_family": child["meta_verification"]["semantic_family"],
                "meta_verification": dict(child["meta_verification"]),
                "judge_scores": child["judge_scores"],
                "pass_count": child["pass_count"],
                "generator_candidate_index": child["candidate_index"],
                "candidate_source": child["candidate_source"],
                "source_candidate_index": child["source_candidate_index"],
                **(
                    {"polarity": child["polarity"]}
                    if float(parent["weight"]) < 0
                    else {}
                ),
                **(
                    {
                        key: child[key]
                        for key in ("source_job_id", "source_step", "source_pass_count")
                    }
                    if child["candidate_source"] == "historical_seed"
                    else {}
                ),
            }
            for child in children
        ],
    }


def process_refinement_job(
    job: Mapping[str, Any],
    *,
    config: WorkerConfig,
    parent_workers: int = 3,
    keep_constant_children: bool = False,
    judge_transport: Any | None = None,
    judge_workers: int | None = None,
    judge_attempts: int | None = None,
    judge_route: Mapping[str, Any] | None = None,
    demonstrations: Any | None = None,
) -> dict[str, Any]:
    """Process one evolution job and return a result dict acceptable as a registry
    `commit_result`. `keep_constant_children` passes through discriminative filtering
    (False = reject constant {0,8} columns, the default protocol policy);
    `demonstrations` may inject a reviewed demonstration library (otherwise use
    built-in few-shot).
    """
    validate_refinement_job(job)
    started = time.monotonic()
    parents = list(job["selected_parents"])
    if len({p["rubric_id"] for p in parents}) != len(parents):
        raise PositiveRefinementProtocolError("selected parent IDs must be unique")

    def prepare(parent: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return prepare_parent(
            job,
            parent,
            config,
            lambda j, p, cfg, call: prepare_parent_candidates(
                j, p, cfg, call, demonstrations=demonstrations
            ),
        )

    worker_count = max(1, min(int(parent_workers), len(parents)))
    if worker_count == 1:
        prepared = [prepare(parent) for parent in parents]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool_executor:
            prepared = list(
                pool_executor.map(prepare, parents)
            )
    flat: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for parent, (pool, _audit) in zip(parents, prepared):
        start = len(flat)
        for candidate in pool:
            mapping.append(
                {
                    "column": len(flat),
                    "parent_rubric_id": parent["rubric_id"],
                    "candidate_index": candidate["candidate_index"],
                }
            )
            flat.append(candidate)
        spans.append((start, len(flat)))
    grid, judge_audit = judge_candidates(
        job,
        flat,
        transport=judge_transport,
        workers=8 if judge_workers is None else int(judge_workers),
        attempts_per_call=judge_attempts,
    )
    if len(grid) != 8 or any((len(row) != len(flat) for row in grid)):
        raise PositiveRefinementProtocolError("batched Judge grid is misaligned")
    response_ids = [str(m["id"]) for m in job["group_members"]]
    decompositions: list[dict[str, Any]] = []
    parent_audits: list[dict[str, Any]] = []
    for parent, (pool, audit), (start, end) in zip(parents, prepared, spans):
        if not pool:
            selected, selection = ([], [])
        else:
            selected, selection = attach_judge_scores(
                pool,
                [row[start:end] for row in grid],
                response_ids=response_ids,
                max_children=CANDIDATE_BUDGET,
                no_filter=keep_constant_children,
            )
        selected = selected[: int(job["max_children_per_parent"])]
        retained_ids = {c["candidate_index"] for c in selected}
        for item in selection:
            if item["retained"] and item["candidate_index"] not in retained_ids:
                item.update(retained=False, rejection_reason="child_cap")
        enough = len(selected) >= int(job["min_children_per_parent"])
        if not enough:
            for item in selection:
                if item["retained"]:
                    item.update(retained=False, rejection_reason="below_min_children")
            selected = []
        audit.update(
            status=(
                "selected"
                if enough
                else audit.get("status", "no_nonconstant_child_pattern")
            ),
            candidate_pass_counts=selection,
            selected_candidate_indices=[c["candidate_index"] for c in selected],
            selected_pass_counts=[c["pass_count"] for c in selected],
            fixed_judge_scope="query_wide",
            judge_columns=[start, end],
            judged_candidates=pool,
        )
        parent_audits.append(audit)
        if enough:
            decompositions.append(decomposition_of(job, parent, selected))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "job_id": str(job["job_id"]),
        "transaction_id": str(job["transaction_id"]),
        "query_id": str(job["query_id"]),
        "source_query_version": int(job["query_version"]),
        "response_set_hash": str(job["response_set_hash"]),
        "seed_bundle_hash": str(job["seed_bundle_hash"]),
        "decompositions": decompositions,
        "audit": {
            "execution_mode": EXECUTION_MODE,
            "candidate_budget_per_parent": CANDIDATE_BUDGET,
            "judge_route": dict(judge_route) if judge_route else {},
            "seed_policy": "exact_parent_first_five_current_response_rescore",
            "selected_parent_count": len(parents),
            "decomposition_count": len(decompositions),
            "selected_child_count": sum(
                (len(d["children"]) for d in decompositions)
            ),
            "parents": parent_audits,
            "parent_workers": min(int(parent_workers), len(parents)),
            "processing_seconds": round(time.monotonic() - started, 3),
            **{
                key: sum((int(a[key]) for a in parent_audits))
                for key in (
                    "generator_call_count",
                    "generator_prompt_character_count",
                    "generator_prompt_utf8_bytes",
                )
            },
            "seed_pre_meta_skipped_count": 0,
            "generator_model": config.model,
            "generator_enable_thinking": config.enable_thinking,
            "meta_model": None,
            "semantic_meta_check_performed": False,
            "fixed_judge_input_mode": "query_answer_descriptions_only",
            "fixed_judge": judge_audit,
            "judge_column_map": mapping,
            "judge_grid": grid,
            "generator_received_rollouts": False,
            "topup_enabled": False,
            "child_pass_count_range": (
                [0, 8] if keep_constant_children else [2, 6]
            ),
            "child_cap_selection": "candidate_order",
            "rollout_pass_rate_filtering": False,
            "constant_children_retained": keep_constant_children,
            "constant_children_rejected": not keep_constant_children,
        },
    }


__all__ = [
    "CANDIDATE_BUDGET",
    "EXECUTION_MODE",
    "MIN_EFFECTIVE_CANDIDATES",
    "decomposition_of",
    "prepare_parent_candidates",
    "process_refinement_job",
]
