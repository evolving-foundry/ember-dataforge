"""Single-round rubric evolution orchestration (design document section 6.3):
`run_evolution_round` is the single entry point. Encapsulates the manual composition
order of P3 components (see tests/rubric/test_evolve_e2e.py) into one call:
`build_refinement_jobs -> reserve_job -> process_refinement_job -> commit_result ->
overlay_decompositions -> apply_decompositions_to_scores`, producing the section 6.3
contract `evolution_result.json` (status / selected_parents / decompositions /
skipped_parents / audit). Discipline: - fail explicitly: any job-processing exception
such as budget exhaustion or Judge interruption first calls `cancel_job` to roll back
pending and then re-raises unchanged; **never write a fabricated success result
file** (a fully committed decomposition is a legitimate unit and is retained); -
result files go through `atomic_write_json_noreplace`: round artifacts are create-
only; never overwrite (idempotent drills depend on this semantics); - audit carries
the rubric-library version write-back chain: each query's before/after overlay
`rubric_library_sha256`; - re-validate the overlaid library with `validate_rubrics`;
never produce an unscoreable library.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..io.atomic import atomic_write_json_noreplace
from ..protocol.fingerprint import rubric_library_digest
from ..protocol.schemas import validate_rubrics
from .decompose import (
    apply_decompositions_to_scores,
    build_refinement_jobs,
    overlay_decompositions,
)
from .registry import PositiveRefinementRegistry
from .seed_library import PositiveRefinementSeedLibrary
from ..generator.client import WorkerConfig
from ..generator.single_pass import process_refinement_job

EVOLUTION_RESULT_SCHEMA = "rubric-evolution-result"
EVOLUTION_RESULT_SCHEMA_VERSION = 1


def run_evolution_round(
    rows: Sequence[Mapping[str, Any]],
    *,
    rubrics: Sequence[Mapping[str, Any]],
    registry: PositiveRefinementRegistry,
    config: WorkerConfig,
    judge_transport: Any,
    run_id: str,
    global_step: int,
    seed_library: PositiveRefinementSeedLibrary | None = None,
    result_path: str | None = None,
    keep_constant_children: bool = False,
    include_negative_weight: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one rubric_evolve round; return (evolution_result, side_effects). `rows` are
    this round's grouped observations
    (response_id/rubric_ids/rubric_scores/rubric_weights per row); `rubrics` are the
    current static rubric libraries for each query. `side_effects` carries
    `overlaid_libraries` (query_id -> overlaid library), `score_adjustments`
    (response_id -> dcd_* correction evidence), and `committed` (registry commit
    summaries) for the caller to persist; when `result_path` is given, atomically
    persist the section 6.3 contract result (noreplace). Queries already evolving are
    excluded via `registry.blocked_query_ids()`.
    """
    base_rubrics = [dict(rubric) for rubric in rubrics]
    jobs = build_refinement_jobs(
        rows,
        run_id=run_id,
        global_step=global_step,
        blocked_query_ids=registry.blocked_query_ids(),
        seed_candidates_by_parent=(
            seed_library.as_lookup_mapping(limit=5) if seed_library else None
        ),
        include_negative_weight=bool(include_negative_weight),
    )
    selected_parents: list[dict[str, Any]] = []
    skipped_parents: list[dict[str, Any]] = []
    decomposition_summaries: list[dict[str, Any]] = []
    job_audits: list[dict[str, Any]] = []
    overlaid_libraries: dict[str, list[dict[str, Any]]] = {}
    score_adjustments: dict[str, dict[str, Any]] = {}
    committed: list[dict[str, Any]] = []
    previous_digest = rubric_library_digest(base_rubrics)
    for job in jobs:
        query_id = str(job["query_id"])
        if not registry.reserve_job(job):
            for parent in job["selected_parents"]:
                skipped_parents.append(
                    {
                        "query_id": query_id,
                        "rubric_id": str(parent["rubric_id"]),
                        "reason": "reserve_refused",
                    }
                )
            continue
        try:
            result = process_refinement_job(
                job,
                config=config,
                judge_transport=judge_transport,
                keep_constant_children=keep_constant_children,
            )
        except Exception:
            registry.cancel_job(query_id, str(job["job_id"]))
            raise
        commit = registry.commit_result(result, created_step=int(job["global_step"]))
        committed.append(commit)
        audit_by_parent = {
            str(item["parent_rubric_id"]): item
            for item in result["audit"]["parents"]
        }
        committed_parent_ids = {
            str(item["parent_rubric_id"])
            for item in result["decompositions"]
        }
        for parent in job["selected_parents"]:
            rubric_id = str(parent["rubric_id"])
            if rubric_id in committed_parent_ids:
                selected_parents.append(
                    {
                        "query_id": query_id,
                        "rubric_id": rubric_id,
                        "pass_count": int(parent["pass_count"]),
                        "weight": float(parent["weight"]),
                    }
                )
            else:
                skipped_parents.append(
                    {
                        "query_id": query_id,
                        "rubric_id": rubric_id,
                        "reason": str(
                            audit_by_parent.get(rubric_id, {}).get("status")
                            or "no_nonconstant_child_pattern"
                        ),
                    }
                )
        for decomposition in commit["applied_decompositions"]:
            children = list(decomposition["children"])
            decomposition_summaries.append(
                {
                    "query_id": query_id,
                    "parent_rubric_id": str(decomposition["parent_rubric_id"]),
                    "parent_fingerprint": str(decomposition["parent_fingerprint"]),
                    "parent_weight": float(decomposition["parent_weight"]),
                    "blend_lambda": float(decomposition["blend_lambda"]),
                    "child_count": len(children),
                    "children": [
                        {
                            "rubric_id": str(child["rubric_id"]),
                            "title": str(child["title"]),
                            "weight": float(child["effective_weight"]),
                        }
                        for child in children
                    ],
                }
            )
        decompositions = registry.snapshot(query_id)["decompositions"]
        overlaid, diagnostics = overlay_decompositions(
            base_rubrics,
            decompositions,
            include_negative_weight=bool(include_negative_weight),
        )
        validate_rubrics(overlaid)
        overlaid_libraries[query_id] = overlaid
        evidence = {
            str(row["response_id"]): {
                "rubric_ids": row["rubric_ids"],
                "rubric_scores": row["rubric_scores"],
                "rubric_weights": row["rubric_weights"],
            }
            for row in rows
            if str(row.get("query_id")) == query_id
            and str(row.get("group_id")) == str(job["group_id"])
        }
        score_adjustments.update(
            apply_decompositions_to_scores(
                evidence,
                commit["applied_decompositions"],
                include_negative_weight=bool(include_negative_weight),
            )
        )
        job_audit = result["audit"]
        job_audits.append(
            {
                "job_id": str(job["job_id"]),
                "query_id": query_id,
                "global_step": int(job["global_step"]),
                "status": "committed",
                "selected_parent_count": int(job_audit["selected_parent_count"]),
                "decomposition_count": int(job_audit["decomposition_count"]),
                "selected_child_count": int(job_audit["selected_child_count"]),
                "generator_call_count": int(job_audit["generator_call_count"]),
                "generator_received_rollouts": bool(
                    job_audit["generator_received_rollouts"]
                ),
                "processing_seconds": job_audit["processing_seconds"],
                "overlay": dict(diagnostics),
            }
        )
    result_payload = {
        "schema": EVOLUTION_RESULT_SCHEMA,
        "schema_version": EVOLUTION_RESULT_SCHEMA_VERSION,
        "status": "ok",
        "run_id": run_id,
        "global_step": global_step,
        "selected_parents": selected_parents,
        "decompositions": decomposition_summaries,
        "skipped_parents": skipped_parents,
        "audit": {
            "job_count": len(jobs),
            "committed_count": len(committed),
            "jobs": job_audits,
            "previous_rubric_library_sha256": previous_digest,
            "rubric_library_sha256_by_query": {
                query_id: rubric_library_digest(library)
                for query_id, library in overlaid_libraries.items()
            },
        },
    }
    if result_path is not None:
        atomic_write_json_noreplace(result_path, result_payload)
    return (
        result_payload,
        {
            "overlaid_libraries": overlaid_libraries,
            "score_adjustments": score_adjustments,
            "committed": committed,
        },
    )


__all__ = [
    "EVOLUTION_RESULT_SCHEMA",
    "EVOLUTION_RESULT_SCHEMA_VERSION",
    "run_evolution_round",
]
