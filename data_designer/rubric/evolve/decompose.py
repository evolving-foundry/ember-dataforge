"""Saturated-decomposition job construction, rubric overlay, and gate-score
redistribution (migrated from the pure-logic parts of RuACT
refinement/protocol.py:352-663 and adapter.py). Semantic translation (design document
section 4.2/4.3): in RL, weighted scores after decomposition change gradient
allocation; on the inference side they change sample gate scores and best-of-n
ordering. The family-total-weight conservation invariant (`parent_effective_weight +
child_total_effective_weight == parent_weight`) is the anchor that keeps gate
thresholds comparable across rounds and must be preserved. Decomposition events are
audited (`dcd_*` fields, following the adapter.py:599-607 diagnostic keys) for the
Proposer layer: 'which criteria were subdivided last round' == 'where the model
capability boundary is'. Deviations from the source (intentional on the inference
side): `RUACT_CRITIC_SELECTION` env -> explicit `selection_mode` parameter
(parameterized in P0); group-level statistics reuse
`protocol/selection.analyze_group` (line-for-line equivalent to the source inline
implementation); torch/verl reward-tensor writes removed -- deltas are returned as
gate-score corrections plus audit; numpy dependency removed, `ordering_metrics`
rewritten in pure Python (semantics preserved item by item).
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Mapping, Sequence

from ..protocol.schemas import (
    PositiveRefinementProtocolError,
    required_text,
    validated_group,
    raw_matrix,
)
from ..protocol.fingerprint import (
    ordered_response_set_hash,
    parent_fingerprint,
)
from ..protocol.transaction import (
    BLEND_LAMBDA,
    REWARD_MODE,
    SCHEMA_VERSION,
    refinement_transaction_id,
    seed_bundle_hash,
)
from ..protocol.selection import analyze_group


def build_refinement_jobs(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    global_step: int,
    expected_group_size: int = 8,
    max_queries: int = 32,
    max_parents_per_query: int = 3,
    min_candidate_count: int = 5,
    max_candidate_count: int = 5,
    min_children_per_parent: int = 1,
    max_children_per_parent: int = 3,
    zero_std_epsilon: float = 1e-08,
    blend_lambda: float = BLEND_LAMBDA,
    selection_mode: str = "saturated",
    seed_candidates_by_parent: (
        Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]] | None
    ) = None,
    seed_candidate_cap: int = 5,
    blocked_query_ids: set[str] | frozenset[str] | None = None,
    include_negative_weight: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Turn grouped-observation rows into evolution jobs (including frozen response
    sets, seed injection, and transaction identity). The job dict field names match
    the source field by field -- `evolve/registry.reserve_job` and
    `generator/candidates.validate_refinement_job` both assert on field names.
    """
    required_text(run_id, "run_id")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise PositiveRefinementProtocolError(
            "global_step must be a non-negative integer"
        )
    if (
        expected_group_size != 8
        or max_queries <= 0
        or (not 1 <= max_parents_per_query <= 3)
        or (not 1 <= min_candidate_count <= max_candidate_count)
        or (not 1 <= min_children_per_parent <= max_children_per_parent <= 3)
        or (not math.isfinite(float(zero_std_epsilon)))
        or (float(zero_std_epsilon) < 0.0)
        or (not 2 <= seed_candidate_cap <= 12)
        or (not math.isfinite(float(blend_lambda)))
        or (float(blend_lambda) != BLEND_LAMBDA)
    ):
        raise PositiveRefinementProtocolError("invalid refinement job settings")
    blocked = set(blocked_query_ids or ())
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[required_text(row.get("group_id"), "group_id")].append(row)
    candidates: list[tuple[float, float, float, str, str, dict[str, Any]]] = []
    for group_id in sorted(grouped):
        group = validated_group(grouped[group_id], expected_group_size)
        query_ids = {required_text(row.get("query_id"), "query_id") for row in group}
        queries = {required_text(row.get("query"), "query") for row in group}
        versions = {int(row.get("rubric_query_version", 0)) for row in group}
        if len(query_ids) != 1 or len(queries) != 1 or len(versions) != 1:
            raise PositiveRefinementProtocolError(
                "query identity or registry version changed within one group"
            )
        query_id = next(iter(query_ids))
        if query_id in blocked:
            continue
        stats = analyze_group(
            group,
            expected_group_size=expected_group_size,
            selection_mode=selection_mode,
            include_negative_weight=bool(include_negative_weight),
            max_parents_per_query=max_parents_per_query,
            zero_std_epsilon=zero_std_epsilon,
        )
        selected_parents = [dict(item) for item in stats["selected_parents"]]
        if not selected_parents:
            continue
        eligible = stats["eligible_parents"]
        saturated_negative_count = int(stats["saturated_negative_count"])
        saturated_positive_fraction = float(stats["saturated_positive_weight_fraction"])
        saturated_weight_fraction = float(stats["saturated_weight_fraction"])
        group_std = float(stats["group_std"])
        zero_std = bool(stats["zero_std"])
        seed_lookup = seed_candidates_by_parent or {}
        for parent in selected_parents:
            lookup_key = (
                query_id,
                str(parent["rubric_id"]),
                str(parent["parent_fingerprint"]),
            )
            raw_seed_candidates = list(seed_lookup.get(lookup_key, ()))[
                :seed_candidate_cap
            ]
            seed_candidates: list[dict[str, Any]] = []
            seen_seed_points: set[str] = set()
            for raw_seed in raw_seed_candidates:
                title = required_text(raw_seed.get("title"), "seed_candidate.title")
                completion_point = required_text(
                    raw_seed.get("completion_point"), "seed_candidate.completion_point"
                )
                canonical_point = " ".join(completion_point.casefold().split())
                if canonical_point in seen_seed_points:
                    continue
                seen_seed_points.add(canonical_point)
                seed_candidates.append(
                    {
                        "title": title,
                        "completion_point": completion_point,
                        "source_job_id": required_text(
                            raw_seed.get("source_job_id"),
                            "seed_candidate.source_job_id",
                        ),
                        "source_step": int(raw_seed.get("source_step", 0)),
                        "source_pass_count": int(raw_seed.get("source_pass_count", -1)),
                        **(
                            {"polarity": str(raw_seed["polarity"])}
                            if raw_seed.get("polarity") in {"positive", "negative"}
                            else {}
                        ),
                    }
                )
            parent["seed_candidates"] = seed_candidates
        members = [
            {
                "id": str(row["response_id"]),
                "rollout_index": int(row["rollout_index"]),
                "original_index": int(row["original_index"]),
                "text": str(row.get("response_text") or ""),
            }
            for row in group
        ]
        response_set_hash = ordered_response_set_hash(members)
        seed_hash = seed_bundle_hash(selected_parents)
        query_version = next(iter(versions))
        job_id = refinement_transaction_id(
            run_id=run_id,
            global_step=global_step,
            group_id=group_id,
            query_id=query_id,
            query_version=query_version,
            response_set_hash=response_set_hash,
            parent_fingerprints=[
                str(item["parent_fingerprint"]) for item in selected_parents
            ],
            seed_bundle_hash=seed_hash,
            blend_lambda=float(blend_lambda),
        )
        job = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "global_step": global_step,
            "group_id": group_id,
            "query_id": query_id,
            "query_version": query_version,
            "response_set_hash": response_set_hash,
            "seed_bundle_hash": seed_hash,
            "blend_lambda": float(blend_lambda),
            "job_id": job_id,
            "transaction_id": job_id,
            "query": next(iter(queries)),
            "group_members": members,
            "selected_parents": selected_parents,
            "positive_saturated_count": len(eligible) - saturated_negative_count,
            "saturated_positive_weight_fraction": saturated_positive_fraction,
            **(
                {
                    "include_negative_weight": True,
                    "negative_saturated_count": saturated_negative_count,
                    "saturated_weight_fraction": saturated_weight_fraction,
                }
                if include_negative_weight
                else {}
            ),
            "min_candidate_count": min_candidate_count,
            "max_candidate_count": max_candidate_count,
            "min_children_per_parent": min_children_per_parent,
            "max_children_per_parent": max_children_per_parent,
            "gate_mode": (
                "saturated_signed_weight_fraction"
                if include_negative_weight
                else "saturated_positive_weight_fraction"
            ),
            "generator_input_mode": "query_and_parent_only",
            "reward_mode": REWARD_MODE,
            "child_activation_mode": "all_meta_valid_fixed_weights",
            "rollout_pass_rate_filtering": False,
            "group_reward_std": group_std,
            "zero_std_gate": zero_std,
            "zero_std_epsilon": float(zero_std_epsilon),
        }
        candidates.append(
            (
                saturated_weight_fraction,
                max((abs(float(item["weight"])) for item in selected_parents)),
                -group_std,
                query_id,
                group_id,
                job,
            )
        )
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))
    selected: list[tuple[float, float, float, str, str, dict[str, Any]]] = []
    seen_groups: set[str] = set()
    for item in candidates:
        if item[4] in seen_groups:
            continue
        seen_groups.add(item[4])
        selected.append(item)
        if len(selected) >= max_queries:
            break
    return tuple((item[5] for item in selected))


def overlay_decompositions(
    rubrics: Sequence[Mapping[str, Any]],
    decompositions: Sequence[Mapping[str, Any]],
    *,
    include_negative_weight: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Overlay registry decompositions onto the current query's rubric set (migrated
    from adapter.py:_overlay_registry). Parent weight becomes `w*(1-lambda)` and
    children are appended; assert that live parent weight and fingerprint have not
    drifted ('active parent semantics changed' -- prevents an old decomposition from
    taking effect silently after the rubric library text changes). Returns (overlaid
    rubrics, overlay diagnostics).
    """
    by_parent = {
        str(item["parent_rubric_id"]): item for item in decompositions
    }
    overlaid: list[dict[str, Any]] = []
    applied: set[str] = set()
    child_count = 0
    for raw_rubric in rubrics:
        rubric = dict(raw_rubric)
        rubric_id = str(rubric.get("rubric_key") or "")
        decomposition = by_parent.get(rubric_id)
        if decomposition is None:
            overlaid.append(rubric)
            continue
        original_weight = float(decomposition["parent_weight"])
        if float(rubric.get("weight", 0.0)) != original_weight or (
            original_weight <= 0.0 and (not include_negative_weight)
        ):
            raise PositiveRefinementProtocolError(
                "active parent no longer matches its decomposition"
            )
        live_fingerprint = parent_fingerprint(
            rubric_id,
            str(rubric.get("title") or ""),
            str(rubric.get("description") or ""),
            original_weight,
        )
        if live_fingerprint != str(decomposition["parent_fingerprint"]):
            raise PositiveRefinementProtocolError(
                "active parent semantics changed after refinement"
            )
        children = list(decomposition["children"])
        if not children:
            raise PositiveRefinementProtocolError(
                "active decomposition has no completion points"
            )
        blend_lambda = float(decomposition["blend_lambda"])
        parent_weight = original_weight * (1.0 - blend_lambda)
        parent = dict(rubric)
        parent["weight"] = parent_weight
        parent["dcd_role"] = "parent"
        parent["dcd_original_weight"] = original_weight
        parent["dcd_blend_lambda"] = blend_lambda
        overlaid.append(parent)
        for child in children:
            child_weight = float(
                child.get(
                    "effective_weight",
                    original_weight * blend_lambda / len(children),
                )
            )
            overlaid.append(
                {
                    "rubric_key": str(child["rubric_id"]),
                    "title": str(child["title"]),
                    "description": str(child["description"]),
                    "weight": child_weight,
                    "source": "dynamic_rubric",
                    "dcd_parent_rubric_key": rubric_id,
                    "dcd_necessity": str(child["necessity"]),
                    "dcd_role": "child",
                    "dcd_original_parent_weight": original_weight,
                    "dcd_blend_lambda": blend_lambda,
                }
            )
        applied.add(rubric_id)
        child_count += len(children)
    if set(by_parent) != applied:
        raise PositiveRefinementProtocolError(
            "decomposition parent is absent from current query rubrics"
        )
    return (
        overlaid,
        {"overlay_parent_count": len(applied), "overlay_child_count": child_count},
    )


def apply_decompositions_to_scores(
    evidence_by_response: Mapping[str, Mapping[str, Sequence[Any]]],
    decompositions: Sequence[Mapping[str, Any]],
    *,
    include_negative_weight: bool = False,
) -> dict[str, dict[str, Any]]:
    """Apply decompositions to per-sample score evidence within a group, producing
    corrected scores and `dcd_*` audit fields (migrated from
    adapter.py:_apply_decompositions; torch tensor writes removed).
    `evidence_by_response`: response_id -> {rubric_ids, rubric_scores,
    rubric_weights}; returns response_id -> {delta, original_reward, combined_reward,
    completion_coverage, parent_ids, child_ids, parent_share, child_share, ...dcd_*
    audit keys}.
    """
    if not decompositions:
        return {}
    indexed: list[tuple[Mapping[str, Any], list[dict[str, int]]]] = []
    expected_ids: set[str] | None = None
    for decomposition in decompositions:
        child_maps = []
        response_ids: set[str] = set()
        for child in decomposition["children"]:
            by_id = {
                str(item["response_id"]): int(item["score"])
                for item in child["judge_scores"]
            }
            if expected_ids is None:
                expected_ids = set(by_id)
            elif set(by_id) != expected_ids:
                raise PositiveRefinementProtocolError(
                    "completion scores do not cover the source group"
                )
            response_ids = set(by_id)
            child_maps.append(by_id)
        indexed.append((decomposition, child_maps))
        if expected_ids is None or set(evidence_by_response) != expected_ids:
            raise PositiveRefinementProtocolError(
                "score evidence must cover exactly the frozen source group"
            )
    results: dict[str, dict[str, Any]] = {}
    blend_lambda = float(indexed[0][0]["blend_lambda"])
    for response_id, evidence in evidence_by_response.items():
        ids = [str(value) for value in evidence.get("rubric_ids") or []]
        scores = [int(value) for value in evidence.get("rubric_scores") or []]
        weights = [float(value) for value in evidence.get("rubric_weights") or []]
        if not len(ids) == len(scores) == len(weights):
            raise PositiveRefinementProtocolError(
                "static rubric evidence is misaligned"
            )
        denominator = sum((weight for weight in weights if weight > 0.0))
        if denominator <= 0.0:
            raise PositiveRefinementProtocolError(
                "reward requires at least one positive rubric"
            )
        original_reward = (
            sum((score * weight for score, weight in zip(scores, weights)))
            / denominator
        )
        numerator_delta = 0.0
        row_coverages: list[float] = []
        parent_ids: list[str] = []
        child_ids: list[str] = []
        for decomposition, child_maps in indexed:
            parent_id = str(decomposition["parent_rubric_id"])
            if parent_id not in ids:
                raise PositiveRefinementProtocolError(
                    "selected parent is absent from current reward evidence"
                )
            position = ids.index(parent_id)
            weight = float(decomposition["parent_weight"])
            if weight != weights[position] or (
                weight <= 0.0 and (not include_negative_weight)
            ):
                raise PositiveRefinementProtocolError(
                    "parent weight changed before same-group DCD"
                )
            coverage = sum(
                (child_map[response_id] for child_map in child_maps)
            ) / len(child_maps)
            parent_score = scores[position]
            numerator_delta += blend_lambda * weight * (coverage - parent_score)
            row_coverages.append(coverage)
            parent_ids.append(parent_id)
            child_ids.extend(
                (str(child["rubric_id"]) for child in decomposition["children"])
            )
        delta = numerator_delta / denominator
        combined = original_reward + delta
        results[response_id] = {
            "delta": delta,
            "original_reward": original_reward,
            "combined_reward": combined,
            "completion_coverage": (
                sum(row_coverages) / len(row_coverages) if row_coverages else 0.0
            ),
            "parent_ids": parent_ids,
            "child_ids": child_ids,
            "parent_share": 1.0 - blend_lambda,
            "child_share": blend_lambda,
            "dcd_parent_ids": parent_ids,
            "dcd_child_ids": child_ids,
            "dcd_original_reward": original_reward,
            "dcd_completion_coverage": (
                sum(row_coverages) / len(row_coverages) if row_coverages else 0.0
            ),
            "dcd_parent_share": 1.0 - blend_lambda,
            "dcd_child_share": blend_lambda,
            "dcd_combined_reward": combined,
            "dcd_reward_delta": delta,
        }
    return results


def ordering_metrics(
    before: Sequence[float], after: Sequence[float]
) -> dict[str, int]:
    """Ordering-quality audit before and after decomposition (migrated from
    adapter.py:_ordering_metrics, pure Python).
    """
    if len(before) != len(after):
        raise PositiveRefinementProtocolError(
            "ordering metrics require aligned before/after vectors"
        )
    reversals = comparable = sign_flips = 0
    mean_before = sum(before) / len(before)
    mean_after = sum(after) / len(after)
    for left, right in combinations(range(len(before)), 2):
        old_diff = before[left] - before[right]
        new_diff = after[left] - after[right]
        if old_diff != 0.0:
            comparable += 1
            reversals += int(old_diff * new_diff < 0.0)
    for old, new in zip(before, after):
        sign_flips += int((old - mean_before) * (new - mean_after) < 0.0)
    return {
        "order_reversal_count": reversals,
        "comparable_pair_count": comparable,
        "sign_flip_count": sign_flips,
    }
