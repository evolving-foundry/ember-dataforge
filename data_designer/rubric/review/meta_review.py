"""Semantic review layer (design document section 6.5; fills the source placeholder
`criterion_semantic_review: 'not_performed'`; required on the inference side -- RL
has training-convergence fallback, inference does not, so bad criteria would
systematically amplify and contaminate the gate across RSI rounds). Design contract:
- trigger points (caller policy; this layer provides tools): (a) before new children
enter the library, weighted-sample at least 10% by weight
(:func:`sample_for_review`); (b) force review of unreachable criteria; (c) review
100% of scenario-B negative criteria (included via `force_keys`); (d) library-wide
review before snapshot export (the freeze gate in :mod:`evolve.snapshot`); - review
actions: `approve | rewrite | retire`; rewrite must provide replacement text, retire
removes the criterion; after apply, the new library is re-validated with
`validate_rubrics`, and the audit carries old/new `rubric_library_sha256` (version
write-back chain propagated with the audit); - meta-judge decoupling discipline: the
prompt contains only criterion text + demonstration cases (data, not instructions),
**never generated samples or rollout statistics**; the model uses
`WorkerConfig.for_meta()`'s meta_model (falling back to the Generator primary model).
"""
from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

from ..generator.client import WorkerConfig, generate_json
from ..protocol.fingerprint import rubric_library_digest
from ..protocol.schemas import PositiveRefinementProtocolError, validate_rubrics

REVIEW_ACTIONS = ("approve", "rewrite", "retire")
DEFAULT_REVIEW_SAMPLE_RATE = 0.1

REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "reason"],
    "properties": {
        "action": {"type": "string", "enum": list(REVIEW_ACTIONS)},
        "rewritten_title": {"type": "string"},
        "rewritten_description": {"type": "string"},
        "reason": {"type": "string", "minLength": 1},
    },
}


def meta_review_prompt(criterion: Mapping[str, Any]) -> str:
    """Build the review prompt: only criterion text (and an optional query), with no
    generated samples.
    """
    query_block = ""
    if str(criterion.get("query") or "").strip():
        query_block = f"\nTarget query (context only):\n{str(criterion['query']).strip()}\n"
    return f"""You are auditing one evaluation criterion for a binary rubric checklist. You never see any model responses or generation samples.\n{query_block}\nCriterion under review:\nTitle: {str(criterion.get('title') or '').strip()}\nDescription: {str(criterion.get('description') or '').strip()}\nWeight polarity: {str(criterion.get('polarity') or 'positive')}\n\nAudit the criterion for semantic quality:\n- Atomicity: it expresses exactly one objective, binary-judgeable requirement.\n- Fidelity: it does not smuggle in a different evaluation objective, style preference, or unverifiable claim.\n- Wording: the description is unambiguous about what counts as satisfied, and matches the stated polarity direction.\n- Drift risk: it is not a paraphrase of another criterion, not a vague demand (comprehensive/detailed/clear), and not a shell that can never be judged.\n\nChoose exactly one action:\n- approve: the criterion is sound as written.\n- rewrite: fix wording or scope; supply rewritten_title and rewritten_description that keep the SAME evaluation objective and polarity.\n- retire: the criterion is unsalvageable (unjudgeable, off-objective, or duplicative); give the reason.\n\nReturn only the required JSON.\n"""


def review_criteria(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: WorkerConfig,
) -> list[dict[str, Any]]:
    """Review each criterion and return a verdict list (including review-call audit).
    Structural validation failures (unknown action, rewrite missing replacement text)
    raise explicitly -- a bad verdict must never be silently treated as approve.
    """
    verdicts: list[dict[str, Any]] = []
    for candidate in candidates:
        rubric_key = str(candidate.get("rubric_key") or "").strip()
        if not rubric_key:
            raise PositiveRefinementProtocolError(
                "review candidate must carry a non-empty rubric_key"
            )
        prompt = meta_review_prompt(candidate)
        value, meta = generate_json(
            prompt, REVIEW_SCHEMA, config=config.for_meta()
        )
        action = str(value.get("action") or "")
        if action not in REVIEW_ACTIONS:
            raise PositiveRefinementProtocolError(
                f"meta review returned unknown action for {rubric_key}: {action!r}"
            )
        reason = str(value.get("reason") or "").strip()
        if not reason:
            raise PositiveRefinementProtocolError(
                f"meta review must give a reason for {rubric_key}"
            )
        verdict: dict[str, Any] = {
            "rubric_key": rubric_key,
            "action": action,
            "reason": reason,
            "review_audit": {
                **meta,
                "prompt_sha256": meta.get("prompt_sha256"),
                "meta_model": config.meta_model or config.model,
            },
        }
        if action == "rewrite":
            title = str(value.get("rewritten_title") or "").strip()
            description = str(value.get("rewritten_description") or "").strip()
            if not title or not description:
                raise PositiveRefinementProtocolError(
                    f"rewrite verdict for {rubric_key} must supply non-empty rewritten title and description"
                )
            verdict["rewritten_title"] = title
            verdict["rewritten_description"] = description
        verdicts.append(verdict)
    return verdicts


def sample_for_review(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rate: float = DEFAULT_REVIEW_SAMPLE_RATE,
    force_keys: Sequence[str] = (),
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Sample before library insertion: weighted sampling without replacement of at
    least `rate` proportion (at least 1 for a non-empty library). `force_keys` are
    always included (force-review triggers such as unreachable / scenario-B negative
    criteria); duplicates are deduplicated. Selected items keep the original library
    order (stable audit).
    """
    if not 0.0 < rate <= 1.0:
        raise PositiveRefinementProtocolError("review sample rate must be in (0,1]")
    if not candidates:
        return []
    active_rng = rng or random.Random()
    force = {str(key) for key in force_keys}
    for candidate in candidates:
        key = str(candidate.get("rubric_key") or "")
        if not key:
            raise PositiveRefinementProtocolError(
                "review candidate must carry a non-empty rubric_key"
            )
    forced = [c for c in candidates if str(c["rubric_key"]) in force]
    rest = [c for c in candidates if str(c["rubric_key"]) not in force]
    # At least rate proportion (force items count toward the quota), rounded up, at
    # least 1 for a non-empty library
    needed_total = max(1, math.ceil(len(candidates) * rate))
    needed = min(max(needed_total - len(forced), 0), len(rest))
    if needed:
        weights = [max(float(c.get("weight") or 0.0), 1e-9) for c in rest]
        chosen = (
            list(rest)
            if needed >= len(rest)
            else _weighted_sample(rest, weights, needed, active_rng)
        )
        forced_keys = {str(c["rubric_key"]) for c in forced}
        forced.extend((c for c in chosen if str(c["rubric_key"]) not in forced_keys))
    ordered_keys = [str(c["rubric_key"]) for c in candidates]
    selected = list(forced)
    selected.sort(key=lambda c: ordered_keys.index(str(c["rubric_key"])))
    return selected


def _weighted_sample(
    candidates: list[Mapping[str, Any]],
    weights: list[float],
    count: int,
    rng: random.Random,
) -> list[Mapping[str, Any]]:
    """Efraimidis-Spirakis weighted sampling without replacement: key = u^(1/w), take
    the largest count items.
    """
    keyed = []
    for candidate, weight in zip(candidates, weights):
        uniform = rng.random()
        keyed.append((uniform ** (1.0 / weight), candidate))
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [candidate for _, candidate in keyed[:count]]


def apply_reviews(
    library: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    *,
    include_negative_weight: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write review verdicts back to the rubric library: rewrite in place, retire
    removes. Returns (new library, audit). The audit carries old/new library sha256
    values (version write-back chain) and per-item verdicts; the new library is re-
    validated with `validate_rubrics` -- explicitly fail if retire empties all
    positive weight, and never produce an unscoreable library.
    """
    by_key: dict[str, dict[str, Any]] = {}
    for verdict in verdicts:
        key = str(verdict.get("rubric_key") or "")
        if not key or key in by_key:
            raise PositiveRefinementProtocolError(
                "review verdicts must carry unique non-empty rubric_key"
            )
        if str(verdict.get("action")) not in REVIEW_ACTIONS:
            raise PositiveRefinementProtocolError(
                f"unknown review action for {key}: {verdict.get('action')!r}"
            )
        by_key[key] = dict(verdict)
    previous_digest = rubric_library_digest(list(library))
    new_library: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for rubric in library:
        key = str(rubric.get("rubric_key") or "")
        verdict = by_key.get(key)
        if verdict is None:
            new_library.append(dict(rubric))
            continue
        action = str(verdict["action"])
        if action == "retire":
            applied.append({**verdict, "retired_weight": float(rubric.get("weight") or 0.0)})
            continue
        updated = dict(rubric)
        if action == "rewrite":
            updated["title"] = str(verdict["rewritten_title"])
            updated["description"] = str(verdict["rewritten_description"])
            updated["semantic_review"] = {
                "action": "rewrite",
                "reason": str(verdict["reason"]),
            }
        else:
            updated["semantic_review"] = {
                "action": "approve",
                "reason": str(verdict["reason"]),
            }
        new_library.append(updated)
        applied.append(dict(verdict))
    missing = sorted(set(by_key) - {str(r.get("rubric_key") or "") for r in library})
    if missing:
        raise PositiveRefinementProtocolError(
            f"review verdicts reference unknown rubric keys: {missing}"
        )
    validate_rubrics(new_library)
    audit = {
        "previous_rubric_library_sha256": previous_digest,
        "rubric_library_sha256": rubric_library_digest(new_library),
        "reviewed_count": len(applied),
        "approved_count": sum((a["action"] == "approve" for a in applied)),
        "rewritten_count": sum((a["action"] == "rewrite" for a in applied)),
        "retired_count": sum((a["action"] == "retire" for a in applied)),
        "verdicts": applied,
    }
    return (new_library, audit)


__all__ = [
    "DEFAULT_REVIEW_SAMPLE_RATE",
    "REVIEW_ACTIONS",
    "REVIEW_SCHEMA",
    "apply_reviews",
    "meta_review_prompt",
    "review_criteria",
    "sample_for_review",
]
