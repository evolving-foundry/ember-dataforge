"""Evolution transaction identity primitives (migrated from the header and identity
functions of RuACT refinement/protocol.py). - `SCHEMA_VERSION` / `REWARD_MODE`:
schema contracts for evolution results and the registry. - `BLEND_LAMBDA`: fraction
of parent weight retained (1-lambda stays with the parent, lambda is split across
children), default 0.5. The source read this from the environment at import time
(invalid values caused SystemExit); this version takes an explicit parameter
validated by `resolve_blend_lambda` with unchanged hash semantics (the default path
produces the same transaction id as the source). The family total-weight conservation
invariant depends on it. - `refinement_transaction_id`: job identity = canonical hash
of all immutable inputs; the registry uses it for idempotent deduplication (re-
submitting the same job is rejected). - `seed_bundle_hash`: integrity hash of the
parent plus frozen seed-candidate bundle.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .schemas import PositiveRefinementProtocolError, required_text
from .fingerprint import stable_hash

SCHEMA_VERSION = 7
REWARD_MODE = "parent_preserved_multi_child_50_50"

BLEND_LAMBDA = 0.5


def resolve_blend_lambda(value: float | None = None) -> float:
    """Validate an explicitly supplied blend_lambda (replacing the source import-time
    environment read). Return a finite float in (0,1]; the default uses the
    `BLEND_LAMBDA` constant so transaction ids match the source default path bit-for-
    bit.
    """
    raw = BLEND_LAMBDA if value is None else float(value)
    if not math.isfinite(raw) or not 0.0 < raw <= 1.0:
        raise PositiveRefinementProtocolError(
            f"blend_lambda must be in (0,1], got {value!r}"
        )
    return raw


def seed_bundle_hash(selected_parents: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash(
        [
            {
                "parent_rubric_id": required_text(
                    parent.get("rubric_id"), "parent.rubric_id"
                ),
                "parent_fingerprint": required_text(
                    parent.get("parent_fingerprint"), "parent.parent_fingerprint"
                ),
                "candidates": list(parent.get("seed_candidates") or ()),
            }
            for parent in selected_parents
        ]
    )


def refinement_transaction_id(
    *,
    run_id: str,
    global_step: int,
    group_id: str,
    query_id: str,
    query_version: int,
    response_set_hash: str,
    parent_fingerprints: Sequence[str],
    seed_bundle_hash: str,
    blend_lambda: float,
) -> str:
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
        or isinstance(query_version, bool)
        or (not isinstance(query_version, int))
        or (query_version < 0)
    ):
        raise PositiveRefinementProtocolError(
            "transaction step and query version must be non-negative integers"
        )
    if not 1 <= len(parent_fingerprints) <= 3:
        raise PositiveRefinementProtocolError(
            "transaction requires one to three parent fingerprints"
        )
    numeric_lambda = float(blend_lambda)
    if not math.isfinite(numeric_lambda) or not 0.0 < numeric_lambda <= 1.0:
        raise PositiveRefinementProtocolError(
            f"blend_lambda must be in (0,1], got {blend_lambda!r}"
        )
    return stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": required_text(run_id, "run_id"),
            "global_step": int(global_step),
            "group_id": required_text(group_id, "group_id"),
            "query_id": required_text(query_id, "query_id"),
            "query_version": int(query_version),
            "response_set_hash": required_text(response_set_hash, "response_set_hash"),
            "parent_fingerprints": [
                required_text(value, "parent_fingerprint")
                for value in parent_fingerprints
            ],
            "seed_bundle_hash": required_text(seed_bundle_hash, "seed_bundle_hash"),
            "blend_lambda": numeric_lambda,
        }
    )
