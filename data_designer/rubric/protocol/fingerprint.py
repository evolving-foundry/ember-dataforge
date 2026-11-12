"""Rubric fingerprints and idempotency (migrated from RuACT refinement/protocol.py;
reuses contracts.provenance). Fingerprint values are byte-compatible with the source:
`_stable_hash` and `contracts.provenance.content_digest` use the same canonical JSON
serialization (ensure_ascii=False / sort_keys / compact separators); this module only
removes the `sha256:` prefix so P3 seed_library exact-match reuse gets identical
hashes.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from data_designer.contracts.provenance import content_digest

from .schemas import PositiveRefinementProtocolError, required_text


def _stable_hash(value: Any) -> str:
    return content_digest(value).removeprefix("sha256:")


def stable_hash(value: Any) -> str:
    return _stable_hash(value)


def parent_fingerprint(
    rubric_id: str, title: str, description: str, weight: float
) -> str:
    numeric_weight = float(weight)
    if not math.isfinite(numeric_weight) or numeric_weight == 0.0:
        raise PositiveRefinementProtocolError(
            "parent weight must be finite and non-zero"
        )
    return _stable_hash(
        {
            "rubric_id": required_text(rubric_id, "rubric_id"),
            "title": required_text(title, "title"),
            "description": required_text(description, "description"),
            "weight": numeric_weight,
        }
    )


def ordered_response_set_hash(members: Sequence[Mapping[str, Any]]) -> str:
    return _stable_hash(
        [
            {
                "id": required_text(item.get("id"), "response.id"),
                "text": str(item.get("text") or ""),
            }
            for item in members
        ]
    )


def rubric_library_digest(
    rubrics: Sequence[Mapping[str, Any]],
) -> str:
    """Content fingerprint of a rubric-library snapshot (sha256-prefixed form), used in
    audits and Gym manifests.
    """
    return content_digest([dict(item) for item in rubrics])


def json_dumps_stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
