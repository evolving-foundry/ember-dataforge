"""Frozen rubric-library snapshot export (design document section 4.5). Rule: **Gym
never references DataDesigner's live rubric library, only snapshots**. Snapshot
export reuses the frozen semantics of `criteria.py export-seed`: -
`atomic_write_bytes_noreplace`: fail if the target exists -- export creates a new
version file and never overwrites an old snapshot (further evolution/export after
export never changes the old snapshot content or sha256, section 7.1 frozen-semantics
case); - `evidence_scope: 'frozen_seed_only'` is recorded in the payload; - final
gate before freezing (section 6.5(d)): by default require library-wide review
evidence -- every rubric_key has a terminal approve/rewrite verdict (retire means
absent from the library); if any is missing, explicitly refuse export; - integrity:
the payload embeds `rubric_library_sha256` and a whole-file `snapshot_sha256`;
:func:`load_frozen_snapshot` recomputes and rejects tampering.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..io.atomic import atomic_write_bytes_noreplace
from ..protocol.fingerprint import rubric_library_digest
from ..protocol.schemas import PositiveRefinementProtocolError

SNAPSHOT_SCHEMA = "rubric-library-snapshot"
SNAPSHOT_SCHEMA_VERSION = 1
EVIDENCE_SCOPE_FROZEN_SEED_ONLY = "frozen_seed_only"
_REVIEW_TERMINAL_ACTIONS = {"approve", "rewrite"}


def full_review_coverage(
    library: Sequence[Mapping[str, Any]],
    review_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that review evidence covers the whole library: every rubric_key has a
    terminal (approve/rewrite) verdict. `review_audit` is the audit from
    :func:`review.meta_review.apply_reviews` (or an isomorphic mapping: a `verdicts`
    list). retire counts as reviewed (the key is absent from the new library); if a
    key in the library lacks a verdict, return the gap without raising (the caller
    decides).
    """
    verdicts = review_audit.get("verdicts") or []
    verdict_by_key = {
        str(verdict.get("rubric_key") or ""): str(verdict.get("action") or "")
        for verdict in verdicts
        if isinstance(verdict, Mapping)
    }
    missing: list[dict[str, str]] = []
    for rubric in library:
        key = str(rubric.get("rubric_key") or "")
        action = verdict_by_key.get(key)
        if action not in _REVIEW_TERMINAL_ACTIONS:
            missing.append({"rubric_key": key, "observed_action": action or "none"})
    return {"missing_count": len(missing), "missing": missing}


def export_frozen_snapshot(
    library: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path,
    review_audit: Mapping[str, Any],
    exported_step: int = 0,
    demonstrations_sha256: str | None = None,
    require_full_review: bool = True,
) -> dict[str, Any]:
    """Export a frozen_seed_only snapshot; return {path, snapshot_sha256,
    rubric_library_sha256}. `require_full_review=True` (default) enforces the section
    6.5(d) library-wide review gate; explicitly setting it to `False` is an escape
    hatch and gaps are recorded in the audit. If the target file already exists,
    noreplace fails explicitly.
    """
    library_list = [dict(rubric) for rubric in library]
    coverage = full_review_coverage(library_list, review_audit)
    if require_full_review and coverage["missing_count"]:
        raise PositiveRefinementProtocolError(
            f"frozen snapshot requires full semantic review; missing {coverage['missing_count']} criteria: "
            + ", ".join((entry["rubric_key"] for entry in coverage["missing"]))
        )
    if isinstance(exported_step, bool) or not isinstance(exported_step, int) or exported_step < 0:
        raise PositiveRefinementProtocolError("exported_step must be a non-negative integer")
    library_digest = rubric_library_digest(library_list)
    payload = {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "evidence_scope": EVIDENCE_SCOPE_FROZEN_SEED_ONLY,
        "exported_step": exported_step,
        "rubric_library": library_list,
        "rubric_library_sha256": library_digest,
        "review": {
            "verdict_count": len(review_audit.get("verdicts") or []),
            "full_review_coverage": coverage,
            "previous_rubric_library_sha256": review_audit.get(
                "previous_rubric_library_sha256"
            ),
            "retired_count": int(review_audit.get("retired_count") or 0),
        },
        **({"demonstrations_sha256": demonstrations_sha256} if demonstrations_sha256 else {}),
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    snapshot_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    annotated = json.dumps(
        {**payload, "snapshot_sha256": snapshot_sha256},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    atomic_write_bytes_noreplace(output_path, annotated.encode("utf-8"))
    return {
        "path": str(output_path),
        "snapshot_sha256": snapshot_sha256,
        "rubric_library_sha256": library_digest,
        "evidence_scope": EVIDENCE_SCOPE_FROZEN_SEED_ONLY,
    }


def load_frozen_snapshot(path: str | Path) -> dict[str, Any]:
    """Load and integrity-check a frozen snapshot: schema, evidence_scope, and
    recomputation of both sha256 values.
    """
    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if (
        not isinstance(payload, Mapping)
        or payload.get("snapshot_schema") != SNAPSHOT_SCHEMA
        or payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
    ):
        raise PositiveRefinementProtocolError(
            f"unsupported rubric snapshot schema: {path}"
        )
    if payload.get("evidence_scope") != EVIDENCE_SCOPE_FROZEN_SEED_ONLY:
        raise PositiveRefinementProtocolError(
            "rubric snapshot must be exported with evidence_scope=frozen_seed_only"
        )
    annotated = str(payload.get("snapshot_sha256") or "")
    body = json.dumps(
        {key: value for key, value in payload.items() if key != "snapshot_sha256"},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if annotated != hashlib.sha256(body.encode("utf-8")).hexdigest():
        raise PositiveRefinementProtocolError(
            f"rubric snapshot integrity check failed: {path}"
        )
    library = payload.get("rubric_library")
    if not isinstance(library, list):
        raise PositiveRefinementProtocolError("snapshot rubric_library must be a list")
    if payload.get("rubric_library_sha256") != rubric_library_digest(library):
        raise PositiveRefinementProtocolError(
            f"snapshot rubric library digest mismatch: {path}"
        )
    return dict(payload)


__all__ = [
    "EVIDENCE_SCOPE_FROZEN_SEED_ONLY",
    "SNAPSHOT_SCHEMA",
    "SNAPSHOT_SCHEMA_VERSION",
    "export_frozen_snapshot",
    "full_review_coverage",
    "load_frozen_snapshot",
]
