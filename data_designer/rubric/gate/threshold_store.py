"""Cross-round persistence of family threshold anchors (design document section 4.1:
theta is initialized from the P25 of the family's previous accepted-sample score
distribution; anchors must survive RSI rounds, so in-memory state is insufficient).
Comparability policy: family scores are comparable only under the same rubric-library
fingerprint -- after decomposition or rewriting (rubric_library_sha256 changes), old
score distributions must not anchor the new library's gate; lookup matches the
fingerprint exactly and returns None on mismatch (callers fall back to the default
threshold). Distributions are never borrowed across libraries. Persistence policy:
atomic JSON replacement (`atomic_replace_bytes`) with an in-process RLock serializing
read-modify-write; files with a mismatched schema version are rejected explicitly.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..gate.acceptance import threshold_from_scores
from ..io.atomic import atomic_replace_bytes

THRESHOLD_STORE_SCHEMA_VERSION = 1


class FamilyThresholdStore:
    """Family threshold-anchor store keyed exactly by (query_id, rubric_library_sha256)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._families: dict[tuple[str, str], dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, Mapping)
            or payload.get("store_schema") != "rubric-family-thresholds"
            or payload.get("schema_version") != THRESHOLD_STORE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported family threshold store schema: {self.path}"
            )
        entries = payload.get("families")
        if not isinstance(entries, list):
            raise ValueError("family threshold store entries must be a list")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("family threshold entry must be an object")
            query_id = entry.get("query_id")
            digest = entry.get("rubric_library_sha256")
            scores = entry.get("accepted_scores")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError("family threshold entry query_id must be non-empty")
            if not isinstance(digest, str) or not digest.strip():
                raise ValueError(
                    "family threshold entry rubric_library_sha256 must be non-empty"
                )
            if (
                not isinstance(scores, list)
                or not scores
                or any(
                    (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not 0.0 <= float(value) <= 1.0
                        for value in scores
                    )
                )
            ):
                raise ValueError(
                    "family threshold entry accepted_scores must be non-empty and within [0,1]"
                )
            key = (query_id, digest)
            if key in self._families:
                raise ValueError(
                    f"duplicate family threshold entry for {query_id!r} at library {digest[:12]}"
                )
            self._families[key] = {
                "accepted_scores": [float(value) for value in scores],
                "updated_step": int(entry.get("updated_step", 0)),
            }

    def _write(self) -> None:
        payload = {
            "store_schema": "rubric-family-thresholds",
            "schema_version": THRESHOLD_STORE_SCHEMA_VERSION,
            "families": [
                {
                    "query_id": query_id,
                    "rubric_library_sha256": digest,
                    "accepted_scores": entry["accepted_scores"],
                    "updated_step": entry["updated_step"],
                }
                for (query_id, digest), entry in sorted(self._families.items())
            ],
        }
        atomic_replace_bytes(
            self.path,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )

    def anchor_threshold(
        self, query_id: str, *, rubric_library_sha256: str
    ) -> float | None:
        """Return the P25 anchor threshold for the family under the given rubric-library
        fingerprint; return None when not recorded.
        """
        with self._lock:
            entry = self._families.get(
                (str(query_id), str(rubric_library_sha256))
            )
            if entry is None:
                return None
            return threshold_from_scores(entry["accepted_scores"])

    def record_accepted(
        self,
        query_id: str,
        accepted_scores: Sequence[float],
        *,
        rubric_library_sha256: str,
        updated_step: int = 0,
    ) -> None:
        """Record or replace the family's accepted-sample score distribution for this
        round (same-key overwrite means latest-round semantics).
        """
        if isinstance(updated_step, bool) or not isinstance(updated_step, int):
            raise ValueError("updated_step must be an integer")
        scores = [float(value) for value in accepted_scores]
        if (
            not scores
            or any(
                (value != value or value < 0.0 or value > 1.0 for value in scores)
            )
        ):
            raise ValueError(
                "accepted_scores must be non-empty and within [0,1]"
            )
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id must be non-empty text")
        if not isinstance(rubric_library_sha256, str) or not rubric_library_sha256.strip():
            raise ValueError("rubric_library_sha256 must be non-empty text")
        with self._lock:
            self._families[(str(query_id), str(rubric_library_sha256))] = {
                "accepted_scores": scores,
                "updated_step": int(updated_step),
            }
            self._write()


__all__ = ["FamilyThresholdStore", "THRESHOLD_STORE_SCHEMA_VERSION"]
