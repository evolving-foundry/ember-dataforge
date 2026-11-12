"""Score -> gate -> audit persistence pipeline (P2: gate only, no evolution; evolution
steps move to evolve/ in P3). RubricGate is the rubric package's public Python API
integration surface (design document section 3.2 integration surface 3): `from
data_designer.rubric import RubricGate`. The caller supplies a full group of samples
(n=group_size) and receives per-sample section 6.1 decision records. When Judge is
unavailable, scoring fails explicitly and propagates -- the gate round fails, with no
all-zero fallback and no partial decisions.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..judge.batching import score_group
from ..judge.transport import JudgeTransport, get_judge_transport
from ..protocol.fingerprint import rubric_library_digest as library_digest
from ..protocol.schemas import validate_rubrics
from ..gate.acceptance import FamilyThresholdAnchor, decide
from ..gate.threshold_store import FamilyThresholdStore
from ..io.atomic import atomic_write_json_noreplace
from .group import discriminative_filter_range, resolve_group_size

DEFAULT_JUDGE_WORKERS = 8
DEFAULT_JUDGE_ATTEMPTS = 3


class RubricGate:
    """Rubric scoring and gate decisions for one group of samples. `threshold` is
    explicit; or `anchor_scores` (scores accepted in the previous round) are used
    through P25 anchoring; if neither is provided, fall back to the
    `RUBRIC_INFER_DEFAULT_THRESHOLD` environment variable (default 0.5).
    """

    def __init__(
        self,
        *,
        rubric_library: Sequence[Mapping[str, Any]],
        threshold: float | None = None,
        anchor_scores: Sequence[float] | None = None,
        anchor_store: FamilyThresholdStore | None = None,
        group_size: int | None = None,
        transport: JudgeTransport | None = None,
        judge_workers: int = DEFAULT_JUDGE_WORKERS,
        judge_attempts: int = DEFAULT_JUDGE_ATTEMPTS,
    ) -> None:
        self.descriptions, self.weights, self.rubric_ids, self.titles = validate_rubrics(
            list(rubric_library)
        )
        if threshold is not None and anchor_scores is not None:
            raise ValueError("provide either threshold or anchor_scores, not both")
        if threshold is not None:
            self.threshold = float(threshold)
        elif anchor_scores:
            self.threshold = FamilyThresholdAnchor(anchor_scores).threshold
        else:
            self.threshold = float(
                os.environ.get("RUBRIC_INFER_DEFAULT_THRESHOLD", "0.5")
            )
        # Cross-round family anchor (section 4.1): only when the caller supplies neither
        # an explicit threshold nor anchor scores, resolve by query family
        # from the store; explicit parameters always win.
        self.anchor_store = anchor_store
        self._consult_anchor_store = threshold is None and anchor_scores is None
        self.group_size = resolve_group_size(group_size)
        self.filter_low, self.filter_high = discriminative_filter_range(self.group_size)
        self.transport = transport
        self.judge_workers = judge_workers
        self.judge_attempts = judge_attempts
        self.library_sha256 = library_digest(list(rubric_library))

    def _judge_route(self) -> dict[str, Any]:
        active = self.transport or get_judge_transport()
        model = active.model or os.environ.get("HEALTHBENCH_JUDGE_MODEL", "")
        return {"model": model}

    def gate_group(
        self,
        *,
        query_id: str,
        query: str,
        responses: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score one complete sample group item by item and produce decision records.
        Each `responses` item contains `sample_id` (or `response_id`),
        `rollout_index`, and `text`; the count must equal group_size.
        """
        if len(responses) != self.group_size:
            raise ValueError(
                f"gate requires a complete group of {self.group_size}, got {len(responses)}"
            )
        ordered = sorted(responses, key=lambda row: int(row["rollout_index"]))
        slots = [int(row["rollout_index"]) for row in ordered]
        if slots != list(range(self.group_size)):
            raise ValueError(
                f"group must contain rollout_index 0..{self.group_size - 1} exactly once"
            )
        members = [
            {
                "id": str(row.get("sample_id") or row.get("response_id")),
                "rollout_index": int(row["rollout_index"]),
                "text": str(row.get("text") or ""),
            }
            for row in ordered
        ]
        rows, judge_audit = score_group(
            query,
            members,
            self.descriptions,
            transport=self.transport,
            workers=self.judge_workers,
            attempts=self.judge_attempts,
        )
        pass_counts = [
            sum(row[index] for row in rows) for index in range(len(self.rubric_ids))
        ]
        anchored_threshold: float | None = None
        if self.anchor_store is not None and self._consult_anchor_store:
            anchored_threshold = self.anchor_store.anchor_threshold(
                query_id, rubric_library_sha256=self.library_sha256
            )
        effective_threshold = (
            anchored_threshold if anchored_threshold is not None else self.threshold
        )
        threshold_source = (
            "family_anchor" if anchored_threshold is not None else "static"
        )
        judge_route = self._judge_route()
        decisions = []
        for index, row in enumerate(ordered):
            decision = decide(
                sample_id=str(members[index]["id"]),
                query_id=query_id,
                rubric_ids=self.rubric_ids,
                rubric_scores=rows[index],
                rubric_weights=self.weights,
                group_pass_counts=pass_counts,
                threshold=effective_threshold,
                rubric_library_sha256=self.library_sha256,
                judge_route=judge_route,
            )
            record = decision.as_dict()
            record["audit"]["threshold_source"] = threshold_source
            record["audit"]["judge_attempts"] = judge_audit["response_task_attempts"]
            record["audit"]["judge_latency_seconds"] = judge_audit["latency_seconds"]
            decisions.append(record)
        return decisions

    def record_family_anchor(
        self,
        query_id: str,
        decisions: Sequence[Mapping[str, Any]],
        *,
        updated_step: int = 0,
    ) -> None:
        """Write this round's accepted sample scores to the family threshold anchor
        store (the cross-round section 4.1 anchoring loop). Only effective when
        `anchor_store` was enabled at construction and neither a threshold nor anchor
        scores were provided explicitly; decisions are the records returned by
        :meth:`gate_group`.
        """
        if self.anchor_store is None or not self._consult_anchor_store:
            return
        accepted = [
            float(record["score"])
            for record in decisions
            if record.get("decision") == "accept"
        ]
        if accepted:
            self.anchor_store.record_accepted(
                query_id,
                accepted,
                rubric_library_sha256=self.library_sha256,
                updated_step=updated_step,
            )

    def filter_bounds(self) -> tuple[int, int]:
        """Discriminative filter range [ceil(0.25n), floor(0.75n)] (for callers /
        subsequent evolution).
        """
        return (self.filter_low, self.filter_high)


def persist_decisions(
    decisions: Sequence[Mapping[str, Any]], directory: str | Path
) -> list[Path]:
    """Atomically persist decision records (one JSON per sample; noreplace refuses
    overwrite).
    """
    out_dir = Path(directory)
    written = []
    for record in decisions:
        target = out_dir / f"{record['sample_id']}.json"
        atomic_write_json_noreplace(target, dict(record))
        written.append(target)
    return written
