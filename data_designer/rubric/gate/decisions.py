"""Gate decision output contract (design document section 6.1). The same evidence
structure is reused on the Gym side as verdict details for the decision adapter. Top-
level fields in persisted JSONL/JSON map one-to-one to section 6.1; audit records
contain only non-credential information such as model routing (credentials never
enter audit logs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

DECISION_ACCEPT = "accept"
DECISION_REJECT = "reject"
DECISION_HOLDOUT = "holdout"

BUCKET_MID = "mid"
BUCKET_SATURATED = "saturated"
BUCKET_UNREACHABLE = "unreachable"

RULE_THRESHOLD = "threshold"
RULE_DISCRIMINATIVE = "discriminative"
RULE_CURRICULUM = "curriculum"


@dataclass(frozen=True)
class RubricEvidence:
    rubric_ids: tuple[str, ...]
    rubric_scores: tuple[int, ...]
    rubric_weights: tuple[float, ...]
    group_pass_counts: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rubric_ids": list(self.rubric_ids),
            "rubric_scores": list(self.rubric_scores),
            "rubric_weights": list(self.rubric_weights),
            "group_pass_counts": list(self.group_pass_counts),
        }


@dataclass(frozen=True)
class GateDecision:
    sample_id: str
    query_id: str
    decision: str
    priority: float
    score: float
    threshold: float
    difficulty_bucket: str
    rubric_evidence: RubricEvidence
    audit: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query_id": self.query_id,
            "decision": self.decision,
            "priority": self.priority,
            "score": self.score,
            "threshold": self.threshold,
            "difficulty_bucket": self.difficulty_bucket,
            "rubric_evidence": self.rubric_evidence.as_dict(),
            "audit": dict(self.audit),
        }


def evidence_from_rows(
    rubric_ids: Sequence[str],
    rubric_scores: Sequence[int],
    rubric_weights: Sequence[float],
    group_pass_counts: Sequence[int],
) -> RubricEvidence:
    if not len(rubric_ids) == len(rubric_scores) == len(rubric_weights) == len(
        group_pass_counts
    ):
        raise ValueError("rubric evidence vectors must be aligned")
    return RubricEvidence(
        tuple(rubric_ids),
        tuple(int(score) for score in rubric_scores),
        tuple(float(weight) for weight in rubric_weights),
        tuple(int(count) for count in group_pass_counts),
    )
