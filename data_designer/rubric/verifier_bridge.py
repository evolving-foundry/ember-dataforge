"""Thin adapter from the rubric_critic verifier to the dispatch protocol (design
document section 3.2, integration surface 1). Parses the spec and wraps
VerificationResult; E1 separates test generation from grading: judge endpoint
credentials are injected explicitly through :func:`configure_judge_endpoints` and are
never published in the payload. artifact: grouped sample group {query, query_id,
responses}; payload: {rubric_library, group_size, threshold, anchor_scores}. Scoring
failures (JudgeGiveUpError / incomplete group) are raised explicitly rather than
being swallowed as zero (existing dispatch policy; an unavailable Judge must fail the
verification round).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .orchestrate.pipeline import RubricGate
from .protocol.schemas import PositiveRefinementProtocolError


def run_rubric_critic(artifact: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise PositiveRefinementProtocolError(
            "rubric_critic artifact must be a mapping with query and responses"
        )
    query = artifact.get("query")
    responses = artifact.get("responses")
    if not isinstance(query, str) or not query.strip():
        raise PositiveRefinementProtocolError("artifact.query must be non-empty")
    if not isinstance(responses, list) or not responses:
        raise PositiveRefinementProtocolError(
            "artifact.responses must be a non-empty list"
        )
    rubric_library = payload.get("rubric_library")
    if not isinstance(rubric_library, list) or not rubric_library:
        raise PositiveRefinementProtocolError(
            "payload.rubric_library must be a non-empty list"
        )
    group_size = int(payload.get("group_size", 8))
    if len(responses) != group_size:
        raise PositiveRefinementProtocolError(
            f"rubric_critic requires a complete group of {group_size}, got {len(responses)}"
        )
    anchor_scores = payload.get("anchor_scores")
    gate = RubricGate(
        rubric_library=rubric_library,
        threshold=payload.get("threshold"),
        anchor_scores=anchor_scores if isinstance(anchor_scores, list) else None,
        group_size=group_size,
    )
    query_id = str(artifact.get("query_id") or "query")
    decisions = gate.gate_group(
        query_id=query_id,
        query=query,
        responses=responses,
    )
    accepted = [record for record in decisions if record["decision"] == "accept"]
    mean_score = sum(record["score"] for record in decisions) / len(decisions)
    detail = json.dumps(
        {
            "query_id": query_id,
            "group_size": group_size,
            "threshold": gate.threshold,
            "filter_bounds": list(gate.filter_bounds()),
            "rubric_library_sha256": gate.library_sha256,
            "accepted": len(accepted),
            "decisions": [record["decision"] for record in decisions],
            "priority": [record["priority"] for record in decisions],
            "difficulty_buckets": [record["difficulty_bucket"] for record in decisions],
            "group_pass_counts": decisions[0]["rubric_evidence"]["group_pass_counts"],
            "rubric_evidence": [record["rubric_evidence"] for record in decisions],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "status": "completed",
        "passed": len(accepted) == len(decisions),
        "score": mean_score,
        "detail": detail,
    }
