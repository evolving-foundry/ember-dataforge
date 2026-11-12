"""Polarity triage for negative-weight parents (migrated from RuACT refinement/worker.py
`_triage_polarity`). Trap rubrics are triaged before decomposition: negation shells
(such as 'Does not mention') are flipped into positive decomposition
(convert_to_positive), while genuine erroneous behavior stays negative
(keep_negative). Triage sees only the query and parent text, following the same
decoupling discipline as the Generator: model responses and rollout statistics are
never visible.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from ..protocol.schemas import PositiveRefinementProtocolError
from .client import WorkerConfig, generate_json

POLARITY_TRIAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["direction"],
    "properties": {
        "direction": {
            "type": "string",
            "enum": ["convert_to_positive", "keep_negative"],
        }
    },
}


def polarity_triage_prompt(job: Mapping[str, Any], parent: Mapping[str, Any]) -> str:
    return f"""You are classifying one negative-weight rubric (a known answer pitfall) before decomposing it. You never see any model responses or rollout statistics.\n\nUser query:\n{job['query']}\n\nPitfall rubric (negative weight; an answer is penalized when the rubric text matches it):\nTitle: {parent['title']}\nDescription: {parent['description']}\n\nClassify the pitfall into exactly one direction:\n- convert_to_positive: the rubric is a NEGATION SHELL — it penalizes answers for MISSING one single positive behavior, and its wording is only the absence of that behavior (typical markers: "Does not mention", "Does not state", "Does not provide", "Without", "Fails to mention", "Omits", "No discussion of", and similar absence-of-X phrasing). After stripping the negation shell there is exactly one positive completion point to refine further.\n- keep_negative: the rubric describes an ERRONEOUS ACTION or wrong claim the answer actively commits (typical markers: "Recommends wrong", "Incorrectly states", "Recommends against", "Suggests a contraindicated", "Administered incorrectly", "Misidentifies", and similar wrong-doing phrasing). The error itself, not an absence, is the penalized behavior.\n\nDeciding hints:\n- Ask whether the answer that gets penalized could be repaired by merely ADDING one missing behavior (shell) versus by REPLACING a stated wrong behavior with the right one (erroneous action).\n- A shell usually has a single negated verb and no wrong content of its own; an erroneous action names concrete wrong content.\n- When a rubric contains both an absence marker and concrete wrong content, prefer keep_negative (the wrong content is the actionable error).\n\nReturn only the direction value.\n"""


def triage_polarity(
    job: Mapping[str, Any], parent: Mapping[str, Any], *, config: WorkerConfig
) -> tuple[str, dict[str, Any]]:
    """Run one polarity triage call; return (direction, meta). Unknown directions
    conservatively fall back to keep_negative.
    """
    prompt = polarity_triage_prompt(job, parent)
    value, worker_meta = generate_json(
        prompt, POLARITY_TRIAGE_SCHEMA, config=config.for_meta()
    )
    worker_meta = {
        **worker_meta,
        "triage_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "triage_prompt_character_count": len(prompt),
        "triage_prompt_utf8_bytes": len(prompt.encode("utf-8")),
    }
    direction = str(value.get("direction") or "keep_negative")
    if direction != "convert_to_positive":
        direction = "keep_negative"
    if direction not in {"convert_to_positive", "keep_negative"}:  # pragma: no cover
        raise PositiveRefinementProtocolError("polarity triage direction is invalid")
    return (direction, worker_meta)
