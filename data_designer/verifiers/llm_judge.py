"""llm_rubric verifier: LLM judge scores against a rubric (§7.2 Security/report category).

E1 hard constraint separating test-setting and grading:
  - the judge endpoint is **not** published with the task (the payload freezes only the full rubric text and threshold),
    and the grading side (``scripts/wb_task_eval.py``) injects it explicitly through ``configure_llm_judge``--physically separated from generator configuration;
  - ``detail`` records the judge model and prompt version; the full rubric enters task provenance.

payload structure (wb-task-v1):
    {
      "rubric": "full scoring rubric text (criterion by criterion)",
      "pass_threshold": 0.8,         # optional, default 0.8
      "score_scale": {"min": 1, "max": 5}   # optional; upstream 1-5 scale linearly mapped to 0-1
    }

artifact: report text produced by the agent (str; dict is first serialized to JSON).
The judge returns JSON ``{"passed": bool, "score": 0.0-1.0, "reason": "..."}``;
when ``score_scale`` is present, the judge's score uses the min-max scale and is compared after conversion.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Optional

JUDGE_PROMPT_VERSION = "wb-judge-v1"
DEFAULT_PASS_THRESHOLD = 0.8

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

class LlmJudgeConfigError(RuntimeError):
    pass


class LlmJudge:
    def __init__(self, client, *, model, provider_version="unknown", temperature=0.0):
        self.client = client
        self.model = model
        self.provider_version = provider_version
        self.temperature = temperature

    def fingerprint(self):
        from data_designer.contracts.provenance import content_digest
        return content_digest({"model": self.model, "provider_version": self.provider_version,
                               "temperature": self.temperature, "prompt_version": JUDGE_PROMPT_VERSION})


def _score_scale(payload: Dict[str, Any]) -> Optional[Dict[str, float]]:
    scale = payload.get("score_scale")
    if scale is None:
        return None
    if (
        not isinstance(scale, dict)
        or set(scale) != {"min", "max"}
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            for v in scale.values()
        )
        or not float(scale["max"]) > float(scale["min"])
    ):
        raise LlmJudgeConfigError(
            f"score_scale must be {{'min','max'}} with max>min: {scale!r}")
    return {"min": float(scale["min"]), "max": float(scale["max"])}


def _judge_system_prompt(rubric: str, scale: Optional[Dict[str, float]]) -> str:
    score_clause = (
        f"score must be {scale['min']:g}–{scale['max']:g} within this range."
        if scale else "score must be between 0.0 and 1.0."
    )
    return (
        "You are a strict scoring judge. Evaluate the submitted report against each rubric item, "
        "and output exactly one JSON object: "
        '{"passed": true|false, "score": number, "reason": "summary of each verdict"}.\n'
        f"{score_clause}"
        f"\n[full rubric]\n{rubric}"
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    for candidate in (raw, *[_m.group(1) for _m in _FENCE_RE.finditer(raw)]):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def judge_with_rubric(artifact: Any, payload: Dict[str, Any], *, judge=None) -> Dict[str, Any]:
    """verifier main entry: artifact=report text; payload freezes the rubric."""
    rubric = payload.get("rubric")
    if not isinstance(rubric, str) or not rubric.strip():
        return {"status": "error", "detail": "verifier payload requires rubric"}
    scale = _score_scale(payload)
    if judge is None:
        raise LlmJudgeConfigError("an explicit marking-side LlmJudge instance is required")
    if isinstance(artifact, dict):
        artifact_text = json.dumps(artifact, ensure_ascii=False, indent=2)
    elif isinstance(artifact, str):
        artifact_text = artifact
    else:
        return {"passed": False, "score": 0.0,
                "detail": f"artifact is not report text: {type(artifact).__name__}"}
    if not artifact_text.strip():
        return {"passed": False, "score": 0.0, "detail": "artifact is an empty report"}

    resp = judge.client.chat(
        messages=[{"role": "system", "content": _judge_system_prompt(rubric, scale)},
                  {"role": "user", "content": artifact_text}], temperature=judge.temperature)
    verdict = _extract_json(str(resp.choices[0].message.content or ""))
    if verdict is None or "score" not in verdict:
        return {"status": "error",
                "detail": f"judge output is not parsable (model={judge.model}, "
                          f"prompt={JUDGE_PROMPT_VERSION})"}
    try:
        raw_score = float(verdict["score"])
    except (TypeError, ValueError):
        return {"status": "error",
                "detail": f"judge score is not numeric: {verdict.get('score')!r}"}
    if scale is not None:
# Upstream 1-5 score linearly mapped to 0-1; out-of-range is treated as unparseable output and fails explicitly.
        if not scale["min"] <= raw_score <= scale["max"]:
            return {"status": "error",
                    "detail": f"judge score is outside the {scale['min']:g}–{scale['max']:g} scale: "
                              f"{raw_score!r}"}
        score = (raw_score - scale["min"]) / (scale["max"] - scale["min"])
    else:
        score = raw_score
    if not 0 <= score <= 1:
        return {"status": "error", "detail": "judge score must be finite and in [0,1]"}
    threshold = float(payload.get("pass_threshold", DEFAULT_PASS_THRESHOLD))
    if not 0 <= threshold <= 1:
        raise LlmJudgeConfigError("pass_threshold must be in [0,1]")
    reason = str(verdict.get("reason") or "").strip()
    passed = bool(verdict.get("passed", score >= threshold)) and score >= threshold
    return {
        "passed": passed,
        "score": round(score, 4),
        "detail": f"judge(model={judge.model}, prompt={JUDGE_PROMPT_VERSION}, "
                  f"threshold={threshold:g}): {reason or 'no reason given'}",
    }


__all__ = ["judge_with_rubric", "LlmJudge", "LlmJudgeConfigError", "JUDGE_PROMPT_VERSION", "DEFAULT_PASS_THRESHOLD"]
