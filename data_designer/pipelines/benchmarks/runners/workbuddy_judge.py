"""wb_composite LLM judge runtime (M4, design §8).

- ``rubric_binary_mean``: binary decision per rubric (question / pass_criteria /
  fail_criteria / scope_limits + cited artifact evidence text), with the mean binary value as the llm component;
  temperature 0.0 (§12 D1: volcano gateway glm-5.3-flash, thinkingLevel low).
- endpoint: injected through ``WB_JUDGE_BASE_URL`` / ``WB_JUDGE_API_KEY`` / ``WB_JUDGE_MODEL``
  environment variables (read at evaluation time, never stored in a database or repo; if any is missing -> no backend,
  ``skipped_no_backend``, offline CI does not touch the network). Response shapes are compatible with OpenAI
  ``choices`` and anthropic ``message`` (the observed shape of the volcano apigw ``/compatible`` gateway).
- cache: four-part key ``canonical_sha256(rubric_id, artifact_sha256, model, prompt_version)``
  (§8.2); same artifact + same model hits at zero cost; JSON is written to the work directory (outside the repo).
- degradation: call failure / unparseable output -> ``judge_error`` -> pure rule score + ``llm_missing``
  flag (explicit, never silent); write_back_policy: critical rule failures
  (factual / output_contract / citation / safety / state_diff / no_mutation /
  parsing / boundary wording) can never be compensated by the judge score -> ``passed=False``.
- bounded concurrency (§12 D1): rubric-level thread pool, ``WB_JUDGE_MAX_CONCURRENCY`` (default 4).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import canonical_sha256

JUDGE_PROMPT_VERSION = "wb-composite-judge-v1"
DEFAULT_PASS_THRESHOLD = 0.8

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Bounded retries for transient network failures (5 attempts, backoff 0/3/8/15/30s): remote disconnects, timeouts, 5xx, 429.
# Do not retry 4xx (auth/config errors); fail fast and explicitly. In observed volcano apigw behavior with large concurrent payloads,
# connections drop randomly (RemoteDisconnected / IncompleteRead); 3 attempts are not enough.
_RETRY_BACKOFF_SECONDS = (0, 3, 8, 15, 30)
_RETRYABLE_MARKERS = (
    "remote disconnected",
    "remote end closed connection",
    "incomplete read",
    "connection reset",
    "connection refused",
    "connection aborted",
    "badstatusline",
    "timed out",
    "timeout",
    "llm api request failed",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
)


def _is_retryable(error: Exception) -> bool:
    text = str(error).lower()
    if "http 4" in text and "http 429" not in text:
        return False
    return any(marker in text for marker in _RETRYABLE_MARKERS)

# write_back_policy (judge.yaml score_merge): rule-failure wording that the judge component must not compensate for.
_CRITICAL_FAILURE_KEYWORDS = (
    "factual",
    "output_contract",
    "output-contract",
    "citation",
    "safety",
    "state_diff",
    "state-diff",
    "no_mutation",
    "no-mutation",
    "parsing",
    "boundary",
)


class JudgeCallError(RuntimeError):
    """judge call failed or output was unparseable (the caller degrades to judge_error, never silently)."""


def backend_config() -> Optional[Dict[str, Any]]:
    """Read the judge endpoint from the environment; valid only when all three core variables are present, otherwise there is no backend.

    The key is injected only through env (never stored in a database or repo); ``WB_JUDGE_TEMPERATURE`` defaults to 0.0,
    ``WB_JUDGE_THINKING_LEVEL`` defaults to low (§12 D1),
    ``WB_JUDGE_MAX_CONCURRENCY`` defaults to 4.
    """
    base_url = os.environ.get("WB_JUDGE_BASE_URL") or ""
    api_key = os.environ.get("WB_JUDGE_API_KEY") or ""
    model = os.environ.get("WB_JUDGE_MODEL") or ""
    if not (base_url and api_key and model):
        return None
    thinking_level = os.environ.get("WB_JUDGE_THINKING_LEVEL") or "low"
    if thinking_level.strip().lower() in ("", "none", "off"):
        thinking_level = None  # Explicit disable switch for when the gateway does not support the thinking field
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": float(os.environ.get("WB_JUDGE_TEMPERATURE") or 0.0),
        "thinking_level": thinking_level,
        "max_concurrency": max(
            1, int(os.environ.get("WB_JUDGE_MAX_CONCURRENCY") or 4)
        ),
    }


def judge_fingerprint(config: Dict[str, Any]) -> str:
    """judge configuration fingerprint (model+temperature+thinking_level+prompt_version).

    Included in scoring results for cache and regression source validation; deliberately excludes api_key / base_url.
    """
    material = {
        "model": config["model"],
        "temperature": float(config.get("temperature") or 0.0),
        "thinking_level": config.get("thinking_level") or "low",
        "prompt_version": JUDGE_PROMPT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    candidates = [raw] + [
        match.group(1) for match in _FENCE_RE.finditer(raw)
    ]
    for candidate in candidates:
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


def _rubric_prompt(rubric: Dict[str, Any]) -> str:
    parts = [
        "You are a strict binary judge. Decide the following question using only the provided artifact evidence. "
        'Output exactly one JSON object: {"passed": true|false, "reason": "rationale"}.',
        f"[Question]\n{rubric.get('question')}",
    ]
    for field, title in (
        ("pass_criteria", "Pass criteria"),
        ("fail_criteria", "Fail criteria"),
        ("scope_limits", "Scope limits"),
    ):
        values = rubric.get(field)
        if values:
            lines = "\n".join(f"- {item}" for item in values)
            parts.append(f"[{title}]\n{lines}")
    return "\n\n".join(parts)


def _evidence_block(
    rubric: Dict[str, Any],
    evidence_by_ref: Dict[str, str],
    max_chars: int,
) -> str:
    refs = rubric.get("artifact_refs") or sorted(evidence_by_ref)
    if not refs:
        return "[Artifact evidence]\n(rubric does not reference artifacts)"
    per_ref = max(1, max_chars // len(refs))
    blocks = []
    for ref in refs:
        text = evidence_by_ref.get(ref)
        if text is None:
            blocks.append(f"[Artifact {ref}]\n(evidence missing)")
            continue
        if len(text) > per_ref:
            text = text[:per_ref] + f"\n... (truncated; original {len(text)} characters)"
        blocks.append(f"[Artifact {ref}]\n{text}")
    return "\n\n".join(blocks)


def _extract_response(resp: Any) -> Tuple[str, Dict[str, int]]:
    """Extract (text, tokens) from the client response, compatible with two gateway shapes.

    OpenAI shape: ``choices[0].message.content`` + ``usage.prompt/completion_tokens``;
    anthropic message shape (the observed shape returned by the volcano apigw ``/compatible`` gateway,
    while the request side remains OpenAI-shaped): concatenated ``content`` text blocks + ``usage.input/output_tokens``.
    If neither matches -> JudgeCallError (explicit failure, no silent guessing).
    """
    choices = getattr(resp, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = str(getattr(message, "content", "") or "") if message else ""
        usage = getattr(choices[0], "usage", None) or getattr(resp, "usage", None)
        tokens: Dict[str, int] = {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, name, None) if usage is not None else None
            if isinstance(value, int):
                tokens[name] = value
        return content, tokens
    blocks = getattr(resp, "content", None)
    if isinstance(blocks, list):
        text = "".join(
            block.get("text") or "" for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = getattr(resp, "usage", None)
        tokens = {}
        if isinstance(usage, dict):
            if isinstance(usage.get("input_tokens"), int):
                tokens["prompt_tokens"] = usage["input_tokens"]
            if isinstance(usage.get("output_tokens"), int):
                tokens["completion_tokens"] = usage["output_tokens"]
            if isinstance(usage.get("cache_read_input_tokens"), int) and usage["cache_read_input_tokens"]:
                tokens["cache_read_tokens"] = usage["cache_read_input_tokens"]
        if text:
            return text, tokens
    raise JudgeCallError(
        "judge response shape unrecognized (neither OpenAI choices nor "
        f"anthropic content blocks; model={getattr(resp, 'model', '?')})"
    )


def judge_rubric_binary(
    rubric: Dict[str, Any],
    evidence_by_ref: Dict[str, str],
    config: Dict[str, Any],
    *,
    max_evidence_chars: int = 120_000,
) -> Dict[str, Any]:
    """Single-rubric binary decision: returns binary(0/1), reason, latency, and token usage."""
    started = time.monotonic()
    from curator.utils.llm_api_config import (
        OpenAICompatibleChatClient,
        OpenAICompatibleConfig,
    )

    client = OpenAICompatibleChatClient(OpenAICompatibleConfig(
        base_url=config["base_url"], api_key=config["api_key"],
        model=config["model"], timeout_seconds=180,
        max_concurrency=int(config.get("max_concurrency") or 4),
    ))
    extra: Dict[str, Any] = {}
    thinking_level = config.get("thinking_level")
    if thinking_level:
        extra["thinking"] = {"type": "enabled", "value": thinking_level}
    messages = [
        {"role": "system", "content": _rubric_prompt(rubric)},
        {"role": "user", "content": _evidence_block(
            rubric, evidence_by_ref, max_evidence_chars)},
    ]
    temperature = float(config.get("temperature") or 0.0)
    resp = None
    for attempt, backoff in enumerate(_RETRY_BACKOFF_SECONDS):
        if backoff:
            time.sleep(backoff)
        try:
            resp = client.chat(messages=messages, temperature=temperature,
                               **extra)
            break
        except Exception as error:  # noqa: BLE001 — retry only identified transient failures
            if attempt == len(_RETRY_BACKOFF_SECONDS) - 1 \
                    or not _is_retryable(error):
                raise
    latency = round(time.monotonic() - started, 3)
    content, tokens = _extract_response(resp)
    verdict = _extract_json(content)
    if verdict is None or "passed" not in verdict:
        raise JudgeCallError(
            f"judge output is not parsable (rubric={rubric.get('id')}, "
            f"model={config['model']}, prompt={JUDGE_PROMPT_VERSION})"
        )
    return {
        "binary": 1.0 if verdict.get("passed") is True else 0.0,
        "reason": str(verdict.get("reason") or "").strip()[:500],
        "latency_seconds": latency,
        "tokens": tokens,
    }


def _cache_path(cache_dir: Optional[str], key: str) -> Optional[Path]:
    if not cache_dir:
        return None
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def _cache_read(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _cache_write(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def rubric_cache_key(
    rubric: Dict[str, Any],
    artifact_shas: Dict[str, str],
    config: Dict[str, Any],
) -> str:
    """Four-part cache key (§8.2): rubric_id + cited artifact sha + model + prompt version."""
    refs = rubric.get("artifact_refs") or sorted(artifact_shas)
    artifact_component = ";".join(
        f"{ref}:{artifact_shas.get(ref, 'missing')}" for ref in sorted(refs)
    )
    return canonical_sha256([
        str(rubric.get("id")),
        artifact_component,
        str(config["model"]),
        JUDGE_PROMPT_VERSION,
    ])


def judge_task_rubrics(
    rubrics: List[Dict[str, Any]],
    evidence_by_ref: Dict[str, str],
    artifact_shas: Dict[str, str],
    config: Dict[str, Any],
    *,
    cache_dir: Optional[str] = None,
    max_evidence_chars: int = 120_000,
) -> Dict[str, Any]:
    """Bounded concurrent decision per rubric -> mean binary value (rubric_binary_mean).

    If any rubric call fails / is unparseable -> ``status="judge_error"`` (explicit degradation,
    the caller falls back to the pure rule score + the llm_missing flag); other rubric results are still recorded.
    """
    if not rubrics:
        raise JudgeCallError("judge payload has no rubrics")

    def _one(rubric: Dict[str, Any]) -> Dict[str, Any]:
        rubric_id = str(rubric.get("id"))
        key = rubric_cache_key(rubric, artifact_shas, config)
        path = _cache_path(cache_dir, key)
        cached = _cache_read(path)
        if cached is not None and "binary" in cached:
            return {"rubric_id": rubric_id, "cached": True,
                    "binary": float(cached["binary"]),
                    "reason": str(cached.get("reason") or ""),
                    "latency_seconds": 0.0,
                    "tokens": dict(cached.get("tokens") or {})}
        result = judge_rubric_binary(
            rubric, evidence_by_ref, config,
            max_evidence_chars=max_evidence_chars,
        )
        _cache_write(path, {
            "binary": result["binary"], "reason": result["reason"],
            "tokens": result["tokens"],
            "model": config["model"],
            "prompt_version": JUDGE_PROMPT_VERSION,
        })
        return {"rubric_id": rubric_id, "cached": False, **result}

    workers = max(1, int(config.get("max_concurrency") or 4))
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, rubric): rubric for rubric in rubrics}
        for future, rubric in futures.items():
            try:
                results.append(future.result())
            except JudgeCallError as error:
                errors.append(f"{rubric.get('id')}: {error}")
            except Exception as error:  # noqa: BLE001 — explicit degradation, never silent
                errors.append(
                    f"{rubric.get('id')}: judge call failed: {error!r}"
                )
    total_latency = round(time.monotonic() - started, 3)
    summary = {
        "fingerprint": judge_fingerprint(config),
        "rubric_results": sorted(
            results, key=lambda item: item["rubric_id"]
        ),
        "call_count": sum(1 for item in results if not item["cached"]),
        "cached_count": sum(1 for item in results if item["cached"]),
        "total_latency_seconds": total_latency,
        "prompt_tokens": sum(
            item["tokens"].get("prompt_tokens", 0) for item in results
        ),
        "completion_tokens": sum(
            item["tokens"].get("completion_tokens", 0) for item in results
        ),
    }
    if errors:
        return {"status": "judge_error", "llm_score": None,
                "errors": errors, **summary}
    return {
        "status": "judge_ok",
        "llm_score": round(
            sum(item["binary"] for item in results) / len(results), 4
        ),
        "errors": [],
        **summary,
    }


def load_judge_evidence(
    logs_dir: str, artifacts: List[Dict[str, Any]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Read sidecar artifact evidence from the sandbox ``/logs/verifier`` (produced by the harness, D6).

    ``artifact_manifest.json`` entries are ``{id, text_path, sha256,
    extract_status}``; cited artifacts whose extraction failed are treated as missing evidence (visible to the judge side)
    and are not silently dropped.
    """
    logs = Path(logs_dir)
    manifest_path = logs / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise JudgeCallError(
            f"judge evidence manifest missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise JudgeCallError("judge evidence manifest has no artifacts list")
    evidence_by_ref: Dict[str, str] = {}
    sha_by_ref: Dict[str, str] = {}
    for record in records:
        artifact_id = str(record.get("id"))
        text_path = logs / str(record.get("text_path") or "")
        if record.get("extract_status") != "ok" or not text_path.is_file():
            evidence_by_ref[artifact_id] = (
                f"(evidence extraction failed: {record.get('error') or 'text missing'})"
            )
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        evidence_by_ref[artifact_id] = text
        sha = record.get("sha256")
        if not sha:
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sha_by_ref[artifact_id] = str(sha)
    for artifact in artifacts:
        artifact_id = str(artifact.get("id"))
        if artifact_id and artifact_id not in evidence_by_ref:
            evidence_by_ref[artifact_id] = "(evidence missing: sidecar did not extract this artifact)"
    return evidence_by_ref, sha_by_ref


def critical_rule_failure(rule_diagnostics: Dict[str, Any]) -> Optional[str]:
    """write_back_policy: critical rule-failure detection (the judge component must not compensate)."""
    for check in rule_diagnostics.get("failed_checks") or []:
        marker = " ".join([
            str(check.get("name") or ""),
            str(check.get("group") or ""),
        ]).lower().replace("-", "_")
        if any(keyword in marker
               for keyword in _CRITICAL_FAILURE_KEYWORDS):
            return f"{check.get('group')}/{check.get('name')}"
    return None


def merge_scores(
    rule_rate: float,
    judge_result: Optional[Dict[str, Any]],
    score_merge: Dict[str, Any],
    rule_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """score_merge weighted_sum (§8.1 step 4) + write_back_policy guard.

    judge_ok: ``score = rule_weight*rule + llm_weight*llm``;
    judge_error / no judge: pure rule score + ``llm_missing`` flag.
    In all cases, a critical rule failure (critical_rule_failure) makes ``passed=False``.
    """
    rule_rate = float(rule_rate)
    if not 0.0 <= rule_rate <= 1.0:
        raise ValueError("rule rate outside [0,1]")
    method = str(score_merge.get("method") or "weighted_sum")
    if method != "weighted_sum":
        raise ValueError(f"unsupported score_merge method: {method}")
    rule_weight = float(score_merge.get("rule_weight", 1.0))
    llm_weight = float(score_merge.get("llm_weight", 0.0))
    threshold = float(score_merge.get("pass_threshold")
                      or DEFAULT_PASS_THRESHOLD)
    critical = critical_rule_failure(rule_diagnostics)
    if judge_result is None or judge_result.get("status") != "judge_ok":
        return {
            "score": round(rule_rate, 4),
            "passed": rule_rate >= threshold and critical is None,
            "llm_missing": True,
            "merge_source": "rule_only",
            "critical_failure": critical,
            "llm_score": None,
            "pass_threshold": threshold,
        }
    llm_score = float(judge_result["llm_score"])
    score = rule_weight * rule_rate + llm_weight * llm_score
    return {
        "score": round(score, 4),
        "passed": score >= threshold and critical is None,
        "llm_missing": False,
        "merge_source": "weighted_sum",
        "critical_failure": critical,
        "llm_score": llm_score,
        "pass_threshold": threshold,
    }


__all__ = [
    "DEFAULT_PASS_THRESHOLD",
    "JUDGE_PROMPT_VERSION",
    "JudgeCallError",
    "backend_config",
    "critical_rule_failure",
    "judge_fingerprint",
    "judge_rubric_binary",
    "judge_task_rubrics",
    "load_judge_evidence",
    "merge_scores",
    "rubric_cache_key",
]
