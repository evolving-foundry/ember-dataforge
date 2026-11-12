"""Generator HTTP client (migrated from RuACT refinement/transport.py, decoupled from
verl/httpx). Physically separated from the Judge path: Generator endpoint credentials
are injected independently (HEALTHBENCH_GENERATOR_API_BASE/KEY/MODEL, design document
section 6.4). Source policies retained: - Requests carry a json_schema
`response_format`; non-JSON replies are fed back as a 'previous reply' correction
request (repair), and retries are counted in the audit. - Content diagnostics record
only length/hash/finish_reason, never the raw text (to prevent data leakage into
logs). - Budget exhaustion fails explicitly -- **never fabricate 'no candidates' as
success** (source README.md:133); the total attempt budget is enforced explicitly by
the caller (evolve pipeline parent budget: 3 attempts / 40s each / 120s total).
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ..protocol.schemas import PositiveRefinementProtocolError

WorkerError = PositiveRefinementProtocolError


@dataclass(frozen=True)
class WorkerConfig:
    base_url: str = ""
    api_key: str = ""
    backend: str = "litellm_http"
    model: str = ""
    enable_thinking: bool = False
    max_tokens: int = 4096
    temperature: float = 0.8
    meta_model: str | None = None
    max_attempts: int = 3
    timeout_seconds: float = 180.0
    retry_backoff_seconds: float = 2.0
    generator_concurrency: int = 64

    def validate(self) -> None:
        if not self.base_url.strip():
            raise WorkerError("LiteLLM base_url must be non-empty")
        if not self.model.strip():
            raise WorkerError("LiteLLM model must be non-empty")
        if self.timeout_seconds <= 0 or self.retry_backoff_seconds < 0:
            raise WorkerError("generator timeout/backoff is invalid")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or (not 1 <= self.max_attempts <= 10)
        ):
            raise WorkerError("generator attempts must be between one and ten")
        if self.meta_model is not None and (not self.meta_model.strip()):
            raise WorkerError("Meta model override must be non-empty")
        if self.max_tokens <= 0 or not 0 <= self.temperature <= 2:
            raise WorkerError("generator max_tokens/temperature is invalid")
        if (
            isinstance(self.generator_concurrency, bool)
            or not isinstance(self.generator_concurrency, int)
            or self.generator_concurrency <= 0
        ):
            raise WorkerError("generator concurrency must be positive")

    def for_meta(self) -> "WorkerConfig":
        return replace(self, model=self.meta_model or self.model)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def content_diagnostic(content: Any, error: BaseException, *, finish_reason: Any) -> str:
    if not isinstance(content, str):
        return f"generator assistant content is not text (type={type(content).__name__}, finish_reason={finish_reason!r})"
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    location = ""
    if isinstance(error, json.JSONDecodeError):
        location = f", line={error.lineno}, column={error.colno}, position={error.pos}"
    return f"generator assistant content is not valid JSON (length={len(content)}, sha256={digest}, finish_reason={finish_reason!r}{location})"


def parse_generator_json(
    content: Any, *, finish_reason: Any
) -> tuple[dict[str, Any], bool]:
    if not isinstance(content, str):
        error = TypeError("assistant content is not text")
        raise WorkerError(
            content_diagnostic(content, error, finish_reason=finish_reason)
        ) from error
    stripped = strip_code_fence(content)
    try:
        value = json.loads(stripped)
        recovered = False
    except json.JSONDecodeError as initial_error:
        object_start = stripped.find("{")
        if object_start < 0:
            raise WorkerError(
                content_diagnostic(content, initial_error, finish_reason=finish_reason)
            ) from initial_error
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped[object_start:])
            recovered = True
        except json.JSONDecodeError as recovery_error:
            raise WorkerError(
                content_diagnostic(
                    content, recovery_error, finish_reason=finish_reason
                )
            ) from recovery_error
    if not isinstance(value, dict):
        error = ValueError("generator response is not a JSON object")
        raise WorkerError(
            content_diagnostic(content, error, finish_reason=finish_reason)
        ) from error
    return (value, recovered)


def repair_messages(prompt: str, invalid_content: Any) -> list[dict[str, str]]:
    prior = invalid_content if isinstance(invalid_content, str) else ""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": prior[:8192]},
        {
            "role": "user",
            "content": "The previous reply was not valid JSON. Return one corrected JSON object that strictly matches the response_format schema. Output JSON only: no markdown fence, explanation, or text before or after the object.",
        },
    ]


def generate_json(
    prompt: str,
    schema: Mapping[str, Any],
    *,
    config: WorkerConfig,
    deadline: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One Generator JSON call with repair retries; return (value, metadata). JSON
    parsing and schema requests complete within `config.max_attempts`; HTTP/network
    errors retry with exponential backoff. Budget exhaustion raises WorkerError --
    explicit failure, never fabricated success.
    """
    config.validate()
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "rubric_generation",
                "strict": True,
                "schema": schema,
            },
        },
        "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    started = time.monotonic()
    last_error: BaseException | None = None
    repair_attempts = 0
    recovered_json = False
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(1, config.max_attempts + 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        payload["messages"] = messages
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            timeout = config.timeout_seconds
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                timeout = min(timeout, remaining)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                raw_body = response.read()
            if status >= 400:
                raise WorkerError(f"generator HTTP status {status}")
            try:
                body = json.loads(raw_body)
            except json.JSONDecodeError as error:
                digest = hashlib.sha256(raw_body).hexdigest()
                raise WorkerError(
                    f"generator HTTP body is not JSON (status={status}, length={len(raw_body)}, sha256={digest}, position={error.pos})"
                ) from error
            choice = body["choices"][0]
            message = choice["message"]
            content = message.get("content")
            try:
                value, recovered_json = parse_generator_json(
                    content, finish_reason=choice.get("finish_reason")
                )
            except WorkerError:
                if attempt < config.max_attempts:
                    repair_attempts += 1
                    messages = repair_messages(prompt, content)
                raise
            return (
                value,
                {
                    "backend": config.backend,
                    "model": config.model,
                    "enable_thinking": config.enable_thinking,
                    "temperature": config.temperature,
                    "attempts": attempt,
                    "repair_attempts": repair_attempts,
                    "json_wrapper_recovered": recovered_json,
                    "latency_seconds": round(time.monotonic() - started, 3),
                },
            )
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            OSError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            WorkerError,
        ) as error:
            last_error = error
            if attempt < config.max_attempts:
                time.sleep(
                    min(30.0, config.retry_backoff_seconds * 2 ** min(attempt - 1, 4))
                )
    error_detail = (
        f"{type(last_error).__name__}: {last_error}"
        if last_error is not None
        else "deadline exhausted"
    )
    raise WorkerError(
        f"generator HTTP failed after {config.max_attempts} attempts ({error_detail})"
    )
