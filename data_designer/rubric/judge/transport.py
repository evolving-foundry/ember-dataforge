"""Judge endpoint transport layer (migrated from RuACT verl_reward.py, decoupled from
verl; global state converted to instance state). Source policies retained: - Circuit
breaker: after consecutive failures reach a threshold (randomized to avoid
synchronized flapping), switch to a backup endpoint; switch back only after a min-
hold plus consecutive failures to escape or consecutive successes during probing. -
Retry budget: attempt-count and deadline constraints; 429 and 5xx go through the
circuit breaker; permanent 4xx errors accumulate to a limit and then give up
explicitly; budget exhaustion raises JudgeGiveUpError -- **never fabricate success**.
- Malformed-response chunk retries: the previous invalid response is fed back in a
corrective conversation request. - Auxiliary Judge: when AUXILIARY_JUDGE_ENABLED=1,
use a fallback path; failures enter a 60s cooldown; optional half-routing by query
hash (Refinement_AUXILIARY_SPLIT_BY_QUERY). Credential injection (design document
section 6.4): endpoint credentials are injected explicitly through
:func:`configure_judge_endpoints`; when not injected, fall back to
HEALTHBENCH_JUDGE_* environment variables. Credentials are never published in task
payloads or written to audit logs.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, NamedTuple

from .prompt import (
    JudgeGiveUpError,
    JudgeProtocolError,
    build_judge_prompt,
    parse_judge_reply,
    require_nonempty_text,
)


class JudgeResult(NamedTuple):
    scores: tuple[int, ...]
    attempts: int
    prompt_sha256: str


def _env_int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


class JudgeCircuitBreaker:
    """Primary/backup endpoint circuit breaker (instance form of the source
    verl_reward.py global state). `flip_threshold` follows the source randomization:
    `randint(2, max(2, env))`, avoiding synchronized flips across workers.
    """

    def __init__(
        self,
        primary_bases: tuple[str, ...],
        backup_base: str,
        *,
        flip_threshold: int | None = None,
        min_hold_seconds: float | None = None,
        backup_escape_after: int | None = None,
        probe_successes: int | None = None,
        probe_interval_seconds: float | None = None,
    ) -> None:
        if backup_base and not primary_bases:
            raise JudgeProtocolError("backup base requires at least one primary base")
        self.primary_bases = tuple(primary_bases)
        self.backup_base = backup_base
        self.flip_threshold = (
            flip_threshold
            if flip_threshold is not None
            else random.randint(2, max(2, _env_int("HEALTHBENCH_JUDGE_FAILOVER_FLIP_AFTER", "3")))
        )
        self.min_hold_seconds = (
            min_hold_seconds
            if min_hold_seconds is not None
            else _env_float("HEALTHBENCH_JUDGE_FAILOVER_MIN_HOLD_SECONDS", "20")
        )
        self.backup_escape_after = (
            backup_escape_after
            if backup_escape_after is not None
            else max(3, _env_int("HEALTHBENCH_JUDGE_FAILOVER_ESCAPE_AFTER", "4"))
        )
        self.probe_successes = (
            probe_successes
            if probe_successes is not None
            else max(8, _env_int("HEALTHBENCH_JUDGE_FAILOVER_PROBE_SUCCESSES", "32"))
        )
        self.probe_interval_seconds = (
            probe_interval_seconds
            if probe_interval_seconds is not None
            else _env_float("HEALTHBENCH_JUDGE_FAILOVER_PROBE_INTERVAL_SECONDS", "240")
        )
        now = time.monotonic()
        self._tripped: dict[str, bool] = {base: False for base in self.primary_bases}
        self._state: dict[str, dict[str, Any]] = {
            base: {"fails": 0, "oks": 0, "since": now}
            for base in self.primary_bases
        }
        self._backup_state: dict[str, Any] = {"fails": 0, "oks": 0, "since": now}

    @property
    def split_enabled(self) -> bool:
        return len(self.primary_bases) > 1

    def assign_base(self, routing_key: str) -> str | None:
        """Stable sharding across primary endpoints by query routing key (split mode)."""
        if not self.split_enabled:
            return None
        digest = hashlib.sha256(routing_key.encode("utf-8")).hexdigest()
        return self.primary_bases[0 if int(digest[:2], 16) < 128 else 1]

    def current_base(self, assigned_base: str | None = None) -> tuple[str, bool]:
        if assigned_base is not None:
            target = assigned_base
        elif self.primary_bases:
            target = self.primary_bases[0]
        else:
            raise JudgeProtocolError(
                "no judge primary endpoint configured "
                "(set HEALTHBENCH_JUDGE_CANONICAL_API_BASE or HEALTHBENCH_JUDGE_API_BASE)"
            )
        if self._tripped.get(target, False):
            return (self.backup_base, True)
        return (target, False)

    def _untrip_all(self) -> None:
        now = time.monotonic()
        for base in self.primary_bases:
            self._tripped[base] = False
            state = self._state[base]
            state["fails"] = 0
            state["oks"] = 0
            state["since"] = now
        self._backup_state["fails"] = 0
        self._backup_state["oks"] = 0
        self._backup_state["since"] = now

    def on_failure(self, is_backup: bool, base_url: str | None = None) -> None:
        now = time.monotonic()
        if is_backup:
            if not any(self._tripped.values()):
                return
            state = self._backup_state
            state["fails"] += 1
            state["oks"] = 0
            if (
                now - state["since"] >= self.min_hold_seconds
                and state["fails"] >= self.backup_escape_after
            ):
                self._untrip_all()
                print(
                    f"JUDGE_FAILOVER action=backup_escape_to_primary base={self.primary_bases[0]} consecutive_failures={self.backup_escape_after}",
                    flush=True,
                )
            return
        target = base_url if base_url is not None else self.primary_bases[0]
        if self._tripped.get(target, False):
            return
        state = self._state[target]
        state["fails"] += 1
        state["oks"] = 0
        if now - state["since"] >= self.min_hold_seconds and state["fails"] >= self.flip_threshold:
            self._tripped[target] = True
            state["fails"] = 0
            state["since"] = now
            self._backup_state["since"] = now
            print(
                f"JUDGE_FAILOVER action=primary_to_backup base={self.backup_base} consecutive_failures={self.flip_threshold}",
                flush=True,
            )

    def on_success(self, is_backup: bool, base_url: str | None = None) -> None:
        now = time.monotonic()
        if is_backup:
            if not any(self._tripped.values()):
                return
            state = self._backup_state
            state["oks"] += 1
            state["fails"] = 0
            if (
                state["oks"] >= self.probe_successes
                and now - state["since"] >= self.probe_interval_seconds
            ):
                self._untrip_all()
                print(
                    f"JUDGE_FAILOVER action=backup_probe_to_primary base={self.primary_bases[0]} consecutive_successes={self.probe_successes}",
                    flush=True,
                )
            return
        target = base_url if base_url is not None else self.primary_bases[0]
        if self._tripped.get(target, False):
            return
        state = self._state[target]
        state["oks"] += 1
        state["fails"] = 0


_AUXILIARY_COOLDOWN_UNTIL = 0.0
_AUXILIARY_ROUTE_LOCAL = threading.local()


def _request_auxiliary_judge(
    request_body: dict, *, rubric_count: int, deadline: float | None = None
) -> dict | None:
    global _AUXILIARY_COOLDOWN_UNTIL
    if (
        os.environ.get("AUXILIARY_JUDGE_ENABLED") != "1"
        or time.monotonic() < _AUXILIARY_COOLDOWN_UNTIL
    ):
        return None
    if os.environ.get("Refinement_AUXILIARY_SPLIT_BY_QUERY") == "1" and (
        not getattr(_AUXILIARY_ROUTE_LOCAL, "use_auxiliary", True)
    ):
        return None
    base = require_nonempty_text(
        os.environ.get("AUXILIARY_JUDGE_API_BASE"), "auxiliary base"
    ).rstrip("/")
    key = require_nonempty_text(
        os.environ.get("AUXILIARY_JUDGE_API_KEY"), "auxiliary key"
    )
    model = require_nonempty_text(
        os.environ.get("AUXILIARY_JUDGE_MODEL", ""), "auxiliary model"
    )
    payload = {
        **request_body,
        "model": model,
        "max_tokens": max(512, int(request_body["max_tokens"])),
    }
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        timeout = float(os.environ.get("AUXILIARY_JUDGE_TIMEOUT_SECONDS", "45"))
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("auxiliary judge deadline exhausted")
            timeout = min(timeout, remaining)
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            body = json.load(reply)
        choice = body["choices"][0]
        content = choice["message"]["content"].strip()
        fence = chr(96) * 3
        if content.startswith(fence):
            content = re.sub(
                "^" + fence + "(?:json)?\\s*|\\s*" + fence + "$", "", content
            ).strip()
        scores = json.loads(content)
        expected = {str(i) for i in range(1, rubric_count + 1)}
        if (
            choice.get("finish_reason") != "stop"
            or not isinstance(scores, dict)
            or set(scores) != expected
            or any((v not in ("PRESENT", "NOT_PRESENT") for v in scores.values()))
        ):
            raise ValueError("incomplete auxiliary Judge result")
    except (
        urllib.error.URLError,
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as error:
        _AUXILIARY_COOLDOWN_UNTIL = time.monotonic() + 60.0
        print(
            json.dumps(
                {
                    "event": "auxiliary_judge_fallback",
                    "error_type": type(error).__name__,
                    "status": getattr(error, "code", None),
                    "cooldown_seconds": 60,
                }
            ),
            flush=True,
        )
        return None
    print(
        json.dumps(
            {
                "event": "auxiliary_judge_success",
                "base": base,
                "model": model,
                "rubric_count": rubric_count,
                "seconds": round(time.monotonic() - started, 3),
            }
        ),
        flush=True,
    )
    body["choices"][0]["message"]["content"] = json.dumps(scores, separators=(",", ":"))
    return body


class JudgeTransport:
    """Singleton Judge request path: request construction, circuit-breaker routing, and
    retry budget.
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        primary_bases: tuple[str, ...] | None = None,
        backup_base: str | None = None,
        breaker: JudgeCircuitBreaker | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/") if api_base else None
        self.api_key = api_key
        self.model = model
        env_primary_bases = tuple(
            base.rstrip("/")
            for base in (
                os.environ.get("HEALTHBENCH_JUDGE_CANONICAL_API_BASE", ""),
                os.environ.get("HEALTHBENCH_JUDGE_SPLIT_SECONDARY_API_BASE", ""),
            )
            if base.rstrip("/")
        )
        if primary_bases is not None:
            resolved_primary_bases = tuple(base.rstrip("/") for base in primary_bases)
        elif self.api_base and not env_primary_bases:
            resolved_primary_bases = (self.api_base,)
        else:
            resolved_primary_bases = env_primary_bases
        if not resolved_primary_bases:
            legacy_base = os.environ.get("HEALTHBENCH_JUDGE_API_BASE", "").rstrip("/")
            if legacy_base:
                resolved_primary_bases = (legacy_base,)
        resolved_backup = (backup_base or os.environ.get("HEALTHBENCH_JUDGE_API_BASE_BACKUP", "")).rstrip("/")
        self.breaker = breaker or JudgeCircuitBreaker(
            resolved_primary_bases,
            resolved_backup,
        )

    def assign_group_base(self, prompt_id: str, routing_step: int) -> str | None:
        return self.breaker.assign_base(f"{prompt_id}\x1f{routing_step}")

    def _resolve(self, value: str | None, env_name: str, field: str) -> str:
        if value:
            return value
        return require_nonempty_text(os.environ.get(env_name), env_name)

    def request_once(
        self,
        prompt: str,
        *,
        rubric_count: int,
        previous_invalid_reply: str | None = None,
        base_url: str | None = None,
        use_response_format: bool = True,
        deadline: float | None = None,
        max_tokens_override: int | None = None,
    ) -> str:
        base_url = self._resolve(
            base_url, "HEALTHBENCH_JUDGE_API_BASE", "judge base"
        ).rstrip("/")
        api_key = self._resolve(self.api_key, "HEALTHBENCH_JUDGE_API_KEY", "judge key")
        model = self._resolve(self.model, "HEALTHBENCH_JUDGE_MODEL", "judge model")
        expected_keys = tuple((str(index) for index in range(1, rubric_count + 1)))
        score_schema = {
            "type": "object",
            "properties": {
                key: {"type": "string", "enum": ["PRESENT", "NOT_PRESENT"]}
                for key in expected_keys
            },
            "required": list(expected_keys),
            "additionalProperties": False,
        }
        messages = [{"role": "user", "content": prompt}]
        if previous_invalid_reply is not None:
            messages.extend(
                [
                    {"role": "assistant", "content": previous_invalid_reply[:8192]},
                    {
                        "role": "user",
                        "content": f"The previous object violated the required key schema. Return a corrected JSON object containing exactly these keys and no others: {', '.join(expected_keys)}. Do not add explanations.",
                    },
                ]
            )
        request_body = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": (
                max_tokens_override
                if max_tokens_override is not None
                else _env_int("HEALTHBENCH_JUDGE_MAX_TOKENS", "4096")
            ),
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if use_response_format:
            request_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "rubric_scores",
                    "strict": True,
                    "schema": score_schema,
                },
            }
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(
                request_body, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        timeout = _env_float("HEALTHBENCH_JUDGE_TIMEOUT_SECONDS", "180")
        body = _request_auxiliary_judge(
            request_body, rubric_count=rubric_count, deadline=deadline
        )
        if body is None:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("judge deadline exhausted before primary request")
                timeout = min(timeout, remaining)
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                body = json.load(reply)
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return ""
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                (part.get("text", "") for part in content if isinstance(part, dict))
            )
        if not isinstance(content, str):
            content = ""
        return content

    def call_judge(
        self,
        *,
        query: str,
        final_answer: str,
        descriptions: tuple[str, ...],
        group_base: str | None = None,
        max_total_attempts: int | None = None,
        deadline: float | None = None,
        strict_response: bool = False,
    ) -> JudgeResult:
        _AUXILIARY_ROUTE_LOCAL.use_auxiliary = (
            int(hashlib.sha256(query.encode("utf-8")).hexdigest(), 16) % 2 == 0
        )
        prompt = build_judge_prompt(
            query=query,
            final_answer=final_answer,
            descriptions=descriptions,
        )
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        # Source difference fix: HEALTHBENCH_JUDGE_MAX_ATTEMPTS never took effect inside
        # the RuACT verl_reward.py loop (dead variable). The inference side applies the
        # 'explicit failure on budget exhaustion' policy, making it the default total
        # attempt budget; callers can still override with max_total_attempts.
        env_attempts = _env_int("HEALTHBENCH_JUDGE_MAX_ATTEMPTS", "8")
        if env_attempts <= 0:
            raise JudgeProtocolError("HEALTHBENCH_JUDGE_MAX_ATTEMPTS must be positive")
        if max_total_attempts is None:
            max_total_attempts = env_attempts
        protocol_failures = 0
        attempt = 0
        previous_invalid_reply = None
        schema_failures = 0
        while True:
            if (
                max_total_attempts is not None
                and attempt >= max_total_attempts
                or (deadline is not None and time.monotonic() >= deadline)
            ):
                raise JudgeGiveUpError("judge total attempt/time budget exhausted")
            attempt += 1
            base_url, is_backup = self.breaker.current_base(group_base)
            try:
                content = self.request_once(
                    prompt,
                    rubric_count=len(descriptions),
                    previous_invalid_reply=previous_invalid_reply,
                    base_url=base_url,
                    use_response_format=not is_backup,
                    deadline=deadline,
                )
                self.breaker.on_success(is_backup, base_url)
                try:
                    if strict_response:
                        try:
                            parsed = json.loads(content)
                        except (ValueError, TypeError) as error:
                            raise JudgeProtocolError(
                                "strict judge response is not JSON"
                            ) from error
                        if (
                            not isinstance(parsed, dict)
                            or set(parsed)
                            != {str(i) for i in range(1, len(descriptions) + 1)}
                            or any(
                                (
                                    value not in ("PRESENT", "NOT_PRESENT")
                                    for value in parsed.values()
                                )
                            )
                        ):
                            raise JudgeProtocolError(
                                "strict judge response has missing/invalid labels"
                            )
                    scores = parse_judge_reply(
                        content,
                        rubric_count=len(descriptions),
                        prompt_sha256=prompt_sha256,
                    )
                except JudgeProtocolError:
                    protocol_failures += 1
                    schema_failures += 1
                    if (
                        max_total_attempts is not None
                        and attempt >= max_total_attempts
                        or schema_failures
                        >= _env_int("HEALTHBENCH_JUDGE_MAX_PROTOCOL_ATTEMPTS", "5")
                    ):
                        print(
                            f"JUDGE_GIVEUP prompt_sha256={prompt_sha256} attempts={attempt} schema_failures={schema_failures} expected_count={len(descriptions)}",
                            flush=True,
                        )
                        raise JudgeGiveUpError(
                            f"judge protocol give up after {schema_failures} schema failures"
                        )
                    previous_invalid_reply = content
                    try:
                        observed = json.loads(content)
                        observed_keys = (
                            sorted(observed)
                            if isinstance(observed, dict)
                            else [type(observed).__name__]
                        )
                    except (json.JSONDecodeError, TypeError):
                        observed_keys = ["non_json"]
                    print(
                        f"JUDGE_PROTOCOL_RETRY prompt_sha256={prompt_sha256} attempt={attempt} expected_count={len(descriptions)} observed_keys={observed_keys}",
                        flush=True,
                    )
                    backoff_cap = _env_float(
                        "HEALTHBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS", "30"
                    )
                    time.sleep(
                        min(max(1.0, backoff_cap), 1.5 * attempt)
                        * random.uniform(0.5, 1.5)
                    )
                    continue
                return JudgeResult(
                    scores=scores, attempts=attempt, prompt_sha256=prompt_sha256
                )
            except JudgeGiveUpError:
                raise
            except urllib.error.HTTPError as error:
                protocol_failures += 1
                print(
                    f"JUDGE_API_RETRY prompt_sha256={prompt_sha256} attempt={attempt} status={error.code} consecutive_failures={protocol_failures}",
                    flush=True,
                )
                if error.code == 429 or not 400 <= error.code < 500:
                    self.breaker.on_failure(is_backup, base_url)
                elif protocol_failures >= _env_int(
                    "HEALTHBENCH_JUDGE_MAX_PERMANENT_4XX", "5"
                ):
                    print(
                        f"JUDGE_GIVEUP prompt_sha256={prompt_sha256} attempts={attempt} status={error.code} reason=permanent_client_error",
                        flush=True,
                    )
                    raise JudgeGiveUpError(
                        f"judge permanent client error {error.code} after {attempt} attempts"
                    )
            except (TimeoutError, ConnectionError, urllib.error.URLError, OSError) as error:
                protocol_failures += 1
                print(
                    f"JUDGE_API_RETRY prompt_sha256={prompt_sha256} attempt={attempt} error={type(error).__name__} consecutive_failures={protocol_failures}",
                    flush=True,
                )
                self.breaker.on_failure(is_backup, base_url)
            except JudgeProtocolError as error:
                protocol_failures += 1
                print(
                    f"JUDGE_PROTOCOL_RETRY prompt_sha256={prompt_sha256} attempt={attempt} expected_count={len(descriptions)} error={type(error).__name__}",
                    flush=True,
                )
            except Exception as error:
                protocol_failures += 1
                print(
                    f"JUDGE_RETRY prompt_sha256={prompt_sha256} attempt={attempt} error={type(error).__name__} consecutive_failures={protocol_failures}",
                    flush=True,
                )
                self.breaker.on_failure(is_backup, base_url)
            if max_total_attempts is not None and attempt >= max_total_attempts:
                raise JudgeGiveUpError("judge total attempt budget exhausted")
            if deadline is not None and time.monotonic() >= deadline:
                raise JudgeGiveUpError("judge time budget exhausted")
            backoff_cap = _env_float(
                "HEALTHBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS", "30"
            )
            time.sleep(
                min(max(1.0, backoff_cap), 1.5 * attempt) * random.uniform(0.5, 1.5)
            )


_default_transport: JudgeTransport | None = None
_transport_lock = threading.Lock()


def configure_judge_endpoints(
    *,
    api_base: str,
    api_key: str,
    model: str,
    primary_bases: tuple[str, ...] | None = None,
    backup_base: str | None = None,
) -> JudgeTransport:
    """Explicitly inject Judge endpoints (grading/production credential policy, aligned
    with configure_llm_judge).
    """
    global _default_transport
    if not api_base.strip() or not api_key.strip() or not model.strip():
        raise JudgeProtocolError(
            "configure_judge_endpoints requires non-empty api_base/api_key/model"
        )
    transport = JudgeTransport(
        api_base=api_base,
        api_key=api_key,
        model=model,
        primary_bases=primary_bases,
        backup_base=backup_base,
    )
    with _transport_lock:
        _default_transport = transport
    return transport


def get_judge_transport() -> JudgeTransport:
    """Return the injected default transport; when none is injected, construct a lazy
    environment-driven instance.
    """
    global _default_transport
    with _transport_lock:
        if _default_transport is None:
            _default_transport = JudgeTransport()
        return _default_transport


def reset_judge_endpoints() -> None:
    """Clear explicit injection (for test isolation)."""
    global _default_transport
    with _transport_lock:
        _default_transport = None
