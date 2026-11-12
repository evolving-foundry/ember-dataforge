"""Judge transport-layer tests: mock HTTP endpoint + circuit-breaker unit semantics."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from data_designer.rubric.judge.prompt import JudgeGiveUpError, JudgeProtocolError
from data_designer.rubric.judge.transport import (
    JudgeCircuitBreaker,
    JudgeTransport,
    configure_judge_endpoints,
    get_judge_transport,
    reset_judge_endpoints,
)

DESCRIPTIONS = ("criterion A", "criterion B")


class _JudgeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # Silence access logs
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        server = self.server
        server.requests.append(body)
        behavior = server.behavior
        index = len(server.requests)
        if behavior == "ok" or (behavior == "flaky" and index >= 2):
            payload = json.dumps({"1": "PRESENT", "2": "NOT_PRESENT"})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": payload},
                            }
                        ]
                    }
                ).encode()
            )
        elif behavior == "http_500":
            self.send_response(500)
            self.end_headers()
        elif behavior == "garbage":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": "not json at all"},
                            }
                        ]
                    }
                ).encode()
            )
        else:
            self.send_response(500)
            self.end_headers()


@pytest.fixture()
def judge_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _JudgeHandler)
    server.requests = []
    server.behavior = "ok"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _transport(server, **kwargs):
    return JudgeTransport(
        api_base=f"http://127.0.0.1:{server.server_port}",
        api_key="test-key",
        model="test-judge",
        primary_bases=(f"http://127.0.0.1:{server.server_port}",),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("AUXILIARY_JUDGE_ENABLED", raising=False)
    monkeypatch.setenv("HEALTHBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS", "0.1")
    monkeypatch.setenv("HEALTHBENCH_JUDGE_MAX_ATTEMPTS", "3")
    reset_judge_endpoints()
    yield
    reset_judge_endpoints()


def test_call_judge_roundtrip(judge_server):
    transport = _transport(judge_server)
    result = transport.call_judge(
        query="q", final_answer="answer", descriptions=DESCRIPTIONS
    )
    assert result.scores == (1, 0)
    assert result.attempts == 1
    assert len(result.prompt_sha256) == 64
    request_body = judge_server.requests[0]
    assert request_body["model"] == "test-judge"
    assert request_body["temperature"] == 0
    assert request_body["response_format"]["json_schema"]["strict"] is True


def test_call_judge_retries_then_succeeds(judge_server):
    judge_server.behavior = "flaky"
    transport = _transport(judge_server)
    result = transport.call_judge(
        query="q", final_answer="answer", descriptions=DESCRIPTIONS
    )
    assert result.scores == (1, 0)
    assert result.attempts == 2


def test_call_judge_gives_up_after_budget(judge_server):
    judge_server.behavior = "http_500"
    transport = _transport(judge_server)
    with pytest.raises(JudgeGiveUpError):
        transport.call_judge(
            query="q",
            final_answer="answer",
            descriptions=DESCRIPTIONS,
            max_total_attempts=2,
        )


def test_env_attempt_budget_is_honored(judge_server, monkeypatch):
    """Source-difference fix regression: HEALTHBENCH_JUDGE_MAX_ATTEMPTS must take effect
    (when no parameter is passed).
    """
    monkeypatch.setenv("HEALTHBENCH_JUDGE_MAX_ATTEMPTS", "2")
    judge_server.behavior = "http_500"
    transport = _transport(judge_server)
    with pytest.raises(JudgeGiveUpError):
        transport.call_judge(
            query="q", final_answer="answer", descriptions=DESCRIPTIONS
        )
    assert len(judge_server.requests) >= 2


def test_strict_response_rejects_malformed(judge_server, monkeypatch):
    monkeypatch.setenv("HEALTHBENCH_JUDGE_MAX_PROTOCOL_ATTEMPTS", "2")
    judge_server.behavior = "garbage"
    transport = _transport(judge_server)
    with pytest.raises(JudgeGiveUpError):
        transport.call_judge(
            query="q",
            final_answer="answer",
            descriptions=DESCRIPTIONS,
            strict_response=True,
            max_total_attempts=3,
        )


def test_breaker_flips_to_backup_and_recovers():
    breaker = JudgeCircuitBreaker(
        ("http://primary",),
        "http://backup",
        flip_threshold=2,
        min_hold_seconds=0.0,
        backup_escape_after=2,
        probe_successes=2,
        probe_interval_seconds=0.0,
    )
    assert breaker.current_base() == ("http://primary", False)
    breaker.on_failure(False, "http://primary")
    assert breaker.current_base() == ("http://primary", False)
    breaker.on_failure(False, "http://primary")
    assert breaker.current_base() == ("http://backup", True)
    # Backup endpoint escape: after min_hold, consecutive failures reach the threshold
    # and switch back to the primary endpoint
    breaker.on_failure(True, "http://backup")
    breaker.on_failure(True, "http://backup")
    assert breaker.current_base() == ("http://primary", False)


def test_breaker_probe_recovery_from_backup():
    breaker = JudgeCircuitBreaker(
        ("http://primary",),
        "http://backup",
        flip_threshold=1,
        min_hold_seconds=0.0,
        probe_successes=2,
        probe_interval_seconds=0.0,
    )
    breaker.on_failure(False, "http://primary")
    assert breaker.current_base() == ("http://backup", True)
    breaker.on_success(True, "http://backup")
    breaker.on_success(True, "http://backup")
    assert breaker.current_base() == ("http://primary", False)


def test_breaker_requires_primary_for_backup():
    with pytest.raises(Exception):
        JudgeCircuitBreaker((), "http://backup")


def test_configure_and_reset_endpoints(judge_server):
    transport = configure_judge_endpoints(
        api_base=f"http://127.0.0.1:{judge_server.server_port}",
        api_key="injected-key",
        model="injected-judge",
    )
    assert get_judge_transport() is transport
    result = transport.call_judge(
        query="q", final_answer="answer", descriptions=DESCRIPTIONS
    )
    assert result.scores == (1, 0)
    assert judge_server.requests[0]["model"] == "injected-judge"
    reset_judge_endpoints()
    # After reset, return to the lazy env instance (a different object)
    assert get_judge_transport() is not transport


def test_configure_rejects_blank_credentials():
    with pytest.raises(Exception):
        configure_judge_endpoints(api_base="http://x", api_key=" ", model="m")


def test_legacy_env_base_becomes_single_primary(monkeypatch):
    """When only HEALTHBENCH_JUDGE_API_BASE is configured (legacy single endpoint), fall
    back to that sole primary endpoint and do not crash before the request
    (IndexError regression guard).
    """
    monkeypatch.delenv("HEALTHBENCH_JUDGE_CANONICAL_API_BASE", raising=False)
    monkeypatch.delenv("HEALTHBENCH_JUDGE_SPLIT_SECONDARY_API_BASE", raising=False)
    monkeypatch.delenv("HEALTHBENCH_JUDGE_API_BASE_BACKUP", raising=False)
    legacy = "http://127.0.0.1:1"
    monkeypatch.setenv("HEALTHBENCH_JUDGE_API_BASE", legacy)
    transport = JudgeTransport(api_key="k", model="m")
    assert transport.breaker.primary_bases == (legacy,)
    base, is_backup = transport.breaker.current_base(None)
    assert base == legacy and is_backup is False


def test_transport_without_any_endpoint_fails_explicitly(monkeypatch):
    monkeypatch.delenv("HEALTHBENCH_JUDGE_CANONICAL_API_BASE", raising=False)
    monkeypatch.delenv("HEALTHBENCH_JUDGE_SPLIT_SECONDARY_API_BASE", raising=False)
    monkeypatch.delenv("HEALTHBENCH_JUDGE_API_BASE_BACKUP", raising=False)
    monkeypatch.delenv("HEALTHBENCH_JUDGE_API_BASE", raising=False)
    transport = JudgeTransport(api_key="k", model="m")
    with pytest.raises(JudgeProtocolError):
        transport.breaker.current_base(None)
