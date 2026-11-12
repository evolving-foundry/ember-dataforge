"""P3: Generator HTTP client -- JSON parsing/repair retries, explicit budget failure,
and credential isolation.
"""
from __future__ import annotations

import json

import pytest

from data_designer.rubric.generator.client import (
    WorkerConfig,
    WorkerError,
    content_diagnostic,
    generate_json,
    parse_generator_json,
    repair_messages,
)
from tests.rubric.evolve_fixture import mock_generator_server

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title"],
                "properties": {"title": {"type": "string"}},
            },
        }
    },
}


def _config(base_url, **overrides):
    config = WorkerConfig(
        base_url=base_url,
        api_key="secret-key-do-not-leak",
        model="mock-model",
        max_attempts=3,
        retry_backoff_seconds=0.0,
        **overrides,
    )
    config.validate()
    return config


def test_parse_generator_json_fences_and_recovery():
    value, recovered = parse_generator_json(
        '{"candidates": []}', finish_reason="stop"
    )
    assert value == {"candidates": []} and recovered is False
    value, recovered = parse_generator_json(
        '```json\n{"candidates": []}\n```', finish_reason="stop"
    )
    assert value == {"candidates": []} and recovered is False
    value, recovered = parse_generator_json(
        'Sure! {"candidates": []} hope that helps', finish_reason="stop"
    )
    assert value == {"candidates": []} and recovered is True
    with pytest.raises(WorkerError, match="not valid JSON"):
        parse_generator_json("no json here", finish_reason="stop")


def test_content_diagnostic_never_leaks_content():
    error = json.JSONDecodeError("expecting value", "super secret raw content", 0)
    diagnostic = content_diagnostic(
        "super secret raw content", error, finish_reason="stop"
    )
    assert "super secret raw content" not in diagnostic
    assert "length=" in diagnostic and "sha256=" in diagnostic
    assert "finish_reason='stop'" in diagnostic


def test_repair_messages_truncates_previous_reply():
    messages = repair_messages("the prompt", "x" * 10000)
    assert len(messages) == 3
    assert messages[1]["content"] == "x" * 8192
    assert "previous reply was not valid JSON" in messages[2]["content"]


def test_generate_json_success_and_request_shape():
    server = mock_generator_server("ok")
    threads = __import__("threading").Thread(target=server.serve_forever, daemon=True)
    threads.start()
    try:
        value, metadata = generate_json(
            "make candidates", SCHEMA, config=_config(f"http://127.0.0.1:{server.server_address[1]}")
        )
        assert value == {
            "candidates": [{"title": "demo candidate 0"}]
        }
        assert metadata["attempts"] == 1
        assert metadata["repair_attempts"] == 0
        request = server.requests[0]
        assert request["model"] == "mock-model"
        assert request["response_format"]["json_schema"]["strict"] is True
        assert request["messages"][0]["content"] == "make candidates"
    finally:
        server.shutdown()


def test_generate_json_repair_retry_on_invalid_json():
    server = mock_generator_server("invalid_json")
    threads = __import__("threading").Thread(target=server.serve_forever, daemon=True)
    threads.start()
    try:
        with pytest.raises(WorkerError, match="after 3 attempts"):
            generate_json(
                "make candidates",
                SCHEMA,
                config=_config(f"http://127.0.0.1:{server.server_address[1]}"),
            )
        # After each failure, feed back a repair message and request again
        assert len(server.requests) == 3
        assert len(server.requests[-1]["messages"]) == 3
        assert "not valid JSON" in server.requests[-1]["messages"][2]["content"]
    finally:
        server.shutdown()


def test_generate_json_endpoint_down_fails_explicitly_with_key_isolation():
    server = mock_generator_server("down")
    threads = __import__("threading").Thread(target=server.serve_forever, daemon=True)
    threads.start()
    try:
        with pytest.raises(WorkerError) as excinfo:
            generate_json(
                "make candidates",
                SCHEMA,
                config=_config(f"http://127.0.0.1:{server.server_address[1]}"),
            )
        # Credential isolation: neither error messages nor request payloads may contain
        # api_key
        assert "secret-key-do-not-leak" not in str(excinfo.value)
        request = server.requests[0]
        assert "secret-key-do-not-leak" not in json.dumps(request)
    finally:
        server.shutdown()


def test_worker_config_validation():
    with pytest.raises(WorkerError, match="base_url"):
        WorkerConfig(base_url="", model="m").validate()
    with pytest.raises(WorkerError, match="model"):
        WorkerConfig(base_url="http://x", model=" ").validate()
    with pytest.raises(WorkerError, match="attempts"):
        WorkerConfig(base_url="http://x", model="m", max_attempts=0).validate()
    with pytest.raises(WorkerError, match="attempts"):
        WorkerConfig(base_url="http://x", model="m", max_attempts=True).validate()
    with pytest.raises(WorkerError, match="concurrency"):
        WorkerConfig(base_url="http://x", model="m", generator_concurrency=0).validate()
    config = WorkerConfig(base_url="http://x", model="m", meta_model="meta-m")
    assert config.for_meta().model == "meta-m"
    assert WorkerConfig(base_url="http://x", model="m").for_meta().model == "m"
