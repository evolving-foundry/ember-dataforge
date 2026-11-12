import pytest

from curator.utils.llm_api_config import (
    is_placeholder_openai_config,
    load_openai_compatible_config,
    mask_api_key,
)


def test_load_openai_compatible_config_from_env_file(tmp_path, monkeypatch):
    for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(key, raising=False)

    config_path = tmp_path / "llm_api.env"
    config_path.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=https://api.example.com/v1",
                "OPENAI_API_KEY=sk-test-value",
                "OPENAI_MODEL=test-model",
                "OPENAI_TIMEOUT_SECONDS=30",
                "OPENAI_MAX_CONCURRENCY=4",
            ]
        ),
        encoding="utf-8",
    )

    config = load_openai_compatible_config(str(config_path))

    assert config.base_url == "https://api.example.com/v1"
    assert config.api_key == "sk-test-value"
    assert config.model == "test-model"
    assert config.timeout_seconds == 30
    assert config.max_concurrency == 4


def test_load_openai_compatible_config_reports_missing(tmp_path, monkeypatch):
    for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(key, raising=False)

    config_path = tmp_path / "llm_api.env"
    config_path.write_text("OPENAI_BASE_URL=https://api.example.com/v1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing OpenAI-compatible API config values"):
        load_openai_compatible_config(str(config_path))


def test_placeholder_and_mask_helpers():
    config = load_openai_compatible_config(require_complete=False)
    object.__setattr__(config, "base_url", "https://your-compatible-endpoint/v1")
    object.__setattr__(config, "api_key", "your_api_key")
    object.__setattr__(config, "model", "your_model_name")

    assert is_placeholder_openai_config(config)
    assert mask_api_key("sk-1234567890") == "sk-1...7890"


def test_openai_compatible_chat_client_posts_to_chat_completions(monkeypatch):
    from curator.utils.llm_api_config import (
        OpenAICompatibleChatClient,
        OpenAICompatibleConfig,
    )

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                b'{"choices":[{"message":{"content":"{\\"score\\":5}"}}]}'
            )

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data.decode("utf-8")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleChatClient(
        OpenAICompatibleConfig(
            base_url="https://api.example.com/v1",
            api_key="sk-test-value",
            model="test-model",
            timeout_seconds=7,
        )
    )
    response = client.chat(
        messages=[{"role": "user", "content": "judge"}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test-value"
    assert captured["headers"]["User-agent"].startswith("Mozilla/5.0")
    assert captured["timeout"] == 7
    assert '"model": "test-model"' in captured["body"]
    assert response.choices[0].message.content == '{"score":5}'
