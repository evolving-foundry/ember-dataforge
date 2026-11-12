import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional
from urllib import request
from urllib.error import HTTPError, URLError


DEFAULT_LLM_API_CONFIG_PATH = os.environ.get(
    "LLM_API_CONFIG_PATH",
    os.path.join("config", "llm_api.env"),
)


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60
    max_concurrency: int = 2
    config_path: str = DEFAULT_LLM_API_CONFIG_PATH


class OpenAICompatibleChatClient:
    """Small OpenAI-compatible chat client used by video QAE components.

    The public surface intentionally matches the narrow call shape used in this
    package: ``client.chat(messages=[...], model=..., temperature=...)``.  The
    implementation avoids a hard dependency on the ``openai`` package, while
    returning an object with ``choices[0].message.content`` for compatibility
    with existing OpenAI SDK-style parsing code.
    """

    def __init__(self, config: OpenAICompatibleConfig):
        if is_placeholder_openai_config(config):
            raise ValueError(
                f"OpenAI-compatible API config at {config.config_path} still contains placeholder values."
            )
        missing = []
        if not config.base_url:
            missing.append("OPENAI_BASE_URL")
        if not config.api_key:
            missing.append("OPENAI_API_KEY")
        if not config.model:
            missing.append("OPENAI_MODEL")
        if missing:
            raise ValueError(f"Missing OpenAI-compatible API config values {missing}.")
        self.config = config

    @classmethod
    def from_config_path(
        cls,
        config_path: Optional[str] = None,
        require_complete: bool = True,
    ) -> "OpenAICompatibleChatClient":
        return cls(load_openai_compatible_config(config_path, require_complete=require_complete))

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        **extra: Any,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra)

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self._chat_completions_url(),
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Some OpenAI-compatible gateways sit behind Cloudflare/WAF rules
                # that reject Python's default urllib User-Agent before the API
                # request reaches the model router.
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
                ),
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc

        try:
            response_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM API returned non-JSON response") from exc

        return _to_openai_chat_response(response_payload)

    def _chat_completions_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"


def _to_openai_chat_response(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    choices = []
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            choices.append(choice)
            continue
        message = choice.get("message") or {}
        if isinstance(message, dict):
            message_obj = SimpleNamespace(**message)
        else:
            message_obj = SimpleNamespace(content=str(message))
        choices.append(SimpleNamespace(**{**choice, "message": message_obj}))
    return SimpleNamespace(**{**payload, "choices": choices})


def create_openai_compatible_client(
    config_path: Optional[str] = None,
    config: Optional[OpenAICompatibleConfig] = None,
    require_complete: bool = True,
) -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        config or load_openai_compatible_config(config_path, require_complete=require_complete)
    )


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_env_file(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}

    values: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            values[key] = _strip_env_value(value)
    return values


def load_openai_compatible_config(
    config_path: Optional[str] = None,
    require_complete: bool = True,
) -> OpenAICompatibleConfig:
    path = config_path or os.environ.get("LLM_API_CONFIG_PATH") or DEFAULT_LLM_API_CONFIG_PATH
    values = parse_env_file(path)
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_TIMEOUT_SECONDS",
        "OPENAI_MAX_CONCURRENCY",
    ):
        if os.environ.get(key):
            values[key] = os.environ[key]

    missing = [
        key
        for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")
        if not values.get(key)
    ]
    if missing and require_complete:
        raise ValueError(
            "Missing OpenAI-compatible API config values "
            f"{missing}. Fill {path} or set them in the environment."
        )

    return OpenAICompatibleConfig(
        base_url=values.get("OPENAI_BASE_URL", ""),
        api_key=values.get("OPENAI_API_KEY", ""),
        model=values.get("OPENAI_MODEL", ""),
        timeout_seconds=int(values.get("OPENAI_TIMEOUT_SECONDS") or 60),
        max_concurrency=int(values.get("OPENAI_MAX_CONCURRENCY") or 2),
        config_path=path,
    )


def is_placeholder_openai_config(config: OpenAICompatibleConfig) -> bool:
    placeholders = {
        "https://your-compatible-endpoint/v1",
        "your_api_key",
        "your_model_name",
    }
    return (
        config.base_url in placeholders
        or config.api_key in placeholders
        or config.model in placeholders
    )


def mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"
