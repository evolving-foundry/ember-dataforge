"""Marking-side model configuration; never stored in a task or generator config."""
import os


def judge_from_environment():
    from data_designer.verifiers.llm_judge import LlmJudge, LlmJudgeConfigError
    from curator.utils.llm_api_config import OpenAICompatibleChatClient, OpenAICompatibleConfig
    values = {key: os.environ.get("DATA_FORGE_JUDGE_" + key.upper(), "") for key in ("base_url", "api_key", "model")}
    if not all(values.values()):
        raise LlmJudgeConfigError("marking-side DATA_FORGE_JUDGE_BASE_URL/API_KEY/MODEL are required")
    return LlmJudge(OpenAICompatibleChatClient(OpenAICompatibleConfig(**values)), model=values["model"],
                    provider_version=os.environ.get("DATA_FORGE_JUDGE_VERSION", "unknown"),
                    temperature=float(os.environ.get("DATA_FORGE_JUDGE_TEMPERATURE", "0")))
