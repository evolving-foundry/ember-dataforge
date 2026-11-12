from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from .device import is_npu_device, resolve_torch_device
from .local_model_paths import local_model_roots_from_params, resolve_local_model_path

if TYPE_CHECKING:
    from PIL import Image


DEFAULT_QWEN_VLM_CANDIDATES = (
    "Qwen2.5-VL-7B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    "Qwen3-VL-2B-Instruct",
)


class LocalQwenVlmClient:
    """Local Qwen VLM client with an OpenAI-compatible chat response shape."""

    supports_images = True

    def __init__(
        self,
        model_path: str | None = None,
        local_model_roots: Any = None,
        local_files_only: bool = True,
        device_preference: str | None = "auto",
        device_map: str | dict[str, Any] | None = "auto",
        torch_dtype: str | None = "auto",
        min_pixels: int = 224 * 224,
        max_pixels: int = 640 * 640,
        max_new_tokens: int = 256,
        trust_remote_code: bool = True,
    ):
        self.local_model_roots = local_model_roots_from_params(local_model_roots)
        self.model_path = select_qwen_vlm_model_path(
            preferred_model_path=model_path,
            local_roots=self.local_model_roots,
        )
        self.local_files_only = bool(local_files_only)
        self.device_preference = device_preference or "auto"
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_new_tokens = int(max_new_tokens)
        self.trust_remote_code = bool(trust_remote_code)
        self._processor = None
        self._model = None
        self._runtime_device = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        self._load()
        qwen_messages, images = self._normalize_messages(messages)
        text = self._processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=images or None,
            return_tensors="pt",
        ).to(self._input_device())

        import torch

        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": float(temperature or 0.0) > 0.0,
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = float(temperature)

        with torch.no_grad():
            output = self._model.generate(**inputs, **generation_kwargs)
        output = output[:, inputs["input_ids"].shape[1] :]
        content = self._processor.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if response_format and response_format.get("type") == "json_object":
            content = normalize_json_content(content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def _load(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        model_type = str(getattr(config, "model_type", "")).lower()
        model_cls = self._model_class(model_type)
        runtime_device = self._resolve_runtime_device()
        use_device_map = self._should_use_device_map(runtime_device)

        processor_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        self._processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.torch_dtype:
            model_kwargs["torch_dtype"] = self.torch_dtype
        if use_device_map:
            model_kwargs["device_map"] = self.device_map
        self._model = model_cls.from_pretrained(self.model_path, **model_kwargs)
        if not use_device_map:
            self._model = self._model.to(runtime_device)
        self._model = self._model.eval()

    def _resolve_runtime_device(self) -> str:
        if self._runtime_device is None:
            self._runtime_device = resolve_torch_device(self.device_preference)
        return self._runtime_device

    def _should_use_device_map(self, runtime_device: str) -> bool:
        if not self.device_map:
            return False
        if is_npu_device(runtime_device):
            return False
        if str(runtime_device).lower() == "cpu":
            return False
        return True

    def _input_device(self):
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            return model_device
        return self._resolve_runtime_device()

    @staticmethod
    def _model_class(model_type: str):
        if model_type == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        if model_type == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        if model_type == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration
        raise ValueError(f"Unsupported local Qwen VLM model_type: {model_type}")

    def _normalize_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        qwen_messages = []
        images = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                qwen_messages.append({"role": message.get("role", "user"), "content": content})
                continue

            qwen_content = []
            for item in content or []:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    qwen_content.append({"type": "text", "text": str(item.get("text") or "")})
                elif item.get("type") == "image":
                    image_value = item.get("image") or item.get("image_url") or item.get("path")
                    image = load_image_value(image_value)
                    if image is None:
                        continue
                    images.append(image)
                    qwen_content.append({"type": "image", "image": image})
            qwen_messages.append(
                {"role": message.get("role", "user"), "content": qwen_content}
            )
        return qwen_messages, images


def select_qwen_vlm_model_path(
    preferred_model_path: str | None = None,
    local_roots: list[str] | None = None,
) -> str:
    roots = list(local_roots or local_model_roots_from_params())
    if preferred_model_path:
        resolved = resolve_local_model_path(preferred_model_path, roots)
        if os.path.isdir(resolved):
            return resolved
        raise FileNotFoundError(f"Local Qwen VLM model not found: {preferred_model_path}")

    for candidate in DEFAULT_QWEN_VLM_CANDIDATES:
        resolved = resolve_local_model_path(candidate, roots)
        if os.path.isdir(resolved):
            return resolved
    raise FileNotFoundError(
        "No local Qwen VLM model found under: " + ", ".join(roots or [])
    )


def load_image_value(value: Any) -> Image.Image | None:
    from PIL import Image

    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        if not os.path.exists(value):
            return None
        return Image.open(value).convert("RGB")
    try:
        return Image.fromarray(value).convert("RGB")
    except Exception:
        return None


def normalize_json_content(content: str) -> str:
    payload = extract_json_object(content)
    if payload is None:
        return content
    return json.dumps(payload, ensure_ascii=False)


def extract_json_object(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None
