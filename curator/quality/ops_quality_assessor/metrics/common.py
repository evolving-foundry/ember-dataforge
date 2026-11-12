from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Any, Iterable, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def round_score(value: float) -> float:
    return round(float(value), 6)


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return round_score(sum(items) / len(items)) if items else 0.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def flatten(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.extend(flatten(value))
        elif value is not None:
            result.append(value)
    return result


def non_empty_text(value: Any) -> str:
    return str(value or "").strip()


def tokenize(text: Any) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;。！？；])\s+", text) if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("embedding", "vector", "features"):
            if key in value:
                return as_vector(value[key])
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if any(isinstance(item, (list, tuple)) for item in value):
        return None
    vector: list[float] = []
    for item in value:
        number = as_float(item)
        if number is None:
            return None
        vector.append(number)
    return vector if vector else None


def mask_box(mask: Any, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if mask is None:
        return None
    try:
        import numpy as np

        array = np.asarray(mask)
        ys, xs = np.where(array > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        width, height = image_size
        left = max(0, min(width - 1, int(xs.min())))
        right = max(left + 1, min(width, int(xs.max()) + 1))
        top = max(0, min(height - 1, int(ys.min())))
        bottom = max(top + 1, min(height, int(ys.max()) + 1))
        return left, top, right, bottom
    except Exception:
        return None

def score_value(value: Any) -> float:
    try:
        if value is None:
            return 1.0
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def mask_area(mask: Any) -> float:
    if mask is None:
        return 0.0
    try:
        import numpy as np

        array = np.asarray(mask)
        return float((array > 0).sum())
    except Exception:
        return 0.0


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return round_score((dot / (left_norm * right_norm) + 1.0) / 2.0)


def get_path_values(data: Any, path: str) -> list[Any]:
    parts = path.split(".")

    def walk(current: Any, index: int) -> list[Any]:
        if index >= len(parts):
            return [current]
        part = parts[index]
        is_list = part.endswith("[]")
        key = part[:-2] if is_list else part

        if isinstance(current, dict):
            next_value = current.get(key)
        else:
            return []

        if is_list:
            values: list[Any] = []
            for item in as_list(next_value):
                values.extend(walk(item, index + 1))
            return values
        return walk(next_value, index + 1)

    return [value for value in walk(data, 0) if value is not None]


def first_available_field(context: Any, candidates: Sequence[str]) -> str | None:
    for field_name in candidates:
        if context.has_field(field_name):
            return field_name
    return None


def has_modality(context: Any, modality: str) -> bool:
    return modality in set(getattr(context.dataset, "modalities", []) or [])


def load_image_array(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        import numpy as np
    except ImportError:
        np = None

    if isinstance(value, str):
        if not os.path.exists(value):
            return None
        try:
            from PIL import Image

            with Image.open(value) as image:
                return np.asarray(image.convert("RGB")) if np is not None else image.copy()
        except Exception:
            return None

    if np is not None:
        try:
            array = np.asarray(value)
            if array.size == 0:
                return None
            return array
        except Exception:
            return None
    return None


def _resolve_torch_device(device: str | None = None) -> str:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for embedding computation.") from exc

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _uses_internvl_image_backend(model_repo_or_path: str) -> bool:
    return "InternVL" in str(model_repo_or_path)


@lru_cache(maxsize=4)
def _load_image_embedding_backend(model_repo_or_path: str, device: str):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for image embedding computation."
        ) from exc

    try:
        processor = AutoImageProcessor.from_pretrained(
            model_repo_or_path,
            trust_remote_code=True,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(
            model_repo_or_path,
            trust_remote_code=True,
        )
    model = AutoModel.from_pretrained(model_repo_or_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    return processor, model, torch


@lru_cache(maxsize=2)
def _load_internvl_image_backend(model_repo_or_path: str, device: str):
    try:
        import importlib.util
        import json
        import sys
        import types
        from pathlib import Path

        import torch
        import torchvision.transforms as transforms
        from safetensors import safe_open
        from torchvision.transforms.functional import InterpolationMode
    except ImportError as exc:
        raise ImportError(
            "safetensors, torch, and torchvision are required for InternVL image embeddings."
        ) from exc

    model_dir = Path(model_repo_or_path)
    with (model_dir / "config.json").open(encoding="utf-8") as config_file:
        model_config = json.load(config_file)

    package_name = f"_internvl_local_{abs(hash(str(model_dir.resolve())))}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(model_dir)]
        sys.modules[package_name] = package

    for module_name in ("configuration_intern_vit", "modeling_intern_vit"):
        full_name = f"{package_name}.{module_name}"
        if full_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            full_name,
            model_dir / f"{module_name}.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load InternVL module: {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    config_module = sys.modules[f"{package_name}.configuration_intern_vit"]
    modeling_module = sys.modules[f"{package_name}.modeling_intern_vit"]
    vision_config = dict(model_config["vision_config"])
    vision_config["use_flash_attn"] = False
    config = config_module.InternVisionConfig(**vision_config)
    model = modeling_module.InternVisionModel(config)

    with (model_dir / "model.safetensors.index.json").open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]

    tensor_names_by_file: dict[str, list[str]] = {}
    for tensor_name, file_name in weight_map.items():
        if tensor_name.startswith("vision_model."):
            tensor_names_by_file.setdefault(file_name, []).append(tensor_name)

    state_dict = {}
    for file_name, tensor_names in tensor_names_by_file.items():
        with safe_open(str(model_dir / file_name), framework="pt", device="cpu") as tensors:
            for tensor_name in tensor_names:
                state_dict[tensor_name.removeprefix("vision_model.")] = tensors.get_tensor(tensor_name)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"InternVL vision weights did not match: missing={missing_keys}, unexpected={unexpected_keys}"
        )

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model.to(device=device, dtype=dtype)
    model.eval()

    image_size = int(vision_config.get("image_size", 448))
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transform, model, torch, dtype


def _internvl_image_embeddings(
    values: Sequence[Any],
    model_repo_or_path: str,
    device: str | None,
) -> list[list[float] | None]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for image embedding computation.") from exc

    resolved_device = _resolve_torch_device(device)
    transform, model, torch, dtype = _load_internvl_image_backend(
        model_repo_or_path,
        resolved_device,
    )

    pixel_values = []
    valid_indices = []
    for index, value in enumerate(values):
        array = load_image_array(value)
        if array is None:
            continue
        image = Image.fromarray(array).convert("RGB")
        pixel_values.append(transform(image))
        valid_indices.append(index)

    results: list[list[float] | None] = [None for _ in values]
    if not pixel_values:
        return results

    batch = torch.stack(pixel_values).to(resolved_device)
    if str(resolved_device).startswith("cuda"):
        batch = batch.to(dtype)

    with torch.inference_mode():
        if hasattr(model, "extract_feature"):
            outputs = model.extract_feature(batch)
        else:
            outputs = model(
                pixel_values=batch,
                output_hidden_states=False,
                return_dict=True,
            )
        vectors_tensor = _embedding_tensor_from_outputs(outputs, torch)
        if getattr(vectors_tensor, "ndim", 0) > 2:
            vectors_tensor = vectors_tensor.mean(dim=1)
    vectors = vectors_tensor.float().detach().cpu().numpy()

    for index, vector in zip(valid_indices, vectors):
        results[index] = vector.astype("float32").tolist()
    return results


@lru_cache(maxsize=4)
def _load_text_embedding_backend(model_repo_or_path: str, device: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for text embedding computation."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_repo_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_repo_or_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    return tokenizer, model, torch


def _vectors_from_outputs(outputs: Any, torch: Any, attention_mask: Any = None) -> Any:
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        return outputs.text_embeds
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        hidden = outputs.last_hidden_state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return hidden.mean(dim=1)
    if isinstance(outputs, (tuple, list)) and outputs:
        first_output = outputs[0]
        if getattr(first_output, "ndim", 0) >= 3:
            return first_output.mean(dim=1)
        return first_output
    raise RuntimeError("Cannot infer embedding tensor from model outputs.")


def _embedding_tensor_from_outputs(outputs: Any, torch: Any, attention_mask: Any = None) -> Any:
    if hasattr(outputs, "detach"):
        return outputs
    return _vectors_from_outputs(outputs, torch, attention_mask=attention_mask)


def image_embeddings(
    values: Sequence[Any],
    model_repo_or_path: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> list[list[float] | None]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for image embedding computation.") from exc

    if _uses_internvl_image_backend(model_repo_or_path):
        return _internvl_image_embeddings(values, model_repo_or_path, device)

    resolved_device = _resolve_torch_device(device)
    processor, model, torch = _load_image_embedding_backend(model_repo_or_path, resolved_device)

    images = []
    valid_indices = []
    for index, value in enumerate(values):
        array = load_image_array(value)
        if array is None:
            continue
        images.append(Image.fromarray(array).convert("RGB"))
        valid_indices.append(index)

    results: list[list[float] | None] = [None for _ in values]
    if not images:
        return results

    inputs = processor(images=images, return_tensors="pt")
    inputs = {
        key: val.to(resolved_device) if hasattr(val, "to") else val
        for key, val in inputs.items()
    }
    with torch.inference_mode():
        if hasattr(model, "get_image_features"):
            outputs = model.get_image_features(**inputs)
        else:
            outputs = model(**inputs)
        vectors_tensor = _embedding_tensor_from_outputs(outputs, torch)
    vectors = vectors_tensor.detach().cpu().numpy()

    for index, vector in zip(valid_indices, vectors):
        results[index] = vector.astype("float32").tolist()
    return results


def text_embeddings(
    texts: Sequence[str],
    model_repo_or_path: str = "BAAI/bge-m3",
    device: str | None = None,
) -> list[list[float] | None]:
    resolved_device = _resolve_torch_device(device)
    tokenizer, model, torch = _load_text_embedding_backend(model_repo_or_path, resolved_device)

    results: list[list[float] | None] = [None for _ in texts]
    valid_pairs = [(index, str(text).strip()) for index, text in enumerate(texts) if str(text).strip()]
    if not valid_pairs:
        return results

    inputs = tokenizer(
        [text for _index, text in valid_pairs],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {
        key: val.to(resolved_device) if hasattr(val, "to") else val
        for key, val in inputs.items()
    }
    with torch.inference_mode():
        if hasattr(model, "get_text_features"):
            outputs = model.get_text_features(**inputs)
        else:
            outputs = model(**inputs)
        vectors_tensor = _embedding_tensor_from_outputs(
            outputs,
            torch,
            attention_mask=inputs.get("attention_mask"),
        )
    vectors = vectors_tensor.detach().cpu().numpy()

    for (index, _text), vector in zip(valid_pairs, vectors):
        results[index] = vector.astype("float32").tolist()
    return results


def video_segment_embeddings(
    segment_inputs: list[Any],
    model_repo_or_path: str,
    device: str | None = None,
) -> list[list[float] | None]:
    """
    segment_inputs already contain prepared video segments.
    For example:
    - decoded frame list
    - temporary clip path
    - model-specific segment object

    This function does not cut video.
    """

    try:
        from transformers import AutoProcessor, AutoModel
        import torch
    except ImportError:
        raise ImportError("transformers and torch are required for video embeddings")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(
        model_repo_or_path,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_repo_or_path,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    embeddings: list[list[float] | None] = []

    for segment_input in segment_inputs:
        if segment_input is None:
            embeddings.append(None)
            continue

        try:
            inputs = processor(
                videos=[segment_input],
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = model(**inputs)

            vector = _extract_video_embedding(outputs)
            embeddings.append(vector)

        except (RuntimeError, ValueError, TypeError):
            embeddings.append(None)

    return embeddings

def _extract_video_embedding(outputs: Any) -> list[float] | None:
    """
    Normalize different model output formats into one vector.
    """

    import torch

    if hasattr(outputs, "video_embeds"):
        tensor = outputs.video_embeds
    elif hasattr(outputs, "pooler_output"):
        tensor = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state"):
        tensor = outputs.last_hidden_state.mean(dim=1)
    elif isinstance(outputs, torch.Tensor):
        tensor = outputs
    else:
        return None

    if tensor.ndim == 2:
        tensor = tensor[0]

    tensor = torch.nn.functional.normalize(tensor.float(), dim=0)

    return tensor.detach().cpu().tolist()


def load_video_frame(
    video_path: Any,
    timestamp_sec: float | None = None,
    frame_index: int | None = None,
) -> Any | None:
    if not video_path or not isinstance(video_path, str) or not os.path.exists(video_path):
        return None
    try:
        import cv2
    except ImportError:
        return None

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            return None
        if frame_index is None:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if timestamp_sec is None or fps <= 0.0:
                return None
            frame_index = max(0, int(round(float(timestamp_sec) * fps)))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            frame_index = min(max(0, int(frame_index)), total_frames - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = capture.read()
        if not ok:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def structural_similarity_score(left: Any, right: Any) -> float | None:
    left_array = load_image_array(left)
    right_array = load_image_array(right)
    if left_array is None or right_array is None:
        return None
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return None

    left_np = np.asarray(left_array)
    right_np = np.asarray(right_array)
    height = min(left_np.shape[0], right_np.shape[0])
    width = min(left_np.shape[1], right_np.shape[1])
    if height <= 0 or width <= 0:
        return None
    left_img = Image.fromarray(left_np.astype("uint8")).resize((width, height))
    right_img = Image.fromarray(right_np.astype("uint8")).resize((width, height))
    left_norm = np.asarray(left_img)
    right_norm = np.asarray(right_img)
    channel_axis = -1 if left_norm.ndim == 3 else None
    return round_score(
        clamp01(
            ssim(
                left_norm,
                right_norm,
                channel_axis=channel_axis,
                data_range=255,
            )
        )
    )
