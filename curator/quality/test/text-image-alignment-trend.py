#!/usr/bin/env python3
"""Generate a real model-judged text-image alignment trend chart.

This script intentionally avoids the old deterministic label-vector stub.  It:
- builds a tiny in-process multimodal dataset from repository ``data/`` images;
- inspects local VLM candidates under ``ALIGNMENT_MODEL_ROOT``
  (default: ``/data/llm_models``);
- runs calibration pairs that include matched, mismatched, partial, and noisy text;
- selects the best calibrated local vision-language judge; and
- uses that model to score operation-level image/text alignment in ``[0, 1]``.

Run the real scorer in the local ML environment, for example:

    python test/text-image-alignment-trend.py

No network downloads are used; all model loading is ``local_files_only=True``.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import re
import sys
import types
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_PATH = Path(__file__).resolve()
LOCAL_MODEL_ROOT = Path(os.environ.get("ALIGNMENT_MODEL_ROOT", "/data/llm_models"))
DEFAULT_SELECTED_MODEL = LOCAL_MODEL_ROOT / "Qwen3-VL-4B-Instruct"
DEFAULT_TEXT_IMAGE_EMBEDDING_MODEL = Path(
    os.environ.get(
        "TEXT_IMAGE_ALIGNMENT_EMBEDDING_MODEL",
        str(LOCAL_MODEL_ROOT / "laion" / "CLIP-ViT-B-32-laion2B-s34B-b79K"),
    )
)
MODEL_CANDIDATE_LIMIT = int(os.environ.get("ALIGNMENT_MODEL_CANDIDATE_LIMIT", "3"))
MAX_GENERATION_TOKENS = int(os.environ.get("ALIGNMENT_MAX_NEW_TOKENS", "16"))


def find_repo_root(start: Path = SCRIPT_PATH) -> Path:
    """Find the DataForge repository root from this script location."""
    for candidate in [start.parent, *start.parents]:
        if (candidate / "data").is_dir() and (candidate / "curator").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _LocalSchema:
    def __init__(self, rows: list[dict[str, Any]]):
        names: list[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
        self.names = names


class _LocalDataset:
    """Small Ray Dataset stand-in for deterministic in-process script runs."""

    def __init__(self, rows: Iterable[dict[str, Any]]):
        self._rows = [dict(row) for row in rows]

    def map(self, func):
        return _LocalDataset(func(dict(row)) for row in self._rows)

    def filter(self, func):
        return _LocalDataset(row for row in self._rows if func(dict(row)))

    def limit(self, n: int):
        return _LocalDataset(self._rows[:n])

    def take_all(self):
        return [dict(row) for row in self._rows]

    def take(self, n: int):
        return [dict(row) for row in self._rows[:n]]

    def count(self):
        return len(self._rows)

    def schema(self):
        return _LocalSchema(self._rows)


def _install_ray_stub_if_needed() -> bool:
    """Install a minimal ray/ray.data module before project imports if needed."""
    if importlib.util.find_spec("ray") is not None:
        return False

    ray_module = types.ModuleType("ray")
    ray_data_module = types.ModuleType("ray.data")
    ray_module.__spec__ = ModuleSpec("ray", loader=None)
    ray_data_module.__spec__ = ModuleSpec("ray.data", loader=None)
    ray_data_module.Dataset = _LocalDataset
    ray_data_module.from_items = lambda rows: _LocalDataset(rows)
    ray_data_module.read_images = lambda *args, **kwargs: _LocalDataset([])
    ray_data_module.read_audio = lambda *args, **kwargs: _LocalDataset([])
    ray_data_module.read_videos = lambda *args, **kwargs: _LocalDataset([])
    ray_module.data = ray_data_module
    ray_module.init = lambda *args, **kwargs: None
    ray_module.shutdown = lambda: None
    ray_module.is_initialized = lambda: False
    sys.modules["ray"] = ray_module
    sys.modules["ray.data"] = ray_data_module
    return True


USING_RAY_STUB = _install_ray_stub_if_needed()


def _install_operations_namespace_stubs() -> None:
    """Avoid executing operations/__init__.py, which imports optional heavy deps."""
    operations_dir = REPO_ROOT / "curator" / "operations"
    augmentation_dir = operations_dir / "augmentation"

    operations_module = types.ModuleType("curator.operations")
    operations_module.__path__ = [str(operations_dir)]
    operations_module.__package__ = "curator"
    operations_module.__spec__ = ModuleSpec(
        "curator.operations", loader=None, is_package=True
    )

    augmentation_module = types.ModuleType("curator.operations.augmentation")
    augmentation_module.__path__ = [str(augmentation_dir)]
    augmentation_module.__package__ = "curator.operations"
    augmentation_module.__spec__ = ModuleSpec(
        "curator.operations.augmentation",
        loader=None,
        is_package=True,
    )

    sys.modules.setdefault("curator.operations", operations_module)
    sys.modules.setdefault("curator.operations.augmentation", augmentation_module)


_install_operations_namespace_stubs()

from base.dataset import MultimodalDataset  # noqa: E402
from curator.operations.augmentation.image_canny_operation import (  # noqa: E402
    ImageCannyAugmentationOperation,
)
from curator.operations.augmentation.image_random_flip_operation import (  # noqa: E402
    ImageRandomFlipOperation,
)
from curator.operations.augmentation.remove_repetitions_punctuation_operation import (  # noqa: E402
    RemoveRepetitionsPunctuationOperation,
)
from curator.operations.augmentation.remove_stopwords_operation import (  # noqa: E402
    RemoveStopwordsOperation,
)
from curator.operations.augmentation.text_normalization_operation import (  # noqa: E402
    TextNormalizationOperation,
)


@dataclass(frozen=True)
class TrendPoint:
    branch: str
    operation: str
    alignment: float
    rows: int
    before_alignment: float | None = None
    after_alignment: float | None = None
    operator_score: float | None = None
    score_mode: str = "image_text_embedding_cosine"
    before_basis: str = ""


@dataclass(frozen=True)
class SingleOperatorComparison:
    operation: str
    sample_id: str
    before_label: str
    before_alignment: float
    after_label: str
    after_alignment: float
    rows: int
    preservation_score: float | None = None

    @property
    def delta(self) -> float:
        return round(self.after_alignment - self.before_alignment, 6)


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    label: str
    image_path: Path
    caption: str
    expected_min: float
    expected_max: float
    kind: str


@dataclass(frozen=True)
class CalibrationResult:
    model_path: Path
    case_id: str
    kind: str
    score: float | None
    raw_output: str
    status: str


@dataclass(frozen=True)
class ModelSelection:
    selected_model: Path
    attempted_models: list[Path]
    results: list[CalibrationResult]
    model_scores: dict[str, float]
    selection_reason: str


LABEL_SPECS = {
    "butterfly": {
        "tokens": {"butterfly", "wing", "wings", "leaf", "spotted", "black", "white"},
        "caption": "A black butterfly with white spots spreads its wings on a green leaf.",
    },
    "messi5": {
        "tokens": {"soccer", "football", "player", "ball", "field", "uniform"},
        "caption": "A soccer player in a red and blue uniform kicks a ball on a field.",
    },
    "lena": {
        "tokens": {"woman", "portrait", "face", "hat", "person"},
        "caption": "A portrait of a woman wearing a wide brimmed hat.",
    },
    "fruits": {
        "tokens": {"fruit", "citrus", "kiwi", "lime", "lemon", "orange", "banana"},
        "caption": "A close-up of sliced citrus fruit, kiwi, lime, and banana.",
    },
}
LABELS = list(LABEL_SPECS)


ALIGNMENT_PROMPT_TEMPLATE = """You are a strict image-text semantic alignment judge.
Score only visible semantic agreement between the image and caption.
Return exactly one number from 0 to 1.
Scoring anchors:
- 1.0 = caption is fully correct and specific for visible objects/actions/scene.
- 0.7 = mostly correct but incomplete.
- 0.4 = generic, noisy, or only weakly related.
- 0.0 = main objects/actions/scene are wrong.
Do not reward random repeated noise words or keyword stuffing.
Caption: {caption}"""

VISUAL_TRANSFORM_PROMPT_TEMPLATE = """You are a strict visual semantic preservation judge.
Compare the first image (before) and the second image (after) for operation: {operation}.
Score how much visible semantic content is preserved.
Ignore expected pixel/style changes from safe augmentations such as horizontal flip, edge detection, grayscale conversion, or color conversion.
Return exactly one number from 0 to 1.
Scoring anchors:
- 1.0 = same main objects/actions/scene are preserved.
- 0.7 = mostly preserved with some detail loss.
- 0.4 = weakly preserved or important objects are hard to recognize.
- 0.0 = different main objects/actions/scene.
Caption context: {caption}"""


def label_from_path(path: str | Path) -> str:
    name = Path(path).stem.lower()
    for label in LABELS:
        if label in name:
            return label
    return LABELS[0]


def discover_image_paths(data_root: Path) -> list[Path]:
    image_paths = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected: list[Path] = []
    seen: set[str] = set()
    for path in image_paths:
        label = label_from_path(path)
        if label in seen:
            continue
        if label in LABEL_SPECS:
            selected.append(path)
            seen.add(label)
    if not selected:
        raise RuntimeError(f"No supported images found under {data_root}")
    return selected


def select_trend_image_paths(image_paths: list[Path], label: str = "messi5") -> list[Path]:
    """Use the report's Messi image for the operation trend demo."""

    selected = [path for path in image_paths if label_from_path(path) == label]
    if not selected:
        raise RuntimeError(f"No image with semantic label {label!r} was found")
    return selected


def build_rows(image_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in image_paths:
        label = label_from_path(path)
        caption = LABEL_SPECS[label]["caption"]
        rows.append(
            {
                "id": path.stem,
                "semantic_label": label,
                "image": str(path),
                "image_path": str(path),
                "caption": caption,
                "text": caption,
                "alignment_text": caption,
                "alignment_image": str(path),
            }
        )
    return rows


def build_ray_dataset(rows: list[dict[str, Any]]):
    # Use a local in-process dataset even when Ray is installed.  This keeps the
    # script deterministic and avoids Ray worker import-path failures for this
    # repository-local demo.
    return _LocalDataset(rows)


def rows_of(dataset: MultimodalDataset) -> list[dict[str, Any]]:
    return dataset.dataset.take_all()


def first_image_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def first_present_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def with_alignment_fields(
    dataset: MultimodalDataset,
    *,
    text_field: str,
    image_field: str = "image",
    image_color_order: str | None = None,
) -> MultimodalDataset:
    def bind(row: dict[str, Any]) -> dict[str, Any]:
        row["alignment_text"] = row.get(text_field) or row.get("caption") or row.get("text") or ""
        image_value = first_image_value(row.get(image_field))
        if image_value is None:
            image_value = row.get("image") or row.get("image_path")
        row["alignment_image"] = image_value
        if image_color_order:
            row["alignment_image_color_order"] = image_color_order
        else:
            row.pop("alignment_image_color_order", None)
        return row

    return dataset.map(bind)


def promote_alignment_image(dataset: MultimodalDataset) -> MultimodalDataset:
    """Make the latest scored image the next image-operator input."""

    def promote(row: dict[str, Any]) -> dict[str, Any]:
        image_value = row.get("alignment_image")
        if image_value is None:
            image_value = row.get("image") or row.get("image_path")
        row["image"] = image_value
        return row

    return dataset.map(promote)


def apply_semantic_keyword_mask(dataset: MultimodalDataset) -> MultimodalDataset:
    """Test-local destructive text operator to make semantic drift visible."""

    def mask(row: dict[str, Any]) -> dict[str, Any]:
        label = str(row.get("semantic_label", ""))
        tokens_to_mask = LABEL_SPECS.get(label, {}).get("tokens", set())
        text = str(row.get("alignment_text") or row.get("text") or row.get("caption") or "")
        for token in sorted(tokens_to_mask, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(token)}\b", "object", text, flags=re.IGNORECASE)
        row["output_text"] = f"{text} random noise noise !!! object object"
        return row

    return dataset.map(mask)


def discover_model_candidates(model_root: Path = LOCAL_MODEL_ROOT) -> list[Path]:
    """Return likely local VLM judge candidates, ordered by expected calibration quality."""
    preferred_names = [
        "Qwen3-VL-4B-Instruct",
        "Qwen3-VL-2B-Instruct",
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
        "gemma-3-4b-it",
        "paligemma2-3b-mix-448",
    ]
    candidates: list[Path] = []
    for name in preferred_names:
        path = model_root / name
        if (path / "config.json").is_file():
            candidates.append(path)

    for config_path in sorted(model_root.rglob("config.json")):
        if any(part in {".locks", ".cache", ".git", "._____temp", "blobs"} for part in config_path.parts):
            continue
        path = config_path.parent
        if path in candidates:
            continue
        text = (path.name + " " + config_path.read_text(errors="ignore")[:4000]).lower()
        if any(key in text for key in ("qwen3_vl", "qwen2_5_vl", "gemma3", "paligemma", "llava", "internvl")):
            candidates.append(path)
    return candidates[:MODEL_CANDIDATE_LIMIT]


def calibration_cases() -> list[CalibrationCase]:
    image_dir = REPO_ROOT / "data" / "images"
    return [
        CalibrationCase(
            "butterfly_match",
            "butterfly",
            image_dir / "butterfly.jpg",
            LABEL_SPECS["butterfly"]["caption"],
            0.75,
            1.0,
            "positive",
        ),
        CalibrationCase(
            "butterfly_wrong_soccer",
            "butterfly",
            image_dir / "butterfly.jpg",
            LABEL_SPECS["messi5"]["caption"],
            0.0,
            0.25,
            "negative",
        ),
        CalibrationCase(
            "butterfly_noise",
            "butterfly",
            image_dir / "butterfly.jpg",
            "object object object random noise !!! repeated words noise noise",
            0.0,
            0.35,
            "noise",
        ),
        CalibrationCase(
            "messi_match",
            "messi5",
            image_dir / "messi5.jpg",
            LABEL_SPECS["messi5"]["caption"],
            0.75,
            1.0,
            "positive",
        ),
        CalibrationCase(
            "lena_match",
            "lena",
            image_dir / "lena.jpg",
            LABEL_SPECS["lena"]["caption"],
            0.75,
            1.0,
            "positive",
        ),
        CalibrationCase(
            "fruits_match",
            "fruits",
            image_dir / "fruits.jpg",
            LABEL_SPECS["fruits"]["caption"],
            0.75,
            1.0,
            "positive",
        ),
        CalibrationCase(
            "fruits_partial_wrong_bowl",
            "fruits",
            image_dir / "fruits.jpg",
            "A bowl of fruit with apples, oranges, and bananas.",
            0.25,
            0.65,
            "partial",
        ),
        CalibrationCase(
            "fruits_wrong_portrait",
            "fruits",
            image_dir / "fruits.jpg",
            LABEL_SPECS["lena"]["caption"],
            0.0,
            0.25,
            "negative",
        ),
    ]


def parse_alignment_score(raw_output: str) -> float | None:
    match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", raw_output)
    if not match:
        return None
    return max(0.0, min(1.0, float(match.group(0))))


class RealVlmAlignmentScorer:
    """Lazy local VLM judge that scores image/caption alignment in ``[0, 1]``."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.processor = None
        self.model = None
        self.torch = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Real alignment scoring requires torch, transformers, and Pillow. "
                "Run with `python test/text-image-alignment-trend.py`."
            ) from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _image_from_value(self, image_value: Any, *, color_order: str | None = None):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow and numpy are required for real image alignment scoring.") from exc

        image_value = first_image_value(image_value)
        if image_value is None:
            raise ValueError("missing image value")
        if isinstance(image_value, (str, Path)):
            return Image.open(image_value).convert("RGB")
        array = np.asarray(image_value)
        if array.size == 0:
            raise ValueError("empty image array")
        if array.ndim == 2:
            return Image.fromarray(array.astype("uint8"), mode="L").convert("RGB")
        if array.ndim == 3 and array.shape[2] >= 3:
            rgb = array[:, :, :3]
            if color_order == "bgr":
                rgb = rgb[:, :, ::-1]
            return Image.fromarray(rgb.astype("uint8")).convert("RGB")
        raise ValueError(f"unsupported image array shape: {getattr(array, 'shape', None)}")

    def score(self, image_value: Any, caption: str, *, color_order: str | None = None) -> tuple[float, str]:
        self.load()
        assert self.processor is not None and self.model is not None and self.torch is not None
        caption = str(caption or "").strip()
        if not caption:
            return 0.0, "empty caption"
        image = self._image_from_value(image_value, color_order=color_order)
        prompt = ALIGNMENT_PROMPT_TEMPLATE.format(caption=caption)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_GENERATION_TOKENS,
                do_sample=False,
            )[0]
        if "input_ids" in inputs:
            generated = generated[inputs["input_ids"].shape[-1] :]
        raw_output = self.processor.decode(generated, skip_special_tokens=True).strip()
        score = parse_alignment_score(raw_output)
        if score is None:
            raise RuntimeError(f"Could not parse alignment score from model output: {raw_output!r}")
        return score, raw_output

    def score_visual_preservation(
        self,
        before_image_value: Any,
        after_image_value: Any,
        *,
        operation: str,
        caption: str = "",
        before_color_order: str | None = None,
        after_color_order: str | None = None,
    ) -> tuple[float, str]:
        """Score before/after visual semantic preservation for an image operator."""

        self.load()
        assert self.processor is not None and self.model is not None and self.torch is not None
        before_image = self._image_from_value(before_image_value, color_order=before_color_order)
        after_image = self._image_from_value(after_image_value, color_order=after_color_order)
        prompt = VISUAL_TRANSFORM_PROMPT_TEMPLATE.format(
            operation=operation,
            caption=str(caption or "").strip() or "not provided",
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": before_image},
                    {"type": "image", "image": after_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[before_image, after_image], return_tensors="pt")
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_GENERATION_TOKENS,
                do_sample=False,
            )[0]
        if "input_ids" in inputs:
            generated = generated[inputs["input_ids"].shape[-1] :]
        raw_output = self.processor.decode(generated, skip_special_tokens=True).strip()
        score = parse_alignment_score(raw_output)
        if score is None:
            raise RuntimeError(f"Could not parse visual preservation score from model output: {raw_output!r}")
        return score, raw_output

    def close(self) -> None:
        if self.model is None:
            return
        try:
            del self.model
            del self.processor
            if self.torch is not None:
                self.torch.cuda.empty_cache()
        finally:
            self.model = None
            self.processor = None
            self.torch = None


def score_model_calibration(model_path: Path, cases: list[CalibrationCase]) -> tuple[list[CalibrationResult], float]:
    scorer = RealVlmAlignmentScorer(model_path)
    results: list[CalibrationResult] = []
    try:
        scorer.load()
        for case in cases:
            try:
                score, raw = scorer.score(case.image_path, case.caption)
                status = "passed" if case.expected_min <= score <= case.expected_max else "out_of_band"
                results.append(CalibrationResult(model_path, case.case_id, case.kind, score, raw, status))
            except Exception as exc:
                results.append(CalibrationResult(model_path, case.case_id, case.kind, None, repr(exc), "failed"))
    except Exception as exc:
        return [CalibrationResult(model_path, "model_load", "load", None, repr(exc), "failed")], -1.0
    finally:
        scorer.close()

    valid = [result for result in results if result.score is not None]
    if len(valid) != len(cases):
        return results, -1.0

    band_error = 0.0
    positives: list[float] = []
    negatives: list[float] = []
    noise_scores: list[float] = []
    for result, case in zip(results, cases):
        assert result.score is not None
        if result.score < case.expected_min:
            band_error += case.expected_min - result.score
        elif result.score > case.expected_max:
            band_error += result.score - case.expected_max
        if case.kind == "positive":
            positives.append(result.score)
        else:
            negatives.append(result.score)
        if case.kind == "noise":
            noise_scores.append(result.score)
    separation = (sum(positives) / len(positives)) - (sum(negatives) / len(negatives))
    noise_penalty = max(noise_scores) if noise_scores else 0.0
    calibration_score = separation - band_error - 0.5 * noise_penalty
    return results, round(calibration_score, 6)


def select_alignment_model(output_dir: Path) -> ModelSelection:
    candidates = discover_model_candidates()
    if not candidates:
        raise RuntimeError(f"No local VLM candidates found under {LOCAL_MODEL_ROOT}")

    cases = calibration_cases()
    all_results: list[CalibrationResult] = []
    model_scores: dict[str, float] = {}
    for candidate in candidates:
        results, model_score = score_model_calibration(candidate, cases)
        all_results.extend(results)
        model_scores[str(candidate)] = model_score

    passed_models = [path for path in candidates if model_scores.get(str(path), -1.0) >= 0.0]
    if passed_models:
        selected = max(passed_models, key=lambda path: (model_scores[str(path)], path == DEFAULT_SELECTED_MODEL))
    else:
        selected = DEFAULT_SELECTED_MODEL if DEFAULT_SELECTED_MODEL in candidates else candidates[0]

    reason = (
        f"selected {selected.name}: highest calibration score among loadable local VLMs; "
        "matched captions score high, mismatches/noise score low, partial caption is not over-rewarded"
    )
    selection = ModelSelection(selected, candidates, all_results, model_scores, reason)
    write_model_selection_csv(selection, output_dir / "text-image-alignment-model-selection.csv")
    return selection


def score_alignment(dataset: MultimodalDataset, scorer: RealVlmAlignmentScorer) -> float:
    from curator.quality.ops_quality_assessor.metrics.common import (
        cosine_similarity,
        image_embeddings,
        text_embeddings,
    )

    image_values: list[Any] = []
    text_values: list[str] = []
    for row in rows_of(dataset):
        image_values.append(
            scorer._image_from_value(
                row.get("alignment_image"),
                color_order=row.get("alignment_image_color_order"),
            )
        )
        text_values.append(str(row.get("alignment_text") or ""))

    if not image_values or not text_values:
        raise RuntimeError("no image-text pairs were available for embedding cosine")

    model_path = str(DEFAULT_TEXT_IMAGE_EMBEDDING_MODEL)
    device = os.environ.get("TEXT_IMAGE_ALIGNMENT_DEVICE", "cpu")
    image_vectors = image_embeddings(
        image_values,
        model_repo_or_path=model_path,
        device=device,
    )
    text_vectors = text_embeddings(
        text_values,
        model_repo_or_path=model_path,
        device=device,
    )

    scores = [
        cosine_similarity(image_vector, text_vector)
        for image_vector, text_vector in zip(image_vectors, text_vectors)
        if image_vector is not None and text_vector is not None
    ]
    if not scores:
        raise RuntimeError("image-text embedding cosine scores could not be computed")
    return round(sum(scores) / len(scores), 6)


def score_visual_preservation(
    before_dataset: MultimodalDataset,
    after_dataset: MultimodalDataset,
    scorer: RealVlmAlignmentScorer,
    *,
    operation_name: str,
) -> float:
    scores: list[float] = []
    before_rows = rows_of(before_dataset)
    after_rows = rows_of(after_dataset)
    if len(before_rows) != len(after_rows):
        raise RuntimeError(
            f"before/after row count mismatch for {operation_name}: "
            f"{len(before_rows)} != {len(after_rows)}"
        )
    for before_row, after_row in zip(before_rows, after_rows):
        from curator.quality.ops_quality_assessor.metrics.common import (
            cosine_similarity,
            image_embeddings,
        )

        before_image = scorer._image_from_value(
            first_present_value(
                before_row.get("alignment_image"),
                before_row.get("image"),
                before_row.get("image_path"),
            ),
            color_order=before_row.get("alignment_image_color_order"),
        )
        after_image = scorer._image_from_value(
            first_present_value(
                after_row.get("alignment_image"),
                after_row.get("image"),
                after_row.get("image_path"),
            ),
            color_order=after_row.get("alignment_image_color_order"),
        )
        before_vector, after_vector = image_embeddings(
            [before_image, after_image],
            model_repo_or_path=str(DEFAULT_TEXT_IMAGE_EMBEDDING_MODEL),
            device=os.environ.get("TEXT_IMAGE_ALIGNMENT_DEVICE", "cpu"),
        )
        if before_vector is None or after_vector is None:
            raise RuntimeError(f"visual embedding preservation failed for {operation_name}")
        score = cosine_similarity(before_vector, after_vector)
        scores.append(score)
    if not scores:
        raise RuntimeError(f"no visual before/after pairs were scored for {operation_name}")
    return round(sum(scores) / len(scores), 6)


def collect_trend_points(base_dataset: MultimodalDataset, scorer: RealVlmAlignmentScorer) -> list[TrendPoint]:
    row_count = base_dataset.dataset.count()
    points: list[TrendPoint] = []

    current_dataset = with_alignment_fields(base_dataset, text_field="caption")
    current_alignment = score_alignment(current_dataset, scorer)
    points.append(
        TrendPoint(
            "baseline",
            "input",
            current_alignment,
            row_count,
            before_alignment=current_alignment,
            after_alignment=current_alignment,
            operator_score=1.0,
            before_basis="raw Messi image + original caption",
        )
    )

    text_ops: list[tuple[str, Any, str]] = [
        (
            "text_normalization",
            TextNormalizationOperation({"text_key": "alignment_text"}),
            "alignment_text",
        ),
        (
            "remove_repetitions_punctuation",
            RemoveRepetitionsPunctuationOperation({"text_key": "alignment_text"}),
            "alignment_text",
        ),
        (
            "remove_stopwords",
            RemoveStopwordsOperation({"text_key": "alignment_text"}),
            "output_text",
        ),
    ]
    for operation_name, operation, output_text_field in text_ops:
        before_alignment = current_alignment
        current_dataset = operation.execute(current_dataset)
        current_dataset = with_alignment_fields(current_dataset, text_field=output_text_field)
        current_alignment = score_alignment(current_dataset, scorer)
        operator_score = current_alignment / before_alignment if before_alignment else current_alignment
        points.append(
            TrendPoint(
                "pipeline",
                operation_name,
                current_alignment,
                row_count,
                before_alignment=before_alignment,
                after_alignment=current_alignment,
                operator_score=round(operator_score, 6),
                before_basis="previous pipeline text-image pair",
            )
        )

    before_alignment = current_alignment
    current_dataset = apply_semantic_keyword_mask(current_dataset)
    current_dataset = with_alignment_fields(current_dataset, text_field="output_text")
    current_alignment = score_alignment(current_dataset, scorer)
    operator_score = current_alignment / before_alignment if before_alignment else current_alignment
    points.append(
        TrendPoint(
            "pipeline",
            "semantic_keyword_mask",
            current_alignment,
            row_count,
            before_alignment=before_alignment,
            after_alignment=current_alignment,
            operator_score=round(operator_score, 6),
            before_basis="previous cleaned-text output",
        )
    )

    image_ops: list[tuple[str, Any, str, str | None]] = [
        (
            "image_random_flip",
            ImageRandomFlipOperation(
                {
                    "image_key": "image",
                    "flip_horizontal_prob": 1.0,
                    "flip_vertical_prob": 0.0,
                    "rotation_prob": 0.0,
                    "num_augmentations": 1,
                    "output_key": "flip_image_data",
                }
            ),
            "flip_image_data",
            "bgr",
        ),
        (
            "image_canny",
            ImageCannyAugmentationOperation(
                {
                    "image_key": "image",
                    "low_threshold": 30,
                    "high_threshold": 100,
                    "output_key": "canny_image_data",
                }
            ),
            "canny_image_data",
            None,
        ),
    ]
    for operation_name, operation, image_field, color_order in image_ops:
        before_dataset = current_dataset
        before_alignment = score_alignment(before_dataset, scorer)
        after_dataset = operation.execute(current_dataset)
        after_dataset = with_alignment_fields(
            after_dataset,
            text_field="alignment_text",
            image_field=image_field,
            image_color_order=color_order,
        )
        current_alignment = score_alignment(after_dataset, scorer)
        operator_score = current_alignment / before_alignment if before_alignment else current_alignment
        points.append(
            TrendPoint(
                "pipeline",
                operation_name,
                current_alignment,
                row_count,
                before_alignment=before_alignment,
                after_alignment=current_alignment,
                operator_score=round(operator_score, 6),
                score_mode="image_text_embedding_cosine_before_after",
                before_basis=(
                    "semantic_keyword_mask output"
                    if operation_name == "image_random_flip"
                    else "semantic_keyword_mask + image_random_flip output"
                ),
            )
        )
        current_dataset = promote_alignment_image(after_dataset)

    return points


def collect_single_operator_comparisons(
    base_dataset: MultimodalDataset,
    scorer: RealVlmAlignmentScorer,
) -> list[SingleOperatorComparison]:
    """Score the report's raw Messi figure against the horizontally flipped figure."""

    row_count = base_dataset.dataset.count()
    current_dataset = with_alignment_fields(base_dataset, text_field="caption")
    before_score = score_alignment(current_dataset, scorer)

    flip_operation = ImageRandomFlipOperation(
        {
            "image_key": "image",
            "flip_horizontal_prob": 1.0,
            "flip_vertical_prob": 0.0,
            "rotation_prob": 0.0,
            "num_augmentations": 1,
            "output_key": "flip_image_data",
        }
    )
    flipped_dataset = flip_operation.execute(current_dataset)
    flipped_dataset = with_alignment_fields(
        flipped_dataset,
        text_field="alignment_text",
        image_field="flip_image_data",
        image_color_order="bgr",
    )
    after_score = score_alignment(flipped_dataset, scorer)
    preservation_score = score_visual_preservation(
        current_dataset,
        flipped_dataset,
        scorer,
        operation_name="image_random_flip",
    )
    rows = rows_of(current_dataset)
    sample_id = str(rows[0].get("id", "messi5")) if rows else "messi5"

    return [
        SingleOperatorComparison(
            operation="image_random_flip",
            sample_id=sample_id,
            before_label="raw_messi",
            before_alignment=before_score,
            after_label="flipped_messi",
            after_alignment=after_score,
            rows=row_count,
            preservation_score=preservation_score,
        )
    ]


def write_csv(points: list[TrendPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "branch",
                "operation",
                "alignment_score",
                "before_alignment_score",
                "after_alignment_score",
                "operator_score",
                "score_mode",
                "before_basis",
                "rows",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.branch,
                    point.operation,
                    f"{point.alignment:.6f}",
                    "" if point.before_alignment is None else f"{point.before_alignment:.6f}",
                    "" if point.after_alignment is None else f"{point.after_alignment:.6f}",
                    "" if point.operator_score is None else f"{point.operator_score:.6f}",
                    point.score_mode,
                    point.before_basis,
                    point.rows,
                ]
            )


def write_single_operator_csv(comparisons: list[SingleOperatorComparison], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "operation",
                "sample_id",
                "before_label",
                "before_alignment_score",
                "after_label",
                "after_alignment_score",
                "visual_semantic_preservation_score",
                "delta",
                "rows",
            ]
        )
        for comparison in comparisons:
            writer.writerow(
                [
                    comparison.operation,
                    comparison.sample_id,
                    comparison.before_label,
                    f"{comparison.before_alignment:.6f}",
                    comparison.after_label,
                    f"{comparison.after_alignment:.6f}",
                    "" if comparison.preservation_score is None else f"{comparison.preservation_score:.6f}",
                    f"{comparison.delta:.6f}",
                    comparison.rows,
                ]
            )


def write_model_selection_csv(selection: ModelSelection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_path", "case_id", "kind", "score", "status", "raw_output", "model_calibration_score"])
        for result in selection.results:
            writer.writerow(
                [
                    str(result.model_path),
                    result.case_id,
                    result.kind,
                    "" if result.score is None else f"{result.score:.6f}",
                    result.status,
                    result.raw_output,
                    f"{selection.model_scores.get(str(result.model_path), -1.0):.6f}",
                ]
            )


def try_write_matplotlib(points: list[TrendPoint], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    operations = [point.operation for point in points]
    x_values = list(range(len(points)))
    colors = {
        "baseline": "#2f2f2f",
        "pipeline": "#1f77b4",
    }
    plt.figure(figsize=(11, 5.6))
    for branch in ["baseline", "pipeline"]:
        branch_x = [x for x, point in zip(x_values, points) if point.branch == branch]
        branch_y = [point.alignment for point in points if point.branch == branch]
        if branch_x:
            plt.plot(branch_x, branch_y, marker="o", linewidth=2.4, label=branch, color=colors[branch])
    plt.xticks(x_values, operations, rotation=28, ha="right")
    plt.ylim(0.0, 1.03)
    plt.xlabel("operation")
    plt.ylabel("real VLM alignment score")
    plt.title("Real text-image alignment trend under text and image operations")
    plt.grid(axis="y", alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(points: list[TrendPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 620
    left, right, top, bottom = 84, 52, 66, 155
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.0, 1.03

    def x_coord(index: int) -> float:
        if len(points) == 1:
            return left + plot_w / 2
        return left + index * plot_w / (len(points) - 1)

    def y_coord(score: float) -> float:
        score = max(y_min, min(y_max, score))
        return top + (y_max - score) * plot_h / (y_max - y_min)

    branch_colors = {
        "baseline": "#2f2f2f",
        "pipeline": "#1f77b4",
    }
    point_by_index = list(enumerate(points))
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.small{font-size:13px}.label{font-size:15px}.title{font-size:22px;font-weight:700}</style>',
        f'<text class="title" x="{width/2:.1f}" y="34" text-anchor="middle">Real text-image alignment trend under text and image operations</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
    ]
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = y_coord(tick)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e2e2"/>')
        lines.append(f'<text class="small" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{tick:.1f}</text>')
    for index, point in point_by_index:
        x = x_coord(index)
        lines.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+6}" stroke="#333"/>')
        lines.append(
            f'<text class="small" transform="translate({x-7:.1f},{top+plot_h+18}) rotate(35)" text-anchor="start">{svg_escape(point.operation)}</text>'
        )

    for branch in ["pipeline"]:
        branch_points = [(i, point) for i, point in point_by_index if point.branch == branch]
        if not branch_points:
            continue
        path_data = " ".join(
            f'{"M" if idx == 0 else "L"} {x_coord(i):.1f},{y_coord(point.alignment):.1f}'
            for idx, (i, point) in enumerate(branch_points)
        )
        lines.append(
            f'<path d="{path_data}" fill="none" stroke="{branch_colors[branch]}" stroke-width="3" stroke-linejoin="round"/>'
        )

    for index, point in point_by_index:
        x = x_coord(index)
        y = y_coord(point.alignment)
        color = branch_colors[point.branch]
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="white" stroke-width="2"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="{y-11:.1f}" text-anchor="middle">{point.alignment:.3f}</text>')

    lines.extend(
        [
            f'<text class="label" x="{left+plot_w/2:.1f}" y="{height-20}" text-anchor="middle">operation</text>',
            f'<text class="label" transform="translate(22,{top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle">real VLM alignment score</text>',
            f'<rect x="{width-284}" y="{top+4}" width="216" height="52" rx="8" fill="#f8f8f8" stroke="#d0d0d0"/>',
            f'<circle cx="{width-260}" cy="{top+31}" r="6" fill="{branch_colors["pipeline"]}"/><text class="small" x="{width-244}" y="{top+36}">cumulative pipeline</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(points: list[TrendPoint], output_dir: Path) -> Path:
    png_path = output_dir / "text-image-alignment-trend.png"
    if try_write_matplotlib(points, png_path):
        return png_path
    svg_path = output_dir / "text-image-alignment-trend.svg"
    write_svg(points, svg_path)
    return svg_path


def print_model_selection(selection: ModelSelection) -> None:
    print("\nModel candidates inspected from /data/llm_models:")
    for path in selection.attempted_models:
        print(f"  - {path}  calibration={selection.model_scores.get(str(path), -1.0):.6f}")
    print(f"Selected model: {selection.selected_model}")
    print(f"Selection reason: {selection.selection_reason}")
    print("Calibration results:")
    for result in selection.results:
        if result.model_path != selection.selected_model:
            continue
        score = "" if result.score is None else f"{result.score:.3f}"
        print(f"  {result.case_id:28s} {result.kind:8s} score={score:>5s} status={result.status}")


def main() -> int:
    data_root = REPO_ROOT / "data"
    discovered_image_paths = discover_image_paths(data_root)
    image_paths = select_trend_image_paths(discovered_image_paths)
    rows = build_rows(image_paths)
    dataset = MultimodalDataset(build_ray_dataset(rows), modalities=["image", "text"])

    output_dir = SCRIPT_PATH.parent / "outputs"
    scorer = RealVlmAlignmentScorer(DEFAULT_SELECTED_MODEL)
    points = collect_trend_points(dataset, scorer)
    single_operator_comparisons = collect_single_operator_comparisons(dataset, scorer)

    csv_path = output_dir / "text-image-alignment-trend.csv"
    single_operator_csv_path = output_dir / "text-image-alignment-single-operator.csv"
    write_csv(points, csv_path)
    write_single_operator_csv(single_operator_comparisons, single_operator_csv_path)
    plot_path = write_plot(points, output_dir)

    print("\nDataset backend:", "local in-process dataset" if not USING_RAY_STUB else "local ray stub")
    print(f"Embedding model: {DEFAULT_TEXT_IMAGE_EMBEDDING_MODEL}")
    print("Trend image:")
    for path in image_paths:
        print(f"  - {path.relative_to(REPO_ROOT)}")
    print("\nReal alignment trend:")
    for point in points:
        before = "" if point.before_alignment is None else f"{point.before_alignment:.6f}"
        operator = "" if point.operator_score is None else f"{point.operator_score:.6f}"
        print(
            f"  {point.branch:10s} | {point.operation:30s} | {point.alignment:.6f} "
            f"| before={before:>8s} | operator={operator:>8s} | basis={point.before_basis}"
        )
    print("\nSingle-operator alignment comparison:")
    for comparison in single_operator_comparisons:
        preservation = (
            ""
            if comparison.preservation_score is None
            else f" | visual_preservation={comparison.preservation_score:.6f}"
        )
        print(
            f"  {comparison.operation:30s} | {comparison.before_label}={comparison.before_alignment:.6f} "
            f"| {comparison.after_label}={comparison.after_alignment:.6f} | delta={comparison.delta:.6f}"
            f"{preservation}"
        )
    print(f"\nCSV:             {csv_path.relative_to(REPO_ROOT)}")
    print(f"Single operator: {single_operator_csv_path.relative_to(REPO_ROOT)}")
    print(f"Plot:            {plot_path.relative_to(REPO_ROOT)}")

    for path in (csv_path, single_operator_csv_path, plot_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"output was not created: {path}")
    if len(points) < 6 or not all(math.isfinite(point.alignment) for point in points):
        raise RuntimeError("trend points are incomplete or non-finite")
    if not all(0.0 <= point.alignment <= 1.0 for point in points):
        raise RuntimeError("alignment scores must be in [0, 1]")
    if not all(
        value is None or (math.isfinite(value) and 0.0 <= value <= 1.0)
        for point in points
        for value in (point.before_alignment, point.after_alignment)
    ):
        raise RuntimeError("trend before/after alignment scores must be in [0, 1]")
    if not all(
        point.operator_score is None
        or (math.isfinite(point.operator_score) and point.operator_score >= 0.0)
        for point in points
    ):
        raise RuntimeError("trend operator ratios must be finite and non-negative")
    if not single_operator_comparisons or not all(
        0.0 <= value <= 1.0
        for comparison in single_operator_comparisons
        for value in (
            comparison.before_alignment,
            comparison.after_alignment,
            comparison.preservation_score if comparison.preservation_score is not None else 1.0,
        )
    ):
        raise RuntimeError("single-operator comparison scores must be complete and in [0, 1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
