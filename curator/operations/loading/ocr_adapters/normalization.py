"""Backend-neutral normalization for OCR runtime output."""
from __future__ import annotations

import inspect
import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .base import OcrAdapter, OcrAdapterError, OcrCandidate, OcrImage, OcrRequest

MISSING = object()
IMAGE_ARGUMENT_NAMES = frozenset(
    {
        "image",
        "img",
        "img_or_path",
        "image_path",
        "input",
        "input_image",
        "numpy_image",
        "pixels",
        "src_img",
    }
)
REQUEST_ARGUMENT_NAMES = frozenset({"request", "ocr_request"})
LANG_ARGUMENT_NAMES = frozenset({"lang", "language"})
DEVICE_ARGUMENT_NAMES = frozenset({"device", "device_target"})
MODEL_PATHS_ARGUMENT_NAMES = frozenset({"model_paths"})
TEXT_KEYS = ("text", "rec_text", "transcription", "content")
CONFIDENCE_KEYS = (
    "confidence",
    "score",
    "rec_score",
    "rec_scores",
    "scores",
    "confidences",
)
BBOX_KEYS = (
    "bbox",
    "box",
    "boxes",
    "bboxes",
    "points",
    "poly",
    "polygon",
    "rec_poly",
    "rec_polys",
    "rec_boxes",
    "rec_box",
    "det_poly",
    "det_polys",
    "det_box",
    "det_boxes",
)
TYPE_KEYS = ("block_type", "type", "category", "layout_type")
NESTED_RESULT_KEYS = (
    "results",
    "result",
    "ocr_result",
    "lines",
    "items",
    "predictions",
    "res",
)


def _execution_error(backend: str, message: str) -> OcrAdapterError:
    return OcrAdapterError("OCR_EXECUTION_FAILED", f"{backend} {message}")


def first_value(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    default: Any = MISSING,
) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    data = getattr(value, "__dict__", None)
    if isinstance(data, Mapping):
        return data
    return None


def is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def clean_text(value: Any, backend: str) -> str:
    if not isinstance(value, str):
        raise _execution_error(backend, "returned a non-string text value")
    text = unicodedata.normalize("NFC", value).replace("\x00", "")
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_confidence(value: Any, backend: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise _execution_error(backend, "returned an invalid confidence")
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise _execution_error(backend, "returned an invalid confidence") from None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise _execution_error(backend, "returned an invalid confidence")
    return score


def looks_like_bbox(value: Any) -> bool:
    try:
        array = np.asarray(value)
    except Exception:
        return False
    if array.ndim == 1:
        return array.size == 4 or (array.size >= 8 and array.size % 2 == 0)
    return array.ndim == 2 and array.shape[1] == 2


def normalize_bbox(
    value: Any, width: int, height: int, backend: str
) -> Tuple[float, float, float, float]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise _execution_error(backend, "returned an invalid bounding box") from None
    if not np.all(np.isfinite(array)):
        raise _execution_error(backend, "returned an invalid bounding box")
    if array.ndim == 1 and array.size == 4:
        x0, y0, x1, y1 = (float(item) for item in array.tolist())
        if x0 >= x1 or y0 >= y1:
            raise _execution_error(backend, "returned an invalid bounding box")
    elif array.ndim == 1 and array.size >= 8 and array.size % 2 == 0:
        points = array.reshape(-1, 2)
        x0, y0 = (float(item) for item in points.min(axis=0))
        x1, y1 = (float(item) for item in points.max(axis=0))
    elif array.ndim == 2 and array.shape[1] == 2 and array.shape[0] >= 2:
        x0, y0 = (float(item) for item in array.min(axis=0))
        x1, y1 = (float(item) for item in array.max(axis=0))
    else:
        raise _execution_error(backend, "returned an invalid bounding box")

    if not all(0.0 <= item <= 1.0 for item in (x0, y0, x1, y1)):
        x0, x1 = x0 / width, x1 / width
        y0, y1 = y0 / height, y1 / height
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise _execution_error(backend, "returned an out-of-range bounding box")
    return (x0, y0, x1, y1)


def normalize_block_type(value: Any, backend: str) -> str:
    if value is None:
        return "paragraph"
    if not isinstance(value, str):
        raise _execution_error(backend, "returned an invalid block type")
    normalized = value.strip().lower()
    aliases = {
        "text": "paragraph",
        "title": "paragraph",
        "heading": "paragraph",
        "plain": "paragraph",
        "image": "figure",
        "picture": "figure",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {
        "paragraph",
        "table",
        "figure",
        "header",
        "footer",
        "list",
    }:
        return normalized
    return "paragraph"


def parallel_value(value: Any, index: int, count: int, backend: str) -> Any:
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == count:
        return value[index]
    if isinstance(value, (list, tuple)):
        if len(value) == count:
            return value[index]
        if count == 1:
            return value
    if count == 1:
        return value
    raise _execution_error(backend, "returned parallel arrays with mismatched lengths")


def tuple_record(value: Sequence[Any]) -> Any:
    text: Any = None
    score: Any = None
    bbox: Any = None
    for item in value:
        if isinstance(item, str) and text is None:
            text = item
            continue
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], str)
            and text is None
        ):
            text = item[0]
            if score is None:
                score = item[1]
            continue
        if isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(
            item, bool
        ):
            if score is None:
                score = item
            continue
        if bbox is None and looks_like_bbox(item):
            bbox = item
    if text is None:
        return value
    record: dict[str, Any] = {"text": text}
    if score is not None:
        record["score"] = score
    if bbox is not None:
        record["bbox"] = bbox
    return record


def mapping_records(mapping: Mapping[str, Any], backend: str) -> Iterable[Any]:
    text_value = first_value(mapping, TEXT_KEYS)
    if text_value is not MISSING and not is_sequence(text_value):
        yield mapping
        return

    for key in NESTED_RESULT_KEYS:
        if key in mapping:
            nested = mapping[key]
            if isinstance(nested, Mapping) or is_sequence(nested):
                yield from extract_records(nested, backend)
                return

    texts = first_value(mapping, ("rec_texts", "texts", "text"))
    if texts is MISSING or isinstance(texts, str) or not is_sequence(texts):
        yield mapping
        return
    count = len(texts)
    scores = first_value(mapping, CONFIDENCE_KEYS, None)
    boxes = first_value(mapping, BBOX_KEYS, None)
    types = first_value(mapping, ("block_types", "types"), None)
    for index in range(count):
        record: dict[str, Any] = {
            "text": parallel_value(texts, index, count, backend)
        }
        if scores is not None:
            record["score"] = parallel_value(scores, index, count, backend)
        if boxes is not None:
            record["bbox"] = parallel_value(boxes, index, count, backend)
        if types is not None:
            record["block_type"] = parallel_value(types, index, count, backend)
        yield record


def extract_records(raw: Any, backend: str) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes, bytearray)):
        raise _execution_error(backend, "returned an unsupported result type")
    mapping = as_mapping(raw)
    if mapping is not None:
        return list(mapping_records(mapping, backend))
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return []
        if raw.ndim == 1:
            raw = raw.tolist()
        else:
            raise _execution_error(backend, "returned an unsupported result type")
    if isinstance(raw, Iterable):
        records: list[Any] = []
        for item in raw:
            item_mapping = as_mapping(item)
            if item_mapping is not None:
                records.extend(mapping_records(item_mapping, backend))
            elif isinstance(item, Sequence) and not isinstance(
                item, (str, bytes, bytearray)
            ):
                record = tuple_record(item)
                if isinstance(record, Mapping):
                    records.append(record)
                elif any(as_mapping(nested) is not None for nested in item):
                    records.extend(extract_records(item, backend))
                else:
                    records.append(record)
            else:
                records.append(item)
        return records
    raise _execution_error(backend, "returned an unsupported result type")


def mapping_candidates(
    records: Iterable[Any],
    image: OcrImage,
    request: OcrRequest,
    backend: str,
) -> list[OcrCandidate]:
    candidates: list[OcrCandidate] = []
    for record in records:
        if isinstance(record, OcrCandidate):
            candidates.append(record)
            continue
        mapping = as_mapping(record)
        if mapping is None:
            raise _execution_error(backend, "returned an invalid candidate type")
        text_value = first_value(mapping, TEXT_KEYS)
        if text_value is MISSING:
            raise _execution_error(backend, "returned a candidate without text")
        text = clean_text(text_value, backend)
        if not text:
            continue
        bbox_value = first_value(mapping, BBOX_KEYS)
        if bbox_value is MISSING:
            raise _execution_error(backend, "returned a candidate without a bounding box")
        bbox = normalize_bbox(bbox_value, image.width, image.height, backend)
        confidence = normalize_confidence(
            first_value(mapping, CONFIDENCE_KEYS, None), backend
        )
        block_type = normalize_block_type(
            first_value(mapping, TYPE_KEYS, None), backend
        )
        if not request.layout_analysis:
            block_type = "paragraph"
        candidates.append(
            OcrCandidate(
                page=1,
                order=len(candidates),
                block_type=block_type,
                text=text,
                bbox=bbox,
                confidence=confidence,
                table=None,
            )
        )
    return candidates


def validate_ocr_call(
    adapter: OcrAdapter, image: OcrImage, request: OcrRequest
) -> None:
    if not isinstance(image, OcrImage):
        raise OcrAdapterError("OCR_EXECUTION_FAILED", "invalid OCR image")
    if not isinstance(request, OcrRequest):
        raise OcrAdapterError("OCR_EXECUTION_FAILED", "invalid OCR request")
    if request.lang not in adapter.capabilities.languages:
        raise OcrAdapterError(
            "OCR_EXECUTION_FAILED",
            f"{adapter.name} received an unsupported language",
        )
    if request.device not in adapter.supported_devices:
        raise OcrAdapterError(
            "OCR_BACKEND_UNAVAILABLE",
            f"{adapter.name} does not support the requested device",
        )
    if request.layout_analysis and not adapter.capabilities.layout:
        raise OcrAdapterError(
            "OCR_BACKEND_UNAVAILABLE",
            f"{adapter.name} does not support layout analysis",
        )


def runner_call_arguments(
    runner: Callable[..., Any],
    pixels: np.ndarray,
    request: OcrRequest,
    *,
    backend: str,
    lang_value: Any,
    device_value: Any,
) -> tuple[list[Any], dict[str, Any]]:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return [pixels], {}
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    image_assigned = False
    has_var_keyword = False
    has_var_positional = False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
            continue
        name = parameter.name
        if name in {"self", "cls"}:
            continue
        value: Any = MISSING
        if name in IMAGE_ARGUMENT_NAMES and not image_assigned:
            value = pixels
            image_assigned = True
        elif name in REQUEST_ARGUMENT_NAMES:
            value = request
        elif name in LANG_ARGUMENT_NAMES:
            value = lang_value
        elif name in DEVICE_ARGUMENT_NAMES:
            value = device_value
        elif name in MODEL_PATHS_ARGUMENT_NAMES:
            value = request.model_paths
        if value is MISSING:
            if parameter.default is not inspect.Parameter.empty:
                continue
            raise OcrAdapterError(
                "OCR_BACKEND_UNAVAILABLE", f"{backend} runtime API is incompatible"
            )
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value
    if not image_assigned:
        if has_var_positional:
            args.append(pixels)
        elif has_var_keyword:
            kwargs["image"] = pixels
        else:
            args.insert(0, pixels)
    return args, kwargs


def invoke_runtime(
    runtime: Any,
    pixels: np.ndarray,
    request: OcrRequest,
    *,
    backend: str,
    lang_value: Any,
    device_value: Any,
) -> Any:
    for method_name in ("recognize", "ocr", "run", "predict"):
        runner = getattr(runtime, method_name, None)
        if callable(runner):
            args, kwargs = runner_call_arguments(
                runner,
                pixels,
                request,
                backend=backend,
                lang_value=lang_value,
                device_value=device_value,
            )
            return runner(*args, **kwargs)
    if callable(runtime):
        args, kwargs = runner_call_arguments(
            runtime,
            pixels,
            request,
            backend=backend,
            lang_value=lang_value,
            device_value=device_value,
        )
        return runtime(*args, **kwargs)
    raise OcrAdapterError(
        "OCR_BACKEND_UNAVAILABLE", f"{backend} runtime does not expose an OCR method"
    )


def runtime_error(
    exc: Exception,
    *,
    backend: str,
    model_tokens: Sequence[str],
    backend_tokens: Sequence[str],
) -> OcrAdapterError:
    message = str(exc).lower()
    if any(token in message for token in model_tokens):
        return OcrAdapterError(
            "OCR_MODEL_UNAVAILABLE",
            f"{backend} model could not be loaded ({type(exc).__name__})",
        )
    if any(token in message for token in backend_tokens):
        return OcrAdapterError(
            "OCR_BACKEND_UNAVAILABLE",
            f"{backend} backend is unavailable ({type(exc).__name__})",
        )
    return OcrAdapterError(
        "OCR_EXECUTION_FAILED",
        f"{backend} execution failed ({type(exc).__name__})",
    )


__all__ = [
    "BBOX_KEYS",
    "CONFIDENCE_KEYS",
    "MISSING",
    "NESTED_RESULT_KEYS",
    "TEXT_KEYS",
    "TYPE_KEYS",
    "as_mapping",
    "extract_records",
    "first_value",
    "invoke_runtime",
    "mapping_candidates",
    "normalize_bbox",
    "normalize_block_type",
    "normalize_confidence",
    "runtime_error",
    "validate_ocr_call",
]
