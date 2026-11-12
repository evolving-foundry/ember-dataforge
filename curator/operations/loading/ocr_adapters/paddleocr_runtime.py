"""Runnable PaddleOCR inference runtime for the C-M3 adapter.

PaddleOCR 3.7 accepts an RGB-external image as an OpenCV BGR array.  The
runtime pins the official PP-OCRv6 medium detector/recognizer package so that
PaddleOcrAdapter.default_model_id stays stable across languages.  Model
downloads are opt-in; cached PaddleX official models are reused when present.
CPU oneDNN is disabled because PaddlePaddle 3.3.1 fails to convert PP-OCRv6
squared-attribute operators through the oneDNN PIR path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import OcrAdapterError, OcrRequest
from .normalization import as_mapping, extract_records, runtime_error

_DET_MODEL_NAME = "PP-OCRv6_medium_det"
_REC_MODEL_NAME = "PP-OCRv6_medium_rec"
_DEVICE_NAMES = {"cpu": "cpu", "cuda": "gpu", "npu": "npu"}
_LANGUAGE_NAMES = {"zh": "ch", "mixed": "ch", "en": "en"}
_UNSUPPORTED_MODEL_KEYS = frozenset({"rec_dict", "layout"})
_MODEL_ERROR_TOKENS = (
    "download",
    "huggingface",
    "model",
    "parameter",
    "weight",
)
_BACKEND_ERROR_TOKENS = (
    "backend",
    "cuda",
    "device",
    "gpu",
    "import",
    "npu",
    "paddle",
    "paddlex",
    "runtime",
)


def _backend_unavailable(message: str) -> OcrAdapterError:
    return OcrAdapterError("OCR_BACKEND_UNAVAILABLE", message)


def _model_unavailable(message: str) -> OcrAdapterError:
    return OcrAdapterError("OCR_MODEL_UNAVAILABLE", message)


def _device_name(device: str) -> str:
    try:
        return _DEVICE_NAMES[device]
    except KeyError:
        raise _backend_unavailable("unsupported PaddleOCR device") from None


def _language_name(lang: str) -> str:
    try:
        return _LANGUAGE_NAMES[lang]
    except KeyError:
        raise _backend_unavailable("unsupported PaddleOCR language") from None


def _cache_dir() -> Path:
    configured = os.environ.get("PADDLE_PDX_CACHE_HOME")
    root = Path(configured).expanduser() if configured else Path.home() / ".paddlex"
    return root / "official_models"


def _is_model_directory(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def _custom_model_dir(raw: Optional[str], kind: str) -> Optional[str]:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not _is_model_directory(path):
        raise _model_unavailable(f"{kind} model directory is missing or empty")
    return str(path)


def _official_model_dir(model_name: str) -> Optional[str]:
    path = _cache_dir() / model_name
    return str(path) if _is_model_directory(path) else None


def _resolve_model_dirs(
    request: OcrRequest, *, allow_model_download: bool
) -> Tuple[Optional[str], Optional[str]]:
    unsupported = sorted(
        key for key in _UNSUPPORTED_MODEL_KEYS if request.model_paths.get(key)
    )
    if unsupported:
        raise _model_unavailable(
            "PaddleOCR does not support model_paths keys: "
            + ", ".join(unsupported)
        )

    det_dir = _custom_model_dir(request.model_paths.get("det"), "detector")
    rec_dir = _custom_model_dir(request.model_paths.get("rec"), "recognizer")
    missing = []
    if det_dir is None:
        det_dir = _official_model_dir(_DET_MODEL_NAME)
        if det_dir is None:
            missing.append("detector")
    if rec_dir is None:
        rec_dir = _official_model_dir(_REC_MODEL_NAME)
        if rec_dir is None:
            missing.append("recognizer")
    if missing and not allow_model_download:
        raise _model_unavailable(
            "PaddleOCR official model is unavailable; set allow_model_download "
            "or provide model_paths.det and model_paths.rec"
        )
    return det_dir, rec_dir


def _engine_kwargs(
    request: OcrRequest, *, allow_model_download: bool
) -> Dict[str, Any]:
    if not isinstance(request, OcrRequest):
        raise ValueError("request must be an OcrRequest")
    if not isinstance(allow_model_download, bool):
        raise ValueError("allow_model_download must be boolean")
    _language_name(request.lang)
    det_dir, rec_dir = _resolve_model_dirs(
        request, allow_model_download=allow_model_download
    )
    kwargs: Dict[str, Any] = {
        "text_detection_model_name": _DET_MODEL_NAME,
        "text_recognition_model_name": _REC_MODEL_NAME,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": _device_name(request.device),
    }
    if request.device == "cpu":
        kwargs["enable_mkldnn"] = False
    if det_dir is not None:
        kwargs["text_detection_model_dir"] = det_dir
    if rec_dir is not None:
        kwargs["text_recognition_model_dir"] = rec_dir
    return kwargs


def _load_backend() -> Any:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise _backend_unavailable(
            f"PaddleOCR backend could not be imported ({type(exc).__name__})"
        ) from None
    return PaddleOCR


def _runtime_error(exc: Exception) -> OcrAdapterError:
    return runtime_error(
        exc,
        backend="PaddleOCR",
        model_tokens=_MODEL_ERROR_TOKENS,
        backend_tokens=_BACKEND_ERROR_TOKENS,
    )


def _to_bgr(pixels: np.ndarray) -> np.ndarray:
    if not isinstance(pixels, np.ndarray) or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise OcrAdapterError("OCR_EXECUTION_FAILED", "invalid OCR image")
    return np.ascontiguousarray(pixels[:, :, ::-1])


def _records_from_value(value: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for value_record in extract_records(value, "PaddleOCR"):
        mapping = as_mapping(value_record)
        if mapping is None:
            raise OcrAdapterError(
                "OCR_EXECUTION_FAILED",
                "PaddleOCR returned an unsupported result type",
            )
        records.append(dict(mapping))
    return records


class PaddleOcrRuntime:
    """PaddleOCR pipeline normalized to simple text/bbox/confidence records."""

    def __init__(
        self,
        request: OcrRequest,
        *,
        allow_model_download: bool = False,
        engine_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        if not isinstance(request, OcrRequest):
            raise ValueError("request must be an OcrRequest")
        if not isinstance(allow_model_download, bool):
            raise ValueError("allow_model_download must be boolean")
        self.request = request
        self.allow_model_download = allow_model_download
        if engine_factory is None:
            paddle_ocr = _load_backend()
            kwargs = _engine_kwargs(
                request, allow_model_download=allow_model_download
            )
            try:
                self._engine = paddle_ocr(**kwargs)
            except OcrAdapterError:
                raise
            except Exception as exc:
                raise _runtime_error(exc) from None
        else:
            self._engine = engine_factory(
                _engine_kwargs(request, allow_model_download=allow_model_download)
            )
        if self._engine is None:
            raise _backend_unavailable("PaddleOCR runtime factory returned no engine")

    def recognize(
        self, pixels: np.ndarray, request: Optional[OcrRequest] = None
    ) -> List[Dict[str, Any]]:
        image = _to_bgr(pixels)
        try:
            raw = self._engine.predict(image)
            return _records_from_value(raw)
        except OcrAdapterError:
            raise
        except Exception as exc:
            raise _runtime_error(exc) from None

    def close(self) -> None:
        close = getattr(self._engine, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


__all__ = ["PaddleOcrRuntime"]
