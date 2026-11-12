"""Runnable MindOCR inference runtime for the C-M3 adapter.

MindOCR 0.5.0 is a model toolbox rather than a one-call inference package. This
module composes its public DBNet detector, CRNN recognizer and postprocessors
into the single ``recognize`` entry point expected by ``MindOcrAdapter``.
"""
from __future__ import annotations

import importlib
import hashlib
import math
import os
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import OcrAdapterError, OcrRequest

_DET_MODEL_NAME = "dbnet_mobilenetv3"
_REC_MODEL_NAMES = {
    "en": "crnn_resnet34",
    "zh": "crnn_resnet34_ch",
    "mixed": "crnn_resnet34_ch",
}
_DET_TARGET_SIZE = (736, 1280)
_DET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_DET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_REC_HEIGHT = 32
_REC_MAX_WIDTH = 320
_REC_MEAN = 127.0
_REC_STD = 127.0
_PPOCR_DICT_LINES = 6623
_PPOCR_DICT_SHA256 = "28b2362ad4ab2dc38769aa72feb535e3a9ddb3fd2a7585a05920e6393b1dc7f7"
_PPOCR_DICT_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/"
    "8cce9b6fd7ccb50226d0c38f94054d81c29b8184/ppocr/utils/ppocr_keys_v1.txt"
)
_DEVICE_TARGETS = {"cpu": "CPU", "npu": "Ascend", "cuda": "GPU"}


def _backend_unavailable(message: str) -> OcrAdapterError:
    return OcrAdapterError("OCR_BACKEND_UNAVAILABLE", message)


def _model_unavailable(message: str) -> OcrAdapterError:
    return OcrAdapterError("OCR_MODEL_UNAVAILABLE", message)


def _apply_mindspore_compat_shim() -> None:
    """Keep MindOCR 0.5.0 importable with MindSpore 2.11.

    MindOCR imports AbiNet at package import time. AbiNet references private
    transformer modules removed from MindSpore, but DBNet/CRNN never use it.
    """
    try:
        importlib.import_module("mindspore.parallel._transformer.layers")
        importlib.import_module("mindspore.parallel._transformer.moe")
        importlib.import_module("mindspore.parallel._transformer.op_parallel_config")
        importlib.import_module("mindspore.parallel._transformer.transformer")
        return
    except Exception:
        pass

    class _Placeholder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> "_Placeholder":
            return self

        def to_float(self, *args: Any, **kwargs: Any) -> "_Placeholder":
            return self

    def _identity_decorator(*args: Any, **kwargs: Any):
        def _wrap(function: Any) -> Any:
            return function

        return _wrap

    layers = types.ModuleType("mindspore.parallel._transformer.layers")
    layers._args_type_validator_check = _identity_decorator
    for name in (
        "_check_input_dtype",
        "_check_past_none_input_none",
        "_valid_type_checks",
        "_valid_value_checks",
    ):
        setattr(layers, name, lambda *args, **kwargs: None)
    layers._LayerInputCheck = _Placeholder
    layers._LayerNorm = _Placeholder

    moe = types.ModuleType("mindspore.parallel._transformer.moe")
    moe.MoE = _Placeholder
    moe._check_moe_config = lambda *args, **kwargs: None
    moe.default_moe_config = _Placeholder()

    op_config = types.ModuleType("mindspore.parallel._transformer.op_parallel_config")
    op_config.MoEParallelConfig = _Placeholder
    op_config.OpParallelConfig = _Placeholder
    op_config._check_config = lambda *args, **kwargs: None
    op_config.default_dpmp_config = _Placeholder()

    transformer = types.ModuleType("mindspore.parallel._transformer.transformer")
    transformer.FeedForward = _Placeholder
    transformer.MultiHeadAttention = _Placeholder
    transformer.TransformerOpParallelConfig = _Placeholder
    transformer._get_lambda_func = lambda *args, **kwargs: (lambda *a, **k: None)
    transformer.default_transformer_config = _Placeholder()

    root = types.ModuleType("mindspore.parallel._transformer")
    root.layers = layers
    root.moe = moe
    root.op_parallel_config = op_config
    root.transformer = transformer
    root.__path__ = []  # type: ignore[attr-defined]

    import mindspore.parallel

    modules = {
        "mindspore.parallel._transformer": root,
        "mindspore.parallel._transformer.layers": layers,
        "mindspore.parallel._transformer.moe": moe,
        "mindspore.parallel._transformer.op_parallel_config": op_config,
        "mindspore.parallel._transformer.transformer": transformer,
    }
    sys.modules.update(modules)
    mindspore.parallel._transformer = root


def _load_backend() -> Tuple[Any, Any, Any, Any]:
    try:
        import mindspore as ms
    except Exception as exc:
        raise _backend_unavailable(
            f"MindSpore backend could not be imported ({type(exc).__name__})"
        ) from None
    _apply_mindspore_compat_shim()
    try:
        from mindocr.models import build_model
        from mindocr.postprocess import build_postprocess
    except Exception as exc:
        raise _backend_unavailable(
            f"MindOCR backend could not be imported ({type(exc).__name__})"
        ) from None
    try:
        import cv2
    except Exception as exc:
        raise _backend_unavailable(
            f"OpenCV backend could not be imported ({type(exc).__name__})"
        ) from None
    return ms, cv2, build_model, build_postprocess


def _device_target(device: str) -> str:
    try:
        return _DEVICE_TARGETS[device]
    except KeyError:
        raise _backend_unavailable("unsupported MindOCR device") from None


def _model_checkpoint(raw: Optional[str], kind: str) -> Optional[str]:
    if not raw:
        return None
    path = Path(raw)
    if path.is_dir():
        candidates = sorted(path.glob("*.ckpt"))
        if not candidates:
            raise _model_unavailable(
                f"{kind} model directory does not contain a checkpoint"
            )
        path = candidates[0]
    if not path.is_file():
        raise _model_unavailable(f"{kind} model path is not a checkpoint file")
    return str(path)


def _cache_dir() -> Path:
    configured = os.environ.get("CURATOR_CACHE_DIR")
    if configured:
        return Path(configured).expanduser() / "mindocr"
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "data-forge" / "mindocr"


def _read_dictionary(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return [line.rstrip("\r\n") for line in handle]
    except (OSError, UnicodeError) as exc:
        raise _model_unavailable(
            f"recognition dictionary is not readable ({type(exc).__name__})"
        ) from None


def _validate_dictionary(path: Path) -> None:
    if len(_read_dictionary(path)) != _PPOCR_DICT_LINES:
        raise _model_unavailable("recognition dictionary has an unexpected size")


def _validate_official_dictionary(path: Path) -> None:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise _model_unavailable(
            f"recognition dictionary is not readable ({type(exc).__name__})"
        ) from None
    if digest != _PPOCR_DICT_SHA256:
        raise _model_unavailable("recognition dictionary checksum does not match")
    _validate_dictionary(path)


def _download_dictionary(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with urllib.request.urlopen(_PPOCR_DICT_URL, timeout=60) as response:
            payload = response.read()
        temporary.write_bytes(payload)
        _validate_official_dictionary(temporary)
        os.replace(temporary, destination)
    except OcrAdapterError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    except Exception as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise _model_unavailable(
            f"recognition dictionary could not be downloaded ({type(exc).__name__})"
        ) from None


def _recognition_dictionary(
    request: OcrRequest, *, allow_model_download: bool
) -> Optional[str]:
    if request.lang == "en":
        return None
    configured = request.model_paths.get("rec_dict")
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise _model_unavailable("recognition dictionary path is not a file")
        _validate_dictionary(path)
        return str(path)
    cached = _cache_dir() / "ppocr_keys_v1.txt"
    if cached.is_file():
        _validate_official_dictionary(cached)
        return str(cached)
    if not allow_model_download:
        raise _model_unavailable(
            "Chinese recognition dictionary is unavailable; "
            "set allow_model_download or provide model_paths.rec_dict"
        )
    _download_dictionary(cached)
    return str(cached)


def _build_model(
    build_model: Any,
    name: str,
    checkpoint: Optional[str],
    *,
    allow_model_download: bool,
    detector: bool = False,
) -> Any:
    if checkpoint:
        kwargs: Dict[str, Any] = {"pretrained": False, "ckpt_load_path": checkpoint}
        if detector:
            kwargs["pretrained_backbone"] = False
        try:
            return build_model(name, **kwargs)
        except Exception as exc:
            raise _model_unavailable(
                f"{name} model could not be loaded ({type(exc).__name__})"
            ) from None
    if not allow_model_download:
        raise _model_unavailable(
            f"{name} weights are not available; "
            "set allow_model_download or provide model_paths"
        )
    try:
        return build_model(name, pretrained=True)
    except Exception as exc:
        raise _model_unavailable(
            f"{name} pretrained model could not be loaded ({type(exc).__name__})"
        ) from None


def _order_points_clockwise(points: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if array.shape[0] != 4:
        raise _backend_unavailable("text detection returned a non-quadrilateral box")
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = array.sum(axis=1)
    differences = np.diff(array, axis=1).ravel()
    ordered[0] = array[np.argmin(sums)]
    ordered[2] = array[np.argmax(sums)]
    ordered[1] = array[np.argmin(differences)]
    ordered[3] = array[np.argmax(differences)]
    return ordered


def _crop_text_region(
    cv2: Any, pixels: np.ndarray, polygon: Sequence[Sequence[float]]
) -> Optional[np.ndarray]:
    points = _order_points_clockwise(polygon)
    width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    if width < 3 or height < 3:
        return None
    destination = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(pixels, transform, (width, height))


def _recognition_input(cv2: Any, crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    target_width = min(
        _REC_MAX_WIDTH,
        max(8, int(math.ceil(_REC_HEIGHT * width / float(height)))),
    )
    resized = cv2.resize(crop, (target_width, _REC_HEIGHT))
    normalized = (resized.astype(np.float32) - _REC_MEAN) / _REC_STD
    return normalized.transpose(2, 0, 1)[None]


class MindOcrRuntime:
    """Minimal DBNet + CRNN runtime over MindOCR model components."""

    def __init__(
        self, request: OcrRequest, *, allow_model_download: bool = False
    ) -> None:
        if not isinstance(request, OcrRequest):
            raise ValueError("request must be an OcrRequest")
        if not isinstance(allow_model_download, bool):
            raise ValueError("allow_model_download must be boolean")
        self.request = request
        self.allow_model_download = allow_model_download
        self._ms, self._cv2, build_model, build_postprocess = _load_backend()
        target = _device_target(request.device)
        try:
            self._ms.set_context(mode=self._ms.PYNATIVE_MODE, device_target=target)
        except Exception as exc:
            raise _backend_unavailable(
                f"MindSpore device could not be initialized ({type(exc).__name__})"
            ) from None

        det_checkpoint = _model_checkpoint(request.model_paths.get("det"), "detector")
        rec_checkpoint = _model_checkpoint(request.model_paths.get("rec"), "recognizer")
        rec_name = _REC_MODEL_NAMES[request.lang]
        self._detector = _build_model(
            build_model,
            _DET_MODEL_NAME,
            det_checkpoint,
            allow_model_download=allow_model_download,
            detector=True,
        )
        self._recognizer = _build_model(
            build_model,
            rec_name,
            rec_checkpoint,
            allow_model_download=allow_model_download,
        )
        try:
            self._detector.set_train(False)
            self._recognizer.set_train(False)
        except Exception:
            pass

        try:
            self._detect_post = build_postprocess(
                {
                    "name": "DBPostprocess",
                    "binary_thresh": 0.3,
                    "box_thresh": 0.6,
                    "max_candidates": 1000,
                    "expand_ratio": 1.5,
                    "box_type": "quad",
                    "pred_name": "binary",
                    "rescale_fields": ["polys"],
                }
            )
            rec_config: Dict[str, Any] = {
                "name": "RecCTCLabelDecode",
                "use_space_char": False,
            }
            dictionary = _recognition_dictionary(
                request, allow_model_download=allow_model_download
            )
            if dictionary is not None:
                rec_config["character_dict_path"] = dictionary
            self._recognize_post = build_postprocess(rec_config)
        except OcrAdapterError:
            raise
        except Exception as exc:
            raise _backend_unavailable(
                f"MindOCR postprocess could not be initialized ({type(exc).__name__})"
            ) from None

    def _detect(self, pixels: np.ndarray) -> Tuple[Sequence[Any], Sequence[Any]]:
        height, width = pixels.shape[:2]
        target_height, target_width = _DET_TARGET_SIZE
        resized = self._cv2.resize(pixels, (target_width, target_height))
        normalized = (resized.astype(np.float32) - _DET_MEAN) / _DET_STD
        tensor = self._ms.Tensor(
            normalized.transpose(2, 0, 1)[None], self._ms.float32
        )
        try:
            raw = self._detector(tensor)
            shape_list = np.array(
                [
                    [
                        height,
                        width,
                        target_height / float(height),
                        target_width / float(width),
                    ]
                ],
                dtype=np.float32,
            )
            result = self._detect_post(raw, shape_list=shape_list)
            return result["polys"][0], result["scores"][0]
        except OcrAdapterError:
            raise
        except Exception as exc:
            raise OcrAdapterError(
                "OCR_EXECUTION_FAILED",
                f"MindOCR text detection failed ({type(exc).__name__})",
            ) from None

    def _recognize_crop(self, crop: np.ndarray) -> Tuple[str, Optional[float]]:
        try:
            tensor = self._ms.Tensor(
                _recognition_input(self._cv2, crop), self._ms.float32
            )
            decoded = self._recognize_post(self._recognizer(tensor))
            texts = decoded["texts"]
            confidences = decoded["confs"]
            if not texts:
                return "", None
            confidence = confidences[0] if confidences else None
            return str(texts[0]), None if confidence is None else float(confidence)
        except Exception as exc:
            raise OcrAdapterError(
                "OCR_EXECUTION_FAILED",
                f"MindOCR text recognition failed ({type(exc).__name__})",
            ) from None

    def recognize(
        self, pixels: np.ndarray, request: Optional[OcrRequest] = None
    ) -> List[Dict[str, Any]]:
        if not isinstance(pixels, np.ndarray):
            raise OcrAdapterError("OCR_EXECUTION_FAILED", "invalid OCR image")
        polygons, _ = self._detect(pixels)
        records: List[Dict[str, Any]] = []
        for polygon in polygons:
            crop = _crop_text_region(self._cv2, pixels, polygon)
            if crop is None:
                continue
            text, confidence = self._recognize_crop(crop)
            if not text:
                continue
            records.append(
                {
                    "text": text,
                    "bbox": np.asarray(polygon, dtype=np.float32).tolist(),
                    "confidence": confidence,
                    "type": "paragraph",
                }
            )
        return records


__all__ = ["MindOcrRuntime"]
