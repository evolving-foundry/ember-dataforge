"""MindOCR adapter for the C-M3a candidate contract."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from typing import Any, Callable, FrozenSet, Optional

from .base import OcrAdapter, OcrAdapterError, OcrCandidate, OcrCapabilities, OcrImage, OcrRequest, merge_candidate_lines
from .normalization import (
    extract_records,
    invoke_runtime,
    mapping_candidates,
    runtime_error,
    validate_ocr_call,
)

_MINDOCR_MODULE = "mindocr"
_DEFAULT_MODEL_ID = (
    "mindocr-official-dbnet_mobilenetv3-crnn_resnet34-"
    "crnn_resnet34_ch-ppocr_keys_v1-v1"
)
_MODEL_ERROR_TOKENS = (
    "checkpoint",
    "ckpt",
    "model",
    "parameter",
    "weight",
    "yaml",
)
_BACKEND_ERROR_TOKENS = (
    "ascend",
    "backend",
    "cuda",
    "device",
    "gpu",
    "import",
    "mindspore",
    "npu",
    "runtime",
)


def _backend_language(lang: str) -> str:
    return "ch" if lang in {"zh", "mixed"} else "en"


def _backend_device(device: str) -> str:
    return {"npu": "Ascend", "cuda": "GPU", "cpu": "CPU"}[device]


def _default_runtime_factory(
    request: OcrRequest, *, allow_model_download: bool = False
) -> Any:
    try:
        from .mindocr_runtime import MindOcrRuntime
    except Exception as exc:
        raise OcrAdapterError(
            "OCR_BACKEND_UNAVAILABLE",
            f"MindOCR runtime could not be imported ({type(exc).__name__})",
        ) from None
    return MindOcrRuntime(request, allow_model_download=allow_model_download)


def _runtime_error(exc: Exception) -> OcrAdapterError:
    return runtime_error(
        exc,
        backend="MindOCR",
        model_tokens=_MODEL_ERROR_TOKENS,
        backend_tokens=_BACKEND_ERROR_TOKENS,
    )


class MindOcrAdapter(OcrAdapter):
    """MindOCR detector/recognizer normalized to the C-M3a contract."""

    name = "mindocr"
    supported_devices: FrozenSet[str] = frozenset({"npu", "cuda", "cpu"})
    capabilities = OcrCapabilities(
        languages=frozenset({"zh", "en", "mixed"}),
        layout=True,
        table_grid=False,
        confidence=True,
    )

    def __init__(
        self,
        runtime_factory: Optional[Callable[[OcrRequest], Any]] = None,
        *,
        allow_model_download: bool = False,
    ) -> None:
        if not isinstance(allow_model_download, bool):
            raise ValueError("allow_model_download must be boolean")
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self.allow_model_download = allow_model_download
        self._runtime: Any = None
        self._runtime_key: Optional[tuple[Any, ...]] = None

    @property
    def version(self) -> str:
        try:
            return importlib.metadata.version(_MINDOCR_MODULE)
        except (importlib.metadata.PackageNotFoundError, ValueError):
            return "unknown"

    @property
    def default_model_id(self) -> Optional[str]:
        return _DEFAULT_MODEL_ID

    @classmethod
    def dependency_installed(cls) -> bool:
        if _MINDOCR_MODULE in sys.modules:
            return sys.modules[_MINDOCR_MODULE] is not None
        try:
            return importlib.util.find_spec(_MINDOCR_MODULE) is not None
        except (ImportError, ValueError):
            return False

    def _runtime_for_request(self, request: OcrRequest) -> Any:
        key = (
            request.lang,
            request.device,
            tuple(sorted(request.model_paths.items())),
        )
        if self._runtime is None or self._runtime_key != key:
            self.close()
            try:
                if self._runtime_factory is _default_runtime_factory:
                    runtime = _default_runtime_factory(
                        request, allow_model_download=self.allow_model_download
                    )
                else:
                    runtime = self._runtime_factory(request)
            except OcrAdapterError:
                raise
            except Exception as exc:
                raise _runtime_error(exc) from None
            if runtime is None:
                raise OcrAdapterError(
                    "OCR_BACKEND_UNAVAILABLE",
                    "MindOCR runtime factory returned no runtime",
                )
            self._runtime = runtime
            self._runtime_key = key
        return self._runtime

    def recognize(self, image: OcrImage, request: OcrRequest) -> list[OcrCandidate]:
        validate_ocr_call(self, image, request)
        runtime = self._runtime_for_request(request)
        try:
            raw = invoke_runtime(
                runtime,
                image.pixels,
                request,
                backend="MindOCR",
                lang_value=_backend_language(request.lang),
                device_value=_backend_device(request.device),
            )
        except OcrAdapterError:
            raise
        except Exception as exc:
            raise _runtime_error(exc) from None
        try:
            return merge_candidate_lines(
                mapping_candidates(
                    extract_records(raw, "MindOCR"),
                    image,
                    request,
                    "MindOCR",
                )
            )
        except OcrAdapterError:
            raise
        except Exception:
            raise OcrAdapterError(
                "OCR_EXECUTION_FAILED", "MindOCR returned an invalid result"
            ) from None


__all__ = ["MindOcrAdapter"]
