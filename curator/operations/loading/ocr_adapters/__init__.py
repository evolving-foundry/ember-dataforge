"""OCR adapter contracts and backends (C-M3a)."""

from .base import (
    OcrAdapter,
    OcrAdapterError,
    OcrCandidate,
    OcrCapabilities,
    OcrCell,
    OcrImage,
    OcrRequest,
    OcrTable,
    merge_candidate_lines,
    validate_candidates,
)
from .mindocr import MindOcrAdapter
from .paddleocr import PaddleOcrAdapter

__all__ = [
    "MindOcrAdapter",
    "PaddleOcrAdapter",
    "OcrAdapter",
    "OcrAdapterError",
    "OcrCandidate",
    "OcrCapabilities",
    "OcrCell",
    "OcrImage",
    "OcrRequest",
    "OcrTable",
    "merge_candidate_lines",
    "validate_candidates",
]
