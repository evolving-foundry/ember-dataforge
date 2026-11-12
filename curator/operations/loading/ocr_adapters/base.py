"""C-M3a OCR adapter contract.

Adapters return candidate facts only. C-IR construction, provenance, block ids
and reading order belong to the common mapping layer.
"""
from __future__ import annotations

import math
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, FrozenSet, Mapping, Optional, Sequence, Tuple

import numpy as np

OCR_ERROR_CODES = frozenset(
    {
        "OCR_BACKEND_UNAVAILABLE",
        "OCR_MODEL_UNAVAILABLE",
        "OCR_EXECUTION_FAILED",
    }
)
OCR_LANGUAGES = frozenset({"zh", "en", "mixed"})
OCR_DEVICES = frozenset({"npu", "cuda", "cpu"})
OCR_BLOCK_TYPES = frozenset(
    {"paragraph", "table", "figure", "header", "footer", "list"}
)


def _single_line(message: str) -> str:
    return " ".join(str(message).replace("\x00", "").splitlines()).strip()


class OcrAdapterError(Exception):
    """Stable adapter error; messages must not contain source paths."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        if code not in OCR_ERROR_CODES:
            raise ValueError(f"unsupported OCR adapter error code: {code}")
        clean = _single_line(message)
        if not clean:
            raise ValueError("OCR adapter error message must be non-empty")
        super().__init__(clean)
        self.code = code
        self.message = clean
        self.retryable = bool(retryable)
        self.stage = "ocr"


def _validate_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must contain four numbers")
    values = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox values must be finite")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("bbox must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return values


def _validate_confidence(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("confidence must be a number or null")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("confidence must be in [0, 1]")
    return result


def _normalized_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("text must be a string")
    normalized = unicodedata.normalize("NFC", value).replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").strip()
    if value != normalized:
        raise ValueError("text must already be normalized")
    return normalized


@dataclass(frozen=True)
class OcrCapabilities:
    languages: FrozenSet[str]
    layout: bool
    table_grid: bool
    confidence: bool

    def __post_init__(self) -> None:
        languages = frozenset(self.languages)
        if not languages or not languages.issubset(OCR_LANGUAGES):
            raise ValueError("languages must be a non-empty subset of zh/en/mixed")
        object.__setattr__(self, "languages", languages)
        for name in ("layout", "table_grid", "confidence"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class OcrImage:
    pixels: np.ndarray
    width: int
    height: int
    page: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or isinstance(self.height, bool):
            raise ValueError("image dimensions must be positive integers")
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValueError("image dimensions must be positive integers")
        if self.width <= 0 or self.height <= 0 or self.page != 1:
            raise ValueError("image width/height must be positive and page must be 1")
        if not isinstance(self.pixels, np.ndarray):
            raise ValueError("pixels must be a numpy array")
        if self.pixels.shape != (self.height, self.width, 3):
            raise ValueError("pixels must have shape (height, width, 3)")
        if self.pixels.dtype != np.uint8:
            raise ValueError("pixels must be uint8 RGB")


@dataclass(frozen=True)
class OcrRequest:
    lang: str
    device: str
    layout_analysis: bool
    model_paths: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lang not in OCR_LANGUAGES:
            raise ValueError("lang must be normalized to zh/en/mixed")
        if self.device not in OCR_DEVICES:
            raise ValueError("device must be one of npu/cuda/cpu")
        if not isinstance(self.layout_analysis, bool):
            raise ValueError("layout_analysis must be boolean")
        if not isinstance(self.model_paths, Mapping):
            raise ValueError("model_paths must be a mapping")
        unknown = set(self.model_paths) - {"det", "rec", "layout", "rec_dict"}
        if unknown:
            raise ValueError(f"unknown model_paths keys: {sorted(unknown)}")
        paths = {}
        for key, value in self.model_paths.items():
            if not isinstance(value, str) or not value:
                raise ValueError("model_paths values must be non-empty strings")
            paths[key] = value
        object.__setattr__(self, "model_paths", MappingProxyType(paths))


@dataclass(frozen=True)
class OcrCell:
    row: int
    col: int
    row_span: int
    col_span: int
    text: str
    bbox: Optional[Tuple[float, float, float, float]]
    confidence: Optional[float]

    def __post_init__(self) -> None:
        for name in ("row", "col", "row_span", "col_span"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.row < 0 or self.col < 0 or self.row_span < 1 or self.col_span < 1:
            raise ValueError("cell coordinates/spans are out of range")
        object.__setattr__(self, "text", _normalized_text(self.text))
        if self.bbox is not None:
            object.__setattr__(self, "bbox", _validate_bbox(self.bbox))
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))


@dataclass(frozen=True)
class OcrTable:
    n_rows: int
    n_cols: int
    cells: Tuple[OcrCell, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_rows, bool)
            or isinstance(self.n_cols, bool)
            or not isinstance(self.n_rows, int)
            or not isinstance(self.n_cols, int)
            or self.n_rows < 1
            or self.n_cols < 1
        ):
            raise ValueError("table dimensions must be positive integers")
        cells = tuple(sorted(self.cells, key=lambda item: (item.row, item.col)))
        expected = {
            (row, col) for row in range(self.n_rows) for col in range(self.n_cols)
        }
        coords = [(cell.row, cell.col) for cell in cells]
        if len(cells) != self.n_rows * self.n_cols or set(coords) != expected:
            raise ValueError("table cells must cover the complete grid exactly once")
        by_coord = dict(zip(coords, cells))
        covered = set()
        for cell in cells:
            if cell.row + cell.row_span > self.n_rows:
                raise ValueError("cell row_span exceeds grid")
            if cell.col + cell.col_span > self.n_cols:
                raise ValueError("cell col_span exceeds grid")
            for row in range(cell.row, cell.row + cell.row_span):
                for col in range(cell.col, cell.col + cell.col_span):
                    if (row, col) == (cell.row, cell.col):
                        continue
                    if (row, col) in covered:
                        raise ValueError("merged cell regions overlap")
                    child = by_coord[(row, col)]
                    if (
                        child.row_span != 1
                        or child.col_span != 1
                        or child.text != ""
                        or child.bbox is not None
                        or child.confidence is not None
                    ):
                        raise ValueError("covered cell must be an empty 1x1 cell")
                    covered.add((row, col))
        object.__setattr__(self, "cells", cells)


@dataclass(frozen=True)
class OcrCandidate:
    page: int
    order: int
    block_type: str
    text: str
    bbox: Tuple[float, float, float, float]
    confidence: Optional[float]
    table: Optional[OcrTable] = None

    def __post_init__(self) -> None:
        if self.page != 1:
            raise ValueError("OCR candidate page must be 1")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("order must be a non-negative integer")
        if self.block_type not in OCR_BLOCK_TYPES:
            raise ValueError(f"unsupported block_type: {self.block_type}")
        object.__setattr__(self, "text", _normalized_text(self.text))
        object.__setattr__(self, "bbox", _validate_bbox(self.bbox))
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))
        if self.table is not None and self.block_type != "table":
            raise ValueError("table is only valid for table candidates")
        if self.table is None and self.block_type == "table" and self.text == "":
            raise ValueError("region table candidate requires text")


class OcrAdapter(ABC):
    name: ClassVar[str]
    supported_devices: ClassVar[FrozenSet[str]]
    capabilities: ClassVar[OcrCapabilities]

    @property
    @abstractmethod
    def version(self) -> str:
        """Runtime backend package version, or 'unknown'."""

    @property
    @abstractmethod
    def default_model_id(self) -> Optional[str]:
        """Stable default model id, or None when it cannot be verified."""

    @classmethod
    @abstractmethod
    def dependency_installed(cls) -> bool:
        """Whether the backend package is importable."""

    @abstractmethod
    def recognize(self, image: OcrImage, request: OcrRequest) -> list[OcrCandidate]:
        """Recognize one EXIF-corrected RGB image and return candidate facts."""

    def close(self) -> None:
        """Release any runtime owned by this adapter."""
        runtime = getattr(self, "_runtime", None)
        close = getattr(runtime, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        if hasattr(self, "_runtime"):
            self._runtime = None
        if hasattr(self, "_runtime_key"):
            self._runtime_key = None


def validate_candidates(
    candidates: Sequence[OcrCandidate],
    request: OcrRequest,
    capabilities: OcrCapabilities,
) -> None:
    if not isinstance(candidates, (list, tuple)):
        raise OcrAdapterError("OCR_EXECUTION_FAILED", "adapter must return a candidate list")
    if request.lang not in capabilities.languages:
        raise OcrAdapterError("OCR_EXECUTION_FAILED", "adapter received an unsupported language")
    if request.layout_analysis and not capabilities.layout:
        raise OcrAdapterError(
            "OCR_BACKEND_UNAVAILABLE", "adapter does not support layout analysis"
        )
    for candidate in candidates:
        if not isinstance(candidate, OcrCandidate):
            raise OcrAdapterError(
                "OCR_EXECUTION_FAILED", "adapter returned an invalid candidate type"
            )
        if candidate.page != 1:
            raise OcrAdapterError("OCR_EXECUTION_FAILED", "candidate page must be 1")
        if not request.layout_analysis and candidate.block_type != "paragraph":
            raise OcrAdapterError(
                "OCR_EXECUTION_FAILED",
                "adapter returned non-paragraph blocks with layout analysis disabled",
            )
        if candidate.table is not None:
            if not capabilities.table_grid:
                raise OcrAdapterError(
                    "OCR_EXECUTION_FAILED", "adapter returned a table grid without capability"
                )
            if candidate.block_type != "table":
                raise OcrAdapterError(
                    "OCR_EXECUTION_FAILED", "table grid must belong to a table block"
                )
        if not capabilities.confidence:
            if candidate.confidence is not None:
                raise OcrAdapterError(
                    "OCR_EXECUTION_FAILED", "adapter returned confidence without capability"
                )
            if candidate.table is not None and any(
                cell.confidence is not None for cell in candidate.table.cells
            ):
                raise OcrAdapterError(
                    "OCR_EXECUTION_FAILED", "adapter returned table confidence without capability"
                )


def _is_cjk(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF
        or 0xF900 <= code <= 0xFAFF
    )


def _join_line_fragments(fragments: Sequence[str]) -> str:
    text = ""
    for fragment in fragments:
        if not text:
            text = fragment
            continue
        previous = text[-1]
        following = fragment[0]
        no_space = (
            following in ".,;:!?%)]}，。；：！？、-—–"
            or previous in "([{‘“"
            or (_is_cjk(previous) and _is_cjk(following))
        )
        text += ("" if no_space else " ") + fragment
    return text


def _line_group_bbox(
    group: Sequence[OcrCandidate],
) -> Tuple[float, float, float, float]:
    return (
        min(candidate.bbox[0] for candidate in group),
        min(candidate.bbox[1] for candidate in group),
        max(candidate.bbox[2] for candidate in group),
        max(candidate.bbox[3] for candidate in group),
    )


def _same_text_line(
    group: Sequence[OcrCandidate], candidate: OcrCandidate
) -> bool:
    group_x0, group_y0, group_x1, group_y1 = _line_group_bbox(group)
    x0, y0, x1, y1 = candidate.bbox
    group_height = group_y1 - group_y0
    candidate_height = y1 - y0
    line_height = max(group_height, candidate_height)
    minimum_height = min(group_height, candidate_height)
    vertical_overlap = max(0.0, min(group_y1, y1) - max(group_y0, y0))
    center_delta = abs(
        (group_y0 + group_y1) / 2.0 - (y0 + y1) / 2.0
    )
    aligned = (
        vertical_overlap / minimum_height >= 0.5
        or center_delta <= 0.4 * line_height
    )
    horizontal_gap = max(group_x0 - x1, x0 - group_x1, 0.0)
    return aligned and horizontal_gap <= max(0.01, 2.5 * line_height)


def merge_candidate_lines(
    candidates: Sequence[OcrCandidate],
) -> list[OcrCandidate]:
    """Merge word-level candidates into line-level facts in reading order.

    MindOCR returns one detection box per word.  The ranking manifest and the
    common C-IR mapping consume line-level facts, so word boxes must be joined
    before text and layout metrics are calculated.  Table candidates are kept
    as standalone facts.
    """
    groups: list[list[OcrCandidate]] = []
    for candidate in sorted(
        candidates, key=lambda item: (item.bbox[1], item.bbox[0], item.order)
    ):
        if not isinstance(candidate, OcrCandidate):
            raise ValueError("candidate must be an OcrCandidate")
        merged = False
        if candidate.table is None:
            matching = [
                index
                for index, group in enumerate(groups)
                if (
                    group[0].table is None
                    and group[0].block_type == candidate.block_type
                    and _same_text_line(group, candidate)
                )
            ]
            if matching:
                group = groups[matching[0]]
                group.append(candidate)
                for index in reversed(matching[1:]):
                    group.extend(groups.pop(index))
                merged = True
        if not merged:
            groups.append([candidate])

    result: list[OcrCandidate] = []
    for group in groups:
        ordered = sorted(group, key=lambda item: (item.bbox[0], item.order))
        confidences = [
            candidate.confidence
            for candidate in ordered
            if candidate.confidence is not None
        ]
        result.append(
            OcrCandidate(
                page=1,
                order=len(result),
                block_type=ordered[0].block_type,
                text=_join_line_fragments(
                    [candidate.text for candidate in ordered]
                ),
                bbox=_line_group_bbox(ordered),
                confidence=min(confidences) if confidences else None,
                table=ordered[0].table if len(ordered) == 1 else None,
            )
        )
    return result
