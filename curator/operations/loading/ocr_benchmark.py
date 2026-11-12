"""Offline OCR backend benchmark and ``ocr-ranking-v1`` writer.

The evaluator consumes a trusted local manifest.  It deliberately does not
invent a bundled ranking: callers run it with an annotated evaluation set and
an installed backend, then publish the measured YAML separately.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from .document_common import DocumentParseError
from .image_ocr import decode_image_file
from .ocr_adapters import (
    OcrAdapter,
    OcrAdapterError,
    OcrCandidate,
    OcrImage,
    OcrRequest,
    validate_candidates,
)
from .ocr_adapters.base import OCR_BLOCK_TYPES, merge_candidate_lines
from .ocr_ranking import (
    MAX_RANKING_FILE_BYTES,
    RANKING_PRIMARY_METRIC,
    RANKING_SCHEMA_VERSION,
    parse_ranking_document,
)
from .ocr_routing import DeviceProbe, select_ocr_routes

MAX_MANIFEST_BYTES = 10 * 1024 * 1024
MAX_CASES = 10_000
MAX_TEXT_CHARS_PER_CASE = 1_000_000
_MANIFEST_KEYS = frozenset({"imagePath", "texts", "boxes", "blockTypes"})
_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_IOU_THRESHOLD = 0.5


class BenchmarkError(ValueError):
    """A deterministic benchmark input or execution error."""


@dataclass(frozen=True)
class BenchmarkCase:
    image_path: str
    texts: Tuple[str, ...]
    boxes: Tuple[Tuple[float, float, float, float], ...]
    block_types: Tuple[str, ...]


@dataclass
class _MetricsAccumulator:
    edit_distance: int = 0
    reference_chars: int = 0
    matched_blocks: int = 0
    reference_blocks: int = 0
    true_positive_boxes: int = 0
    predicted_boxes: int = 0
    reference_boxes: int = 0


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        raise BenchmarkError("manifest texts must be strings")
    normalized = unicodedata.normalize("NFC", value).replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise BenchmarkError("manifest texts must not be empty")
    return normalized


def _comparison_text(value: str) -> str:
    """Normalize formatting that must not dominate cross-vocabulary ranking.

    OCR adapters have different alphabet sizes.  MindOCR's English CRNN, for
    example, can only emit lowercase letters and digits.  Ranking compares the
    recognized content after Unicode/case/whitespace/punctuation normalization;
    the OCR output itself remains unchanged.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold()
    compact = "".join(character for character in normalized if character.isalnum())
    return compact or normalized.strip()


def _box(value: Any) -> Tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise BenchmarkError("manifest boxes must be four-number arrays")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise BenchmarkError("manifest boxes must be four-number arrays")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise BenchmarkError("manifest boxes must be finite")
    x0, y0, x1, y1 = result
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise BenchmarkError("manifest boxes must be normalized and non-empty")
    return result  # type: ignore[return-value]


def load_benchmark_manifest(
    manifest_path: str | Path,
    *,
    allowed_roots: Optional[Sequence[str]] = None,
) -> list[BenchmarkCase]:
    path = Path(manifest_path).expanduser()
    try:
        info = path.stat()
    except OSError as exc:
        raise BenchmarkError("benchmark manifest is not readable") from exc
    if not path.is_file():
        raise BenchmarkError("benchmark manifest must be a regular file")
    if info.st_size > MAX_MANIFEST_BYTES:
        raise BenchmarkError("benchmark manifest exceeds 10 MiB")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BenchmarkError("benchmark manifest is not readable UTF-8") from exc
    lines = text.splitlines()
    if len(lines) > MAX_CASES:
        raise BenchmarkError("benchmark manifest exceeds 10000 cases")
    roots = tuple(
        os.path.realpath(os.path.abspath(root)) for root in (allowed_roots or ())
    )
    cases: list[BenchmarkCase] = []
    seen_paths = set()
    total_chars = 0
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(
                f"benchmark manifest line {line_no} is not valid JSON"
            ) from exc
        if not isinstance(raw, dict) or set(raw) != _MANIFEST_KEYS:
            raise BenchmarkError(
                f"benchmark manifest line {line_no} must contain exactly "
                "imagePath, texts, boxes and blockTypes"
            )
        image_path = raw["imagePath"]
        if not isinstance(image_path, str) or not image_path:
            raise BenchmarkError("imagePath must be a non-empty string")
        if _URL_RE.match(image_path) or image_path.startswith("file:"):
            raise BenchmarkError("imagePath must be a local path")
        resolved = Path(image_path)
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        try:
            resolved = resolved.resolve(strict=False)
        except OSError as exc:
            raise BenchmarkError("imagePath is not readable") from exc
        if not resolved.is_file():
            raise BenchmarkError("imagePath must point to a regular file")
        if roots:
            real = os.path.realpath(str(resolved))
            inside_root = False
            for root in roots:
                try:
                    if os.path.commonpath([real, root]) == root:
                        inside_root = True
                        break
                except ValueError:
                    continue
            if not inside_root:
                raise BenchmarkError("imagePath is outside allowed roots")
            resolved = Path(real)
        if str(resolved) in seen_paths:
            raise BenchmarkError("benchmark manifest contains a duplicate imagePath")
        seen_paths.add(str(resolved))

        texts_raw = raw["texts"]
        boxes_raw = raw["boxes"]
        types_raw = raw["blockTypes"]
        if not isinstance(texts_raw, list) or not texts_raw:
            raise BenchmarkError("texts must be a non-empty array")
        if not isinstance(boxes_raw, list) or len(boxes_raw) != len(texts_raw):
            raise BenchmarkError("boxes must have one entry per text")
        if not isinstance(types_raw, list) or len(types_raw) != len(texts_raw):
            raise BenchmarkError("blockTypes must have one entry per text")
        texts = tuple(_normalize_text(value) for value in texts_raw)
        boxes = tuple(_box(value) for value in boxes_raw)
        block_types: list[str] = []
        for value in types_raw:
            if not isinstance(value, str) or value not in OCR_BLOCK_TYPES:
                raise BenchmarkError("blockTypes contains an unsupported value")
            block_types.append(value)
        case_chars = sum(len(value) for value in texts)
        total_chars += case_chars
        if case_chars > MAX_TEXT_CHARS_PER_CASE:
            raise BenchmarkError("case text exceeds 1000000 characters")
        cases.append(
            BenchmarkCase(
                image_path=str(resolved),
                texts=texts,
                boxes=boxes,
                block_types=tuple(block_types),
            )
        )
    if not cases:
        raise BenchmarkError("benchmark manifest contains no cases")
    if total_chars > MAX_CASES * MAX_TEXT_CHARS_PER_CASE:
        raise BenchmarkError("benchmark manifest text exceeds safety limit")
    return cases


def _levenshtein_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for col, hyp_char in enumerate(hypothesis, start=1):
            insert = current[col - 1] + 1
            delete = previous[col] + 1
            replace = previous[col - 1] + (ref_char != hyp_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _candidate_texts(candidates: Sequence[OcrCandidate]) -> list[str]:
    result: list[str] = []
    for candidate in candidates:
        if candidate.table is not None:
            result.extend(
                cell.text for cell in candidate.table.cells if cell.text
            )
        elif candidate.text:
            result.append(candidate.text)
    return result


def _iou(
    first: Sequence[float], second: Sequence[float]
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x0 >= x1 or y0 >= y1:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _match_layout(
    candidates: Sequence[OcrCandidate], case: BenchmarkCase
) -> int:
    pairs = []
    for candidate_index, candidate in enumerate(candidates):
        for reference_index, (box, block_type) in enumerate(
            zip(case.boxes, case.block_types)
        ):
            if candidate.block_type != block_type:
                continue
            overlap = _iou(candidate.bbox, box)
            if overlap >= _IOU_THRESHOLD:
                pairs.append((overlap, candidate_index, reference_index))
    matched_candidates = set()
    matched_references = set()
    for _, candidate_index, reference_index in sorted(
        pairs, key=lambda item: (-item[0], item[1], item[2])
    ):
        if candidate_index in matched_candidates or reference_index in matched_references:
            continue
        matched_candidates.add(candidate_index)
        matched_references.add(reference_index)
    return len(matched_candidates)


def _accumulate_case(
    accumulator: _MetricsAccumulator,
    candidates: Sequence[OcrCandidate],
    case: BenchmarkCase,
) -> None:
    predicted_texts = [
        normalized
        for normalized in (
            _comparison_text(text) for text in _candidate_texts(candidates)
        )
        if normalized
    ]
    reference_texts = [_comparison_text(text) for text in case.texts]
    reference_text = "".join(reference_texts)
    predicted_text = "".join(predicted_texts)
    accumulator.edit_distance += _levenshtein_distance(
        reference_text, predicted_text
    )
    accumulator.reference_chars += len(reference_text)

    remaining = Counter(predicted_texts)
    for text in reference_texts:
        if remaining[text] > 0:
            accumulator.matched_blocks += 1
            remaining[text] -= 1
    accumulator.reference_blocks += len(reference_texts)
    accumulator.predicted_boxes += len(candidates)
    accumulator.reference_boxes += len(case.boxes)
    accumulator.true_positive_boxes += _match_layout(candidates, case)


def _metrics(accumulator: _MetricsAccumulator) -> Dict[str, float]:
    cer = accumulator.edit_distance / max(1, accumulator.reference_chars)
    line_accuracy = accumulator.matched_blocks / max(1, accumulator.reference_blocks)
    precision = accumulator.true_positive_boxes / max(1, accumulator.predicted_boxes)
    recall = accumulator.true_positive_boxes / max(1, accumulator.reference_boxes)
    layout_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "cer": round(cer, 8),
        "line_accuracy": round(line_accuracy, 8),
        "layout_f1": round(layout_f1, 8),
    }


def run_benchmark(
    manifest_path: str | Path,
    *,
    backend: str,
    device: str,
    lang: str,
    layout_analysis: bool,
    ranking_version: str,
    evaluation_set_version: str,
    model_paths: Optional[Mapping[str, str]] = None,
    allow_model_download: bool = False,
    max_image_megapixels: float = 100.0,
    max_file_mb: float = 100.0,
    allowed_roots: Optional[Sequence[str]] = None,
    backend_version: Optional[str] = None,
    model_id: Optional[str] = None,
    adapter_factories: Optional[Mapping[str, Callable[[], OcrAdapter]]] = None,
    device_probe: Optional[DeviceProbe] = None,
) -> Dict[str, Any]:
    if backend not in {"mindocr", "paddleocr"}:
        raise BenchmarkError("benchmark backend must be mindocr or paddleocr")
    if not isinstance(ranking_version, str) or not ranking_version:
        raise BenchmarkError("ranking_version must be a non-empty string")
    if not isinstance(evaluation_set_version, str) or not evaluation_set_version:
        raise BenchmarkError("evaluation_set_version must be a non-empty string")
    if backend_version is not None and (
        not isinstance(backend_version, str) or not backend_version
    ):
        raise BenchmarkError("backend_version must be a non-empty string")
    if model_id is not None and (not isinstance(model_id, str) or not model_id):
        raise BenchmarkError("model_id must be a non-empty string")
    if (
        isinstance(max_image_megapixels, bool)
        or not isinstance(max_image_megapixels, (int, float))
        or not math.isfinite(max_image_megapixels)
        or max_image_megapixels <= 0
    ):
        raise BenchmarkError("max_image_megapixels must be positive")
    if (
        isinstance(max_file_mb, bool)
        or not isinstance(max_file_mb, (int, float))
        or not math.isfinite(max_file_mb)
        or max_file_mb <= 0
    ):
        raise BenchmarkError("max_file_mb must be positive")
    cases = load_benchmark_manifest(manifest_path, allowed_roots=allowed_roots)
    max_bytes = int(max_file_mb * 1024 * 1024)
    for case in cases:
        if os.path.getsize(case.image_path) > max_bytes:
            raise BenchmarkError("benchmark image exceeds max_file_mb")
    routing_kwargs: Dict[str, Any] = {
        "backend": backend,
        "device": device,
        "lang": lang,
        "layout_analysis": layout_analysis,
        "model_paths": dict(model_paths or {}),
        "allow_model_download": allow_model_download,
    }
    if adapter_factories is not None:
        routing_kwargs["adapter_factories"] = adapter_factories
    if device_probe is not None:
        routing_kwargs["device_probe"] = device_probe
    routes = select_ocr_routes(**routing_kwargs)
    if len(routes) != 1:
        for candidate in routes:
            candidate.adapter.close()
        raise BenchmarkError("benchmark requires exactly one OCR route")
    route = routes[0]
    if route.backend != backend:
        route.adapter.close()
        raise BenchmarkError("OCR adapter name does not match the requested backend")
    resolved_version = backend_version or route.backend_version
    resolved_model = model_id or route.model_id
    if not resolved_version or resolved_version == "unknown":
        route.adapter.close()
        raise BenchmarkError("backend version is unknown; pass --backend-version")
    if not resolved_model:
        route.adapter.close()
        raise BenchmarkError("model id is unknown; pass --model-id")

    request = OcrRequest(
        lang=lang,
        device=route.device,
        layout_analysis=layout_analysis,
        model_paths=dict(model_paths or {}),
    )
    accumulator = _MetricsAccumulator()
    started = time.perf_counter()
    try:
        for case in cases:
            pixels, width, height, _ = decode_image_file(
                case.image_path, max_image_megapixels
            )
            image = OcrImage(pixels=pixels, width=width, height=height)
            try:
                candidates = route.adapter.recognize(image, request)
                candidates = merge_candidate_lines(candidates)
                validate_candidates(
                    candidates, request, route.adapter.capabilities
                )
            except OcrAdapterError as exc:
                raise BenchmarkError(
                    f"OCR backend failed with {exc.code}"
                ) from exc
            except Exception as exc:
                raise BenchmarkError("OCR backend returned an invalid result") from exc
            _accumulate_case(accumulator, candidates, case)
    except DocumentParseError as exc:
        raise BenchmarkError(f"image decoding failed with {exc.code}") from exc
    finally:
        route.adapter.close()
    elapsed = max(time.perf_counter() - started, 1e-9)
    metrics = _metrics(accumulator)
    payload: Dict[str, Any] = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "ranking_version": ranking_version,
        "evaluation_set_version": evaluation_set_version,
        "primary_metric": RANKING_PRIMARY_METRIC,
        "rows": [
            {
                "backend": route.backend,
                "backend_version": resolved_version,
                "device": route.device,
                "model_id": resolved_model,
                **metrics,
                "throughput_pages_s": round(len(cases) / elapsed, 8),
                "eligible": True,
            }
        ],
    }
    parse_ranking_document(payload)
    return payload


def merge_ranking_tables(
    existing: Mapping[str, Any], update: Mapping[str, Any]
) -> Dict[str, Any]:
    base = parse_ranking_document(dict(existing))
    current = parse_ranking_document(dict(update))
    if base.ranking_version != current.ranking_version:
        raise BenchmarkError("ranking_version does not match the existing table")
    if base.evaluation_set_version != current.evaluation_set_version:
        raise BenchmarkError("evaluation_set_version does not match the existing table")
    rows = {
        (row.backend, row.backend_version, row.device, row.model_id): row.__dict__
        for row in base.rows
    }
    for row in current.rows:
        rows[(row.backend, row.backend_version, row.device, row.model_id)] = row.__dict__
    merged = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "ranking_version": base.ranking_version,
        "evaluation_set_version": base.evaluation_set_version,
        "primary_metric": RANKING_PRIMARY_METRIC,
        "rows": list(rows.values()),
    }
    parse_ranking_document(merged)
    return merged


def write_ranking_table(
    payload: Mapping[str, Any], output_path: str | Path, *, merge: bool = False
) -> Path:
    table = parse_ranking_document(dict(payload))
    target = Path(output_path).expanduser()
    if merge and target.exists():
        try:
            if target.stat().st_size > MAX_RANKING_FILE_BYTES:
                raise BenchmarkError("existing ranking file exceeds 10 MiB")
            existing = yaml.safe_load(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise BenchmarkError("existing ranking file could not be read") from exc
        payload = merge_ranking_tables(existing, dict(payload))
        table = parse_ranking_document(dict(payload))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BenchmarkError("ranking directory could not be created") from exc
    serialized = yaml.safe_dump(
        {
            "schema_version": RANKING_SCHEMA_VERSION,
            "ranking_version": table.ranking_version,
            "evaluation_set_version": table.evaluation_set_version,
            "primary_metric": RANKING_PRIMARY_METRIC,
            "rows": [row.__dict__ for row in table.rows],
        },
        allow_unicode=True,
        sort_keys=False,
    )
    fd = -1
    temp_name = ""
    try:
        fd, temp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            fd = -1
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        temp_name = ""
    except OSError as exc:
        raise BenchmarkError("ranking file could not be written") from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    return target


def _model_paths(values: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise BenchmarkError("--model-path must use key=path")
        key, path = value.split("=", 1)
        if key not in {"det", "rec", "layout"} or not path:
            raise BenchmarkError("--model-path key must be det, rec or layout")
        result[key] = path
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark an OCR backend and write ocr-ranking-v1 YAML")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", required=True, choices=("mindocr", "paddleocr"))
    parser.add_argument("--device", default="auto", choices=("auto", "npu", "cuda", "cpu"))
    parser.add_argument("--lang", default="zh", choices=("zh", "en", "mixed"))
    parser.add_argument("--no-layout-analysis", action="store_true")
    parser.add_argument("--ranking-version", required=True)
    parser.add_argument("--evaluation-set-version", required=True)
    parser.add_argument("--model-path", action="append", default=[])
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-image-megapixels", type=float, default=100.0)
    parser.add_argument("--max-file-mb", type=float, default=100.0)
    parser.add_argument("--allowed-root", action="append", default=[])
    parser.add_argument("--backend-version")
    parser.add_argument("--model-id")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge this row into an existing output ranking table",
    )
    args = parser.parse_args(argv)
    try:
        payload = run_benchmark(
            args.manifest,
            backend=args.backend,
            device=args.device,
            lang=args.lang,
            layout_analysis=not args.no_layout_analysis,
            ranking_version=args.ranking_version,
            evaluation_set_version=args.evaluation_set_version,
            model_paths=_model_paths(args.model_path),
            allow_model_download=args.allow_model_download,
            max_image_megapixels=args.max_image_megapixels,
            max_file_mb=args.max_file_mb,
            allowed_roots=args.allowed_root or None,
            backend_version=args.backend_version,
            model_id=args.model_id,
        )
        write_ranking_table(payload, args.output, merge=args.merge)
    except (BenchmarkError, DocumentParseError) as exc:
        print(f"ocr benchmark failed: {exc}", file=sys.stderr)
        return 1
    summary = dict(payload["rows"][0])
    summary["ranking_version"] = payload["ranking_version"]
    summary["evaluation_set_version"] = payload["evaluation_set_version"]
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
