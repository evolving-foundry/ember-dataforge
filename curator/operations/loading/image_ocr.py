"""Image OCR orchestration, decoding and C-IR-2 mapping."""
from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .document_common import (
    DocumentParseError,
    REQUEST_LEVEL_CODES,
    detect_language,
    package_version,
    validate_output_key,
)
from .document_io import (
    InputItem,
    Snapshot,
    canonicalize_roots,
    default_roots,
    discover_inputs,
    snapshot_file,
)
from .document_parsing import row_payload
from .ocr_adapters import (
    OcrAdapterError,
    OcrCandidate,
    OcrImage,
    OcrRequest,
    validate_candidates,
)
from .ocr_routing import (
    OcrRoute,
    default_adapter_factories,
    more_specific_reason,
    select_ocr_routes,
)
from .parsed_document import (
    READING_ORDER_VERSION,
    Block,
    Cell,
    ErrorInfo,
    ParsedDocument,
    Provenance,
    Source,
    Table,
    make_error_row,
    make_success_row,
)
from .process_limits import apply_child_resource_limits
from .subprocess_runner import (
    read_result,
    result_error,
    run_json_child,
    start_child,
    stop_child,
    wait_for_result,
    write_result,
)

IMAGE_EXTENSIONS = {".png": "png", ".jpg": "jpg", ".jpeg": "jpg"}
IMAGE_FORMATS = {"png": frozenset({"PNG"}), "jpg": frozenset({"JPEG"})}
DEFAULT_MAX_IMAGE_MEGAPIXELS = 100.0
DEFAULT_OCR_TIMEOUT_S = 300.0
DEFAULT_OCR_MEMORY_MB = 4096
MAX_TABLE_CELLS = 1_000_000
IMAGE_OCR_PARAM_KEYS = frozenset(
    {
        "data_paths",
        "output_key",
        "backend",
        "device",
        "lang",
        "layout_analysis",
        "max_file_mb",
        "max_image_megapixels",
        "parse_timeout_s",
        "max_memory_mb",
        "on_error",
        "allowed_roots",
        "backend_ranking_path",
        "model_paths",
        "allow_model_download",
    }
)
OP_IMAGE_OCR_PARAM_KEYS = IMAGE_OCR_PARAM_KEYS - {"data_paths"}
_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_USE_LOADER = object()


@dataclass(frozen=True)
class ImageOcrOptions:
    output_key: str = "parsed_doc"
    backend: str = "auto"
    device: str = "auto"
    lang: str = "zh"
    layout_analysis: bool = True
    max_file_mb: float = 100.0
    max_image_megapixels: float = DEFAULT_MAX_IMAGE_MEGAPIXELS
    parse_timeout_s: float = DEFAULT_OCR_TIMEOUT_S
    max_memory_mb: int = DEFAULT_OCR_MEMORY_MB
    on_error: str = "raise"
    allowed_roots: Tuple[str, ...] = ()
    backend_ranking_path: Optional[str] = None
    model_paths: Tuple[Tuple[str, str], ...] = ()
    allow_model_download: bool = False

    @property
    def model_path_map(self) -> Dict[str, str]:
        return dict(self.model_paths)


def _invalid(message: str) -> DocumentParseError:
    return DocumentParseError("INVALID_PARAMETER", message, stage="validate")


def _number(name: str, value: Any, *, minimum: float, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"{name} must be a number")
    if integer and not isinstance(value, int):
        raise _invalid(f"{name} must be an integer")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise _invalid(f"{name} must be finite and >= {minimum}")
    return result


def _validate_local_model_paths(raw: Any) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(raw, dict):
        raise _invalid("model_paths must be an object")
    unknown = set(raw) - {"det", "rec", "layout", "rec_dict"}
    if unknown:
        raise _invalid(f"model_paths has unknown keys: {sorted(unknown)}")
    result: List[Tuple[str, str]] = []
    for key in sorted(raw):
        value = raw[key]
        if not isinstance(value, str) or not value:
            raise _invalid("model_paths values must be non-empty strings")
        if value.startswith("file:") or _URL_RE.match(value):
            raise _invalid("model_paths values must be local paths")
        path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            resolved = path.resolve(strict=False)
        except OSError as exc:
            raise DocumentParseError(
                "OCR_MODEL_UNAVAILABLE", "model path is not readable", stage="validate"
            ) from exc
        if not resolved.exists() or not os.access(str(resolved), os.R_OK):
            raise DocumentParseError(
                "OCR_MODEL_UNAVAILABLE", "model path is missing or unreadable", stage="validate"
            )
        result.append((key, str(resolved)))
    return tuple(result)


def build_image_ocr_options(
    params: Dict[str, Any],
    *,
    allowed_keys: frozenset = IMAGE_OCR_PARAM_KEYS,
    require_allowed_roots: bool = False,
) -> ImageOcrOptions:
    if not isinstance(params, dict):
        raise _invalid("params must be an object")
    unknown = set(params) - allowed_keys
    if unknown:
        raise _invalid(f"unknown params: {sorted(unknown)}")
    output_key = params.get("output_key", "parsed_doc")
    try:
        validate_output_key(output_key)
    except ValueError as exc:
        raise _invalid(str(exc)) from exc
    backend = params.get("backend", "auto")
    if backend not in {"auto", "mindocr", "paddleocr"}:
        raise _invalid("backend must be auto, mindocr or paddleocr")
    device = params.get("device", "auto")
    if device not in {"auto", "npu", "cuda", "cpu"}:
        raise _invalid("device must be auto, npu, cuda or cpu")
    lang = params.get("lang", "zh")
    if lang == "ch":
        lang = "zh"
    if lang not in {"zh", "en", "mixed"}:
        raise _invalid("lang must be zh, en or mixed")
    layout_analysis = params.get("layout_analysis", True)
    if not isinstance(layout_analysis, bool):
        raise _invalid("layout_analysis must be boolean")
    max_file_mb = _number("max_file_mb", params.get("max_file_mb", 100.0), minimum=0.000001)
    max_image_megapixels = _number(
        "max_image_megapixels",
        params.get("max_image_megapixels", DEFAULT_MAX_IMAGE_MEGAPIXELS),
        minimum=0.000001,
    )
    timeout_s = _number(
        "parse_timeout_s", params.get("parse_timeout_s", DEFAULT_OCR_TIMEOUT_S), minimum=0.000001
    )
    max_memory_mb = int(
        _number(
            "max_memory_mb",
            params.get("max_memory_mb", DEFAULT_OCR_MEMORY_MB),
            minimum=512,
            integer=True,
        )
    )
    on_error = params.get("on_error", "raise")
    if on_error not in {"raise", "skip"}:
        raise _invalid("on_error must be raise or skip")
    allowed_roots = params.get("allowed_roots", [])
    if not isinstance(allowed_roots, list) or not all(
        isinstance(root, str) and root for root in allowed_roots
    ):
        raise _invalid("allowed_roots must be an array of non-empty strings")
    if require_allowed_roots and not allowed_roots:
        raise _invalid("allowed_roots must be non-empty")
    ranking_path = params.get("backend_ranking_path")
    if ranking_path is not None:
        if not isinstance(ranking_path, str):
            raise _invalid("backend_ranking_path must be a string")
        if not ranking_path:
            ranking_path = None
        elif ranking_path.startswith("file:") or _URL_RE.match(ranking_path):
            raise _invalid("backend_ranking_path must be a local path")
    model_paths = _validate_local_model_paths(params.get("model_paths", {}))
    if model_paths and backend == "auto":
        raise _invalid("model_paths requires an explicit backend")
    allow_model_download = params.get("allow_model_download", False)
    if not isinstance(allow_model_download, bool):
        raise _invalid("allow_model_download must be boolean")
    return ImageOcrOptions(
        output_key=output_key,
        backend=backend,
        device=device,
        lang=lang,
        layout_analysis=layout_analysis,
        max_file_mb=max_file_mb,
        max_image_megapixels=max_image_megapixels,
        parse_timeout_s=timeout_s,
        max_memory_mb=max_memory_mb,
        on_error=on_error,
        allowed_roots=tuple(allowed_roots),
        backend_ranking_path=ranking_path,
        model_paths=model_paths,
        allow_model_download=allow_model_download,
    )


def decode_image_file(path: str, max_image_megapixels: float) -> Tuple[Any, int, int, str]:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:
        raise DocumentParseError(
            "DEPENDENCY_MISSING", "Pillow is required for image OCR", stage="validate"
        ) from exc
    extension = os.path.splitext(path)[1].lower()
    media_type = IMAGE_EXTENSIONS.get(extension)
    if media_type is None:
        raise DocumentParseError(
            "UNSUPPORTED_FORMAT", "unsupported image extension", stage="validate"
        )
    max_pixels = int(max_image_megapixels * 1_000_000)
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        try:
            with Image.open(path) as probe:
                probe.verify()
        except Image.DecompressionBombError as exc:
            raise DocumentParseError(
                "RESOURCE_LIMIT", "image exceeds max_image_megapixels", stage="parse", retryable=True
            ) from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise DocumentParseError(
                "CORRUPT_DOCUMENT", "image file is corrupt or unsupported", stage="parse"
            ) from exc
        try:
            with Image.open(path) as source:
                if source.format not in IMAGE_FORMATS[media_type]:
                    raise DocumentParseError(
                        "UNSUPPORTED_FORMAT", "image extension and signature disagree", stage="validate"
                    )
                width, height = source.size
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    raise DocumentParseError(
                        "RESOURCE_LIMIT",
                        "image exceeds max_image_megapixels",
                        stage="parse",
                        retryable=True,
                    )
                oriented = ImageOps.exif_transpose(source)
                if oriented is source:
                    oriented = source.copy()
                rgb = oriented.convert("RGB")
                rgb.load()
                width, height = rgb.size
                import numpy as np

                pixels = np.asarray(rgb, dtype=np.uint8).copy()
        except DocumentParseError:
            raise
        except Image.DecompressionBombError as exc:
            raise DocumentParseError(
                "RESOURCE_LIMIT", "image exceeds max_image_megapixels", stage="parse", retryable=True
            ) from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise DocumentParseError(
                "CORRUPT_DOCUMENT", "image file is corrupt or unsupported", stage="parse"
            ) from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    if pixels.shape != (height, width, 3):
        raise DocumentParseError(
            "CORRUPT_DOCUMENT", "decoded image has an unexpected shape", stage="parse"
        )
    return pixels, width, height, media_type


def _rounded_bbox(raw: Sequence[float]) -> List[float]:
    values = [round(float(value), 6) for value in raw]
    if not (0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1):
        raise DocumentParseError(
            "OCR_EXECUTION_FAILED", "OCR bounding box becomes invalid after rounding", stage="ocr"
        )
    return values


def _table_from_candidate(candidate: OcrCandidate) -> Tuple[Table, Optional[float]]:
    if candidate.table is None:
        return Table(structure="region", n_rows=None, n_cols=None, cells=[]), None
    if candidate.table.n_rows * candidate.table.n_cols > MAX_TABLE_CELLS:
        raise DocumentParseError(
            "TABLE_TOO_LARGE", "table cell count exceeds limit", stage="parse"
        )
    anchors: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for cell in candidate.table.cells:
        for row in range(cell.row, cell.row + cell.row_span):
            for col in range(cell.col, cell.col + cell.col_span):
                if (row, col) != (cell.row, cell.col):
                    anchors[(row, col)] = (cell.row, cell.col)
    cells: List[Cell] = []
    confidences: List[float] = []
    for cell in candidate.table.cells:
        anchor = anchors.get((cell.row, cell.col))
        if anchor is not None:
            cells.append(
                Cell(
                    row=cell.row,
                    col=cell.col,
                    row_span=1,
                    col_span=1,
                    text="",
                    cell_status="covered",
                    bbox=None,
                    ocr_confidence=None,
                    covered_by=[anchor[0], anchor[1]],
                )
            )
            continue
        cell_bbox = _rounded_bbox(cell.bbox) if cell.bbox is not None else None
        confidence = cell.confidence if cell.text else None
        if confidence is not None:
            confidences.append(float(confidence))
        cells.append(
            Cell(
                row=cell.row,
                col=cell.col,
                row_span=cell.row_span,
                col_span=cell.col_span,
                text=cell.text,
                cell_status="value" if cell.text else "empty",
                bbox=cell_bbox,
                ocr_confidence=confidence,
                covered_by=None,
            )
        )
    table_confidence = min(confidences) if confidences else None
    return (
        Table(
            structure="grid",
            n_rows=candidate.table.n_rows,
            n_cols=candidate.table.n_cols,
            cells=cells,
        ),
        table_confidence,
    )


def candidates_to_blocks(candidates: Sequence[OcrCandidate]) -> List[Block]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.bbox[1],
            candidate.bbox[0],
            candidate.order,
        ),
    )
    blocks: List[Block] = []
    for index, candidate in enumerate(ordered, start=1):
        block_bbox = _rounded_bbox(candidate.bbox)
        if candidate.block_type == "table":
            table, confidence = _table_from_candidate(candidate)
            text = "" if candidate.table is not None else candidate.text
        else:
            table = None
            text = candidate.text
            confidence = candidate.confidence if candidate.text else None
        blocks.append(
            Block(
                block_id=f"b-{index:06d}",
                type=candidate.block_type,
                page=1,
                sheet=None,
                bbox=block_bbox,
                text=text,
                table=table,
                kv_pair=None,
                style=None,
                ocr_confidence=confidence,
            )
        )
    return blocks


def _document_text(blocks: Sequence[Block]) -> str:
    parts: List[str] = []
    for block in blocks:
        if block.text:
            parts.append(block.text)
        if block.table is not None:
            parts.extend(cell.text for cell in block.table.cells if cell.text)
    return "\n".join(parts)


def _make_document(
    snapshot: Snapshot,
    media_type: str,
    candidates: Sequence[OcrCandidate],
    route: OcrRoute,
    routing_reason: str,
) -> ParsedDocument:
    blocks = candidates_to_blocks(candidates)
    provenance = Provenance(
        extractor=route.backend,
        extractor_version=route.backend_version,
        operation="image_ocr",
        op_version=package_version("data-forge", "0.2.0"),
        reading_order_version=READING_ORDER_VERSION,
        routing_reason=routing_reason,
        ranking_version=route.ranking_version,
        ranking_evaluation_set=route.ranking_evaluation_set,
        backend=route.backend,
        backend_version=route.backend_version,
        device=route.device,
        model_id=route.model_id,
    )
    return ParsedDocument(
        schema_version="parsed-doc-v2",
        doc_id=snapshot.doc_id,
        source=Source(
            media_type=media_type,
            sha256=snapshot.sha256,
            byte_size=snapshot.byte_size,
            page_count=1,
        ),
        language=detect_language(_document_text(blocks)),
        provenance=provenance,
        blocks=blocks,
    )


def _map_adapter_error(error: OcrAdapterError) -> DocumentParseError:
    stage = "validate" if error.code == "OCR_MODEL_UNAVAILABLE" else "ocr"
    return DocumentParseError(
        error.code, error.message, stage=stage, retryable=error.retryable
    )


def _recognize(route: OcrRoute, image: OcrImage, request: OcrRequest) -> List[OcrCandidate]:
    try:
        candidates = route.adapter.recognize(image, request)
        validate_candidates(candidates, request, route.adapter.capabilities)
        return candidates
    except OcrAdapterError:
        raise
    except Exception as exc:
        raise OcrAdapterError(
            "OCR_EXECUTION_FAILED", f"OCR backend failed ({type(exc).__name__})"
        ) from None


def _route_request(options: ImageOcrOptions, route: OcrRoute) -> OcrRequest:
    return OcrRequest(
        lang=options.lang,
        device=route.device,
        layout_analysis=options.layout_analysis,
        model_paths=options.model_path_map,
    )


def _select_routes(
    options: ImageOcrOptions,
    *,
    adapter_factories: Optional[Mapping[str, Callable[[], Any]]] = None,
    device_probe: Optional[Callable[[str], bool]] = None,
    ranking_override: Any = _USE_LOADER,
) -> List[OcrRoute]:
    routing_kwargs: Dict[str, Any] = {
        "backend": options.backend,
        "device": options.device,
        "lang": options.lang,
        "layout_analysis": options.layout_analysis,
        "model_paths": options.model_path_map,
        "allow_model_download": options.allow_model_download,
        "backend_ranking_path": options.backend_ranking_path,
    }
    if adapter_factories is not None:
        routing_kwargs["adapter_factories"] = adapter_factories
    if device_probe is not None:
        routing_kwargs["device_probe"] = device_probe
    if ranking_override is not _USE_LOADER:
        routing_kwargs["ranking"] = ranking_override
    return select_ocr_routes(**routing_kwargs)


def _close_routes(routes: Sequence[OcrRoute]) -> None:
    closed: set[int] = set()
    for route in routes:
        if id(route.adapter) in closed:
            continue
        closed.add(id(route.adapter))
        route.adapter.close()


def _parse_image_file_with_routes(
    path: str,
    options: ImageOcrOptions,
    routes: Sequence[OcrRoute],
    work_dir: Optional[str],
) -> ParsedDocument:
    snapshot = snapshot_file(path, options, work_dir)
    try:
        pixels, width, height, media_type = decode_image_file(
            snapshot.path, options.max_image_megapixels
        )
        image = OcrImage(pixels=pixels, width=width, height=height)
        last_failure: Optional[OcrAdapterError] = None
        used_fallback = False
        for route in routes:
            request = _route_request(options, route)
            try:
                candidates = _recognize(route, image, request)
            except OcrAdapterError as exc:
                if options.backend != "auto" or exc.code not in {
                    "OCR_BACKEND_UNAVAILABLE",
                    "OCR_MODEL_UNAVAILABLE",
                }:
                    failure = _map_adapter_error(exc)
                    failure.doc_id = snapshot.doc_id
                    raise failure from None
                last_failure = exc
                used_fallback = True
                continue
            routing_reason = route.routing_reason
            if used_fallback:
                routing_reason = more_specific_reason(routing_reason, "backend_unavailable")
            return _make_document(snapshot, media_type, candidates, route, routing_reason)
        if last_failure is not None:
            raise DocumentParseError(
                "OCR_BACKEND_UNAVAILABLE",
                "no OCR backend could execute the image",
                stage="ocr",
            )
        raise DocumentParseError(
            "OCR_BACKEND_UNAVAILABLE", "no OCR backend could execute the image", stage="ocr"
        )
    except DocumentParseError as exc:
        if exc.doc_id is None:
            exc.doc_id = snapshot.doc_id
        raise
    finally:
        try:
            os.unlink(snapshot.path)
        except OSError:
            pass


def parse_image_file(
    path: str,
    options: ImageOcrOptions,
    *,
    work_dir: Optional[str] = None,
    adapter_factories: Optional[Mapping[str, Callable[[], Any]]] = None,
    device_probe: Optional[Callable[[str], bool]] = None,
    ranking_override: Any = _USE_LOADER,
) -> ParsedDocument:
    routes = _select_routes(
        options,
        adapter_factories=adapter_factories,
        device_probe=device_probe,
        ranking_override=ranking_override,
    )
    try:
        return _parse_image_file_with_routes(path, options, routes, work_dir)
    finally:
        _close_routes(routes)


def _error_payload(
    exc: DocumentParseError, *, stage: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": exc.code,
            "message": exc.message,
            "stage": exc.stage if stage is None else stage,
            "retryable": exc.retryable,
            "doc_id": exc.doc_id,
        },
    }


def _write_worker_error(
    result_path: str,
    *,
    code: str,
    message: str,
    retryable: bool,
) -> None:
    write_result(
        result_path,
        {
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "stage": "ocr",
                "retryable": retryable,
                "doc_id": None,
            },
        },
    )


def _subprocess_ocr(
    path: str, options: ImageOcrOptions, work_dir: str, result_path: str
) -> None:
    apply_child_resource_limits(
        max_memory_mb=options.max_memory_mb,
        max_file_mb=options.max_file_mb,
        cpu_timeout_s=options.parse_timeout_s,
    )
    try:
        document = parse_image_file(path, options, work_dir=work_dir)
        write_result(
            result_path,
            {"ok": True, "document": document.model_dump(mode="json", exclude_none=False)},
        )
    except DocumentParseError as exc:
        write_result(result_path, _error_payload(exc))
    except MemoryError:
        _write_worker_error(
            result_path,
            code="RESOURCE_LIMIT",
            message="OCR process exceeded memory limit",
            retryable=True,
        )
    except Exception:
        _write_worker_error(
            result_path,
            code="OCR_EXECUTION_FAILED",
            message="OCR process failed unexpectedly",
            retryable=False,
        )


def _subprocess_ocr_batch(
    paths: Tuple[str, ...],
    options: ImageOcrOptions,
    result_dir: str,
    start_index: int,
    cpu_timeout_s: float,
) -> None:
    apply_child_resource_limits(
        max_memory_mb=options.max_memory_mb,
        max_file_mb=options.max_file_mb,
        cpu_timeout_s=cpu_timeout_s,
    )
    try:
        routes = _select_routes(options)
    except DocumentParseError as exc:
        write_result(
            os.path.join(result_dir, f"result-{start_index}.json"),
            _error_payload(exc),
        )
        return
    except Exception:
        _write_worker_error(
            os.path.join(result_dir, f"result-{start_index}.json"),
            code="OCR_EXECUTION_FAILED",
            message="OCR backend could not be initialized",
            retryable=False,
        )
        return

    try:
        for offset, path in enumerate(paths):
            index = start_index + offset
            result_path = os.path.join(result_dir, f"result-{index}.json")
            try:
                document = _parse_image_file_with_routes(
                    path, options, routes, result_dir
                )
                write_result(
                    result_path,
                    {
                        "ok": True,
                        "document": document.model_dump(
                            mode="json", exclude_none=False
                        ),
                    },
                )
            except DocumentParseError as exc:
                write_result(result_path, _error_payload(exc))
            except MemoryError:
                _write_worker_error(
                    result_path,
                    code="RESOURCE_LIMIT",
                    message="OCR process exceeded memory limit",
                    retryable=True,
                )
            except Exception:
                _write_worker_error(
                    result_path,
                    code="OCR_EXECUTION_FAILED",
                    message="OCR process failed unexpectedly",
                    retryable=False,
                )
    finally:
        _close_routes(routes)


def parse_image_in_subprocess(path: str, options: ImageOcrOptions) -> ParsedDocument:
    work_dir = tempfile.mkdtemp(prefix=".data-forge-ocr-")
    result_path = os.path.join(work_dir, "result.json")
    try:
        payload = run_json_child(
            _subprocess_ocr,
            (path, options, work_dir, result_path),
            result_path=result_path,
            timeout_s=options.parse_timeout_s,
            timeout_message="image OCR timed out",
            stage="ocr",
            missing_message="OCR process produced no result",
        )
        error = result_error(payload, default_stage="ocr")
        if error is None:
            return ParsedDocument.model_validate(payload["document"])
        raise error
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def iter_image_results_in_subprocess(
    paths: Sequence[str], options: ImageOcrOptions
) -> Iterator[Union[ParsedDocument, DocumentParseError]]:
    """Yield OCR results in order while reusing one model-loaded worker."""
    pending = tuple(paths)
    if not pending:
        return
    work_dir = tempfile.mkdtemp(prefix=".data-forge-ocr-batch-")
    try:
        next_index = 0
        while next_index < len(pending):
            remaining = len(pending) - next_index
            process = start_child(
                _subprocess_ocr_batch,
                (
                    pending[next_index:],
                    options,
                    work_dir,
                    next_index,
                    options.parse_timeout_s * remaining,
                ),
            )
            current = next_index
            try:
                while current < len(pending):
                    result_path = os.path.join(work_dir, f"result-{current}.json")
                    if wait_for_result(
                        result_path, process, options.parse_timeout_s
                    ):
                        try:
                            payload = read_result(result_path)
                            error = result_error(payload, default_stage="ocr")
                        except (OSError, ValueError):
                            error = DocumentParseError(
                                "RESOURCE_LIMIT",
                                "OCR process produced an invalid result",
                                stage="ocr",
                                retryable=True,
                            )
                            payload = {}
                        if error is None:
                            try:
                                yield ParsedDocument.model_validate(
                                    payload["document"]
                                )
                            except (KeyError, ValueError):
                                yield DocumentParseError(
                                    "RESOURCE_LIMIT",
                                    "OCR process produced an invalid document",
                                    stage="ocr",
                                    retryable=True,
                                )
                        else:
                            yield error
                        current += 1
                        next_index = current
                        if current < len(pending) and not process.is_alive():
                            break
                        continue
                    if process.is_alive():
                        stop_child(process)
                        yield DocumentParseError(
                            "PARSE_TIMEOUT",
                            "image OCR timed out",
                            stage="ocr",
                            retryable=True,
                        )
                    else:
                        yield DocumentParseError(
                            "RESOURCE_LIMIT",
                            "OCR process produced no result",
                            stage="ocr",
                            retryable=True,
                        )
                    current += 1
                    next_index = current
                    break
            finally:
                stop_child(process)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _error_row(
    index: int, error: DocumentParseError, output_key: str
) -> Dict[str, Any]:
    info = ErrorInfo(
        code=error.code,
        stage=error.stage,
        message=error.message,
        retryable=error.retryable,
    )
    return row_payload(
        make_error_row(index, info, output_key, doc_id=error.doc_id), output_key
    )


def parse_image_inputs(
    data_paths: Sequence[str],
    params: Dict[str, Any],
    *,
    work_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    options = build_image_ocr_options(params)
    roots = canonicalize_roots(params.get("allowed_roots", [])) if params.get("allowed_roots") else ()
    items = discover_inputs(
        data_paths, roots, supported_extensions=IMAGE_EXTENSIONS
    )
    if not items:
        raise _invalid("no supported input files found")
    effective_roots = roots or default_roots(data_paths)
    options = replace(options, allowed_roots=effective_roots)
    _close_routes(_select_routes(options))
    rows: List[Dict[str, Any]] = []
    index = 0
    while index < len(items):
        item = items[index]
        if item.error is not None:
            if options.on_error == "raise" or item.error.code in REQUEST_LEVEL_CODES:
                raise item.error
            rows.append(_error_row(index, item.error, options.output_key))
            index += 1
            continue
        run_start = index
        valid_paths: List[str] = []
        while index < len(items) and items[index].error is None:
            valid_paths.append(items[index].path)
            index += 1
        for offset, result in enumerate(
            iter_image_results_in_subprocess(valid_paths, options)
        ):
            result_index = run_start + offset
            if isinstance(result, DocumentParseError):
                if (
                    options.on_error == "raise"
                    or result.code in REQUEST_LEVEL_CODES
                ):
                    raise result
                rows.append(
                    _error_row(result_index, result, options.output_key)
                )
                continue
            rows.append(
                row_payload(
                    make_success_row(result_index, result, options.output_key),
                    options.output_key,
                )
            )
    return rows


__all__ = [
    "IMAGE_EXTENSIONS",
    "IMAGE_OCR_PARAM_KEYS",
    "ImageOcrOptions",
    "OP_IMAGE_OCR_PARAM_KEYS",
    "build_image_ocr_options",
    "candidates_to_blocks",
    "decode_image_file",
    "iter_image_results_in_subprocess",
    "parse_image_file",
    "parse_image_in_subprocess",
    "parse_image_inputs",
]
