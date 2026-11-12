"""Input expansion, safety prechecks, and parsing orchestration for docx/xlsx/pdf."""
from __future__ import annotations

import os
import posixpath
import re
import shutil
import tempfile
import zipfile
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from .document_common import (
    DocumentParseError,
    ParseOptions,
    REQUEST_LEVEL_CODES,
    build_parse_options,
    detect_language,
    make_document_provenance,
)
from .document_io import (
    DEFAULT_DOCUMENT_EXTENSIONS,
    InputItem,
    Snapshot,
    canonicalize_roots,
    default_roots,
    discover_inputs,
    sha256_file,
    snapshot_file,
    validate_source_file,
)
from .parsed_document import (
    ErrorInfo,
    ParsedDocument,
    Source,
    make_error_row,
    make_success_row,
)
from .process_limits import apply_child_resource_limits
from .subprocess_runner import result_error, run_json_child, write_result

SUPPORTED_EXTENSIONS = DEFAULT_DOCUMENT_EXTENSIONS
ENTRY_LIMIT = 256 * 1024 * 1024
TOTAL_LIMIT = 1024 * 1024 * 1024
MAX_ENTRIES = 100_000
MAX_COMPRESSION_RATIO = 100
MAX_XML_PART = 64 * 1024 * 1024
_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def _ooxml_content_type(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("[Content_Types].xml")
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        raise DocumentParseError("CORRUPT_DOCUMENT", "OOXML content types are missing", stage="parse") from exc
    text = raw.decode("utf-8", errors="replace").lower()
    if "macroenabled" in text or "vbaproject" in text:
        raise DocumentParseError("MACRO_CONTENT_REJECTED", "macro-enabled OOXML is not supported", stage="parse")
    if "wordprocessingml.document.main+xml" in text:
        return "docx"
    if "spreadsheetml.sheet.main+xml" in text:
        return "xlsx"
    raise DocumentParseError("UNSUPPORTED_FORMAT", "OOXML content type is not docx/xlsx", stage="parse")


def _safe_zip_name(name: str) -> str:
    slash_name = name.replace("\\", "/")
    if any(segment == ".." for segment in slash_name.split("/")):
        raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "ZIP entry contains a parent segment", stage="parse")
    normalized = posixpath.normpath(slash_name)
    if name.startswith("/") or name.startswith("\\") or _DRIVE_RE.match(name):
        raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "absolute ZIP entry is not allowed", stage="parse")
    if normalized == ".." or normalized.startswith("../"):
        raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "ZIP entry escapes root", stage="parse")
    return normalized


def precheck_ooxml(path: str) -> None:
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise DocumentParseError("CORRUPT_DOCUMENT", "OOXML is not a valid ZIP", stage="parse") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ENTRIES:
            raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "ZIP entry count exceeds limit", stage="parse")
        names: set = set()
        total_declared = 0
        content_types_info = None
        xml_infos = []
        for info in infos:
            if info.flag_bits & 0x1:
                raise DocumentParseError("MACRO_CONTENT_REJECTED", "encrypted ZIP entry is not supported", stage="parse")
            normalized = _safe_zip_name(info.filename)
            if normalized in names:
                raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "duplicate ZIP entry name", stage="parse")
            names.add(normalized)
            if info.filename.lower().endswith("vbaproject.bin"):
                raise DocumentParseError("MACRO_CONTENT_REJECTED", "VBA payload is not supported", stage="parse")
            if info.file_size > ENTRY_LIMIT:
                raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "ZIP entry exceeds size limit", stage="parse")
            if info.file_size > 0 and info.compress_size == 0:
                raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "ZIP entry has invalid compression ratio", stage="parse")
            if info.compress_size > 0 and info.file_size > info.compress_size * MAX_COMPRESSION_RATIO:
                raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "ZIP compression ratio exceeds limit", stage="parse")
            total_declared += info.file_size
            if total_declared > TOTAL_LIMIT:
                raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "ZIP declared size exceeds limit", stage="parse")
            if normalized == "[Content_Types].xml":
                content_types_info = info
            lower_name = normalized.lower()
            if lower_name.endswith(".xml") or lower_name.endswith(".rels"):
                xml_infos.append(info)
        if content_types_info is None:
            raise DocumentParseError("CORRUPT_DOCUMENT", "OOXML content types are missing", stage="parse")

        try:
            with archive.open(content_types_info) as handle:
                content_types = handle.read(MAX_XML_PART + 1)
        except zipfile.BadZipFile as exc:
            raise DocumentParseError("CORRUPT_DOCUMENT", "OOXML content types are corrupt", stage="parse") from exc
        if len(content_types) > MAX_XML_PART:
            raise DocumentParseError("XML_PARSE_LIMIT", "XML part exceeds size limit", stage="parse")
        lowered_content_types = content_types.lower()
        if b"macroenabled" in lowered_content_types or b"vbaproject" in lowered_content_types:
            raise DocumentParseError("MACRO_CONTENT_REJECTED", "macro-enabled OOXML is not supported", stage="parse")

        actual_total = 0
        for info in infos:
            if info.is_dir():
                continue
            actual = 0
            lower_name = _safe_zip_name(info.filename).lower()
            is_xml = lower_name.endswith(".xml") or lower_name.endswith(".rels")
            scan_tail = b""
            try:
                with archive.open(info) as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        actual += len(chunk)
                        actual_total += len(chunk)
                        if actual > ENTRY_LIMIT or actual_total > TOTAL_LIMIT:
                            raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "ZIP actual size exceeds limit", stage="parse")
                        if actual > info.file_size:
                            raise DocumentParseError("ZIP_LIMIT_EXCEEDED", "ZIP actual size exceeds declared size", stage="parse")
                        if is_xml:
                            lowered = scan_tail + chunk.lower()
                            if b"<!doctype" in lowered or b"<!entity" in lowered:
                                raise DocumentParseError(
                                    "XML_PARSE_LIMIT", "DTD or entity declaration is not allowed", stage="parse"
                                )
                            scan_tail = lowered[-16:]
            except zipfile.BadZipFile as exc:
                raise DocumentParseError("CORRUPT_DOCUMENT", "ZIP CRC or stream is corrupt", stage="parse") from exc
        try:
            from defusedxml import ElementTree as DefusedElementTree
        except ImportError as exc:
            raise DocumentParseError(
                "DEPENDENCY_MISSING",
                "install the docs extra: pip install 'data-forge[docs]'",
                stage="validate",
            ) from exc
        for info in xml_infos:
            if info.file_size > MAX_XML_PART:
                raise DocumentParseError("XML_PARSE_LIMIT", "XML part exceeds size limit", stage="parse")
            try:
                with archive.open(info) as handle:
                    data = handle.read(MAX_XML_PART + 1)
            except zipfile.BadZipFile as exc:
                raise DocumentParseError("CORRUPT_DOCUMENT", "XML part is corrupt", stage="parse") from exc
            try:
                DefusedElementTree.fromstring(data)
            except Exception as exc:
                raise DocumentParseError("XML_PARSE_LIMIT", "XML part failed safe parsing", stage="parse") from exc


def _detect_pdf(path: str) -> None:
    try:
        with open(path, "rb") as handle:
            head = handle.read(1024)
    except OSError as exc:
        raise DocumentParseError("FILE_UNREADABLE", "source file is not readable", stage="read") from exc
    if b"%PDF-" not in head:
        raise DocumentParseError("UNSUPPORTED_FORMAT", "PDF signature is missing", stage="validate")


def _language_text(blocks: Sequence[Any]) -> str:
    parts: List[str] = []
    for block in blocks:
        if block.text:
            parts.append(block.text)
        if block.table is not None:
            parts.extend(cell.text for cell in block.table.cells if cell.text)
        if block.kv_pair is not None:
            parts.extend((block.kv_pair.key, block.kv_pair.value))
    return "\n".join(parts)


def parse_document_file(
    path: str,
    options: ParseOptions,
    work_dir: Optional[str] = None,
) -> ParsedDocument:
    snapshot = snapshot_file(path, options, work_dir)
    try:
        ext = os.path.splitext(path)[1].lower()
        media_type = SUPPORTED_EXTENSIONS.get(ext)
        if media_type is None:
            raise DocumentParseError("UNSUPPORTED_FORMAT", "unsupported file extension", stage="validate")
        if media_type in {"docx", "xlsx"}:
            precheck_ooxml(snapshot.path)
            detected = _ooxml_content_type(snapshot.path)
            if detected != media_type:
                raise DocumentParseError("UNSUPPORTED_FORMAT", "file extension and OOXML content type disagree", stage="validate")
            from .document_parsers import parse_docx, parse_xlsx

            result = parse_docx(snapshot.path, options) if media_type == "docx" else parse_xlsx(snapshot.path, options)
        else:
            _detect_pdf(snapshot.path)
            from .document_parsers import parse_pdf

            result = parse_pdf(snapshot.path, options)
        text = _language_text(result.blocks)
        return ParsedDocument(
            schema_version="parsed-doc-v2",
            doc_id=snapshot.doc_id,
            source=Source(
                media_type=media_type,
                sha256=snapshot.sha256,
                byte_size=snapshot.byte_size,
                page_count=result.page_count,
            ),
            language=detect_language(text),
            provenance=make_document_provenance(result.extractor, result.extractor_version),
            blocks=result.blocks,
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


def _subprocess_parse(path: str, options: ParseOptions, work_dir: str, result_path: str) -> None:
    apply_child_resource_limits(
        max_memory_mb=options.max_memory_mb,
        max_file_mb=options.max_file_mb,
        cpu_timeout_s=options.parse_timeout_s,
    )
    try:
        document = parse_document_file(path, options, work_dir)
        write_result(
            result_path,
            {
                "ok": True,
                "document": document.model_dump(mode="json", exclude_none=False),
            },
        )
    except DocumentParseError as exc:
        write_result(
            result_path,
            {
                "ok": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "stage": exc.stage,
                    "retryable": exc.retryable,
                    "doc_id": exc.doc_id,
                },
            },
        )
    except MemoryError:
        write_result(
            result_path,
            {
                "ok": False,
                "error": {
                    "code": "RESOURCE_LIMIT",
                    "message": "parser exceeded memory limit",
                    "stage": "parse",
                    "retryable": True,
                    "doc_id": None,
                },
            },
        )
    except Exception:
        write_result(
            result_path,
            {
                "ok": False,
                "error": {
                    "code": "CORRUPT_DOCUMENT",
                    "message": "parser failed unexpectedly",
                    "stage": "parse",
                    "retryable": False,
                    "doc_id": None,
                },
            },
        )


def parse_in_subprocess(path: str, options: ParseOptions) -> ParsedDocument:
    work_dir = tempfile.mkdtemp(prefix=".data-forge-work-")
    result_path = os.path.join(work_dir, "result.json")
    try:
        payload = run_json_child(
            _subprocess_parse,
            (path, options, work_dir, result_path),
            result_path=result_path,
            timeout_s=options.parse_timeout_s,
            timeout_message="document parsing timed out",
            stage="parse",
            missing_message="parser process produced no result",
        )
        error = result_error(payload)
        if error is None:
            return ParsedDocument.model_validate(payload["document"])
        raise error
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def row_payload(row: Dict[str, Any], output_key: str = "parsed_doc") -> Dict[str, Any]:
    result = dict(row)
    document = result.get(output_key)
    if document is not None and not isinstance(document, dict):
        result[output_key] = document.model_dump(mode="json", exclude_none=False)
    error = result.get("error")
    if error is not None and not isinstance(error, dict):
        result["error"] = error.model_dump(mode="json", exclude_none=False)
    return result


def parse_inputs(
    data_paths: Sequence[str],
    params: Dict[str, Any],
    *,
    work_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    options = build_parse_options(params)
    roots = canonicalize_roots(params.get("allowed_roots", [])) if params.get("allowed_roots") else ()
    items = discover_inputs(data_paths, roots)
    if not items:
        raise ValueError("no supported input files found")
    effective_roots = roots or default_roots(data_paths)
    options = replace(options, allowed_roots=effective_roots)
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        if item.error is not None:
            if options.on_error == "raise":
                raise item.error
            error = ErrorInfo(
                code=item.error.code,
                stage=item.error.stage,
                message=item.error.message,
                retryable=item.error.retryable,
            )
            rows.append(
                row_payload(
                    make_error_row(index, error, options.output_key, doc_id=item.error.doc_id),
                    options.output_key,
                )
            )
            continue
        try:
            document = parse_in_subprocess(item.path, options)
        except DocumentParseError as exc:
            if options.on_error == "raise" or exc.code in REQUEST_LEVEL_CODES:
                raise
            error = ErrorInfo(code=exc.code, stage=exc.stage, message=exc.message, retryable=exc.retryable)
            rows.append(row_payload(make_error_row(index, error, options.output_key, doc_id=exc.doc_id), options.output_key))
            continue
        rows.append(row_payload(make_success_row(index, document, options.output_key), options.output_key))
    return rows
