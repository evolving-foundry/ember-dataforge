"""C-IR-2 consumer helpers for document and image environment synthesis.

The design contract for user-example ingestion is Curator's ``parsed-doc-v2``
row envelope.  DataForge consumes only the default ``parsed_doc`` output
key.  Validation has two layers:

* Curator's exported JSON Schema catches structural violations;
* Curator's own ``ParsedDocument`` model catches cross-field semantics that
  JSON Schema cannot express (derived doc id, sequential block ids, page and
  table invariants, OCR field rules).

The model module is loaded from its source file instead of importing
``curator``.  The package initializer eagerly imports the Ray-backed
runtime, which is unnecessary for contract validation and makes offline CI
fragile.  Document parsing itself is lazy-imported in :func:`parse_documents`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class CuratorContractError(RuntimeError):
    """Curator contract is unavailable, invalid, or violated."""


PARSED_DOC_SCHEMA_VERSION = "parsed-doc-v2"
PARSED_DOC_OUTPUT_KEY = "parsed_doc"
ROW_SCHEMA_NAME = "parsed-document-row-v2.schema.json"
PARSED_DOC_MODEL_NAME = "parsed_document.py"


def curator_package_dir() -> Path:
    """Locate the in-repo or installed ``curator`` package directory.

    Resolution order:
      1. ``CURATOR_ROOT`` (package dir or repository root; dev/CI convenience)
      2. importlib's spec for ``curator`` (does not execute the package)
    """
    configured = os.environ.get("CURATOR_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if (candidate / "curator").is_dir():
            return candidate / "curator"
        if candidate.name == "curator" and candidate.is_dir():
            return candidate
        raise CuratorContractError(
            f"CURATOR_ROOT={configured!r} does not point to the curator package directory"
        )

    try:
        spec = importlib.util.find_spec("curator")
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        locations = list(spec.submodule_search_locations or [])
        if locations:
            return Path(locations[0])
        if spec.origin:
            return Path(spec.origin).parent

    raise CuratorContractError(
        "Cannot find the curator package: install from the repository root (pip install -e .),"
        "or set CURATOR_ROOT to the curator package directory"
    )


def _contract_path(*parts: str) -> Path:
    path = curator_package_dir().joinpath("operations", "loading", *parts)
    if not path.is_file():
        raise CuratorContractError(f"Curator C-IR-2 contract file is missing: {path}")
    return path


@lru_cache(maxsize=1)
def load_row_schema() -> Dict[str, Any]:
    """Load Curator's frozen row-envelope JSON schema."""
    try:
        return json.loads(_contract_path("schemas", ROW_SCHEMA_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CuratorContractError(f"Unable to read the Curator row-envelope schema: {error}") from error


@lru_cache(maxsize=1)
def _row_validator():
    import jsonschema

    return jsonschema.Draft202012Validator(load_row_schema())


@lru_cache(maxsize=1)
def _curator_model_module():
    """Load Curator's contract model without executing its package initializer."""
    path = _contract_path(PARSED_DOC_MODEL_NAME)
    module_name = "_data_designer_curator_parsed_document"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create a module loader for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise CuratorContractError(
            f"Unable to load the Curator semantic validation model ({path}): {error}"
        ) from error


def _validation_details(error: Exception, *, limit: int = 3) -> str:
    errors = getattr(error, "errors", None)
    if callable(errors):
        try:
            details = []
            for item in errors()[:limit]:
                location = ".".join(str(part) for part in item.get("loc", ()))
                details.append(
                    f"{location or '<parsed_doc>'}: {item.get('msg', 'invalid')}"
                )
            if details:
                return "; ".join(details)
        except (TypeError, ValueError):
            pass
    text = str(error).strip()
    return text.splitlines()[0] if text else type(error).__name__


def _validate_document_semantics(document: Dict[str, Any]) -> None:
    model = getattr(_curator_model_module(), "ParsedDocument", None)
    if model is None:
        raise CuratorContractError("Curator parsed_document.py does not export ParsedDocument")
    try:
        model.model_validate(document)
    except Exception as error:
        raise CuratorContractError(
            "C-IR-2 parsed_doc semantic validation failed: " + _validation_details(error)
        ) from error


def validate_parsed_row(row: Any) -> Dict[str, Any]:
    """Validate one C-IR-2 row envelope; raise ``CuratorContractError`` on failure."""
    if not isinstance(row, dict):
        raise CuratorContractError(
            f"C-IR-2 row must be an object; received {type(row).__name__}"
        )
    try:
        validator = _row_validator()
    except ModuleNotFoundError as error:  # pragma: no cover - dependency error
        raise CuratorContractError(
            "Validating the C-IR-2 row envelope requires jsonschema (a Curator base dependency)"
        ) from error
    errors = sorted(validator.iter_errors(row), key=lambda item: list(item.path))
    if errors:
        details = []
        for error in errors[:3]:
            location = "/".join(str(part) for part in error.path) or "<row>"
            details.append(f"{location}: {error.message}")
        raise CuratorContractError(
            "C-IR-2 row-envelope validation failed: " + "; ".join(details)
        )

    document = row.get(PARSED_DOC_OUTPUT_KEY)
    if document is None:
        return row
    if not isinstance(document, dict):
        raise CuratorContractError("parsed_doc must be an object or null")
    if document.get("schema_version") != PARSED_DOC_SCHEMA_VERSION:
        raise CuratorContractError(
            f"parsed_doc.schema_version must be {PARSED_DOC_SCHEMA_VERSION}"
        )
    _validate_document_semantics(document)
    if row.get("doc_id") != document.get("doc_id"):
        raise CuratorContractError("C-IR-2 row doc_id must match parsed_doc.doc_id")
    return row


def validate_parsed_rows(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    validated = [validate_parsed_row(row) for row in rows]
    indexes = [row.get("input_index") for row in validated]
    if len(indexes) != len(set(indexes)):
        raise CuratorContractError("C-IR-2 row input_index must be unique")
    if sorted(indexes) != list(range(len(indexes))):
        raise CuratorContractError("C-IR-2 row input_index must increase contiguously from 0")
    return sorted(validated, key=lambda row: row["input_index"])


def parse_documents(
    paths: Sequence[str],
    *,
    allowed_roots: Optional[Sequence[str]] = None,
    parse_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Parse docx/xlsx/pdf through Curator's C-IR-2 operator surface.

    This is the only function in the envsynth package that requires the full
    Curator runtime (Ray + parser optional dependencies).
    """
    if not paths:
        raise CuratorContractError("parse_documents requires at least one input path")
    options: Dict[str, Any] = dict(parse_options or {})
    output_key = options.get("output_key", PARSED_DOC_OUTPUT_KEY)
    if output_key != PARSED_DOC_OUTPUT_KEY:
        raise CuratorContractError(
            "DataForge consumes only output_key='parsed_doc';"
            f"received {output_key!r}"
        )
    options.setdefault("on_error", "raise")
    options.setdefault("extract_style", False)
    if allowed_roots:
        options["allowed_roots"] = list(allowed_roots)
    options.pop("data_paths", None)
    try:
        from curator.operations.loading.document_parsing import parse_inputs
    except Exception as error:  # pragma: no cover - environment dependent
        raise CuratorContractError(
            "Unable to import the Curator document parsing operation (install curator and the docs extra): "
            f"{error}"
        ) from error

    try:
        rows = parse_inputs(list(paths), options)
    except Exception as error:
        raise CuratorContractError(f"Curator document parsing failed: {error}") from error
    return validate_parsed_rows(rows)


def parse_images(
    paths: Sequence[str],
    *,
    allowed_roots: Optional[Sequence[str]] = None,
    ocr_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Parse png/jpg through Curator's C-IR-2 image OCR surface."""
    if not paths:
        raise CuratorContractError("parse_images requires at least one input path")
    options: Dict[str, Any] = dict(ocr_options or {})
    output_key = options.get("output_key", PARSED_DOC_OUTPUT_KEY)
    if output_key != PARSED_DOC_OUTPUT_KEY:
        raise CuratorContractError(
            "DataForge consumes only output_key='parsed_doc';"
            f"received {output_key!r}"
        )
    options.setdefault("on_error", "raise")
    if allowed_roots:
        options["allowed_roots"] = list(allowed_roots)
    options.pop("data_paths", None)
    try:
        from curator.operations.loading.image_ocr import parse_image_inputs
    except Exception as error:  # pragma: no cover - environment dependent
        raise CuratorContractError(
            "Unable to import the Curator image OCR operation (install curator and its OCR dependencies): "
            f"{error}"
        ) from error

    try:
        rows = parse_image_inputs(list(paths), options)
    except Exception as error:
        raise CuratorContractError(f"Curator image OCR failed: {error}") from error
    return validate_parsed_rows(rows)


__all__ = [
    "CuratorContractError",
    "PARSED_DOC_OUTPUT_KEY",
    "PARSED_DOC_SCHEMA_VERSION",
    "curator_package_dir",
    "load_row_schema",
    "parse_documents",
    "parse_images",
    "validate_parsed_row",
    "validate_parsed_rows",
]
