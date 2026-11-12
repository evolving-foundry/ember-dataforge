"""C-IR-2 (``parsed-doc-v2``) contract models and row envelope validation."""
from __future__ import annotations

import math
import re
from typing import Annotated, Any, Dict, List, Literal, Optional

try:
    import pydantic
    from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
except ImportError as exc:
    raise ImportError(
        "C-IR-2 requires pydantic>=2.0,<3; install 'data-forge' or upgrade pydantic"
    ) from exc

if not str(getattr(pydantic, "VERSION", "")).startswith("2."):
    raise ImportError(
        "C-IR-2 requires pydantic>=2.0,<3; install 'data-forge' or upgrade pydantic"
    )

SCHEMA_VERSION = "parsed-doc-v2"
READING_ORDER_VERSION = "ro-v1"

MediaType = Literal["docx", "xlsx", "pdf", "png", "jpg"]
Language = Literal["zh", "en", "mixed", "unknown"]
BlockType = Literal["paragraph", "table", "kv_pair", "list", "figure", "header", "footer"]
CellStatus = Literal[
    "value",
    "cached_formula",
    "formula_cache_missing",
    "empty",
    "covered",
    "error",
]
SheetState = Literal["visible", "hidden", "very_hidden"]
RoutingReason = Literal[
    "explicit_backend",
    "ranking",
    "ranking_missing",
    "backend_unavailable",
    "device_unavailable",
    "ranking_row_unavailable",
]
Device = Literal["npu", "cuda", "cpu"]

_DOC_ID_RE = re.compile(r"^pd-[0-9a-f]{16}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OUTPUT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
RESERVED_ROW_KEYS = frozenset({"input_index", "doc_id", "error"})


def _coerce_json_number(value: Any) -> Any:
    """Accept both JSON integer and float; bool is rejected by strict validation."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return float(value)
    return value


StrictFloat = Annotated[float, BeforeValidator(_coerce_json_number)]
BBox = Annotated[List[StrictFloat], Field(min_length=4, max_length=4)]
CoveredBy = Annotated[List[int], Field(min_length=2, max_length=2)]


def validate_output_key(value: str) -> str:
    if not isinstance(value, str) or not _OUTPUT_KEY_RE.fullmatch(value):
        raise ValueError("output_key must match ^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    if value in RESERVED_ROW_KEYS:
        raise ValueError("output_key must not be a reserved row field")
    return value


def _check_bbox(bbox: Optional[List[float]]) -> None:
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("bbox must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    for value in bbox:
        if not math.isfinite(value):
            raise ValueError("bbox values must be finite")


class Source(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    media_type: MediaType
    sha256: str = Field(pattern=_SHA256_RE.pattern)
    byte_size: int = Field(ge=0)
    page_count: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_page_count(self) -> "Source":
        if self.media_type in {"docx", "xlsx"} and self.page_count is not None:
            raise ValueError("docx/xlsx page_count must be null")
        if self.media_type in {"png", "jpg"} and self.page_count != 1:
            raise ValueError("image page_count must be 1")
        return self


class Provenance(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    extractor: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    operation: Literal["document_parsing", "image_ocr"]
    op_version: str = Field(min_length=1)
    reading_order_version: str = Field(min_length=1)
    routing_reason: Optional[RoutingReason]
    ranking_version: Optional[str]
    ranking_evaluation_set: Optional[str]
    backend: Optional[str]
    backend_version: Optional[str]
    device: Optional[Device]
    model_id: Optional[str]

    @model_validator(mode="after")
    def _validate_operation_fields(self) -> "Provenance":
        if self.operation == "document_parsing":
            if any(
                value is not None
                for value in (
                    self.routing_reason,
                    self.ranking_version,
                    self.ranking_evaluation_set,
                    self.backend,
                    self.backend_version,
                    self.device,
                    self.model_id,
                )
            ):
                raise ValueError("document_parsing provenance must keep OCR fields null")
        else:
            if self.backend is None or self.backend_version is None or self.device is None:
                raise ValueError("image_ocr provenance requires backend, backend_version and device")
            if self.extractor != self.backend:
                raise ValueError("image_ocr extractor must equal backend")
            if self.extractor_version != self.backend_version:
                raise ValueError("image_ocr extractor_version must equal backend_version")
        return self


class Cell(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    row_span: int = Field(ge=1)
    col_span: int = Field(ge=1)
    text: str
    cell_status: CellStatus
    bbox: Optional[BBox]
    ocr_confidence: Optional[StrictFloat] = Field(default=None, ge=0, le=1)
    covered_by: Optional[CoveredBy]

    @model_validator(mode="after")
    def _validate_cell(self) -> "Cell":
        _check_bbox(self.bbox)
        if self.cell_status == "covered":
            if self.covered_by is None:
                raise ValueError("covered cell requires covered_by")
            if self.row_span != 1 or self.col_span != 1 or self.text != "":
                raise ValueError("covered cell must be a 1x1 empty cell")
            if self.bbox is not None or self.ocr_confidence is not None:
                raise ValueError("covered cell must not carry bbox or confidence")
        elif self.covered_by is not None:
            raise ValueError("covered_by is only valid for covered cells")
        if self.covered_by is not None:
            if self.covered_by == [self.row, self.col]:
                raise ValueError("covered_by must not point at itself")
        if self.cell_status == "formula_cache_missing" and self.text != "":
            raise ValueError("formula_cache_missing cell must have empty text")
        return self


class Table(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    structure: Literal["grid", "region"]
    n_rows: Optional[int] = Field(default=None, ge=1)
    n_cols: Optional[int] = Field(default=None, ge=1)
    cells: List[Cell]

    @model_validator(mode="after")
    def _validate_table(self) -> "Table":
        if self.structure == "region":
            if self.n_rows is not None or self.n_cols is not None or self.cells:
                raise ValueError("region table must have null dimensions and no cells")
            return self
        if self.n_rows is None or self.n_cols is None:
            raise ValueError("grid table requires n_rows and n_cols")
        if len(self.cells) != self.n_rows * self.n_cols:
            raise ValueError("grid table cell count must equal n_rows * n_cols")
        coords = [(cell.row, cell.col) for cell in self.cells]
        if len(set(coords)) != len(coords):
            raise ValueError("grid table coordinates must be unique")
        if coords != sorted(coords):
            raise ValueError("grid table cells must be sorted by row then col")
        by_coord = {coord: cell for coord, cell in zip(coords, self.cells)}
        for coord in ((row, col) for row in range(self.n_rows) for col in range(self.n_cols)):
            if coord not in by_coord:
                raise ValueError(f"grid table is missing cell {coord}")
        for cell in self.cells:
            if cell.row + cell.row_span > self.n_rows:
                raise ValueError("cell row_span exceeds grid")
            if cell.col + cell.col_span > self.n_cols:
                raise ValueError("cell col_span exceeds grid")
            if cell.cell_status == "covered":
                anchor = by_coord.get(tuple(cell.covered_by))
                if anchor is None:
                    raise ValueError("covered_by does not point at a cell")
                if not (
                    anchor.row <= cell.row < anchor.row + anchor.row_span
                    and anchor.col <= cell.col < anchor.col + anchor.col_span
                ):
                    raise ValueError("covered_by anchor does not cover the cell")
        for cell in self.cells:
            if cell.cell_status == "covered":
                continue
            for row in range(cell.row, cell.row + cell.row_span):
                for col in range(cell.col, cell.col + cell.col_span):
                    if (row, col) != (cell.row, cell.col):
                        child = by_coord[(row, col)]
                        if child.cell_status != "covered" or child.covered_by != [cell.row, cell.col]:
                            raise ValueError("merged cell must cover every coordinate")
        return self


class KvPair(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    key: str = Field(min_length=1)
    value: str


class Style(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    font_size_pt: Optional[StrictFloat] = Field(default=None, gt=0)
    bold: Optional[bool]


class Sheet(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    name: str
    index: int = Field(ge=0)
    state: SheetState


class Block(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    block_id: str = Field(pattern=r"^b-[0-9]{6}$")
    type: BlockType
    page: Optional[int] = Field(default=None, ge=1)
    sheet: Optional[Sheet]
    bbox: Optional[BBox]
    text: str
    table: Optional[Table]
    kv_pair: Optional[KvPair]
    style: Optional[Style]
    ocr_confidence: Optional[StrictFloat] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _validate_block(self) -> "Block":
        _check_bbox(self.bbox)
        if self.type == "table":
            if self.table is None:
                raise ValueError("table block requires table")
        elif self.table is not None:
            raise ValueError("table must be null unless type=table")
        if self.type == "kv_pair":
            if self.kv_pair is None:
                raise ValueError("kv_pair block requires kv_pair")
        elif self.kv_pair is not None:
            raise ValueError("kv_pair must be null unless type=kv_pair")
        if self.type != "table" and self.sheet is not None:
            raise ValueError("sheet is only valid on table blocks")
        return self


class ParsedDocument(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal["parsed-doc-v2"]
    doc_id: str = Field(pattern=_DOC_ID_RE.pattern)
    source: Source
    language: Language
    provenance: Provenance
    blocks: List[Block]

    @model_validator(mode="after")
    def _validate_doc_id(self) -> "ParsedDocument":
        expected = f"pd-{self.source.sha256[:16]}"
        if self.doc_id != expected:
            raise ValueError("doc_id must be derived from source.sha256")
        return self

    @model_validator(mode="after")
    def _validate_block_order(self) -> "ParsedDocument":
        expected = [f"b-{index:06d}" for index in range(1, len(self.blocks) + 1)]
        actual = [block.block_id for block in self.blocks]
        if actual != expected:
            raise ValueError("block_id must be sequential from b-000001")
        return self

    @model_validator(mode="after")
    def _validate_media_semantics(self) -> "ParsedDocument":
        media_type = self.source.media_type
        is_ocr = self.provenance.operation == "image_ocr"
        if is_ocr and media_type not in {"png", "jpg"}:
            raise ValueError("image_ocr provenance requires png/jpg media_type")
        if not is_ocr and media_type not in {"docx", "xlsx", "pdf"}:
            raise ValueError("document_parsing provenance requires docx/xlsx/pdf media_type")
        if media_type == "pdf" and self.source.page_count is None:
            raise ValueError("pdf source requires page_count")

        allowed_cell_statuses = {
            "docx": frozenset({"value", "empty", "covered"}),
            "xlsx": frozenset(
                {
                    "value",
                    "cached_formula",
                    "formula_cache_missing",
                    "empty",
                    "covered",
                    "error",
                }
            ),
            "pdf": frozenset({"value", "empty", "covered"}),
            "png": frozenset({"value", "empty", "covered"}),
            "jpg": frozenset({"value", "empty", "covered"}),
        }[media_type]

        for block in self.blocks:
            if media_type in {"docx", "xlsx"}:
                if block.page is not None:
                    raise ValueError("docx/xlsx block page must be null")
            elif media_type == "pdf":
                if block.page is None:
                    raise ValueError("pdf block page must be 1-based")
            elif block.page != 1:
                raise ValueError("image block page must be 1")

            if media_type == "xlsx":
                if block.type == "table" and block.sheet is None:
                    raise ValueError("xlsx table block requires sheet")
            elif block.sheet is not None:
                raise ValueError("sheet is only valid for xlsx table blocks")

            if media_type != "docx" and block.style is not None:
                raise ValueError("style is only valid for docx blocks")
            if not is_ocr and block.ocr_confidence is not None:
                raise ValueError("document_parsing blocks cannot carry OCR confidence")

            if block.table is None:
                continue
            for cell in block.table.cells:
                if cell.cell_status not in allowed_cell_statuses:
                    raise ValueError(
                        f"cell_status {cell.cell_status!r} is not valid for {media_type}"
                    )
                if not is_ocr and cell.ocr_confidence is not None:
                    raise ValueError("document_parsing cells cannot carry OCR confidence")
                if media_type in {"docx", "xlsx"} and cell.bbox is not None:
                    raise ValueError("native docx/xlsx cells must keep bbox null")
        return self


class ErrorInfo(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    stage: Literal["validate", "read", "parse", "ocr", "output"]
    message: str = Field(min_length=1)
    retryable: bool


JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_ROW_OUTPUT_KEY_PATTERN = r"^(?!input_index$|doc_id$|error$)[A-Za-z_][A-Za-z0-9_]{0,127}$"


def parsed_document_json_schema() -> Dict[str, Any]:
    """Return the strict C-IR-2 schema used by downstream consumers."""
    return ParsedDocument.model_json_schema(
        mode="validation",
        ref_template="#/$defs/{model}",
    )


def _schema_definitions() -> Dict[str, Any]:
    document_schema = dict(parsed_document_json_schema())
    definitions = dict(document_schema.pop("$defs", {}))
    definitions["ParsedDocument"] = document_schema
    error_schema = dict(ErrorInfo.model_json_schema(
        mode="validation",
        ref_template="#/$defs/{model}",
    ))
    error_schema.pop("$defs", None)
    definitions["ErrorInfo"] = error_schema
    return definitions


def parsed_document_row_json_schema(output_key: str = "parsed_doc") -> Dict[str, Any]:
    """Return a schema for one row with a concrete ``output_key``.

    The row envelope has a dynamic success field, so the static schema uses
    ``patternProperties`` for the general shape and an explicit ``oneOf`` for
    the generated key to preserve the exactly-one-of invariant.
    """
    validate_output_key(output_key)
    definitions = _schema_definitions()
    output_pattern = f"^{re.escape(output_key)}$"
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": "urn:curator:parsed-doc-row-v2",
        "title": "ParsedDocumentRowEnvelope",
        "type": "object",
        "additionalProperties": False,
        "required": ["input_index", "doc_id", output_key, "error"],
        "properties": {
            "input_index": {"type": "integer", "minimum": 0},
            "doc_id": {
                "anyOf": [
                    {"type": "string", "pattern": _DOC_ID_RE.pattern},
                    {"type": "null"},
                ]
            },
            "error": {
                "anyOf": [
                    {"$ref": "#/$defs/ErrorInfo"},
                    {"type": "null"},
                ]
            },
        },
        "patternProperties": {
            output_pattern: {
                "anyOf": [
                    {"$ref": "#/$defs/ParsedDocument"},
                    {"type": "null"},
                ]
            }
        },
        "allOf": [
            {
                "oneOf": [
                    {
                        "required": [output_key],
                        "properties": {
                            output_key: {"$ref": "#/$defs/ParsedDocument"},
                            "error": {"type": "null"},
                        },
                    },
                    {
                        "required": ["error"],
                        "properties": {
                            output_key: {"type": "null"},
                            "error": {"$ref": "#/$defs/ErrorInfo"},
                        },
                    },
                ]
            }
        ],
        "$defs": definitions,
    }


def parsed_document_row_pattern_json_schema() -> Dict[str, Any]:
    """Return the output-key-agnostic row schema for consumer validation."""
    definitions = _schema_definitions()
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": "urn:curator:parsed-doc-row-pattern-v2",
        "title": "ParsedDocumentRowEnvelopePattern",
        "type": "object",
        "additionalProperties": False,
        "minProperties": 4,
        "maxProperties": 4,
        "required": ["input_index", "doc_id", "error"],
        "properties": {
            "input_index": {"type": "integer", "minimum": 0},
            "doc_id": {
                "anyOf": [
                    {"type": "string", "pattern": _DOC_ID_RE.pattern},
                    {"type": "null"},
                ]
            },
            "error": {
                "anyOf": [
                    {"$ref": "#/$defs/ErrorInfo"},
                    {"type": "null"},
                ]
            },
        },
        "patternProperties": {
            _ROW_OUTPUT_KEY_PATTERN: {
                "anyOf": [
                    {"$ref": "#/$defs/ParsedDocument"},
                    {"type": "null"},
                ]
            }
        },
        "allOf": [
            {
                "anyOf": [
                    {"properties": {"error": {"$ref": "#/$defs/ErrorInfo"}}},
                    {
                        "patternProperties": {
                            _ROW_OUTPUT_KEY_PATTERN: {"$ref": "#/$defs/ParsedDocument"}
                        }
                    },
                ]
            },
            {
                "not": {
                    "allOf": [
                        {"properties": {"error": {"$ref": "#/$defs/ErrorInfo"}}},
                        {
                            "patternProperties": {
                                _ROW_OUTPUT_KEY_PATTERN: {"$ref": "#/$defs/ParsedDocument"}
                            }
                        },
                    ]
                }
            },
        ],
        "$defs": definitions,
    }


def write_consumer_schemas(output_dir: str, output_key: str = "parsed_doc") -> None:
    """Write the committed consumer schemas as canonical UTF-8 JSON files."""
    import json
    from pathlib import Path

    validate_output_key(output_key)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    row_filename = (
        "parsed-document-row-v2.schema.json"
        if output_key == "parsed_doc"
        else f"parsed-document-row-{output_key}.schema.json"
    )
    payloads = {
        "parsed-document-v2.schema.json": parsed_document_json_schema(),
        "parsed-document-row-pattern-v2.schema.json": parsed_document_row_pattern_json_schema(),
        row_filename: parsed_document_row_json_schema(output_key),
    }
    for filename, payload in payloads.items():
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        (target / filename).write_text(text, encoding="utf-8", newline="\n")


def validate_row_envelope(row: Dict[str, Any], output_key: str = "parsed_doc") -> Dict[str, Any]:
    validate_output_key(output_key)
    expected_keys = {"input_index", "doc_id", output_key, "error"}
    if set(row) != expected_keys:
        raise ValueError(f"row envelope keys must be exactly {sorted(expected_keys)}")
    input_index = row["input_index"]
    if isinstance(input_index, bool) or not isinstance(input_index, int) or input_index < 0:
        raise ValueError("input_index must be a non-negative integer")
    doc_id = row["doc_id"]
    if doc_id is not None and (not isinstance(doc_id, str) or not _DOC_ID_RE.fullmatch(doc_id)):
        raise ValueError("doc_id must be null or match ^pd-[0-9a-f]{16}$")
    document = row[output_key]
    error = row["error"]
    if (document is None) == (error is None):
        raise ValueError(f"exactly one of {output_key} and error must be non-null")
    if document is not None:
        parsed = document if isinstance(document, ParsedDocument) else ParsedDocument.model_validate(document)
        if doc_id != parsed.doc_id:
            raise ValueError("row doc_id must equal parsed document doc_id")
        return row
    error_obj = error if isinstance(error, ErrorInfo) else ErrorInfo.model_validate(error)
    row["error"] = error_obj
    return row


def make_success_row(input_index: int, document: ParsedDocument, output_key: str = "parsed_doc") -> Dict[str, Any]:
    row = {
        "input_index": input_index,
        "doc_id": document.doc_id,
        output_key: document,
        "error": None,
    }
    return validate_row_envelope(row, output_key)


def make_error_row(
    input_index: int,
    error: ErrorInfo,
    output_key: str = "parsed_doc",
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    row = {
        "input_index": input_index,
        "doc_id": doc_id,
        output_key: None,
        "error": error,
    }
    return validate_row_envelope(row, output_key)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export C-IR-2 consumer JSON Schemas")
    parser.add_argument("output_dir", help="directory for schema files")
    parser.add_argument("--output-key", default="parsed_doc")
    args = parser.parse_args()
    write_consumer_schemas(args.output_dir, args.output_key)


if __name__ == "__main__":
    main()
