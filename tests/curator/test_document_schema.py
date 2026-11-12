"""C-IR-2 JSON Schema export and row-envelope validation tests."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("jsonschema")

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from jsonschema import Draft202012Validator

from curator.operations.loading.document_parsing import row_payload
from curator.operations.loading.parsed_document import (
    Block,
    Cell,
    ErrorInfo,
    ParsedDocument,
    Provenance,
    Sheet,
    Source,
    Table,
    make_error_row,
    make_success_row,
    parsed_document_json_schema,
    parsed_document_row_json_schema,
    parsed_document_row_pattern_json_schema,
    write_consumer_schemas,
)

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "curator/operations/loading/schemas"
SHA = "b" * 64


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    """Schema tests do not need a Ray cluster."""
    yield


def _document() -> ParsedDocument:
    return ParsedDocument(
        schema_version="parsed-doc-v2",
        doc_id=f"pd-{SHA[:16]}",
        source=Source(media_type="xlsx", sha256=SHA, byte_size=12, page_count=None),
        language="en",
        provenance=Provenance(
            extractor="openpyxl",
            extractor_version="3.1.5",
            operation="document_parsing",
            op_version="0.2.0",
            reading_order_version="ro-v1",
            routing_reason=None,
            ranking_version=None,
            ranking_evaluation_set=None,
            backend=None,
            backend_version=None,
            device=None,
            model_id=None,
        ),
        blocks=[],
    )


def _error() -> ErrorInfo:
    return ErrorInfo(code="FILE_UNREADABLE", stage="read", message="source is not readable", retryable=False)


def _assert_invalid(schema, instance):
    errors = list(Draft202012Validator(schema).iter_errors(instance))
    assert errors, "schema unexpectedly accepted invalid instance"


def test_committed_schemas_match_exports_and_are_valid():
    expected = {
        "parsed-document-v2.schema.json": parsed_document_json_schema(),
        "parsed-document-row-v2.schema.json": parsed_document_row_json_schema(),
        "parsed-document-row-pattern-v2.schema.json": parsed_document_row_pattern_json_schema(),
    }
    for filename, schema in expected.items():
        committed = json.loads((SCHEMA_DIR / filename).read_text(encoding="utf-8"))
        assert committed == json.loads(json.dumps(schema)), filename
        Draft202012Validator.check_schema(committed)


def test_schema_field_sets_match_pydantic_models():
    schema = parsed_document_json_schema()
    assert set(schema["properties"]) == set(ParsedDocument.model_fields)
    assert schema["additionalProperties"] is False
    models = {
        "Source": Source,
        "Provenance": Provenance,
        "Block": Block,
        "Table": Table,
        "Cell": Cell,
        "Sheet": Sheet,
    }
    for model_name, model in models.items():
        assert set(schema["$defs"][model_name]["properties"]) == set(model.model_fields)
        assert schema["$defs"][model_name]["additionalProperties"] is False
    row_schema = parsed_document_row_pattern_json_schema()
    assert set(row_schema["$defs"]["ErrorInfo"]["properties"]) == set(ErrorInfo.model_fields)
    assert row_schema["$defs"]["ErrorInfo"]["additionalProperties"] is False
    assert "$defs" not in row_schema["$defs"]["ParsedDocument"]
    assert "$defs" not in row_schema["$defs"]["ErrorInfo"]


def test_success_and_error_rows_validate_against_both_schemas(tmp_path):
    document = _document()
    success = row_payload(make_success_row(0, document))
    failure = row_payload(make_error_row(1, _error()))
    failure_with_id = row_payload(make_error_row(2, _error(), doc_id=document.doc_id))

    for schema in (
        parsed_document_row_json_schema(),
        parsed_document_row_pattern_json_schema(),
    ):
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for row in (success, failure, failure_with_id):
            errors = sorted(validator.iter_errors(row), key=lambda item: list(item.path))
            assert not errors, errors

    write_consumer_schemas(str(tmp_path))
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "parsed-document-row-pattern-v2.schema.json",
        "parsed-document-row-v2.schema.json",
        "parsed-document-v2.schema.json",
    ]


def test_custom_output_key_schema_and_pattern_schema(tmp_path):
    document = _document()
    custom = row_payload(make_success_row(0, document, output_key="document"), "document")
    exact = parsed_document_row_json_schema("document")
    Draft202012Validator(exact).validate(custom)
    _assert_invalid(
        exact,
        {
            "input_index": 0,
            "doc_id": document.doc_id,
            "document": document.model_dump(mode="json"),
            "parsed_doc": document.model_dump(mode="json"),
            "error": None,
        },
    )

    pattern = parsed_document_row_pattern_json_schema()
    Draft202012Validator(pattern).validate(custom)
    _assert_invalid(pattern, {**custom, "extra": 1})
    _assert_invalid(
        pattern,
        {"input_index": 0, "doc_id": None, "error": None, "document": None},
    )
    _assert_invalid(
        pattern,
        {"input_index": 0, "doc_id": None, "error": _error().model_dump(), "document": document.model_dump(mode="json")},
    )
    write_consumer_schemas(str(tmp_path), "document")
    assert (tmp_path / "parsed-document-row-document.schema.json").is_file()


def test_row_schema_rejects_unknown_and_invalid_combinations():
    document = _document()
    success = make_success_row(0, document)
    failure = make_error_row(1, _error())
    schema = parsed_document_row_json_schema()
    _assert_invalid(schema, {**success, "extra": 1})
    _assert_invalid(schema, {**success, "doc_id": "pd-xyz"})
    _assert_invalid(schema, {**success, "error": _error().model_dump()})
    _assert_invalid(schema, {"input_index": 0, "doc_id": None, "parsed_doc": None, "error": None})
    _assert_invalid(schema, {key: value for key, value in failure.items() if key != "parsed_doc"})


def test_reserved_or_malformed_output_key_rejected():
    for key in ("input_index", "doc_id", "error", "bad-key", "1bad"):
        with pytest.raises(ValueError):
            parsed_document_row_json_schema(key)
        with pytest.raises(ValueError):
            write_consumer_schemas("unused", key)
