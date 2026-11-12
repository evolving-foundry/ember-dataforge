"""Golden fixture tests for docx/xlsx/pdf parsing (C-IR-2)."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("docx")
pytest.importorskip("openpyxl")
pytest.importorskip("pdfplumber")

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from curator.operations.loading.document_common import DocumentParseError, ParseOptions
from curator.operations.loading.document_parsing import parse_document_file

GOLDEN = Path(__file__).resolve().parent / "golden/document_parsing"
FILES = GOLDEN / "files"
EXPECTED = GOLDEN / "expected"


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    """Golden parsing tests do not need a Ray cluster."""
    yield


def _fixtures():
    return sorted(
        path for path in FILES.rglob("*") if path.suffix.lower() in {".docx", ".xlsx", ".pdf"}
    )


def _snapshot_path(path: Path) -> Path:
    return EXPECTED / path.relative_to(FILES).with_suffix(".json")


def _normalized(document):
    payload = document.model_dump(mode="json", exclude_none=False)
    payload["provenance"]["extractor_version"] = "<version>"
    payload["provenance"]["op_version"] = "<version>"
    return payload


def test_golden_fixture_matrix_is_complete():
    fixtures = _fixtures()
    formats = {path.suffix.lower().lstrip(".") for path in fixtures}
    assert formats == {"docx", "xlsx", "pdf"}
    for media_type in formats:
        names = {path.stem for path in fixtures if path.suffix.lower().lstrip(".") == media_type}
        assert {"chinese", "table", "empty"} <= names
        assert names & {"multipage", "multisheet"}, (media_type, names)
        assert len(names) >= 4
    expected = {path.relative_to(EXPECTED).with_suffix("").as_posix() for path in EXPECTED.rglob("*.json")}
    actual = {path.relative_to(FILES).with_suffix("").as_posix() for path in fixtures}
    assert expected == actual
    for path in fixtures:
        snapshot = json.loads(_snapshot_path(path).read_text(encoding="utf-8"))
        assert snapshot["fixture"] == path.relative_to(FILES).as_posix()


@pytest.mark.parametrize("path", _fixtures(), ids=lambda item: item.relative_to(FILES).as_posix())
def test_golden_document(path: Path):
    expected = json.loads(_snapshot_path(path).read_text(encoding="utf-8"))["result"]
    if expected["ok"]:
        document = parse_document_file(str(path), ParseOptions())
        actual = {"ok": True, "document": _normalized(document)}
        assert actual == expected
        return
    with pytest.raises(DocumentParseError) as exc_info:
        parse_document_file(str(path), ParseOptions())
    exc = exc_info.value
    actual = {
        "ok": False,
        "error": {
            "code": exc.code,
            "stage": exc.stage,
            "message": exc.message,
            "retryable": exc.retryable,
            "doc_id": exc.doc_id,
        },
    }
    assert actual == expected