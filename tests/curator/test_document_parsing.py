"""Document parsing / C-IR-2 / trusted_worker op-mode regression tests."""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys
import types
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

# The repository imports torch eagerly through operations/__init__.py. These
# tests only exercise document parsing, so a bare module stub is sufficient.
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from pydantic import ValidationError

from curator.operations.loading.document_common import ParseOptions
from curator.operations.loading.document_parsing import (
    DocumentParseError,
    discover_inputs,
    parse_document_file,
    parse_in_subprocess,
    parse_inputs,
    precheck_ooxml,
    validate_source_file,
)
from curator.operations.loading.document_parsers import (
    _pdf_line_blocks,
    _pdf_table_block,
    _pdf_table_has_upright_text,
)
from curator.operations.loading.parsed_document import (
    Cell,
    ErrorInfo,
    ParsedDocument,
    Provenance,
    Sheet,
    Source,
    Style,
    Table,
    make_error_row,
    make_success_row,
    validate_output_key,
    validate_row_envelope,
)


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    """Document parsing uses process isolation, not a Ray cluster."""
    yield


SHA = "a" * 64


def _provenance(operation: str = "document_parsing") -> Provenance:
    return Provenance(
        extractor="test-extractor",
        extractor_version="1.0",
        operation=operation,
        op_version="0.2.0",
        reading_order_version="ro-v1",
        routing_reason=None,
        ranking_version=None,
        ranking_evaluation_set=None,
        backend=None,
        backend_version=None,
        device=None,
        model_id=None,
    )


def _document(media_type: str = "xlsx", blocks=None, page_count=None) -> ParsedDocument:
    if media_type == "pdf":
        page_count = 1 if page_count is None else page_count
    if media_type in {"png", "jpg"}:
        page_count = 1
    operation = "image_ocr" if media_type in {"png", "jpg"} else "document_parsing"
    return ParsedDocument(
        schema_version="parsed-doc-v2",
        doc_id=f"pd-{SHA[:16]}",
        source=Source(media_type=media_type, sha256=SHA, byte_size=10, page_count=page_count),
        language="en",
        provenance=_provenance(operation),
        blocks=blocks or [],
    )


def _cell(
    row: int,
    col: int,
    text: str = "",
    status: str = "value",
    row_span: int = 1,
    col_span: int = 1,
    covered_by=None,
) -> Cell:
    return Cell(
        row=row,
        col=col,
        row_span=row_span,
        col_span=col_span,
        text=text,
        cell_status=status,
        bbox=None,
        ocr_confidence=None,
        covered_by=covered_by,
    )


def _xlsx_table_block(cells, n_rows: int, n_cols: int):
    from curator.operations.loading.parsed_document import Block

    return Block(
        block_id="b-000001",
        type="table",
        page=None,
        sheet=Sheet(name="Sheet1", index=0, state="visible"),
        bbox=None,
        text="",
        table=Table(structure="grid", n_rows=n_rows, n_cols=n_cols, cells=cells),
        kv_pair=None,
        style=None,
        ocr_confidence=None,
    )


class TestContract:
    def test_image_ocr_provenance_requires_consistent_backend(self):
        fields = dict(
            extractor="mindocr",
            extractor_version="1.2.3",
            operation="image_ocr",
            op_version="0.2.0",
            reading_order_version="ro-v1",
            routing_reason="explicit_backend",
            ranking_version=None,
            ranking_evaluation_set=None,
            backend="mindocr",
            backend_version="1.2.3",
            device="cpu",
            model_id="model-a",
        )
        Provenance(**fields)
        with pytest.raises(ValidationError, match="requires backend"):
            Provenance(**{**fields, "backend": None})
        with pytest.raises(ValidationError, match="extractor_version"):
            Provenance(**{**fields, "extractor_version": "1.2.4"})

    def test_numbers_coerce_and_integer_stays_strict(self):
        from curator.operations.loading.parsed_document import Block

        block = Block(
            block_id="b-000001",
            type="paragraph",
            page=1,
            sheet=None,
            bbox=[0, 0, 1, 1],
            text="hello",
            table=None,
            kv_pair=None,
            style=None,
            ocr_confidence=None,
        )
        doc = _document("pdf", blocks=[block])
        assert doc.blocks[0].bbox == [0.0, 0.0, 1.0, 1.0]
        assert Style(font_size_pt=12, bold=True).font_size_pt == 12.0

        with pytest.raises(ValidationError):
            Block(
                block_id="b-000001",
                type="paragraph",
                page=1.0,
                sheet=None,
                bbox=None,
                text="hello",
                table=None,
                kv_pair=None,
                style=None,
                ocr_confidence=None,
            )

    def test_unknown_fields_are_rejected(self):
        payload = _document().model_dump(mode="json")
        payload["extra"] = 1
        with pytest.raises(ValidationError):
            ParsedDocument.model_validate(payload)

    def test_doc_id_must_match_sha(self):
        payload = _document().model_dump(mode="json")
        payload["doc_id"] = "pd-0123456789abcdef"
        with pytest.raises(ValidationError):
            ParsedDocument.model_validate(payload)

    def test_block_ids_must_be_sequential(self):
        from curator.operations.loading.parsed_document import Block

        def block(block_id: str) -> Block:
            return Block(
                block_id=block_id,
                type="paragraph",
                page=1,
                sheet=None,
                bbox=None,
                text="hello",
                table=None,
                kv_pair=None,
                style=None,
                ocr_confidence=None,
            )

        with pytest.raises(ValidationError):
            _document("pdf", blocks=[block("b-000001"), block("b-000003")])
        with pytest.raises(ValidationError):
            _document("pdf", blocks=[block("b-000002")])

    def test_media_semantics_are_enforced(self):
        from curator.operations.loading.parsed_document import Block

        bad_docx = Block(
            block_id="b-000001",
            type="paragraph",
            page=1,
            sheet=None,
            bbox=None,
            text="x",
            table=None,
            kv_pair=None,
            style=None,
            ocr_confidence=None,
        )
        with pytest.raises(ValidationError):
            _document("docx", blocks=[bad_docx])

        cached = _cell(0, 0, "3", "cached_formula")
        with pytest.raises(ValidationError):
            _document("docx", blocks=[_xlsx_table_block([cached], 1, 1)])

    def test_grid_merge_contract(self):
        cells = [
            _cell(0, 0, "A", "value", 1, 2),
            _cell(0, 1, "", "covered", covered_by=[0, 0]),
            _cell(1, 0, "", "empty"),
            _cell(1, 1, "D"),
        ]
        table = Table(structure="grid", n_rows=2, n_cols=2, cells=cells)
        assert table.n_rows == 2
        with pytest.raises(ValidationError):
            Table(structure="grid", n_rows=2, n_cols=2, cells=cells[:3])
        with pytest.raises(ValidationError):
            Table(
                structure="grid",
                n_rows=2,
                n_cols=2,
                cells=[
                    _cell(0, 0, "A", "value", 1, 2),
                    _cell(0, 1, "", "covered", covered_by=[1, 1]),
                    _cell(1, 0, "", "empty"),
                    _cell(1, 1, "", "empty"),
                ],
            )
        with pytest.raises(ValidationError):
            Cell(
                row=0,
                col=1,
                row_span=1,
                col_span=1,
                text="",
                cell_status="covered",
                bbox=[0.1, 0.1, 0.2, 0.2],
                ocr_confidence=None,
                covered_by=[0, 0],
            )

    def test_region_table_has_no_grid(self):
        with pytest.raises(ValidationError):
            Table(structure="region", n_rows=1, n_cols=1, cells=[_cell(0, 0)])

    def test_row_envelope(self):
        doc = _document()
        row = make_success_row(0, doc)
        assert set(row) == {"input_index", "doc_id", "parsed_doc", "error"}
        assert row["parsed_doc"] is doc
        assert row["error"] is None
        validate_row_envelope(row)

        custom = make_success_row(1, doc, output_key="document")
        assert set(custom) == {"input_index", "doc_id", "document", "error"}
        validate_row_envelope(custom, output_key="document")

        error = ErrorInfo(code="CORRUPT_DOCUMENT", stage="parse", message="bad", retryable=False)
        failed = make_error_row(2, error)
        assert failed["parsed_doc"] is None
        assert failed["error"] is error
        validate_row_envelope(failed)

        with pytest.raises(ValueError):
            validate_row_envelope({**row, "extra": 1})
        with pytest.raises(ValueError):
            validate_row_envelope({**row, "error": error})
        with pytest.raises(ValueError):
            validate_output_key("input_index")
        with pytest.raises(ValueError):
            validate_output_key("bad-key")


def _write_zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def _pdf_bytes(
    lines: list[str] | None = None,
    prefix: bytes = b"",
    raw_content: bytes | None = None,
) -> bytes:
    lines = lines or []
    content_lines = []
    y = 720
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_lines.append(f"BT /F1 12 Tf 72 {y} Td ({escaped}) Tj ET")
        y -= 20
    content = raw_content if raw_content is not None else "\n".join(content_lines).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header = prefix + b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = b""
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(header) + len(body))
        body += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_offset = len(header) + len(body)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        xref += f"{offset:010d} 00000 n \n".encode()
    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF\n"
    )
    return header + body + xref + trailer


def _write_docx(path: Path) -> None:
    docx = pytest.importorskip("docx")
    from docx.shared import Pt

    document = docx.Document()
    document.add_paragraph("正文第一段，用于文档解析测试。")
    styled = document.add_paragraph()
    run = styled.add_run("Styled paragraph")
    run.bold = True
    run.font.size = Pt(14)
    document.add_paragraph("列表项", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = "B"
    table.cell(1, 0).text = "C"
    table.cell(1, 1).text = "D"
    table.cell(0, 0).merge(table.cell(0, 1))
    table.cell(1, 1).add_paragraph("第二段")
    document.sections[0].header.paragraphs[0].text = "页眉"
    document.sections[0].footer.paragraphs[0].text = "页脚"
    document.save(path)


def _write_xlsx(path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from openpyxl.styles import PatternFill

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "台账"
    sheet["A1"] = "Ticket ID"
    sheet["B1"] = "状态"
    sheet["A2"] = 2
    sheet["B2"] = True
    sheet["A3"] = 1.5
    sheet["B3"] = dt.date(2024, 1, 2)
    sheet["C1"] = "=A2+1"
    sheet["D1"] = "#DIV/0!"
    sheet.merge_cells("E1:F1")
    sheet["E1"] = "合并"
    sheet["XFD1048576"].fill = PatternFill(fill_type="solid", fgColor="FFFF00")
    hidden = workbook.create_sheet("隐藏")
    hidden["A1"] = "hidden"
    hidden.sheet_state = "hidden"
    very_hidden = workbook.create_sheet("很隐藏")
    very_hidden["A1"] = "very hidden"
    very_hidden.sheet_state = "veryHidden"
    workbook.save(path)


class TestParsers:
    def test_docx_blocks_tables_headers_and_styles(self, tmp_path):
        path = tmp_path / "sample.docx"
        _write_docx(path)
        document = parse_document_file(str(path), ParseOptions(extract_style=True))

        assert document.source.media_type == "docx"
        assert document.source.page_count is None
        assert document.doc_id == f"pd-{document.source.sha256[:16]}"
        texts = [block.text for block in document.blocks]
        assert any("正文第一段" in text for text in texts)
        assert any(block.type == "list" for block in document.blocks)
        assert any(block.type == "header" and block.text == "页眉" for block in document.blocks)
        assert any(block.type == "footer" and block.text == "页脚" for block in document.blocks)
        assert any(block.style is not None and block.style.bold is True for block in document.blocks)

        table_blocks = [block for block in document.blocks if block.type == "table"]
        assert len(table_blocks) == 1
        table = table_blocks[0].table
        assert table.n_rows == 2 and table.n_cols == 2
        assert len(table.cells) == 4
        anchor = next(cell for cell in table.cells if cell.row == 0 and cell.col == 0)
        covered = next(cell for cell in table.cells if cell.row == 0 and cell.col == 1)
        assert anchor.row_span == 1 and anchor.col_span == 2
        assert covered.cell_status == "covered" and covered.covered_by == [0, 0]
        merged_text = next(cell.text for cell in table.cells if cell.row == 1 and cell.col == 1)
        assert "\n第二段" in merged_text

    def test_xlsx_values_states_and_effective_dimensions(self, tmp_path):
        path = tmp_path / "sample.xlsx"
        _write_xlsx(path)
        document = parse_document_file(str(path), ParseOptions())
        assert document.source.media_type == "xlsx"
        assert len(document.blocks) == 3
        assert [block.sheet.state for block in document.blocks] == ["visible", "hidden", "very_hidden"]

        table = document.blocks[0].table
        assert table.n_rows == 3
        assert table.n_cols == 6
        cells = {(cell.row, cell.col): cell for cell in table.cells}
        assert cells[(1, 0)].text == "2" and cells[(1, 0)].cell_status == "value"
        assert cells[(1, 1)].text == "TRUE"
        assert cells[(2, 0)].text == "1.5"
        assert cells[(2, 1)].text == "2024-01-02T00:00:00"
        assert cells[(0, 2)].cell_status == "formula_cache_missing"
        assert cells[(0, 2)].text == ""
        assert cells[(0, 3)].cell_status == "error"
        assert cells[(0, 3)].text == "#DIV/0!"
        assert cells[(0, 4)].row_span == 1 and cells[(0, 4)].col_span == 2
        assert cells[(0, 5)].cell_status == "covered"
        assert cells[(0, 5)].covered_by == [0, 4]

    def test_xlsx_language_uses_table_cells(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")

        path = tmp_path / "language.xlsx"
        workbook = openpyxl.Workbook()
        workbook.active["A1"] = "中文内容" * 6
        workbook.save(path)

        document = parse_document_file(str(path), ParseOptions())
        assert document.language == "zh"

    def test_xlsx_cached_formula_is_preserved(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")

        source = tmp_path / "formula.xlsx"
        patched = tmp_path / "formula-cached.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet["A1"] = 2
        sheet["B1"] = "=A1+1"
        workbook.save(source)
        with zipfile.ZipFile(source) as archive, zipfile.ZipFile(patched, "w") as target:
            for item in archive.infolist():
                data = archive.read(item.filename)
                if item.filename == "xl/worksheets/sheet1.xml":
                    data = data.replace(b"</f>", b"</f><v>3</v>", 1)
                target.writestr(item, data)
        document = parse_document_file(str(patched), ParseOptions())
        cell = next(cell for cell in document.blocks[0].table.cells if cell.row == 0 and cell.col == 1)
        assert cell.cell_status == "cached_formula"
        assert cell.text == "3"

    def test_pdf_text_and_scan_rejection(self, tmp_path):
        pytest.importorskip("pdfplumber")

        text_pdf = tmp_path / "text.pdf"
        text_pdf.write_bytes(_pdf_bytes(["Hello PDF text extraction works"]))
        document = parse_document_file(
            str(text_pdf), ParseOptions(pdf_min_text_chars_per_page=1)
        )
        assert document.source.media_type == "pdf"
        assert document.source.page_count == 1
        assert any(block.page == 1 and "Hello" in block.text for block in document.blocks)

        blank_pdf = tmp_path / "blank.pdf"
        blank_pdf.write_bytes(_pdf_bytes())
        with pytest.raises(DocumentParseError) as exc:
            parse_document_file(str(blank_pdf), ParseOptions(pdf_min_text_chars_per_page=1))
        assert exc.value.code == "SCANNED_PDF_REQUIRES_OCR"

    def test_pdf_signature_and_none_bbox(self, tmp_path):
        pytest.importorskip("pdfplumber")
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf")
        with pytest.raises(DocumentParseError) as exc:
            parse_document_file(str(bad), ParseOptions())
        assert exc.value.code == "UNSUPPORTED_FORMAT"

        class FakeTable:
            bbox = None

            def extract(self):
                return [["a", "b"], ["c", "d"]]

        block = _pdf_table_block(FakeTable(), 1, 100.0, 100.0, "b-000001", ParseOptions())
        assert block is not None
        assert block.bbox is None

    def test_pdf_all_empty_table_is_not_emitted(self):
        class FakeEmptyTable:
            bbox = None

            def extract(self):
                return [["", ""]]

        block = _pdf_table_block(
            FakeEmptyTable(), 1, 100.0, 100.0, "b-000001", ParseOptions(max_table_cells=1)
        )
        assert block is None

    def test_pdf_rotated_text_only_table_is_rejected(self):
        class FakePage:
            chars = [{"x0": 10.0, "top": 10.0, "upright": False}]

        class FakeTable:
            bbox = (0.0, 0.0, 20.0, 20.0)

        assert _pdf_table_has_upright_text(FakePage(), FakeTable()) is False
        FakePage.chars = [
            {"x0": 10.0, "top": 10.0, "upright": True},
            *[{"x0": 10.0, "top": 10.0, "upright": False}] * 4,
        ]
        assert _pdf_table_has_upright_text(FakePage(), FakeTable()) is False
        FakePage.chars = [{"x0": 10.0, "top": 10.0, "upright": True}]
        assert _pdf_table_has_upright_text(FakePage(), FakeTable()) is True

    def test_pdf_tight_word_spacing_is_preserved(self, tmp_path):
        pytest.importorskip("pdfplumber")

        path = tmp_path / "tight.pdf"
        path.write_bytes(
            _pdf_bytes(raw_content=b'BT /F1 12 Tf 72 720 Td [(Hello) -240 (world)] TJ ET')
        )
        document = parse_document_file(
            str(path), ParseOptions(pdf_min_text_chars_per_page=1)
        )
        assert any(block.text == "Hello world" for block in document.blocks)
        assert not any(block.text == "Helloworld" for block in document.blocks)

    def test_pdf_words_inside_table_are_not_emitted_as_paragraphs(self):
        class FakePage:
            def extract_words(self, **kwargs):
                return [
                    {
                        "text": "cell",
                        "x0": 12.0,
                        "x1": 30.0,
                        "top": 12.0,
                        "bottom": 20.0,
                    },
                    {
                        "text": "caption",
                        "x0": 60.0,
                        "x1": 90.0,
                        "top": 12.0,
                        "bottom": 20.0,
                    },
                ]

        blocks = _pdf_line_blocks(
            FakePage(),
            1,
            100.0,
            100.0,
            1,
            [(10.0, 10.0, 50.0, 50.0)],
        )
        assert [block.text for block in blocks] == ["caption"]


class TestSecurityAndPaths:
    def test_ooxml_macro_and_zip_escape_rejected(self, tmp_path):
        macro = tmp_path / "macro.docx"
        _write_zip(
            macro,
            {
                "[Content_Types].xml": (
                    b'<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                    b'<Override PartName="/word/document.xml" '
                    b'ContentType="application/vnd.ms-word.document.macroEnabled.main+xml"/></Types>'
                ),
            },
        )
        with pytest.raises(DocumentParseError) as exc:
            parse_document_file(str(macro), ParseOptions())
        assert exc.value.code == "MACRO_CONTENT_REJECTED"

        traversal = tmp_path / "traversal.docx"
        _write_zip(traversal, {"../evil.xml": b"<x/>"})
        with pytest.raises(DocumentParseError) as exc:
            precheck_ooxml(str(traversal))
        assert exc.value.code == "ZIP_LIMIT_EXCEEDED"

        embedded_parent = tmp_path / "embedded-parent.docx"
        _write_zip(embedded_parent, {"word/../evil.xml": b"<x/>"})
        with pytest.raises(DocumentParseError) as exc:
            precheck_ooxml(str(embedded_parent))
        assert exc.value.code == "ZIP_LIMIT_EXCEEDED"

    def test_ooxml_dtd_rejected(self, tmp_path):
        path = tmp_path / "dtd.docx"
        _write_zip(
            path,
            {
                "[Content_Types].xml": (
                    b'<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                    b'<Override PartName="/word/document.xml" '
                    b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
                ),
                "word/document.xml": b'<!DOCTYPE foo [<!ENTITY x "y">]><document>&x;</document>',
            },
        )
        with pytest.raises(DocumentParseError) as exc:
            precheck_ooxml(str(path))
        assert exc.value.code == "XML_PARSE_LIMIT"

    def test_allowed_roots(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside.xlsx"
        outside.write_bytes(b"x")
        with pytest.raises(DocumentParseError) as exc:
            validate_source_file(str(outside), (str(allowed),))
        assert exc.value.code == "PATH_OUTSIDE_ALLOWED_ROOT"

        rows = parse_inputs(
            [str(outside)],
            {"allowed_roots": [str(allowed)], "on_error": "skip"},
        )
        assert len(rows) == 1
        assert rows[0]["error"]["code"] == "PATH_OUTSIDE_ALLOWED_ROOT"

    def test_discovery_ignores_directory_non_documents(self, tmp_path):
        folder = tmp_path / "folder"
        folder.mkdir()
        (folder / "ignore.txt").write_text("x", encoding="utf-8")
        assert discover_inputs([str(folder)], (str(folder),)) == []

    def test_timeout_terminates_parser_child(self, tmp_path):
        path = tmp_path / "slow.xlsx"
        path.write_bytes(b"not parsed before timeout")
        with pytest.raises(DocumentParseError) as exc:
            parse_in_subprocess(str(path), ParseOptions(parse_timeout_s=0.000001))
        assert exc.value.code == "PARSE_TIMEOUT"

    def test_skip_does_not_downgrade_request_level_dependency_error(
        self, tmp_path, monkeypatch
    ):
        import curator.operations.loading.document_parsing as module

        source = tmp_path / "input.xlsx"
        source.write_bytes(b"fixture")

        def fail(*args, **kwargs):
            raise DocumentParseError(
                "DEPENDENCY_MISSING", "install the docs extra", stage="validate"
            )

        monkeypatch.setattr(module, "parse_in_subprocess", fail)
        with pytest.raises(DocumentParseError) as exc:
            parse_inputs(
                [str(source)],
                {"allowed_roots": [str(tmp_path)], "on_error": "skip"},
            )
        assert exc.value.code == "DEPENDENCY_MISSING"


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _op_request(tmp_path: Path, rows: list[dict], params: dict, output_dir: str | None) -> dict:
    manifest = _write_manifest(tmp_path / "manifest.jsonl", rows)
    return {
        "schemaVersion": 1,
        "op": "parse_document",
        "params": params,
        "dataPath": str(manifest),
        "outputDir": output_dir,
    }


def _run_op(request: dict) -> tuple[int, str, str]:
    pytest.importorskip("ray")
    from curator.trusted_worker.op.protocol import handle

    stdout, stderr = io.StringIO(), io.StringIO()
    code = handle(json.dumps(request), stdout, stderr)
    return code, stdout.getvalue(), stderr.getvalue()


class TestOpMode:
    def test_parse_document_inline_and_metrics(self, tmp_path):
        pytest.importorskip("openpyxl")
        _write_xlsx(tmp_path / "good.xlsx")
        request = _op_request(
            tmp_path,
            [{"sourcePath": "good.xlsx", "ignored": True}],
            {"allowed_roots": [str(tmp_path)]},
            None,
        )
        code, out, err = _run_op(request)
        assert code == 0, err
        response = json.loads(out)
        assert response["op"] == "parse_document"
        assert response["metrics"] == {
            "input_count": 1.0,
            "document_count": 1.0,
            "error_count": 0.0,
        }
        row = response["items"][0]
        assert set(row) == {"input_index", "doc_id", "parsed_doc", "error"}
        assert row["parsed_doc"]["schema_version"] == "parsed-doc-v2"
        assert str(tmp_path) not in out

    def test_artifact_and_skip_errors(self, tmp_path):
        pytest.importorskip("openpyxl")
        _write_xlsx(tmp_path / "good.xlsx")
        (tmp_path / "bad.docx").write_bytes(b"not a docx")
        out_dir = tmp_path / "out"
        request = _op_request(
            tmp_path,
            [{"sourcePath": "bad.docx"}, {"sourcePath": "good.xlsx"}],
            {"allowed_roots": [str(tmp_path)], "on_error": "skip"},
            str(out_dir),
        )
        code, out, err = _run_op(request)
        assert code == 0, err
        response = json.loads(out)
        assert response["artifacts"] == [{"relativePath": "parsed-documents.jsonl"}]
        assert response["metrics"] == {
            "input_count": 2.0,
            "document_count": 1.0,
            "error_count": 1.0,
        }
        lines = (out_dir / "parsed-documents.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["error"]["code"] == "CORRUPT_DOCUMENT"

    def test_raise_does_not_write_artifact_or_stdout(self, tmp_path):
        (tmp_path / "bad.docx").write_bytes(b"not a docx")
        out_dir = tmp_path / "out"
        request = _op_request(
            tmp_path,
            [{"sourcePath": "bad.docx"}],
            {"allowed_roots": [str(tmp_path)]},
            str(out_dir),
        )
        code, out, err = _run_op(request)
        assert code == 1
        assert out == ""
        assert json.loads(err.strip().splitlines()[-1])["code"] == "CORRUPT_DOCUMENT"
        assert not (out_dir / "parsed-documents.jsonl").exists()

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"allowed_roots": [""]},
            {"allowed_roots": ["{tmp_path}"] , "data_paths": ["x"]},
        ],
    )
    def test_request_level_validation_exits_2(self, tmp_path, params):
        request = _op_request(
            tmp_path,
            [{"sourcePath": "missing.xlsx"}],
            {key: (str(tmp_path) if value == "{tmp_path}" else value) for key, value in params.items()},
            None,
        )
        code, out, _ = _run_op(request)
        assert code == 2
        assert out == ""

    def test_empty_source_path_exits_2(self, tmp_path):
        request = _op_request(
            tmp_path,
            [{"sourcePath": ""}],
            {"allowed_roots": [str(tmp_path)]},
            None,
        )
        code, out, _ = _run_op(request)
        assert code == 2
        assert out == ""

    def test_inline_limit_requires_artifact(self, tmp_path, monkeypatch):
        pytest.importorskip("openpyxl")
        _write_xlsx(tmp_path / "good.xlsx")
        import curator.trusted_worker.ops.parse_document as parse_document_module

        monkeypatch.setattr(parse_document_module, "MAX_INLINE_BYTES", 1)
        request = _op_request(
            tmp_path,
            [{"sourcePath": "good.xlsx"}],
            {"allowed_roots": [str(tmp_path)]},
            None,
        )
        code, out, err = _run_op(request)
        assert code == 1
        assert out == ""
        assert json.loads(err.strip().splitlines()[-1])["code"] == "OUTPUT_TOO_LARGE"

    def test_registries_keep_evaluation_surface_frozen(self):
        pytest.importorskip("ray")
        from curator.core.operation_registry import OperationRegistry
        from curator.trusted_worker.op.registry import OP_MODE_REGISTRY
        from curator.trusted_worker.registry import OP_REGISTRY

        assert OP_MODE_REGISTRY["parse_document"]["required_fields"] == {"sourcePath": str}
        assert "parse_document" not in OP_REGISTRY
        OperationRegistry._ensure_initialized()
        operation_class = OperationRegistry.get_operation_class("document_parsing")
        assert operation_class is not None
        instance = OperationRegistry.create_operation("document_parsing", {"data_paths": []})
        assert instance.__class__ is operation_class
