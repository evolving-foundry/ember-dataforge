"""Generate deterministic document fixtures and canonical golden snapshots.

Run from the repository root:

    uv run --isolated --no-project --python 3.13 \
        --index-url https://pypi.org/simple \
        --with reportlab --with python-docx --with openpyxl \
        --with pdfplumber --with pydantic --with pyyaml \
        python tests/golden/document_parsing/generate_fixtures.py --generate

Then, with the repository test environment installed:

    python tests/golden/document_parsing/generate_fixtures.py --snapshots
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ROOT / "files"
EXPECTED = ROOT / "expected"
FIXED_TIMESTAMP = dt.datetime(2026, 1, 1, 0, 0, 0)


def _ensure_dirs() -> None:
    for name in ("docx", "xlsx", "pdf"):
        (FILES / name).mkdir(parents=True, exist_ok=True)
        (EXPECTED / name).mkdir(parents=True, exist_ok=True)


def _normalize_zip(path: Path) -> None:
    """Rewrite OOXML containers with fixed metadata and entry order."""
    with zipfile.ZipFile(path, "r") as source:
        entries = [(info.filename, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(path, "w") as target:
        for filename, data in sorted(entries):
            if filename == "docProps/core.xml":
                text = data.decode("utf-8")
                for tag in ("created", "modified"):
                    text = re.sub(
                        rf"(<dcterms:{tag}\b[^>]*>).*?(</dcterms:{tag}>)",
                        rf"\g<1>2026-01-01T00:00:00Z\g<2>",
                        text,
                    )
                data = text.encode("utf-8")
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0o600 << 16
            target.writestr(info, data)


def _docx_properties(document) -> None:
    props = document.core_properties
    props.author = "Curator"
    props.created = FIXED_TIMESTAMP
    props.modified = FIXED_TIMESTAMP
    props.last_modified_by = "Curator"
    props.revision = 1
    props.title = "Curator deterministic fixture"
    props.comments = ""


def _save_docx(document, name: str) -> None:
    path = FILES / "docx" / name
    _docx_properties(document)
    document.save(str(path))
    _normalize_zip(path)


def _generate_docx() -> None:
    from docx import Document

    document = Document()
    document.add_paragraph("项目验收报告")
    document.add_paragraph("这是第一段中文内容，包含足够多的中文字符以触发语言检测。")
    document.add_paragraph("第二段补充说明：所有字段均来自可复现的 golden 样本。")
    _save_docx(document, "chinese.docx")

    document = Document()
    document.add_paragraph("设备清单")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "设备名称"
    table.cell(0, 1).text = "状态"
    table.cell(1, 0).text = "训练服务器"
    table.cell(1, 1).text = "已验收"
    _save_docx(document, "table.docx")

    document = Document()
    document.add_paragraph("第一页内容：本文档包含多个页面，用于验证分页文本输出。")
    document.add_page_break()
    document.add_paragraph("第二页内容：分页之后的中文段落仍然需要保持阅读顺序。")
    _save_docx(document, "multipage.docx")

    document = Document()
    _save_docx(document, "empty.docx")


def _xlsx_properties(workbook) -> None:
    props = workbook.properties
    props.creator = "Curator"
    props.lastModifiedBy = "Curator"
    props.created = FIXED_TIMESTAMP
    props.modified = FIXED_TIMESTAMP
    props.title = "Curator deterministic fixture"


def _save_xlsx(workbook, name: str) -> None:
    path = FILES / "xlsx" / name
    _xlsx_properties(workbook)
    workbook.save(str(path))
    _normalize_zip(path)


def _generate_xlsx() -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "订单"
    sheet["A1"] = "订单编号"
    sheet["B1"] = "状态"
    sheet["A2"] = "GD-2026-0011"
    sheet["B2"] = "已完成"
    sheet["A3"] = "备注"
    sheet["B3"] = "中文内容需要保持稳定输出"
    _save_xlsx(workbook, "chinese.xlsx")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "合并表"
    sheet["A1"] = "设备"
    sheet.merge_cells("A1:B1")
    sheet["A2"] = "服务器"
    sheet["B2"] = "就绪"
    sheet["A3"] = "交换机"
    sheet["B3"] = "就绪"
    _save_xlsx(workbook, "table.xlsx")

    workbook = Workbook()
    first = workbook.active
    first.title = "总览"
    first["A1"] = "多工作表样本"
    first["A2"] = 1
    second = workbook.create_sheet("明细")
    second["A1"] = "条目"
    second["A2"] = "第二条内容"
    third = workbook.create_sheet("归档")
    third["A1"] = "归档数据"
    second.sheet_state = "hidden"
    third.sheet_state = "veryHidden"
    _save_xlsx(workbook, "multisheet.xlsx")

    workbook = Workbook()
    workbook.active.title = "空表"
    _save_xlsx(workbook, "empty.xlsx")


def _pdf_canvas(path: Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    value = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    value.setTitle("Curator deterministic fixture")
    value.setAuthor("Curator")
    value.setSubject("document-parsing-golden")
    return value


def _pdf_text(value, x: float, y: float, text: str, size: int = 13) -> None:
    value.setFont("STSong-Light", size)
    value.drawString(x, y, text)


def _generate_pdf() -> None:
    path = FILES / "pdf" / "chinese.pdf"
    value = _pdf_canvas(path)
    _pdf_text(value, 72, 760, "项目验收报告")
    _pdf_text(value, 72, 736, "这是一个中文测试文档，用于验证 PDF 文本抽取和语言检测。")
    _pdf_text(value, 72, 712, "所有内容均来自可复现的 golden 样本。")
    value.showPage()
    value.save()

    path = FILES / "pdf" / "table.pdf"
    value = _pdf_canvas(path)
    _pdf_text(value, 72, 790, "设备清单", 16)
    left, right = 72, 420
    top, bottom = 760, 680
    row_height = 40
    value.setLineWidth(1)
    value.rect(left, bottom, right - left, top - bottom)
    value.line(left, top - row_height, right, top - row_height)
    value.line(left + 180, bottom, left + 180, top)
    _pdf_text(value, left + 12, top - 26, "设备名称", 12)
    _pdf_text(value, left + 192, top - 26, "状态", 12)
    _pdf_text(value, left + 12, top - row_height - 26, "训练服务器", 12)
    _pdf_text(value, left + 192, top - row_height - 26, "已验收", 12)
    value.showPage()
    value.save()

    path = FILES / "pdf" / "multipage.pdf"
    value = _pdf_canvas(path)
    _pdf_text(value, 72, 760, "第一页：多页 PDF 需要保留页码和阅读顺序。")
    value.showPage()
    _pdf_text(value, 72, 760, "第二页：第二页的中文段落也必须完整输出。")
    value.showPage()
    value.save()

    path = FILES / "pdf" / "empty.pdf"
    value = _pdf_canvas(path)
    value.showPage()
    value.save()


def generate() -> None:
    _ensure_dirs()
    _generate_docx()
    _generate_xlsx()
    _generate_pdf()


def _normalized_document(document: dict) -> dict:
    provenance = document["provenance"]
    provenance["extractor_version"] = "<version>"
    provenance["op_version"] = "<version>"
    return document


def _write_snapshot(relative: Path, payload: dict) -> None:
    target = EXPECTED / relative.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")


def snapshots() -> None:
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.ModuleType("torch")
    from curator.operations.loading.document_common import DocumentParseError, ParseOptions
    from curator.operations.loading.document_parsing import parse_document_file

    for path in sorted(FILES.rglob("*")):
        if path.suffix.lower() not in {".docx", ".xlsx", ".pdf"}:
            continue
        relative = path.relative_to(FILES)
        try:
            document = parse_document_file(str(path), ParseOptions())
        except DocumentParseError as exc:
            payload = {
                "fixture": relative.as_posix(),
                "result": {
                    "ok": False,
                    "error": {
                        "code": exc.code,
                        "stage": exc.stage,
                        "message": exc.message,
                        "retryable": exc.retryable,
                        "doc_id": exc.doc_id,
                    },
                },
            }
        else:
            payload = {
                "fixture": relative.as_posix(),
                "result": {
                    "ok": True,
                    "document": _normalized_document(
                        document.model_dump(mode="json", exclude_none=False)
                    ),
                },
            }
        _write_snapshot(relative, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="create binary fixtures")
    parser.add_argument("--snapshots", action="store_true", help="parse fixtures and write snapshots")
    args = parser.parse_args()
    if not args.generate and not args.snapshots:
        parser.error("choose --generate and/or --snapshots")
    if args.generate:
        generate()
    if args.snapshots:
        snapshots()


if __name__ == "__main__":
    main()
