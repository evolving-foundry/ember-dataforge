"""Format fact extraction for docx/xlsx/pdf."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .document_common import (
    DocumentParseError,
    ParseOptions,
    excel_cell_text,
    join_cell_paragraphs,
    normalize_bbox,
    normalize_text,
    package_version,
)
from .parsed_document import Block, Cell, Sheet, Style, Table

_EXCEL_ERRORS = {
    "#NULL!",
    "#DIV/0!",
    "#VALUE!",
    "#REF!",
    "#NAME?",
    "#NUM!",
    "#N/A",
}

PDF_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "intersection_tolerance": 3,
}
PDF_WORD_X_TOLERANCE = 2
PDF_TABLE_MIN_UPRIGHT_RATIO = 0.8


@dataclass(frozen=True)
class ParseResult:
    blocks: List[Block]
    page_count: Optional[int]
    extractor: str
    extractor_version: str


def _block_id(index: int) -> str:
    return f"b-{index:06d}"


def _docx_style(paragraph: Any, enabled: bool) -> Optional[Style]:
    if not enabled:
        return None
    runs = [run for run in paragraph.runs if normalize_text(run.text or "")]
    if not runs:
        return None
    sizes = {run.font.size.pt for run in runs if run.font.size is not None}
    bolds = {bool(run.bold) for run in runs if run.bold is not None}
    size = sizes.pop() if len(sizes) == 1 else None
    bold = bolds.pop() if len(bolds) == 1 else None
    if size is None and bold is None:
        return None
    return Style(font_size_pt=size, bold=bold)


def _docx_paragraph_block(
    paragraph: Any,
    block_id: str,
    block_type: str,
    enabled_style: bool,
) -> Optional[Block]:
    text = normalize_text(paragraph.text or "")
    if not text:
        return None
    if block_type == "paragraph":
        from docx.oxml.ns import qn

        style_name = (getattr(getattr(paragraph, "style", None), "name", "") or "").lower()
        p_pr = getattr(paragraph._p, "pPr", None)
        has_numbering = p_pr is not None and p_pr.find(qn("w:numPr")) is not None
        if style_name.startswith("list") or has_numbering:
            block_type = "list"
    return Block(
        block_id=block_id,
        type=block_type,
        page=None,
        sheet=None,
        bbox=None,
        text=text,
        table=None,
        kv_pair=None,
        style=_docx_style(paragraph, enabled_style),
        ocr_confidence=None,
    )


def _docx_table_block(table: Any, block_id: str, options: ParseOptions) -> Optional[Block]:
    rows = list(table.rows)
    if not rows:
        return None
    row_cells = [list(row.cells) for row in rows]
    n_rows = len(row_cells)
    n_cols = max((len(cells) for cells in row_cells), default=0)
    if n_rows == 0 or n_cols == 0:
        return None
    if n_rows * n_cols > options.max_table_cells:
        raise DocumentParseError("TABLE_TOO_LARGE", "table cell count exceeds limit", stage="parse")
    groups: Dict[int, List[Tuple[int, int]]] = {}
    objects: Dict[int, Any] = {}
    for row_index, cells in enumerate(row_cells):
        for col_index in range(n_cols):
            cell = cells[col_index] if col_index < len(cells) else cells[-1]
            key = id(cell._tc)
            groups.setdefault(key, []).append((row_index, col_index))
            objects[key] = cell
    cells: List[Cell] = []
    for key, coords in groups.items():
        min_row = min(row for row, _ in coords)
        max_row = max(row for row, _ in coords)
        min_col = min(col for _, col in coords)
        max_col = max(col for _, col in coords)
        expected = {
            (row, col)
            for row in range(min_row, max_row + 1)
            for col in range(min_col, max_col + 1)
        }
        if set(coords) != expected:
            raise DocumentParseError("CORRUPT_DOCUMENT", "docx merged cell is not rectangular", stage="parse")
        anchor = (min_row, min_col)
        text = join_cell_paragraphs([paragraph.text for paragraph in objects[key].paragraphs])
        cells.append(
            Cell(
                row=min_row,
                col=min_col,
                row_span=max_row - min_row + 1,
                col_span=max_col - min_col + 1,
                text=text,
                cell_status="empty" if text == "" else "value",
                bbox=None,
                ocr_confidence=None,
                covered_by=None,
            )
        )
    covered: List[Cell] = []
    for coords in groups.values():
        min_row = min(row for row, _ in coords)
        max_row = max(row for row, _ in coords)
        min_col = min(col for _, col in coords)
        max_col = max(col for _, col in coords)
        anchor = (min_row, min_col)
        for row, col in coords:
            if (row, col) == anchor:
                continue
            covered.append(
                Cell(
                    row=row,
                    col=col,
                    row_span=1,
                    col_span=1,
                    text="",
                    cell_status="covered",
                    bbox=None,
                    ocr_confidence=None,
                    covered_by=[anchor[0], anchor[1]],
                )
            )
    all_cells = sorted(cells + covered, key=lambda item: (item.row, item.col))
    return Block(
        block_id=block_id,
        type="table",
        page=None,
        sheet=None,
        bbox=None,
        text="",
        table=Table(structure="grid", n_rows=n_rows, n_cols=n_cols, cells=all_cells),
        kv_pair=None,
        style=None,
        ocr_confidence=None,
    )


def parse_docx(path: str, options: ParseOptions) -> ParseResult:
    try:
        import docx
        from docx.oxml.ns import qn
        from docx.table import Table as DocxTable
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise DocumentParseError(
            "DEPENDENCY_MISSING", "install the docs extra: pip install 'data-forge[docs]'", stage="validate"
        ) from exc
    try:
        document = docx.Document(path)
    except Exception as exc:
        raise DocumentParseError("CORRUPT_DOCUMENT", "docx cannot be opened", stage="parse") from exc
    blocks: List[Block] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            block = _docx_paragraph_block(Paragraph(child, document), _block_id(len(blocks) + 1), "paragraph", options.extract_style)
        elif tag == "tbl":
            block = _docx_table_block(DocxTable(child, document), _block_id(len(blocks) + 1), options)
        else:
            continue
        if block is not None:
            blocks.append(block)
    seen = set()
    for section in document.sections:
        for part_name, part_type in ((section.header, "header"), (section.footer, "footer")):
            element_id = id(part_name._element)
            if element_id in seen:
                continue
            seen.add(element_id)
            for paragraph in part_name.paragraphs:
                block = _docx_paragraph_block(
                    paragraph, _block_id(len(blocks) + 1), part_type, options.extract_style
                )
                if block is not None:
                    blocks.append(block)
            for table in part_name.tables:
                block = _docx_table_block(table, _block_id(len(blocks) + 1), options)
                if block is not None:
                    blocks.append(block)
    return ParseResult(blocks, None, "python-docx", package_version("python-docx"))


def _is_excel_error(cell: Any, value: Any) -> bool:
    if getattr(cell, "data_type", None) == "e":
        return True
    return isinstance(value, str) and value.strip().upper() in _EXCEL_ERRORS


def _sheet_state(value: str) -> str:
    return {"visible": "visible", "hidden": "hidden", "veryHidden": "very_hidden"}.get(value, "hidden")


def _xlsx_effective_dimensions(sheet: Any) -> Tuple[int, int]:
    """Compute the grid extent only from actual values/formulas and merged regions.

    openpyxl's ``max_row`` / ``max_column`` include empty cells with only styling set;
    treating those values directly as the grid size can falsely trigger ``TABLE_TOO_LARGE``. ``_cells`` is
    the mapping of cells already allocated by openpyxl, avoiding traversal of a virtual rectangle enlarged by styling.
    """
    allocated = getattr(sheet, "_cells", None)
    if isinstance(allocated, dict):
        candidates = (
            ((int(row), int(col)), cell) for (row, col), cell in allocated.items()
        )
    else:
        candidates = (
            ((cell.row, cell.column), cell)
            for row in sheet.iter_rows()
            for cell in row
        )
    max_row = 0
    max_col = 0
    non_empty = False
    for (row, col), cell in candidates:
        if cell.value is not None or getattr(cell, "data_type", None) == "f":
            non_empty = True
            max_row = max(max_row, row)
            max_col = max(max_col, col)
    if not non_empty:
        return 0, 0
    for merged in sheet.merged_cells.ranges:
        max_row = max(max_row, merged.max_row)
        max_col = max(max_col, merged.max_col)
    return max_row, max_col


def _xlsx_table(
    formula_sheet: Any,
    cached_sheet: Any,
    sheet: Sheet,
    options: ParseOptions,
    block_id: str,
) -> Optional[Block]:
    max_row, max_col = _xlsx_effective_dimensions(formula_sheet)
    if max_row <= 0 or max_col <= 0:
        return None
    if max_row * max_col > options.max_table_cells:
        raise DocumentParseError("TABLE_TOO_LARGE", "table cell count exceeds limit", stage="parse")
    covered: Dict[Tuple[int, int], Tuple[int, int]] = {}
    spans: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for merged in formula_sheet.merged_cells.ranges:
        top = (merged.min_row - 1, merged.min_col - 1)
        spans[top] = (merged.max_row - merged.min_row + 1, merged.max_col - merged.min_col + 1)
        for row in range(merged.min_row, merged.max_row + 1):
            for col in range(merged.min_col, merged.max_col + 1):
                if (row - 1, col - 1) != top:
                    covered[(row - 1, col - 1)] = top
    cells: List[Cell] = []
    for row in range(1, max_row + 1):
        for col in range(1, max_col + 1):
            coord = (row - 1, col - 1)
            if coord in covered:
                anchor = covered[coord]
                cells.append(
                    Cell(
                        row=coord[0],
                        col=coord[1],
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
            formula_cell = formula_sheet.cell(row=row, column=col)
            cached_cell = cached_sheet.cell(row=row, column=col)
            is_formula = getattr(formula_cell, "data_type", None) == "f"
            if is_formula:
                cached_value = cached_cell.value
                if cached_value is None:
                    text = ""
                    status = "formula_cache_missing"
                elif _is_excel_error(cached_cell, cached_value):
                    text = normalize_text(str(cached_value))
                    status = "error"
                else:
                    text = excel_cell_text(cached_value)
                    status = "cached_formula"
            elif formula_cell.value is None:
                text = ""
                status = "empty"
            elif _is_excel_error(formula_cell, formula_cell.value):
                text = normalize_text(str(formula_cell.value))
                status = "error"
            else:
                text = excel_cell_text(formula_cell.value)
                status = "value"
            row_span, col_span = spans.get(coord, (1, 1))
            cells.append(
                Cell(
                    row=coord[0],
                    col=coord[1],
                    row_span=row_span,
                    col_span=col_span,
                    text=text,
                    cell_status=status,
                    bbox=None,
                    ocr_confidence=None,
                    covered_by=None,
                )
            )
    cells.sort(key=lambda item: (item.row, item.col))
    return Block(
        block_id=block_id,
        type="table",
        page=None,
        sheet=sheet,
        bbox=None,
        text="",
        table=Table(structure="grid", n_rows=max_row, n_cols=max_col, cells=cells),
        kv_pair=None,
        style=None,
        ocr_confidence=None,
    )


def parse_xlsx(path: str, options: ParseOptions) -> ParseResult:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise DocumentParseError(
            "DEPENDENCY_MISSING", "install the docs extra: pip install 'data-forge[docs]'", stage="validate"
        ) from exc
    formula_book = None
    cached_book = None
    try:
        formula_book = load_workbook(path, data_only=False, read_only=False)
        cached_book = load_workbook(path, data_only=True, read_only=False)
    except Exception as exc:
        if formula_book is not None:
            formula_book.close()
        if cached_book is not None:
            cached_book.close()
        raise DocumentParseError("CORRUPT_DOCUMENT", "xlsx cannot be opened", stage="parse") from exc
    blocks: List[Block] = []
    try:
        for index, name in enumerate(formula_book.sheetnames):
            formula_sheet = formula_book[name]
            cached_sheet = cached_book[name]
            sheet = Sheet(name=name, index=index, state=_sheet_state(formula_sheet.sheet_state))
            block = _xlsx_table(formula_sheet, cached_sheet, sheet, options, _block_id(len(blocks) + 1))
            if block is not None:
                blocks.append(block)
    finally:
        formula_book.close()
        cached_book.close()
    return ParseResult(blocks, None, "openpyxl", package_version("openpyxl"))


def _pdf_word_in_bbox(word: Dict[str, Any], bbox: Sequence[float]) -> bool:
    left, top, right, bottom = bbox
    center_x = (float(word["x0"]) + float(word["x1"])) / 2.0
    center_y = (float(word["top"]) + float(word["bottom"])) / 2.0
    return left <= center_x <= right and top <= center_y <= bottom


def _pdf_line_blocks(
    page: Any,
    page_number: int,
    page_width: float,
    page_height: float,
    start: int,
    table_bboxes: Sequence[Sequence[float]] = (),
) -> List[Block]:
    words = page.extract_words(
        use_text_flow=False,
        keep_blank_chars=False,
        x_tolerance=PDF_WORD_X_TOLERANCE,
    ) or []
    if table_bboxes:
        words = [
            word
            for word in words
            if not any(_pdf_word_in_bbox(word, bbox) for bbox in table_bboxes)
        ]
    lines: Dict[float, List[Dict[str, Any]]] = {}
    for word in words:
        lines.setdefault(round(float(word["top"]), 1), []).append(word)
    blocks: List[Block] = []
    for top in sorted(lines):
        line = sorted(lines[top], key=lambda item: float(item["x0"]))
        text = normalize_text(" ".join(str(item["text"]) for item in line))
        if not text:
            continue
        bbox = normalize_bbox(
            [
                min(float(item["x0"]) for item in line),
                min(float(item["top"]) for item in line),
                max(float(item["x1"]) for item in line),
                max(float(item["bottom"]) for item in line),
            ],
            page_width,
            page_height,
        )
        blocks.append(
            Block(
                block_id=_block_id(start + len(blocks)),
                type="paragraph",
                page=page_number,
                sheet=None,
                bbox=bbox,
                text=text,
                table=None,
                kv_pair=None,
                style=None,
                ocr_confidence=None,
            )
        )
    return blocks


def _pdf_table_block(table: Any, page_number: int, page_width: float, page_height: float, block_id: str, options: ParseOptions) -> Optional[Block]:
    data = table.extract()
    if not data:
        return None
    n_rows = len(data)
    n_cols = max((len(row) for row in data), default=0)
    if n_rows == 0 or n_cols == 0:
        return None
    if not any(normalize_text(cell or "") for row in data for cell in (row or [])):
        return None
    if n_rows * n_cols > options.max_table_cells:
        raise DocumentParseError("TABLE_TOO_LARGE", "table cell count exceeds limit", stage="parse")
    bbox = normalize_bbox(table.bbox, page_width, page_height) if table.bbox is not None else None
    cells: List[Cell] = []
    for row_index in range(n_rows):
        for col_index in range(n_cols):
            values = data[row_index] or []
            raw = values[col_index] if col_index < len(values) else None
            text = normalize_text(raw or "")
            cells.append(
                Cell(
                    row=row_index,
                    col=col_index,
                    row_span=1,
                    col_span=1,
                    text=text,
                    cell_status="empty" if text == "" else "value",
                    bbox=None,
                    ocr_confidence=None,
                    covered_by=None,
                )
            )
    return Block(
        block_id=block_id,
        type="table",
        page=page_number,
        sheet=None,
        bbox=bbox,
        text="",
        table=Table(structure="grid", n_rows=n_rows, n_cols=n_cols, cells=cells),
        kv_pair=None,
        style=None,
        ocr_confidence=None,
    )


def _pdf_sort_position(block: Block) -> Tuple[float, float]:
    if block.bbox is None:
        return 0.0, 0.0
    return block.bbox[1], block.bbox[0]


def _pdf_table_has_upright_text(page: Any, table: Any) -> bool:
    bbox = getattr(table, "bbox", None)
    if bbox is None:
        return True
    left, top, right, bottom = (float(value) for value in bbox)
    upright_count = 0
    char_count = 0
    for char in getattr(page, "chars", []) or []:
        if left <= float(char["x0"]) <= right and top <= float(char["top"]) <= bottom:
            char_count += 1
            if bool(char.get("upright", True)):
                upright_count += 1
    if char_count == 0:
        return False
    return upright_count / char_count >= PDF_TABLE_MIN_UPRIGHT_RATIO


def parse_pdf(path: str, options: ParseOptions) -> ParseResult:
    try:
        import pdfplumber
    except ImportError as exc:
        raise DocumentParseError(
            "DEPENDENCY_MISSING", "install the docs extra: pip install 'data-forge[docs]'", stage="validate"
        ) from exc
    try:
        pdf = pdfplumber.open(path)
    except Exception as exc:
        raise DocumentParseError("CORRUPT_DOCUMENT", "pdf cannot be opened", stage="parse") from exc
    with pdf:
        if getattr(pdf, "is_encrypted", False) or getattr(getattr(pdf, "doc", None), "is_encrypted", False):
            raise DocumentParseError("PDF_ENCRYPTED", "encrypted PDF is not supported", stage="parse")
        pages = list(pdf.pages)
        if len(pages) > options.max_pages:
            raise DocumentParseError("PDF_PAGE_LIMIT", "PDF page count exceeds limit", stage="parse")
        texts: List[str] = []
        for page in pages:
            if page.width > 14400 or page.height > 14400:
                raise DocumentParseError("PDF_PAGE_TOO_LARGE", "PDF page dimensions exceed limit", stage="parse")
            texts.append(page.extract_text() or "")
        if sum(len(text) for text in texts) < max(1, len(pages) * options.pdf_min_text_chars_per_page):
            raise DocumentParseError(
                "SCANNED_PDF_REQUIRES_OCR",
                "PDF has too little extractable text; use image_ocr explicitly",
                stage="parse",
            )
        ordered: List[Tuple[int, float, float, Block]] = []
        block_number = 1
        for page_number, page in enumerate(pages, start=1):
            table_blocks: List[Block] = []
            table_bboxes: List[Sequence[float]] = []
            if hasattr(page, "find_tables"):
                for table in page.find_tables(table_settings=PDF_TABLE_SETTINGS):
                    if not _pdf_table_has_upright_text(page, table):
                        continue
                    block = _pdf_table_block(
                        table,
                        page_number,
                        page.width,
                        page.height,
                        _block_id(block_number + len(table_blocks)),
                        options,
                    )
                    if block is not None:
                        table_blocks.append(block)
                        if table.bbox is not None:
                            table_bboxes.append(tuple(float(value) for value in table.bbox))
            for block in _pdf_line_blocks(
                page,
                page_number,
                page.width,
                page.height,
                block_number,
                table_bboxes,
            ):
                top, left = _pdf_sort_position(block)
                ordered.append((page_number, top, left, block))
                block_number += 1
            for block in table_blocks:
                top, left = _pdf_sort_position(block)
                ordered.append((page_number, top, left, block))
                block_number += 1
        ordered.sort(key=lambda item: (item[0], item[1], item[2]))
        blocks = [block.model_copy(update={"block_id": _block_id(index)}) for index, (_, _, _, block) in enumerate(ordered, start=1)]
    return ParseResult(blocks, len(pages), "pdfplumber", package_version("pdfplumber"))
