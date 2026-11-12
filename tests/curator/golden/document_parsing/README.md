# Document Parsing Golden Fixtures

本目录固定 `parsed-doc-v2` 的格式层回归样本。

## 矩阵

每种格式各 4 份：

| 格式 | 中文 | 表格 | 多页/多 sheet | 空内容 |
|---|---|---|---|---|
| docx | `chinese.docx` | `table.docx` | `multipage.docx` | `empty.docx` |
| xlsx | `chinese.xlsx` | `table.xlsx` | `multisheet.xlsx` | `empty.xlsx` |
| pdf | `chinese.pdf` | `table.pdf` | `multipage.pdf` | `empty.pdf` |

PDF 空内容按解析契约返回 `SCANNED_PDF_REQUIRES_OCR`，快照记录该错误而不是成功文档。

## 快照

`expected/<format>/<name>.json` 由 `parse_document_file` 生成，比较前仅把
`provenance.extractor_version` 和 `provenance.op_version` 归一化为 `<version>`。
其余字段（包括 `sha256`、`doc_id`、文本、bbox 和表格坐标）均逐字段比较。

## 重新生成

二进制样本需要 `reportlab`、`python-docx`、`openpyxl`：

```bash
uv run --isolated --no-project --python 3.13 \
  --index-url https://pypi.org/simple \
  --with reportlab --with python-docx --with openpyxl \
  python tests/golden/document_parsing/generate_fixtures.py --generate
```

生成后，在已安装 docs 依赖的 Python 环境中重写快照：

```bash
PYTHONPATH=. python tests/golden/document_parsing/generate_fixtures.py --snapshots
```

生成脚本固定 OOXML ZIP 条目顺序/时间戳和 PDF invariant 时间戳；重复生成应保持
SHA256 不变。若有意更新样本，必须同时提交二进制文件与全部受影响快照。