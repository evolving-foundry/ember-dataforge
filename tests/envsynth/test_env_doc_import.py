"""End-to-end source import for real docx seed + pdf regulation files.

This test intentionally exercises Curator's document parser instead of the
hand-built C-IR-2 fixtures.  It has no OCR/NPU dependency: docx uses
``python-docx`` and pdf uses text extraction through ``pdfplumber``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from data_designer.pipelines.envsynth import artifacts, gates, package, seed_import
from data_designer.pipelines.envsynth import synthesizer


REPO_ROOT = Path(__file__).resolve().parents[2]


TITLE = "\u7535\u529b\u5de5\u5355\u53f0\u8d26"
HEADERS = ("\u5de5\u5355\u7f16\u53f7", "\u6807\u9898", "\u72b6\u6001")
TICKETS = (
    ("GD-2026-1001", "\u7ebf\u8def\u68c0\u4fee", "\u5df2\u5ba1\u6279"),
    ("GD-2026-1002", "\u7f3a\u9677\u5904\u7406", "\u5ba1\u6838\u4e2d"),
)


def _require_document_runtime() -> None:
    missing = [
        module
        for module in ("docx", "pdfplumber", "openpyxl", "jsonschema")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise AssertionError(
            "Document parsing test dependencies are missing: {}".format(", ".join(missing))
        )
    try:
        from curator.operations.loading.document_parsing import (  # noqa: F401
            parse_inputs,
        )
    except Exception as error:  # optional Curator runtime/deps may be absent
        raise AssertionError(
            "Curator document parser unavailable: {}".format(error)
        ) from error


def _write_docx(path) -> None:
    _require_document_runtime()
    docx = importlib.import_module("docx")
    document = docx.Document()
    document.add_paragraph(TITLE)
    table = document.add_table(rows=len(TICKETS) + 1, cols=len(HEADERS))
    for col, text in enumerate(HEADERS):
        table.cell(0, col).text = text
    for row_index, ticket in enumerate(TICKETS, start=1):
        for col, text in enumerate(ticket):
            table.cell(row_index, col).text = text
    document.save(str(path))


def _write_pdf(path) -> None:
    lines = (
        "Work order regulation",
        "Every work order requires two-person approval before closure.",
        "Rejected work orders return to draft status for revision.",
    )
    content = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index:
            content.append("0 -18 Td")
        content.append("({}) Tj".format(escaped))
    content.append("ET")
    stream = ("\n".join(content) + "\n").encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"endstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header = b"%PDF-1.4\n"
    body = b""
    offsets = []
    for number, value in enumerate(objects, start=1):
        offsets.append(len(header) + len(body))
        body += "{} 0 obj\n".format(number).encode("ascii") + value + b"\nendobj\n"
    xref_offset = len(header) + len(body)
    xref = "xref\n0 {}\n0000000000 65535 f \n".format(len(objects) + 1).encode("ascii")
    xref += b"".join(
        "{:010d} 00000 n \n".format(offset).encode("ascii") for offset in offsets
    )
    trailer = (
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n".format(
            len(objects) + 1, xref_offset
        ).encode("ascii")
    )
    path.write_bytes(header + body + xref + trailer)


def _write_xlsx(path) -> None:
    _require_document_runtime()
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "\u5de5\u5355"
    sheet.append(list(HEADERS))
    for ticket in TICKETS:
        sheet.append(list(ticket))
    workbook.save(str(path))


def _statuses(domain):
    return {ticket["ticket_id"]: ticket["status"] for ticket in domain["tickets"]}


def test_regulation_paths_outside_source_paths_are_inventoried(tmp_path):
    docx_path = tmp_path / "seed.docx"
    pdf_path = tmp_path / "regulation.pdf"
    docx_path.write_bytes(b"docx")
    pdf_path.write_bytes(b"pdf")

    records = seed_import.discover_source_files(
        [str(docx_path)], regulation_paths=[str(pdf_path)]
    )
    assert {item["relativeRef"]: item["role"] for item in records} == {
        "seed.docx": "seed",
        "regulation.pdf": "regulation",
    }


def test_xlsx_seed_builds_a_seed_bundle(tmp_path):
    _require_document_runtime()

    xlsx_path = tmp_path / "seed.xlsx"
    _write_xlsx(xlsx_path)

    manifest = seed_import.build_seed_import_from_source(
        [str(xlsx_path)],
        str(tmp_path / "seed-import"),
        anchor_holdout_ratio=0.3,
    )
    bundle = seed_import.load_seed_bundle(
        str(tmp_path / "seed-import" / "seed_manifest.json")
    )

    assert manifest["blockingQuestionCount"] == 0
    assert len(bundle.parsed_rows) == 1
    row = bundle.parsed_rows[0]
    assert row["parsed_doc"]["source"]["media_type"] == "xlsx"
    assert row["parsed_doc"]["source"]["sha256"] == seed_import.sha256_file(xlsx_path)
    assert bundle.files[0]["role"] == "seed"
    assert _statuses(bundle.domain) == {
        "GD-2026-1001": "approved",
        "GD-2026-1002": "reviewing",
    }
    assert bundle.domain["questions"] == []


def test_docx_seed_and_pdf_regulation_build_a_frozen_package(tmp_path):
    _require_document_runtime()

    docx_path = tmp_path / "seed.docx"
    pdf_path = tmp_path / "regulation.pdf"
    _write_docx(docx_path)
    _write_pdf(pdf_path)

    manifest = seed_import.build_seed_import_from_source(
        [str(docx_path)],
        str(tmp_path / "seed-import"),
        regulation_paths=[str(pdf_path)],
        anchor_holdout_ratio=0.3,
    )
    bundle = seed_import.load_seed_bundle(
        str(tmp_path / "seed-import" / "seed_manifest.json")
    )

    assert manifest["blockingQuestionCount"] == 0
    assert len(bundle.parsed_rows) == 2
    media_types = {
        row["parsed_doc"]["source"]["media_type"]
        for row in bundle.parsed_rows
        if row.get("parsed_doc")
    }
    assert media_types == {"docx", "pdf"}

    roles = {item["mediaType"]: item["role"] for item in bundle.files}
    assert roles == {"docx": "seed", "pdf": "regulation"}
    assert _statuses(bundle.domain) == {
        "GD-2026-1001": "approved",
        "GD-2026-1002": "reviewing",
    }
    assert bundle.domain["questions"] == []
    assert {item["role"] for item in bundle.domain["documents"]} == {
        "seed",
        "regulation",
    }
    assert bundle.domain["regulations"]
    assert any(
        "two-person approval" in item["text"].lower()
        for item in bundle.domain["regulations"]
    )

    result = synthesizer.synthesize(bundle.domain, mode="template", seed=42)
    assert result.status == "synthesized"
    env_package = package.build_env_package(
        bundle.domain,
        result.draft,
        ref_id="office-power",
        version=1,
        seed=42,
        anchor_ratio=0.3,
        source_files=bundle.files,
        seed_manifest_sha256=bundle.manifest_sha256,
        expert_review={"mode": "waived", "sampleRate": 0.0},
    )
    report = gates.run_gates(env_package, quality_threshold=0.5)
    assert report["passed"], report

    final = package.finalize_bundle(env_package, report)
    release = package.write_release(str(tmp_path / "release"), final)
    assert release["assetSha256"] == final.asset_sha256
    assert str(tmp_path).encode("utf-8") not in b"".join(
        package.load_env_package(str(tmp_path / "release" / "package"))[1].values()
    )


def test_docx_pdf_cli_chain_reaches_gym_native_package_without_rsi(tmp_path):
    _require_document_runtime()

    docx_path = tmp_path / "seed.docx"
    pdf_path = tmp_path / "regulation.pdf"
    _write_docx(docx_path)
    _write_pdf(pdf_path)

    source_input = tmp_path / "source-input.json"
    source_output = tmp_path / "seed-import"
    source_input.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "request": {
                    "sourcePaths": [str(docx_path)],
                    "regulationPaths": [str(pdf_path)],
                    "family": "power-ticket",
                    "anchorHoldoutRatio": 0.3,
                    "maxQuestions": 20,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    source = subprocess.run(
        [
            sys.executable,
            "-m",
            "data_designer.pipelines.envsynth.cli.source_import",
            str(source_input),
            str(source_output),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert source.returncode == 0, source.stderr
    source_summary = json.loads(source.stdout)

    compile_input = tmp_path / "compile-input.json"
    compile_result = tmp_path / "compile-result.json"
    compile_input.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "request": {
                    "assetRoot": str(tmp_path / "assets"),
                    "refId": "office-power-docx",
                    "family": "power-ticket",
                    "split": "train",
                    "seedManifestPath": source_summary["seedManifestPath"],
                    "seedManifestSha256": source_summary["seedManifestSha256"],
                    "envProfile": "office",
                    "synthesis": {
                        "mode": "template",
                        "maxRounds": 3,
                        "grounding": {"embedModel": "char-ngram-v1", "topK": 4},
                    },
                    "qualityThreshold": 0.5,
                    "anchorHoldoutRatio": 0.3,
                    "expertReview": {"mode": "waived", "sampleRate": 0.0},
                    "seed": 42,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    compiled = subprocess.run(
        [
            sys.executable,
            "-m",
            "data_designer.pipelines.envsynth.cli.env_compile",
            str(compile_input),
            str(compile_result),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = json.loads(compile_result.read_text(encoding="utf-8"))
    assert result["schemaVersion"] == "env-compile-result-v1"
    release_dir = Path(result["envRelease"]["assetDir"]).parent
    release_meta, taskset_payload = artifacts.load_release_metadata(release_dir)
    assert release_meta["source"]["curatorSchema"] == "parsed-doc-v2"
    assert taskset_payload["itemCount"] > 0
    assert taskset_payload["itemCount"] == len(taskset_payload["tasks"])

    gym_root = tmp_path / "gym"
    (gym_root / "overlay").mkdir(parents=True)
    publisher = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "gym_publish_env.py"),
            "--gym-root",
            str(gym_root),
        ],
        cwd=str(REPO_ROOT),
        input=json.dumps(
            {"compileResultPath": str(compile_result), "name": "office-power-docx"},
            ensure_ascii=False,
        ),
        capture_output=True,
        text=True,
    )
    assert publisher.returncode == 0, publisher.stderr
    published = json.loads(publisher.stdout)
    assert published["status"] == "published"
    assert published["runnable"] is True
    assert published["benchmark"] == "synth-office-power-docx"
    assert (gym_root / "overlay" / published["assets"]["packageZip"]).is_file()
    assert (gym_root / "overlay" / published["assets"]["prepareScript"]).is_file()
