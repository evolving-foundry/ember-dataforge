"""Source-import tests for mixed document and image OCR inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from data_designer.pipelines.envsynth import seed_import
from data_designer.pipelines.envsynth.cli import source_import
from data_designer.pipelines.envsynth.cli.common import WorkerError
from tests.envsynth import env_fixtures


REPO_ROOT = Path(__file__).resolve().parents[2]
TICKETS = (("GD-2026-4001", "Inspection", "approved"),)


def _image_row(sha256: str, *, media_type: str = "png") -> Dict[str, Any]:
    doc_id = "pd-" + sha256[:16]
    lines = (
        "ticket id: GD-2026-4002",
        "title: OCRInspection",
        "status: approved",
    )
    blocks = []
    for index, line in enumerate(lines, start=1):
        y0 = 0.05 + (index - 1) * 0.12
        blocks.append(
            {
                "block_id": "b-{:06d}".format(index),
                "type": "paragraph",
                "page": 1,
                "sheet": None,
                "bbox": [0.05, y0, 0.9, y0 + 0.08],
                "text": line,
                "table": None,
                "kv_pair": None,
                "style": None,
                "ocr_confidence": 0.99,
            }
        )
    document = {
        "schema_version": "parsed-doc-v2",
        "doc_id": doc_id,
        "source": {
            "media_type": media_type,
            "sha256": sha256,
            "byte_size": 128,
            "page_count": 1,
        },
        "language": "en",
        "provenance": {
            "extractor": "paddleocr",
            "extractor_version": "3.7.0",
            "operation": "image_ocr",
            "op_version": "0.2.0",
            "reading_order_version": "ro-v1",
            "routing_reason": "explicit_backend",
            "ranking_version": None,
            "ranking_evaluation_set": None,
            "backend": "paddleocr",
            "backend_version": "3.7.0",
            "device": "cpu",
            "model_id": "paddleocr-official-ppocr-default-v1",
        },
        "blocks": blocks,
    }
    return {
        "input_index": 0,
        "doc_id": doc_id,
        "parsed_doc": document,
        "error": None,
    }


def test_image_files_are_inventoried_with_media_types(tmp_path):
    source_image = tmp_path / "scan.png"
    regulation_image = tmp_path / "policy.jpeg"
    source_image.write_bytes(b"png")
    regulation_image.write_bytes(b"jpg")

    records = seed_import.discover_source_files(
        [str(source_image)], regulation_paths=[str(regulation_image)]
    )

    assert {
        item["relativeRef"]: (item["mediaType"], item["role"])
        for item in records
    } == {
        "scan.png": ("png", "seed"),
        "policy.jpeg": ("jpg", "regulation"),
    }


def test_mixed_documents_and_images_are_routed_and_merged(tmp_path, monkeypatch):
    xlsx_path = tmp_path / "seed.xlsx"
    png_path = tmp_path / "scan.png"
    xlsx_path.write_bytes(b"xlsx")
    png_path.write_bytes(b"png")

    files = seed_import.discover_source_files([str(xlsx_path), str(png_path)])
    by_type = {item["mediaType"]: item for item in files}
    document_row = env_fixtures.parsed_row(
        TICKETS,
        sha256=by_type["xlsx"]["sha256"],
        media_type="xlsx",
    )
    image_row = _image_row(by_type["png"]["sha256"])
    calls: Dict[str, Any] = {}

    def fake_parse_documents(paths, *, allowed_roots, parse_options):
        calls["documents"] = {
            "paths": list(paths),
            "allowed_roots": list(allowed_roots),
            "parse_options": parse_options,
        }
        return [document_row]

    def fake_parse_images(paths, *, allowed_roots, ocr_options):
        calls["images"] = {
            "paths": list(paths),
            "allowed_roots": list(allowed_roots),
            "ocr_options": ocr_options,
        }
        return [image_row]

    monkeypatch.setattr(seed_import.curator_contract, "parse_documents", fake_parse_documents)
    monkeypatch.setattr(seed_import.curator_contract, "parse_images", fake_parse_images)

    manifest = seed_import.build_seed_import_from_source(
        [str(xlsx_path), str(png_path)],
        str(tmp_path / "seed-import"),
        parse_options={"extract_style": False},
        ocr_options={
            "backend": "paddleocr",
            "device": "cpu",
            "lang": "en",
        },
    )
    bundle = seed_import.load_seed_bundle(
        str(tmp_path / "seed-import" / "seed_manifest.json")
    )

    assert calls["documents"]["paths"] == [str(by_type["xlsx"]["pathRef"])]
    assert calls["documents"]["parse_options"] == {"extract_style": False}
    assert calls["images"]["paths"] == [str(by_type["png"]["pathRef"])]
    assert calls["images"]["ocr_options"]["device"] == "cpu"
    assert calls["images"]["ocr_options"]["backend"] == "paddleocr"
    assert calls["documents"]["allowed_roots"] == calls["images"]["allowed_roots"]
    assert [row["input_index"] for row in bundle.parsed_rows] == [0, 1]
    assert {row["parsed_doc"]["source"]["media_type"] for row in bundle.parsed_rows} == {
        "xlsx",
        "png",
    }
    assert {item["mediaType"] for item in bundle.files} == {"xlsx", "png"}
    assert {item["media_type"] for item in bundle.domain["documents"]} == {
        "xlsx",
        "png",
    }
    assert manifest["blockingQuestionCount"] == 0
    assert {ticket["ticket_id"] for ticket in bundle.domain["tickets"]} == {
        "GD-2026-4001",
        "GD-2026-4002",
    }


def test_source_input_accepts_and_validates_ocr_options(tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"png")
    payload = {
        "schemaVersion": 1,
        "request": {
            "sourcePaths": [str(image)],
            "ocrOptions": {
                "backend": "paddleocr",
                "device": "cpu",
                "lang": "en",
            },
        },
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = source_import.parse_input(str(input_path))
    assert parsed.ocr_options == {
        "backend": "paddleocr",
        "device": "cpu",
        "lang": "en",
    }

    payload["request"]["ocrOptions"] = ["invalid"]
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkerError, match="ocrOptions"):
        source_import.parse_input(str(input_path))


def test_ocr_paragraph_without_punctuation_maps_ticket():
    row = _image_row('a' * 64)
    for block, text in zip(
        row['parsed_doc']['blocks'],
        ('ticket id gd2026 4002', 'title OCRInspection', 'status approved'),
    ):
        block['text'] = text

    from data_designer.pipelines.envsynth.domains.user_examples_seed_adapter import map_parsed_rows

    domain = map_parsed_rows([row])

    assert domain['tickets'][0]['ticket_id'] == 'gd2026 4002'
    assert domain['tickets'][0]['status'] == 'approved'
    assert domain['questions'] == []
