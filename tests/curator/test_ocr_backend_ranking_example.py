"""Contract tests for the real OCR backend ranking example artifacts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PIL")
pytest.importorskip("ray")

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from curator.operations.loading.ocr_benchmark import load_benchmark_manifest
from curator.operations.loading.ocr_ranking import parse_ranking_document
from curator.operations.loading.parsed_document import ParsedDocument


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "curator"
    / "single_ops"
    / "loading"
    / "ocr_backend_ranking_example.py"
)
ASSET_DIR = EXAMPLE_PATH.parent / "assets" / "ocr_ranking"
MANIFEST_PATH = ASSET_DIR / "manifest.jsonl"
RANKING_PATH = ASSET_DIR / "ocr_backend_ranking.yaml"
AUTO_OUTPUT_PATH = ASSET_DIR / "ocr-auto-route.jsonl"


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "ocr_backend_ranking_example_under_test", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ranking_table():
    return parse_ranking_document(
        yaml.safe_load(RANKING_PATH.read_text(encoding="utf-8"))
    )


def test_manifest_has_three_distinct_annotated_documents():
    cases = load_benchmark_manifest(
        MANIFEST_PATH, allowed_roots=[str(ASSET_DIR)]
    )

    assert len(cases) == 3
    assert len({case.image_path for case in cases}) == 3
    assert sum(len(case.texts) for case in cases) == 18
    for case in cases:
        assert Path(case.image_path).is_file()
        assert len(case.texts) == len(case.boxes) == len(case.block_types)
        assert all(box[0] < box[2] and box[1] < box[3] for box in case.boxes)


def test_ranking_artifact_contains_real_mindocr_and_paddleocr_rows():
    table = _ranking_table()
    rows = {row.backend: row for row in table.rows}

    assert table.ranking_version == "ocr-ranking-demo-v1"
    assert table.evaluation_set_version == "ocr-ranking-demo-en-v1"
    assert set(rows) == {"mindocr", "paddleocr"}
    assert rows["mindocr"].model_id == (
        "mindocr-official-dbnet_mobilenetv3-crnn_resnet34-"
        "crnn_resnet34_ch-ppocr_keys_v1-v1"
    )
    assert rows["paddleocr"].model_id == "paddleocr-official-ppocr-default-v1"
    assert rows["paddleocr"].cer < rows["mindocr"].cer
    assert rows["paddleocr"].line_accuracy > rows["mindocr"].line_accuracy
    assert rows["paddleocr"].layout_f1 > rows["mindocr"].layout_f1
    assert rows["paddleocr"].throughput_pages_s > 0
    assert rows["mindocr"].throughput_pages_s > 0


def test_auto_route_artifact_uses_ranking_and_selects_best_backend():
    table = _ranking_table()
    expected = min(
        table.rows,
        key=lambda row: (
            row.cer,
            -row.line_accuracy,
            -row.layout_f1,
            -row.throughput_pages_s,
            row.backend,
        ),
    )
    row = json.loads(AUTO_OUTPUT_PATH.read_text(encoding="utf-8"))
    assert row["error"] is None

    document = ParsedDocument.model_validate(row["parsed_doc"])
    provenance = document.provenance
    assert provenance.routing_reason == "ranking"
    assert provenance.backend == expected.backend == "paddleocr"
    assert provenance.backend_version == expected.backend_version
    assert provenance.model_id == expected.model_id
    assert provenance.ranking_version == table.ranking_version
    assert provenance.ranking_evaluation_set == table.evaluation_set_version
    assert document.source.sha256 == hashlib.sha256(
        (ASSET_DIR / "invoice.png").read_bytes()
    ).hexdigest()
    assert document.blocks


def test_create_sample_assets_is_reproducible(tmp_path):
    module = _load_example()
    manifest = module.create_sample_assets(tmp_path)
    cases = load_benchmark_manifest(manifest, allowed_roots=[str(tmp_path)])

    assert len(cases) == 3
    assert (tmp_path / "invoice.png").is_file()
    assert (tmp_path / "receipt.png").is_file()
    assert (tmp_path / "report.png").is_file()
    assert manifest.read_text(encoding="utf-8") == MANIFEST_PATH.read_text(
        encoding="utf-8"
    )


def test_dry_run_does_not_load_models(tmp_path, capsys):
    module = _load_example()
    args = argparse.Namespace(
        manifest=MANIFEST_PATH,
        output_dir=tmp_path,
        ranking_output=None,
        backend=["paddleocr"],
        device="cpu",
        lang="en",
        no_layout_analysis=False,
        allow_model_download=True,
        ranking_version=module.DEFAULT_RANKING_VERSION,
        evaluation_set_version=module.DEFAULT_EVALUATION_SET_VERSION,
        merge=False,
        route_image=None,
        no_route=False,
        route_only=False,
        dry_run=True,
    )

    assert module.run_example(args) == 0
    output = capsys.readouterr().out
    assert "cases: 3" in output
    assert "invoice.png (8 lines)" in output
