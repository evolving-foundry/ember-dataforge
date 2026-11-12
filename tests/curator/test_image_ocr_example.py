"""Contract tests for the runnable Image OCR example.

These tests do not load MindOCR models. They keep the CPU command, dictionary
argument and dry-run output wired to the operation parameters. Real CPU
inference is recorded in the v0.2 design document, section 15.1.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("ray")

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from PIL import Image


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "curator"
    / "single_ops"
    / "loading"
    / "image_ocr_example.py"
)
ASSET_PATH = EXAMPLE_PATH.parent / "assets" / "ocr_sample.png"
OUTPUT_PATH = EXAMPLE_PATH.parent / "assets" / "ocr_sample_output.jsonl"
PADDLE_OUTPUT_PATH = (
    EXAMPLE_PATH.parent / "assets" / "ocr_sample_paddleocr_output.jsonl"
)


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    yield


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "image_ocr_example_under_test", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_path_args(**overrides):
    values = {
        "backend": "mindocr",
        "model_det": None,
        "model_rec": None,
        "model_layout": None,
        "model_rec_dict": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_model_paths_includes_recognition_dictionary(tmp_path):
    module = _load_example()
    dictionary = tmp_path / "ppocr_keys_v1.txt"
    dictionary.write_text("a\n", encoding="utf-8")

    paths = module._build_model_paths(_model_path_args(model_rec_dict=str(dictionary)))

    assert paths == {"rec_dict": str(dictionary.resolve())}


def test_model_paths_require_explicit_backend(tmp_path):
    module = _load_example()

    with pytest.raises(SystemExit):
        module._build_model_paths(
            _model_path_args(backend="auto", model_det=str(tmp_path))
        )


def test_build_params_carries_cpu_and_dictionary(tmp_path):
    module = _load_example()
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    dictionary = tmp_path / "ppocr_keys_v1.txt"
    dictionary.write_text("a\n", encoding="utf-8")
    paths = module._build_model_paths(_model_path_args(model_rec_dict=str(dictionary)))
    args = argparse.Namespace(
        output_key="parsed_doc",
        backend="mindocr",
        device="cpu",
        lang="zh",
        no_layout_analysis=True,
        on_error="raise",
        allow_model_download=True,
        ranking_path=None,
    )

    params = module._build_params(args, image, paths)

    assert params["backend"] == "mindocr"
    assert params["device"] == "cpu"
    assert params["lang"] == "zh"
    assert params["allow_model_download"] is True
    assert params["model_paths"] == {"rec_dict": str(dictionary.resolve())}


def test_cpu_dry_run_prints_runnable_command(tmp_path, capsys):
    module = _load_example()
    image = tmp_path / "sample.png"
    Image.new("RGB", (40, 20), (255, 255, 255)).save(image)
    dictionary = tmp_path / "ppocr_keys_v1.txt"
    dictionary.write_text("a\n", encoding="utf-8")

    code = module.main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(tmp_path / "out"),
            "--backend",
            "mindocr",
            "--device",
            "cpu",
            "--lang",
            "zh",
            "--model-rec-dict",
            str(dictionary),
            "--allow-model-download",
            "--dry-run",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "device: cpu" in output
    assert "CPU command example" in output
    assert "--lang zh" in output
    assert "--backend paddleocr" in output


def test_sample_cpu_output_matches_image_and_contract():
    from curator.operations.loading.parsed_document import ParsedDocument

    payload = OUTPUT_PATH.read_bytes()
    assert payload.endswith(b"\n")
    assert payload.count(b"\n") == 1
    row = json.loads(payload)
    assert row["error"] is None

    document = ParsedDocument.model_validate(row["parsed_doc"])
    assert document.source.sha256 == hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest()
    assert document.provenance.backend == "mindocr"
    assert document.provenance.device == "cpu"
    assert len(document.blocks) == 27
    assert document.blocks[0].text == "agentdesigner"
    assert "ocr-2026-001" in [block.text for block in document.blocks]




def test_paddleocr_sample_cpu_output_matches_image_and_contract():
    from curator.operations.loading.parsed_document import ParsedDocument

    payload = PADDLE_OUTPUT_PATH.read_bytes()
    assert payload.endswith(b"\n")
    assert payload.count(b"\n") == 1
    row = json.loads(payload)
    assert row["error"] is None

    document = ParsedDocument.model_validate(row["parsed_doc"])
    assert document.source.sha256 == hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest()
    assert document.provenance.backend == "paddleocr"
    assert document.provenance.backend_version == "3.7.0"
    assert document.provenance.device == "cpu"
    assert document.provenance.model_id == "paddleocr-official-ppocr-default-v1"
    assert len(document.blocks) == 11
    assert document.blocks[0].text == "DataForge Curator OCR Sample"
    assert "Invoice No: OCR-2026-001" in [block.text for block in document.blocks]
