"""Unit tests for the PaddleOCR runtime helpers that do not load PaddlePaddle."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from curator.operations.loading.ocr_adapters import OcrAdapterError, OcrRequest
from curator.operations.loading.ocr_adapters import (
    paddleocr_runtime as runtime,
)


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    """PaddleOCR runtime helper tests do not need a Ray cluster."""
    yield


def _request(lang: str = "zh", device: str = "cpu", model_paths=None) -> OcrRequest:
    return OcrRequest(
        lang=lang,
        device=device,
        layout_analysis=True,
        model_paths=model_paths or {},
    )


def _model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "inference.yml").write_text("model", encoding="utf-8")
    return path


def test_device_and_language_mapping():
    assert runtime._device_name("cpu") == "cpu"
    assert runtime._device_name("cuda") == "gpu"
    assert runtime._device_name("npu") == "npu"
    assert runtime._language_name("zh") == "ch"
    assert runtime._language_name("mixed") == "ch"
    assert runtime._language_name("en") == "en"
    with pytest.raises(OcrAdapterError) as caught:
        runtime._device_name("tpu")
    assert caught.value.code == "OCR_BACKEND_UNAVAILABLE"


def test_missing_official_models_require_download_permission(tmp_path, monkeypatch):
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path))
    with pytest.raises(OcrAdapterError) as caught:
        runtime._engine_kwargs(_request(), allow_model_download=False)
    assert caught.value.code == "OCR_MODEL_UNAVAILABLE"


def test_cached_official_models_are_used_without_download(tmp_path, monkeypatch):
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path))
    det = _model_dir(tmp_path / "official_models" / runtime._DET_MODEL_NAME)
    rec = _model_dir(tmp_path / "official_models" / runtime._REC_MODEL_NAME)

    kwargs = runtime._engine_kwargs(_request(), allow_model_download=False)

    assert kwargs["text_detection_model_name"] == runtime._DET_MODEL_NAME
    assert kwargs["text_recognition_model_name"] == runtime._REC_MODEL_NAME
    assert kwargs["text_detection_model_dir"] == str(det)
    assert kwargs["text_recognition_model_dir"] == str(rec)
    assert kwargs["device"] == "cpu"
    assert kwargs["enable_mkldnn"] is False


def test_custom_model_directories_are_passed_to_paddleocr(tmp_path):
    det = _model_dir(tmp_path / "det")
    rec = _model_dir(tmp_path / "rec")
    kwargs = runtime._engine_kwargs(
        _request(model_paths={"det": str(det), "rec": str(rec)}),
        allow_model_download=False,
    )
    assert kwargs["text_detection_model_dir"] == str(det)
    assert kwargs["text_recognition_model_dir"] == str(rec)


def test_rec_dict_is_rejected_instead_of_silently_ignored(tmp_path):
    dictionary = tmp_path / "keys.txt"
    dictionary.write_text("a\n", encoding="utf-8")
    with pytest.raises(OcrAdapterError) as caught:
        runtime._engine_kwargs(
            _request(model_paths={"rec_dict": str(dictionary)}),
            allow_model_download=False,
        )
    assert caught.value.code == "OCR_MODEL_UNAVAILABLE"
    assert "rec_dict" in caught.value.message


def test_empty_custom_model_directory_is_model_unavailable(tmp_path):
    with pytest.raises(OcrAdapterError) as caught:
        runtime._engine_kwargs(
            _request(model_paths={"det": str(tmp_path)}), allow_model_download=True
        )
    assert caught.value.code == "OCR_MODEL_UNAVAILABLE"


def test_rgb_input_is_converted_to_contiguous_bgr():
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[0, 0] = (1, 2, 3)
    converted = runtime._to_bgr(pixels)
    assert converted.flags["C_CONTIGUOUS"]
    assert converted[0, 0].tolist() == [3, 2, 1]
    assert converted.shape == pixels.shape


def test_runtime_returns_normalized_records_from_predict(tmp_path):
    det = _model_dir(tmp_path / "det")
    rec = _model_dir(tmp_path / "rec")
    seen = {}

    class Engine:
        def predict(self, image):
            seen["image"] = image
            return [
                {
                    "rec_texts": ["hello", "world"],
                    "rec_scores": [0.9, 0.8],
                    "rec_polys": [
                        np.array([[0, 0], [20, 0], [20, 10], [0, 10]]),
                        np.array([[0, 20], [20, 20], [20, 30], [0, 30]]),
                    ],
                }
            ]

    def factory(kwargs):
        seen["kwargs"] = kwargs
        return Engine()

    instance = runtime.PaddleOcrRuntime(
        _request(model_paths={"det": str(det), "rec": str(rec)}),
        engine_factory=factory,
    )
    pixels = np.zeros((100, 200, 3), dtype=np.uint8)
    records = instance.recognize(pixels)

    assert [record["text"] for record in records] == ["hello", "world"]
    assert records[0]["bbox"].shape == (4, 2)
    assert seen["kwargs"]["text_detection_model_dir"] == str(det)
    assert seen["image"].flags["C_CONTIGUOUS"]


def test_runtime_error_classification_does_not_leak_details():
    model_error = runtime._runtime_error(
        RuntimeError("model weights could not be downloaded from /secret")
    )
    backend_error = runtime._runtime_error(RuntimeError("npu device import failed"))
    execution_error = runtime._runtime_error(RuntimeError("unexpected failure"))
    assert model_error.code == "OCR_MODEL_UNAVAILABLE"
    assert backend_error.code == "OCR_BACKEND_UNAVAILABLE"
    assert execution_error.code == "OCR_EXECUTION_FAILED"
    for error in (model_error, backend_error, execution_error):
        assert "/secret" not in error.message
        assert "unexpected failure" not in error.message
