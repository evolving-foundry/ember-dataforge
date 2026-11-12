"""Unit tests for the MindOCR runtime helpers that do not load models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from curator.operations.loading.ocr_adapters import OcrAdapterError, OcrRequest
from curator.operations.loading.ocr_adapters import mindocr_runtime as runtime


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    """MindOCR runtime helper tests do not need a Ray cluster."""
    yield


def _request(lang: str = "zh", model_paths=None) -> OcrRequest:
    return OcrRequest(
        lang=lang,
        device="cpu",
        layout_analysis=True,
        model_paths=model_paths or {},
    )


def _dictionary(path: Path, lines: int = runtime._PPOCR_DICT_LINES) -> Path:
    path.write_text("".join(f"c{index}\n" for index in range(lines)), encoding="utf-8")
    return path


def test_device_target_mapping():
    assert runtime._device_target("cpu") == "CPU"
    assert runtime._device_target("npu") == "Ascend"
    assert runtime._device_target("cuda") == "GPU"
    with pytest.raises(OcrAdapterError) as caught:
        runtime._device_target("tpu")
    assert caught.value.code == "OCR_BACKEND_UNAVAILABLE"


def test_checkpoint_file_is_selected_from_directory(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    assert runtime._model_checkpoint(str(tmp_path), "recognizer") == str(checkpoint)
    assert runtime._model_checkpoint(str(checkpoint), "recognizer") == str(checkpoint)
    assert runtime._model_checkpoint(None, "recognizer") is None


def test_checkpoint_directory_without_ckpt_is_model_unavailable(tmp_path):
    with pytest.raises(OcrAdapterError) as caught:
        runtime._model_checkpoint(str(tmp_path), "detector")
    assert caught.value.code == "OCR_MODEL_UNAVAILABLE"


def test_english_does_not_require_a_dictionary():
    assert runtime._recognition_dictionary(_request("en"), allow_model_download=False) is None


def test_explicit_recognition_dictionary_is_validated(tmp_path):
    dictionary = _dictionary(tmp_path / "keys.txt")
    assert (
        runtime._recognition_dictionary(
            _request(model_paths={"rec_dict": str(dictionary)}),
            allow_model_download=False,
        )
        == str(dictionary)
    )


def test_invalid_recognition_dictionary_size_is_model_unavailable(tmp_path):
    dictionary = _dictionary(tmp_path / "keys.txt", lines=3)
    with pytest.raises(OcrAdapterError) as caught:
        runtime._recognition_dictionary(
            _request(model_paths={"rec_dict": str(dictionary)}),
            allow_model_download=False,
        )
    assert caught.value.code == "OCR_MODEL_UNAVAILABLE"


def test_non_utf8_recognition_dictionary_is_model_unavailable(tmp_path):
    dictionary = tmp_path / "keys.txt"
    dictionary.write_bytes(b"\xff\xfe")
    with pytest.raises(OcrAdapterError) as caught:
        runtime._recognition_dictionary(
            _request(model_paths={"rec_dict": str(dictionary)}),
            allow_model_download=False,
        )
    assert caught.value.code == "OCR_MODEL_UNAVAILABLE"


def test_missing_dictionary_requires_download_permission(tmp_path, monkeypatch):
    monkeypatch.setenv("CURATOR_CACHE_DIR", str(tmp_path))
    with pytest.raises(OcrAdapterError) as caught:
        runtime._recognition_dictionary(_request(), allow_model_download=False)
    assert caught.value.code == "OCR_MODEL_UNAVAILABLE"


def test_download_rejects_dictionary_checksum_mismatch(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"tampered dictionary"

    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    destination = tmp_path / "ppocr_keys_v1.txt"
    with pytest.raises(OcrAdapterError) as caught:
        runtime._download_dictionary(destination)
    assert caught.value.code == "OCR_MODEL_UNAVAILABLE"
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".tmp").exists()


def test_order_points_clockwise_returns_tl_tr_br_bl():
    points = np.array(
        [[100, 40], [10, 20], [100, 20], [10, 40]], dtype=np.float32
    )
    ordered = runtime._order_points_clockwise(points)
    assert ordered.tolist() == [[10, 20], [100, 20], [100, 40], [10, 40]]


def test_recognition_input_is_normalized_nchw():
    cv2 = pytest.importorskip("cv2")
    crop = np.full((20, 100, 3), 127, dtype=np.uint8)
    tensor = runtime._recognition_input(cv2, crop)
    assert tensor.shape == (1, 3, runtime._REC_HEIGHT, 160)
    assert tensor.dtype == np.float32
    assert np.max(np.abs(tensor)) < 0.01
