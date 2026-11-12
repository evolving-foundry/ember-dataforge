"""C-M3a OCR adapter contract tests."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

# Importing curator.operations.loading currently passes through modules
# that import torch. The OCR contract tests do not need torch.
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from curator.operations.loading.ocr_adapters import (
    MindOcrAdapter,
    PaddleOcrAdapter,
    OcrAdapterError,
    OcrCandidate,
    OcrCapabilities,
    OcrCell,
    OcrImage,
    OcrRequest,
    OcrTable,
    merge_candidate_lines,
    validate_candidates,
)


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    """OCR adapter tests do not need the repo-wide Ray fixture."""
    yield


def _image(width: int = 200, height: int = 100) -> OcrImage:
    return OcrImage(
        pixels=np.zeros((height, width, 3), dtype=np.uint8),
        width=width,
        height=height,
    )


def _request(layout_analysis: bool = True) -> OcrRequest:
    return OcrRequest(
        lang="zh",
        device="npu",
        layout_analysis=layout_analysis,
    )


class _Runtime:
    def __init__(self, result):
        self.result = result
        self.seen = None

    def run(self, image, request):
        self.seen = (image, request)
        return self.result


def _adapter(result, adapter_type=MindOcrAdapter):
    runtime = _Runtime(result)
    return adapter_type(runtime_factory=lambda request: runtime), runtime


class TestMindOcrAdapter:
    def test_identity_and_conservative_metadata(self):
        adapter = MindOcrAdapter(runtime_factory=lambda request: _Runtime([]))
        assert adapter.name == "mindocr"
        assert adapter.supported_devices == frozenset({"npu", "cuda", "cpu"})
        assert adapter.capabilities.languages == frozenset({"zh", "en", "mixed"})
        assert adapter.capabilities.table_grid is False
        assert adapter.default_model_id == (
            "mindocr-official-dbnet_mobilenetv3-crnn_resnet34-"
            "crnn_resnet34_ch-ppocr_keys_v1-v1"
        )
        assert adapter.version in {"unknown"} or isinstance(adapter.version, str)

    def test_dependency_installed_is_false_when_module_is_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mindocr", None)
        assert MindOcrAdapter.dependency_installed() is False

    def test_fake_module_is_detected_without_importing_a_real_backend(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mindocr", types.ModuleType("mindocr"))
        assert MindOcrAdapter.dependency_installed() is True

    def test_missing_backend_maps_to_backend_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mindocr", None)
        with pytest.raises(OcrAdapterError) as caught:
            MindOcrAdapter().recognize(_image(), _request())
        assert caught.value.code == "OCR_BACKEND_UNAVAILABLE"

    def test_parallel_result_is_normalized(self):
        result = {
            "rec_texts": [" hello "],
            "rec_scores": [0.9],
            "rec_polys": [np.array([[10, 20], [100, 20], [100, 40], [10, 40]])],
        }
        adapter, runtime = _adapter(result)
        candidates = adapter.recognize(_image(), _request())
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.page == 1
        assert candidate.order == 0
        assert candidate.block_type == "paragraph"
        assert candidate.text == "hello"
        assert candidate.confidence == 0.9
        assert candidate.bbox == pytest.approx((0.05, 0.2, 0.5, 0.4))
        assert runtime.seen is not None
        assert runtime.seen[0].shape == (100, 200, 3)

    def test_parallel_word_boxes_are_merged_into_one_line(self):
        result = {
            "rec_texts": ["world", "hello"],
            "rec_scores": [0.8, 0.9],
            "rec_boxes": [[100, 20, 180, 40], [10, 20, 90, 40]],
        }
        adapter, _ = _adapter(result)
        candidates = adapter.recognize(_image(), _request())

        assert len(candidates) == 1
        assert candidates[0].text == "hello world"
        assert candidates[0].bbox == pytest.approx((0.05, 0.2, 0.9, 0.4))
        assert candidates[0].confidence == 0.8

    def test_tuple_result_and_empty_result_are_supported(self):
        box = ((10, 20), (100, 20), (100, 40), (10, 40))
        adapter, _ = _adapter([(box, "hello", 0.8)])
        assert adapter.recognize(_image(), _request())[0].text == "hello"

        adapter, _ = _adapter([])
        assert adapter.recognize(_image(), _request()) == []

    def test_varargs_runtime_receives_the_pixel_array(self):
        class VarArgsRuntime:
            def run(self, *args):
                assert len(args) == 1
                assert args[0].shape == (100, 200, 3)
                return {"text": "ok", "score": 0.6, "bbox": [0, 0, 20, 10]}

        adapter = MindOcrAdapter(runtime_factory=lambda request: VarArgsRuntime())
        assert adapter.recognize(_image(), _request())[0].text == "ok"

    def test_default_factory_uses_real_runtime_contract(self, monkeypatch):
        from curator.operations.loading.ocr_adapters import mindocr_runtime

        seen = {}

        class FakeRuntime:
            def __init__(self, request, *, allow_model_download=False):
                seen["request"] = request
                seen["allow_model_download"] = allow_model_download

            def recognize(self, img, request=None):
                return {
                    "rec_texts": ["ok"],
                    "rec_scores": [0.7],
                    "rec_boxes": [[0, 0, 20, 10]],
                }

        monkeypatch.setattr(mindocr_runtime, "MindOcrRuntime", FakeRuntime)
        request = OcrRequest(
            lang="zh",
            device="cpu",
            layout_analysis=True,
            model_paths={
                "det": "D:/models/det",
                "rec": "D:/models/rec",
                "rec_dict": "D:/models/ppocr_keys_v1.txt",
            },
        )
        candidates = MindOcrAdapter(allow_model_download=True).recognize(
            _image(), request
        )
        assert candidates[0].text == "ok"
        assert seen["request"] is request
        assert seen["allow_model_download"] is True

    def test_invalid_bbox_is_execution_failure(self):
        adapter, _ = _adapter(
            {"rec_texts": ["x"], "rec_scores": [0.5], "rec_boxes": [[0, 0, 300, 20]]}
        )
        with pytest.raises(OcrAdapterError) as caught:
            adapter.recognize(_image(), _request())
        assert caught.value.code == "OCR_EXECUTION_FAILED"

    @pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True])
    def test_invalid_confidence_is_execution_failure(self, score):
        adapter, _ = _adapter(
            {"rec_texts": ["x"], "rec_scores": [score], "rec_boxes": [[0, 0, 20, 10]]}
        )
        with pytest.raises(OcrAdapterError) as caught:
            adapter.recognize(_image(), _request())
        assert caught.value.code == "OCR_EXECUTION_FAILED"

    def test_invalid_candidate_type_is_execution_failure(self):
        adapter, _ = _adapter([object()])
        with pytest.raises(OcrAdapterError) as caught:
            adapter.recognize(_image(), _request())
        assert caught.value.code == "OCR_EXECUTION_FAILED"

    def test_layout_disabled_forces_paragraph(self):
        adapter, _ = _adapter(
            {"text": "region", "type": "table", "bbox": [0, 0, 1, 1], "score": 0.4}
        )
        candidate = adapter.recognize(_image(), _request(layout_analysis=False))[0]
        assert candidate.block_type == "paragraph"
        assert candidate.table is None

    def test_runtime_errors_do_not_leak_source_paths_or_text(self):
        class FailingRuntime:
            def run(self, image):
                raise RuntimeError(r"failed at D:\secret\source.png with document body")

        adapter = MindOcrAdapter(runtime_factory=lambda request: FailingRuntime())
        with pytest.raises(OcrAdapterError) as caught:
            adapter.recognize(_image(), _request())
        message = caught.value.message
        assert "D:\\secret" not in message
        assert "source.png" not in message
        assert "document body" not in message
        assert caught.value.stage == "ocr"


class TestPaddleOcrAdapter:
    def test_identity_and_default_model(self):
        adapter = PaddleOcrAdapter(runtime_factory=lambda request: _Runtime([]))
        assert adapter.name == "paddleocr"
        assert adapter.supported_devices == frozenset({"npu", "cuda", "cpu"})
        assert adapter.capabilities.languages == frozenset({"zh", "en", "mixed"})
        assert adapter.capabilities.layout is True
        assert adapter.capabilities.table_grid is False
        assert adapter.default_model_id == "paddleocr-official-ppocr-default-v1"

    def test_dependency_detection_uses_paddleocr_module(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "paddleocr", None)
        assert PaddleOcrAdapter.dependency_installed() is False
        monkeypatch.setitem(sys.modules, "paddleocr", types.ModuleType("paddleocr"))
        assert PaddleOcrAdapter.dependency_installed() is True

    def test_parallel_result_is_normalized(self):
        result = {
            "rec_texts": [" hello "],
            "rec_scores": [0.9],
            "rec_polys": [np.array([[10, 20], [100, 20], [100, 40], [10, 40]])],
        }
        adapter, runtime = _adapter(result, adapter_type=PaddleOcrAdapter)
        candidates = adapter.recognize(_image(), _request())
        assert len(candidates) == 1
        assert candidates[0].text == "hello"
        assert candidates[0].confidence == 0.9
        assert candidates[0].bbox == pytest.approx((0.05, 0.2, 0.5, 0.4))
        assert runtime.seen is not None

    def test_nested_legacy_tuple_result_is_supported(self):
        result = [
            [
                {"rec_texts": ["ok"], "rec_scores": [0.7], "rec_boxes": [[0, 0, 20, 10]]}
            ]
        ]
        adapter, _ = _adapter(result, adapter_type=PaddleOcrAdapter)
        assert adapter.recognize(_image(), _request())[0].text == "ok"

    def test_default_factory_passes_download_flag(self, monkeypatch):
        from curator.operations.loading.ocr_adapters import paddleocr_runtime

        seen = {}

        class FakeRuntime:
            def __init__(self, request, *, allow_model_download=False):
                seen["request"] = request
                seen["allow_model_download"] = allow_model_download

            def recognize(self, img, request=None):
                return {
                    "rec_texts": ["ok"],
                    "rec_scores": [0.7],
                    "rec_boxes": [[0, 0, 20, 10]],
                }

        monkeypatch.setattr(paddleocr_runtime, "PaddleOcrRuntime", FakeRuntime)
        request = OcrRequest(
            lang="en",
            device="cpu",
            layout_analysis=True,
            model_paths={"det": "D:/models/det", "rec": "D:/models/rec"},
        )
        candidates = PaddleOcrAdapter(allow_model_download=True).recognize(
            _image(), request
        )
        assert candidates[0].text == "ok"
        assert seen["request"] is request
        assert seen["allow_model_download"] is True

    def test_changed_runtime_key_closes_previous_runtime(self):
        class Runtime:
            def __init__(self):
                self.closed = False

            def run(self, image, request):
                return []

            def close(self):
                self.closed = True

        created = []

        def factory(request):
            runtime = Runtime()
            created.append(runtime)
            return runtime

        adapter = PaddleOcrAdapter(runtime_factory=factory)
        adapter.recognize(_image(), _request())
        adapter.recognize(
            _image(),
            OcrRequest(lang="en", device="npu", layout_analysis=True),
        )

        assert len(created) == 2
        assert created[0].closed is True
        assert created[1].closed is False

        adapter.close()
        assert created[1].closed is True

    def test_runtime_errors_do_not_leak_text(self):
        class FailingRuntime:
            def run(self, image):
                raise RuntimeError(r"failed at D:\secret\source.png with document body")

        adapter = PaddleOcrAdapter(runtime_factory=lambda request: FailingRuntime())
        with pytest.raises(OcrAdapterError) as caught:
            adapter.recognize(_image(), _request())
        message = caught.value.message
        assert r"D:\secret" not in message
        assert "source.png" not in message
        assert "document body" not in message


class TestCandidateValidation:
    def _table_candidate(self):
        cells = (
            OcrCell(0, 0, 1, 2, "merged", (0.0, 0.0, 1.0, 0.5), 0.9),
            OcrCell(0, 1, 1, 1, "", None, None),
            OcrCell(1, 0, 1, 1, "a", (0.0, 0.5, 0.5, 1.0), 0.8),
            OcrCell(1, 1, 1, 1, "b", (0.5, 0.5, 1.0, 1.0), 0.8),
        )
        return OcrCandidate(
            page=1,
            order=0,
            block_type="table",
            text="merged",
            bbox=(0.0, 0.0, 1.0, 1.0),
            confidence=0.9,
            table=OcrTable(n_rows=2, n_cols=2, cells=cells),
        )

    def test_merged_table_grid_is_valid(self):
        request = _request(layout_analysis=True)
        capabilities = OcrCapabilities(
            languages=frozenset({"zh"}),
            layout=True,
            table_grid=True,
            confidence=True,
        )
        validate_candidates([self._table_candidate()], request, capabilities)

    def test_table_grid_requires_capability(self):
        request = _request(layout_analysis=True)
        capabilities = OcrCapabilities(
            languages=frozenset({"zh"}),
            layout=True,
            table_grid=False,
            confidence=True,
        )
        with pytest.raises(OcrAdapterError) as caught:
            validate_candidates([self._table_candidate()], request, capabilities)
        assert caught.value.code == "OCR_EXECUTION_FAILED"

    def test_layout_disabled_rejects_non_paragraph_blocks(self):
        candidate = OcrCandidate(
            page=1,
            order=0,
            block_type="table",
            text="region",
            bbox=(0.0, 0.0, 1.0, 1.0),
            confidence=0.5,
            table=None,
        )
        with pytest.raises(OcrAdapterError) as caught:
            validate_candidates(
                [candidate],
                _request(layout_analysis=False),
                OcrCapabilities(
                    languages=frozenset({"zh"}),
                    layout=True,
                    table_grid=False,
                    confidence=True,
                ),
            )
        assert caught.value.code == "OCR_EXECUTION_FAILED"

    def test_table_grid_must_cover_complete_grid(self):
        with pytest.raises(ValueError):
            OcrTable(
                n_rows=2,
                n_cols=2,
                cells=(OcrCell(0, 0, 1, 1, "a", None, None),),
            )


class TestCandidateLineMerging:
    def _candidate(self, text, bbox, *, order=0, confidence=0.8, block_type="paragraph"):
        return OcrCandidate(
            page=1,
            order=order,
            block_type=block_type,
            text=text,
            bbox=bbox,
            confidence=confidence,
        )

    def test_merges_words_and_returns_top_down_reading_order(self):
        candidates = [
            self._candidate("second", (0.2, 0.7, 0.4, 0.8), order=0),
            self._candidate("world", (0.3, 0.1, 0.5, 0.2), order=1),
            self._candidate("hello", (0.1, 0.1, 0.2, 0.2), order=2),
            self._candidate("first", (0.1, 0.7, 0.2, 0.8), order=3),
        ]

        merged = merge_candidate_lines(candidates)

        assert [candidate.text for candidate in merged] == [
            "hello world",
            "first second",
        ]
        assert [candidate.order for candidate in merged] == [0, 1]
        assert merged[0].bbox == pytest.approx((0.1, 0.1, 0.5, 0.2))

    def test_joins_cjk_without_spaces_and_keeps_latin_word_spacing(self):
        candidates = [
            self._candidate("你好", (0.0, 0.0, 0.1, 0.1), order=0),
            self._candidate("世界", (0.1, 0.0, 0.2, 0.1), order=1),
            self._candidate("Invoice", (0.0, 0.3, 0.2, 0.4), order=2),
            self._candidate("No:", (0.2, 0.3, 0.3, 0.4), order=3),
            self._candidate("INV-2026", (0.3, 0.3, 0.6, 0.4), order=4),
        ]

        merged = merge_candidate_lines(candidates)

        assert [candidate.text for candidate in merged] == [
            "你好世界",
            "Invoice No: INV-2026",
        ]

    def test_bridging_word_merges_all_groups_on_the_same_line(self):
        candidates = [
            self._candidate(
                "northwind", (0.048571, 0.065556, 0.202143, 0.103333), order=0
            ),
            self._candidate(
                "invoice", (0.306429, 0.067778, 0.406429, 0.104444), order=1
            ),
            self._candidate(
                "supply", (0.212857, 0.068889, 0.296429, 0.103333), order=2
            ),
        ]

        merged = merge_candidate_lines(candidates)

        assert [candidate.text for candidate in merged] == [
            "northwind supply invoice"
        ]

    def test_does_not_merge_different_block_types(self):
        candidates = [
            self._candidate("header", (0.0, 0.0, 0.2, 0.1), block_type="header"),
            self._candidate("body", (0.2, 0.0, 0.4, 0.1), block_type="paragraph"),
        ]

        merged = merge_candidate_lines(candidates)

        assert [candidate.text for candidate in merged] == ["header", "body"]
