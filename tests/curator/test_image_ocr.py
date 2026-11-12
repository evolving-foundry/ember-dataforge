"""Image OCR decoding, C-IR mapping, operator and op-mode tests."""
from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("PIL")

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from PIL import Image

from curator.operations.loading.document_common import DocumentParseError
from curator.operations.loading.image_ocr import (
    ImageOcrOptions,
    _subprocess_ocr_batch,
    build_image_ocr_options,
    candidates_to_blocks,
    decode_image_file,
    iter_image_results_in_subprocess,
    parse_image_file,
    parse_image_inputs,
)
from curator.operations.loading import image_ocr as image_ocr_module
from curator.operations.loading.ocr_routing import OcrRoute
from curator.operations.loading.ocr_adapters import (
    OcrAdapterError,
    OcrCandidate,
    OcrCell,
    OcrRequest,
    OcrTable,
    validate_candidates,
)
from curator.operations.loading.ocr_ranking import parse_ranking_document
from curator.operations.loading.parsed_document import Block
from .test_ocr_routing import FakeAdapter, _factory, _probe, _row


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    yield


def _install_fake_mindocr(directory: Path, monkeypatch) -> None:
    mindspore = directory / "mindspore"
    mindspore.mkdir()
    (mindspore / "__init__.py").write_text(
        "import numpy as np\n"
        "PYNATIVE_MODE = 0\n"
        "float32 = np.float32\n"
        "def Tensor(value, dtype=None):\n"
        "    return np.asarray(value, dtype=dtype)\n"
        "def set_context(**kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (mindspore / "parallel.py").write_text("", encoding="utf-8")

    mindocr = directory / "mindocr"
    mindocr.mkdir()
    (mindocr / "__init__.py").write_text("", encoding="utf-8")
    (mindocr / "models.py").write_text(
        "import numpy as np\n"
        "class _Model:\n"
        "    def __call__(self, tensor):\n"
        "        return np.zeros((1, 1, 736, 1280), dtype=np.float32)\n"
        "    def set_train(self, value):\n"
        "        return None\n"
        "def build_model(name, **kwargs):\n"
        "    return _Model()\n",
        encoding="utf-8",
    )
    (mindocr / "postprocess.py").write_text(
        "def build_postprocess(config):\n"
        "    if config['name'] == 'DBPostprocess':\n"
        "        def detect(raw, shape_list=None):\n"
        "            height, width = shape_list[0][:2]\n"
        "            box = [[width * 0.1, height * 0.1],\n"
        "                   [width * 0.9, height * 0.1],\n"
        "                   [width * 0.9, height * 0.2],\n"
        "                   [width * 0.1, height * 0.2]]\n"
        "            return {'polys': [[box]], 'scores': [[0.9]]}\n"
        "        return detect\n"
        "    def recognize(raw):\n"
        "        return {'texts': ['hello world'], 'confs': [0.9]}\n"
        "    return recognize\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(directory))
    sys.modules.pop("mindspore", None)
    sys.modules.pop("mindocr", None)
    import importlib

    importlib.invalidate_caches()


def _fake_ocr_params(**overrides):
    params = {
        "backend": "mindocr",
        "device": "cpu",
        "lang": "en",
        "allow_model_download": True,
    }
    params.update(overrides)
    return params


def _png(path: Path, *, size=(40, 20), color=(255, 255, 255)) -> Path:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _jpeg(path: Path, *, size=(40, 20), color=(10, 20, 30)) -> Path:
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


def _paragraph(text: str, bbox=(0.1, 0.1, 0.9, 0.2), confidence=0.9, order=0):
    return OcrCandidate(
        page=1,
        order=order,
        block_type="paragraph",
        text=text,
        bbox=bbox,
        confidence=confidence,
    )


class TestDecode:
    def test_png_pixels(self, tmp_path):
        path = _png(tmp_path / "image.png")
        pixels, width, height, media_type = decode_image_file(str(path), 100)
        assert (width, height) == (40, 20)
        assert media_type == "png"
        assert pixels.shape == (20, 40, 3)
        assert pixels.dtype.name == "uint8"

    def test_jpeg_and_jpeg_extension(self, tmp_path):
        path = _jpeg(tmp_path / "image.jpeg")
        _, width, height, media_type = decode_image_file(str(path), 100)
        assert (width, height) == (40, 20)
        assert media_type == "jpg"

    def test_exif_orientation_is_applied(self, tmp_path):
        path = tmp_path / "exif.jpg"
        image = Image.new("RGB", (40, 20), (1, 2, 3))
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, format="JPEG", exif=exif.tobytes())
        pixels, width, height, _ = decode_image_file(str(path), 100)
        assert (width, height) == (20, 40)
        assert pixels.shape == (40, 20, 3)

    def test_extension_signature_mismatch(self, tmp_path):
        path = tmp_path / "wrong.png"
        _jpeg(path)
        with pytest.raises(DocumentParseError) as exc:
            decode_image_file(str(path), 100)
        assert exc.value.code == "UNSUPPORTED_FORMAT"

    def test_corrupt_image(self, tmp_path):
        path = tmp_path / "bad.png"
        path.write_bytes(b"not an image")
        with pytest.raises(DocumentParseError) as exc:
            decode_image_file(str(path), 100)
        assert exc.value.code == "CORRUPT_DOCUMENT"

    def test_pixel_limit(self, tmp_path):
        path = _png(tmp_path / "big.png", size=(20, 20))
        with pytest.raises(DocumentParseError) as exc:
            decode_image_file(str(path), 0.0001)
        assert exc.value.code == "RESOURCE_LIMIT"

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "image.bmp"
        path.write_bytes(b"BM")
        with pytest.raises(DocumentParseError) as exc:
            decode_image_file(str(path), 100)
        assert exc.value.code == "UNSUPPORTED_FORMAT"


class TestMapping:
    def test_reading_order_top_then_left_then_order(self):
        blocks = candidates_to_blocks(
            [
                _paragraph("bottom", bbox=(0.1, 0.8, 0.9, 0.9), order=0),
                _paragraph("right", bbox=(0.6, 0.1, 0.9, 0.2), order=1),
                _paragraph("left", bbox=(0.1, 0.1, 0.4, 0.2), order=2),
            ]
        )
        assert [block.text for block in blocks] == ["left", "right", "bottom"]
        assert [block.block_id for block in blocks] == [
            "b-000001",
            "b-000002",
            "b-000003",
        ]
        assert all(block.page == 1 for block in blocks)
        assert all(block.sheet is None and block.style is None for block in blocks)

    def test_grid_table_merged_cells(self):
        table = OcrTable(
            n_rows=2,
            n_cols=2,
            cells=(
                OcrCell(0, 0, 2, 2, "merged", (0.1, 0.1, 0.9, 0.9), 0.8),
                OcrCell(0, 1, 1, 1, "", None, None),
                OcrCell(1, 0, 1, 1, "", None, None),
                OcrCell(1, 1, 1, 1, "", None, None),
            ),
        )
        candidate = OcrCandidate(
            page=1,
            order=0,
            block_type="table",
            text="",
            bbox=(0.1, 0.1, 0.9, 0.9),
            confidence=0.9,
            table=table,
        )
        block = candidates_to_blocks([candidate])[0]
        assert block.type == "table"
        assert block.text == ""
        assert block.table.structure == "grid"
        assert (block.table.n_rows, block.table.n_cols) == (2, 2)
        assert [(cell.row, cell.col) for cell in block.table.cells] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        assert block.table.cells[0].cell_status == "value"
        assert block.table.cells[0].row_span == 2
        assert block.table.cells[0].col_span == 2
        assert [cell.cell_status for cell in block.table.cells[1:]] == [
            "covered",
            "covered",
            "covered",
        ]
        assert block.table.cells[1].covered_by == [0, 0]
        assert block.ocr_confidence == 0.8

    def test_region_table_keeps_null_confidence(self):
        candidate = OcrCandidate(
            page=1,
            order=0,
            block_type="table",
            text="grid not recognized",
            bbox=(0.1, 0.1, 0.9, 0.9),
            confidence=0.9,
        )
        block = candidates_to_blocks([candidate])[0]
        assert block.table.structure == "region"
        assert block.table.n_rows is None
        assert block.table.n_cols is None
        assert block.table.cells == []
        assert block.text == "grid not recognized"
        assert block.ocr_confidence is None

    def test_empty_text_paragraph_keeps_null_confidence(self):
        candidate = _paragraph("", confidence=0.9)
        block = candidates_to_blocks([candidate])[0]
        assert block.text == ""
        assert block.ocr_confidence is None

    def test_table_cell_limit(self, monkeypatch):
        import curator.operations.loading.image_ocr as module

        table = OcrTable(
            n_rows=2,
            n_cols=2,
            cells=(
                OcrCell(0, 0, 1, 1, "a", None, None),
                OcrCell(0, 1, 1, 1, "b", None, None),
                OcrCell(1, 0, 1, 1, "c", None, None),
                OcrCell(1, 1, 1, 1, "d", None, None),
            ),
        )
        candidate = OcrCandidate(
            page=1,
            order=0,
            block_type="table",
            text="",
            bbox=(0.1, 0.1, 0.9, 0.9),
            confidence=None,
            table=table,
        )
        monkeypatch.setattr(module, "MAX_TABLE_CELLS", 3)
        with pytest.raises(DocumentParseError) as exc:
            candidates_to_blocks([candidate])
        assert exc.value.code == "TABLE_TOO_LARGE"

    def test_layout_disabled_rejects_non_paragraph_candidate(self):
        candidate = OcrCandidate(
            page=1,
            order=0,
            block_type="table",
            text="region",
            bbox=(0.1, 0.1, 0.9, 0.9),
            confidence=None,
        )
        adapter = FakeAdapter(table_grid=False)
        with pytest.raises(OcrAdapterError) as exc:
            validate_candidates(
                [candidate],
                OcrRequest(lang="zh", device="cpu", layout_analysis=False),
                adapter.capabilities,
            )
        assert exc.value.code == "OCR_EXECUTION_FAILED"


class TestOptions:
    def test_lang_ch_alias(self):
        options = build_image_ocr_options({"lang": "ch"})
        assert options.lang == "zh"

    def test_auto_with_model_paths_rejected(self):
        with pytest.raises(DocumentParseError) as exc:
            build_image_ocr_options(
                {"backend": "auto", "model_paths": {"det": str(Path.cwd())}}
            )
        assert exc.value.code == "INVALID_PARAMETER"

    def test_model_path_url_rejected(self):
        with pytest.raises(DocumentParseError) as exc:
            build_image_ocr_options(
                {"backend": "mindocr", "model_paths": {"det": "https://example.test/m"}}
            )
        assert exc.value.code == "INVALID_PARAMETER"

    def test_missing_model_path_is_model_unavailable(self, tmp_path):
        with pytest.raises(DocumentParseError) as exc:
            build_image_ocr_options(
                {
                    "backend": "mindocr",
                    "model_paths": {"det": str(tmp_path / "missing")},
                }
            )
        assert exc.value.code == "OCR_MODEL_UNAVAILABLE"

    def test_unknown_model_key_rejected(self, tmp_path):
        with pytest.raises(DocumentParseError) as exc:
            build_image_ocr_options(
                {"backend": "mindocr", "model_paths": {"other": str(tmp_path)}}
            )
        assert exc.value.code == "INVALID_PARAMETER"

    def test_recognition_dictionary_path_is_allowed(self, tmp_path):
        dictionary = tmp_path / "ppocr_keys_v1.txt"
        dictionary.write_text("a\nb\n", encoding="utf-8")
        options = build_image_ocr_options(
            {"backend": "mindocr", "model_paths": {"rec_dict": str(dictionary)}}
        )
        assert options.model_path_map["rec_dict"] == str(dictionary.resolve())


class TestParseImageFile:
    def test_success_and_deterministic_doc_id(self, tmp_path):
        path = _png(tmp_path / "image.png")
        options = ImageOcrOptions()
        adapter = FakeAdapter(result=(_paragraph("hello"),))
        document = parse_image_file(
            str(path),
            options,
            adapter_factories={"mindocr": _factory(adapter)},
            device_probe=_probe("cpu"),
            ranking_override=None,
        )
        again = parse_image_file(
            str(path),
            options,
            adapter_factories={"mindocr": _factory(adapter)},
            device_probe=_probe("cpu"),
            ranking_override=None,
        )
        assert document.doc_id == again.doc_id
        assert document.source.media_type == "png"
        assert document.source.page_count == 1
        assert document.provenance.backend == "mindocr"
        assert document.provenance.device == "cpu"
        assert document.provenance.routing_reason == "ranking_missing"
        assert [block.text for block in document.blocks] == ["hello"]

    def test_auto_falls_back_to_second_backend(self, tmp_path):
        path = _png(tmp_path / "image.png")
        mindocr = FakeAdapter(
            name="mindocr",
            error=OcrAdapterError(
                "OCR_MODEL_UNAVAILABLE", "MindOCR model could not be loaded"
            ),
        )
        paddle = FakeAdapter(
            name="paddleocr", result=(_paragraph("from paddle"),)
        )
        document = parse_image_file(
            str(path),
            ImageOcrOptions(),
            adapter_factories={
                "mindocr": _factory(mindocr),
                "paddleocr": _factory(paddle),
            },
            device_probe=_probe("cpu"),
            ranking_override=None,
        )
        assert document.provenance.backend == "paddleocr"
        assert document.provenance.routing_reason == "ranking_missing"
        assert [block.text for block in document.blocks] == ["from paddle"]

    def test_ranked_failure_falls_back_without_writing_ranking(self, tmp_path):
        path = _png(tmp_path / "image.png")
        mindocr = FakeAdapter(
            name="mindocr",
            model_id="mindocr-official-dbnet_mobilenetv3-crnn_resnet34-crnn_resnet34_ch-ppocr_keys_v1-v1",
            error=OcrAdapterError(
                "OCR_MODEL_UNAVAILABLE", "MindOCR model could not be loaded"
            ),
        )
        paddle = FakeAdapter(
            name="paddleocr",
            model_id="paddleocr-official-ppocr-default-v1",
            result=(_paragraph("from paddle"),),
        )
        table = parse_ranking_document(
            {
                "schema_version": "ocr-ranking-v1",
                "ranking_version": "rank-v1",
                "evaluation_set_version": "eval-v1",
                "primary_metric": "cer",
                "rows": [
                    _row(
                        "mindocr",
                        version=mindocr.version,
                        model=mindocr.default_model_id,
                        cer=0.1,
                    ),
                    _row(
                        "paddleocr",
                        version=paddle.version,
                        model=paddle.default_model_id,
                        cer=0.2,
                    ),
                ],
            }
        )
        document = parse_image_file(
            str(path),
            ImageOcrOptions(),
            adapter_factories={
                "mindocr": _factory(mindocr),
                "paddleocr": _factory(paddle),
            },
            device_probe=_probe("cpu"),
            ranking_override=table,
        )
        assert document.provenance.backend == "paddleocr"
        assert document.provenance.routing_reason == "backend_unavailable"
        assert document.provenance.ranking_version == "rank-v1"
        assert [block.text for block in document.blocks] == ["from paddle"]

    def test_explicit_backend_does_not_fallback(self, tmp_path):
        path = _png(tmp_path / "image.png")
        mindocr = FakeAdapter(
            name="mindocr",
            error=OcrAdapterError(
                "OCR_MODEL_UNAVAILABLE", "MindOCR model could not be loaded"
            ),
        )
        paddle = FakeAdapter(
            name="paddleocr", result=(_paragraph("from paddle"),)
        )
        with pytest.raises(DocumentParseError) as exc:
            parse_image_file(
                str(path),
                ImageOcrOptions(backend="mindocr"),
                adapter_factories={
                    "mindocr": _factory(mindocr),
                    "paddleocr": _factory(paddle),
                },
                device_probe=_probe("cpu"),
            )
        assert exc.value.code == "OCR_MODEL_UNAVAILABLE"
        assert exc.value.stage == "validate"

    def test_backend_failure_does_not_leak_path(self, tmp_path):
        path = _png(tmp_path / "secret-name.png")
        mindocr = FakeAdapter(
            name="mindocr",
            error=OcrAdapterError("OCR_EXECUTION_FAILED", "backend exploded"),
        )
        with pytest.raises(DocumentParseError) as exc:
            parse_image_file(
                str(path),
                ImageOcrOptions(),
                adapter_factories={"mindocr": _factory(mindocr)},
                device_probe=_probe("cpu"),
                ranking_override=None,
            )
        assert "secret-name.png" not in exc.value.message
        assert str(tmp_path) not in exc.value.message

    def test_layout_disabled_rejects_adapter_table_candidate(self, tmp_path):
        path = _png(tmp_path / "image.png")
        table = OcrCandidate(
            page=1,
            order=0,
            block_type="table",
            text="region",
            bbox=(0.1, 0.1, 0.9, 0.9),
            confidence=None,
        )
        adapter = FakeAdapter(result=(table,))
        with pytest.raises(DocumentParseError) as exc:
            parse_image_file(
                str(path),
                ImageOcrOptions(layout_analysis=False),
                adapter_factories={"mindocr": _factory(adapter)},
                device_probe=_probe("cpu"),
                ranking_override=None,
            )
        assert exc.value.code == "OCR_EXECUTION_FAILED"


class TestBatchWorker:
    def test_subprocess_batch_selects_routes_once_and_writes_every_result(
        self, tmp_path, monkeypatch
    ):
        class TrackingAdapter(FakeAdapter):
            def __init__(self):
                super().__init__(result=(_paragraph("hello"),))
                self.close_calls = 0

            def close(self):
                self.close_calls += 1

        first = _png(tmp_path / "first.png")
        second = _png(tmp_path / "second.png")
        result_dir = tmp_path / "results"
        result_dir.mkdir()
        options = build_image_ocr_options(
            _fake_ocr_params(allowed_roots=[str(tmp_path)])
        )
        adapter = TrackingAdapter()
        route = OcrRoute(
            adapter=adapter,
            backend="mindocr",
            backend_version="1.0",
            device="cpu",
            model_id="model-a",
            routing_reason="explicit_backend",
            ranking_version=None,
            ranking_evaluation_set=None,
        )
        select_calls = []

        def select_routes(options):
            select_calls.append(options)
            return [route]

        monkeypatch.setattr(image_ocr_module, "_select_routes", select_routes)
        monkeypatch.setattr(
            image_ocr_module, "apply_child_resource_limits", lambda **kwargs: None
        )

        _subprocess_ocr_batch(
            (str(first), str(second)), options, str(result_dir), 0, 30.0
        )

        assert len(select_calls) == 1
        for index in range(2):
            payload = json.loads(
                (result_dir / f"result-{index}.json").read_text(encoding="utf-8")
            )
            assert payload["ok"] is True
            assert payload["document"]["blocks"][0]["text"] == "hello"
        assert adapter.close_calls == 1

    def test_batch_timeout_restarts_worker_for_next_image(self, monkeypatch):
        class Process:
            def __init__(self):
                self.stop_calls = 0

            def is_alive(self):
                return True

        starts = []
        processes = []

        def start_child(target, args):
            process = Process()
            starts.append((target, args))
            processes.append(process)
            return process

        monkeypatch.setattr(image_ocr_module, "start_child", start_child)
        monkeypatch.setattr(
            image_ocr_module,
            "wait_for_result",
            lambda path, process, timeout_s: False,
        )
        monkeypatch.setattr(
            image_ocr_module,
            "stop_child",
            lambda process: setattr(
                process, "stop_calls", process.stop_calls + 1
            ),
        )

        results = list(
            iter_image_results_in_subprocess(
                ("/first.png", "/second.png"),
                ImageOcrOptions(parse_timeout_s=1.0),
            )
        )

        assert len(results) == 2
        assert all(
            isinstance(result, DocumentParseError)
            and result.code == "PARSE_TIMEOUT"
            for result in results
        )
        assert len(starts) == 2
        assert starts[1][1][0] == ("/second.png",)
        assert starts[1][1][3] == 1
        assert all(process.stop_calls >= 1 for process in processes)


class TestParseImageInputs:
    def test_directory_order_and_deduplication(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        folder = tmp_path / "images"
        folder.mkdir()
        _png(folder / "b.png", color=(1, 1, 1))
        _png(folder / "a.png", color=(2, 2, 2))
        nested = folder / "nested"
        nested.mkdir()
        _png(nested / "c.png", color=(3, 3, 3))
        rows = parse_image_inputs(
            [str(folder), str(folder / "a.png")],
            _fake_ocr_params(allowed_roots=[str(folder)]),
        )
        assert [row["input_index"] for row in rows] == [0, 1, 2]
        assert all(row["error"] is None for row in rows)
        assert len({row["doc_id"] for row in rows}) == 3

    def test_skip_writes_error_rows(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        folder = tmp_path / "images"
        folder.mkdir()
        _png(folder / "good.png")
        (folder / "bad.png").write_bytes(b"not an image")
        rows = parse_image_inputs(
            [str(folder)],
            _fake_ocr_params(allowed_roots=[str(folder)], on_error="skip"),
        )
        assert len(rows) == 2
        assert rows[0]["error"]["code"] == "CORRUPT_DOCUMENT"
        assert rows[1]["error"] is None

    def test_raise_stops_on_corrupt_image(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        folder = tmp_path / "images"
        folder.mkdir()
        (folder / "bad.png").write_bytes(b"not an image")
        with pytest.raises(DocumentParseError) as exc:
            parse_image_inputs(
                [str(folder)],
                _fake_ocr_params(allowed_roots=[str(folder)], on_error="raise"),
            )
        assert exc.value.code == "CORRUPT_DOCUMENT"

    def test_unsupported_explicit_file_is_row_level(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        rows = parse_image_inputs(
            [str(path)],
            _fake_ocr_params(allowed_roots=[str(tmp_path)], on_error="skip"),
        )
        assert rows[0]["error"]["code"] == "UNSUPPORTED_FORMAT"

    def test_all_extras_missing_is_request_level(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "mindocr", None)
        monkeypatch.setitem(sys.modules, "paddleocr", None)
        folder = tmp_path / "images"
        folder.mkdir()
        _png(folder / "good.png")
        with pytest.raises(DocumentParseError) as exc:
            parse_image_inputs([str(folder)], {"allowed_roots": [str(folder)]})
        assert exc.value.code == "DEPENDENCY_MISSING"
        assert "ocr-mindocr" in exc.value.message
        assert "ocr-paddleocr" in exc.value.message

    def test_empty_directory_is_invalid_parameter(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        folder = tmp_path / "empty"
        folder.mkdir()
        with pytest.raises(DocumentParseError) as exc:
            parse_image_inputs([str(folder)], {"allowed_roots": [str(folder)]})
        assert exc.value.code == "INVALID_PARAMETER"


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _op_request(
    tmp_path: Path, rows: list[dict], params: dict, output_dir: str | None
) -> dict:
    manifest = _write_manifest(tmp_path / "manifest.jsonl", rows)
    merged_params = _fake_ocr_params(**params)
    return {
        "schemaVersion": 1,
        "op": "ocr_image",
        "params": merged_params,
        "dataPath": str(manifest),
        "outputDir": output_dir,
    }


def _run_op(request: dict) -> tuple[int, str, str]:
    from curator.trusted_worker.op.protocol import handle

    stdout, stderr = io.StringIO(), io.StringIO()
    code = handle(json.dumps(request), stdout, stderr)
    return code, stdout.getvalue(), stderr.getvalue()


class TestOpMode:
    def test_inline_and_metrics(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        _png(tmp_path / "good.png")
        request = _op_request(
            tmp_path,
            [{"sourcePath": "good.png", "ignored": True}],
            {"allowed_roots": [str(tmp_path)]},
            None,
        )
        code, out, err = _run_op(request)
        assert code == 0, err
        response = json.loads(out)
        assert response["op"] == "ocr_image"
        assert response["metrics"] == {
            "input_count": 1.0,
            "document_count": 1.0,
            "error_count": 0.0,
        }
        row = response["items"][0]
        assert set(row) == {"input_index", "doc_id", "parsed_doc", "error"}
        assert row["parsed_doc"]["provenance"]["backend"] == "mindocr"
        assert str(tmp_path) not in out

    def test_artifact_and_skip(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        _png(tmp_path / "good.png")
        (tmp_path / "bad.png").write_bytes(b"not an image")
        out_dir = tmp_path / "out"
        request = _op_request(
            tmp_path,
            [{"sourcePath": "bad.png"}, {"sourcePath": "good.png"}],
            {"allowed_roots": [str(tmp_path)], "on_error": "skip"},
            str(out_dir),
        )
        code, out, err = _run_op(request)
        assert code == 0, err
        response = json.loads(out)
        assert response["artifacts"] == [{"relativePath": "ocr-documents.jsonl"}]
        assert response["metrics"] == {
            "input_count": 2.0,
            "document_count": 1.0,
            "error_count": 1.0,
        }
        lines = (out_dir / "ocr-documents.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["error"]["code"] == "CORRUPT_DOCUMENT"

    def test_raise_does_not_write_artifact_or_stdout(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        (tmp_path / "bad.png").write_bytes(b"not an image")
        out_dir = tmp_path / "out"
        request = _op_request(
            tmp_path,
            [{"sourcePath": "bad.png"}],
            {"allowed_roots": [str(tmp_path)]},
            str(out_dir),
        )
        code, out, err = _run_op(request)
        assert code == 1
        assert out == ""
        assert json.loads(err.strip().splitlines()[-1])["code"] == "CORRUPT_DOCUMENT"
        assert not (out_dir / "ocr-documents.jsonl").exists()

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"allowed_roots": [""]},
            {"allowed_roots": ["{tmp_path}"], "data_paths": ["x"]},
        ],
    )
    def test_request_level_validation_exits_2(self, tmp_path, params):
        request = _op_request(
            tmp_path,
            [{"sourcePath": "missing.png"}],
            {
                key: (str(tmp_path) if value == "{tmp_path}" else value)
                for key, value in params.items()
            },
            None,
        )
        code, out, _ = _run_op(request)
        assert code == 2
        assert out == ""

    def test_empty_source_path_exits_2(self, tmp_path):
        request = _op_request(
            tmp_path,
            [{"sourcePath": ""}],
            {"allowed_roots": [str(tmp_path)]},
            None,
        )
        code, out, _ = _run_op(request)
        assert code == 2
        assert out == ""

    def test_relative_source_path_resolves_against_manifest(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        folder = tmp_path / "data"
        folder.mkdir()
        _png(folder / "image.png")
        request = _op_request(
            tmp_path,
            [{"sourcePath": "data/image.png"}],
            {"allowed_roots": [str(tmp_path)]},
            None,
        )
        code, out, err = _run_op(request)
        assert code == 0, err
        assert json.loads(out)["metrics"]["document_count"] == 1.0

    def test_unsupported_extension_row_level(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        skip = _op_request(
            tmp_path,
            [{"sourcePath": "notes.txt"}],
            {"allowed_roots": [str(tmp_path)], "on_error": "skip"},
            None,
        )
        code, out, err = _run_op(skip)
        assert code == 0, err
        assert json.loads(out)["items"][0]["error"]["code"] == "UNSUPPORTED_FORMAT"

        raise_request = _op_request(
            tmp_path,
            [{"sourcePath": "notes.txt"}],
            {"allowed_roots": [str(tmp_path)], "on_error": "raise"},
            None,
        )
        code, out, err = _run_op(raise_request)
        assert code == 1
        assert out == ""
        assert json.loads(err.strip().splitlines()[-1])["code"] == "UNSUPPORTED_FORMAT"

    def test_dependency_missing_is_exit_1(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "mindocr", None)
        monkeypatch.setitem(sys.modules, "paddleocr", None)
        _png(tmp_path / "good.png")
        request = _op_request(
            tmp_path,
            [{"sourcePath": "good.png"}],
            {"backend": "auto", "allowed_roots": [str(tmp_path)]},
            None,
        )
        code, out, err = _run_op(request)
        assert code == 1
        assert out == ""
        diagnostic = json.loads(err.strip().splitlines()[-1])
        assert diagnostic["code"] == "DEPENDENCY_MISSING"
        assert "ocr-mindocr" in diagnostic["message"]
        assert "ocr-paddleocr" in diagnostic["message"]

    def test_inline_limit(self, tmp_path, monkeypatch):
        _install_fake_mindocr(tmp_path, monkeypatch)
        _png(tmp_path / "good.png")
        import curator.trusted_worker.ops.ocr_image as module

        monkeypatch.setattr(module, "MAX_INLINE_BYTES", 1)
        request = _op_request(
            tmp_path,
            [{"sourcePath": "good.png"}],
            {"allowed_roots": [str(tmp_path)]},
            None,
        )
        code, out, err = _run_op(request)
        assert code == 1
        assert out == ""
        assert json.loads(err.strip().splitlines()[-1])["code"] == "OUTPUT_TOO_LARGE"

    def test_registries_and_operation_class(self):
        from curator.core.operation_registry import OperationRegistry
        from curator.trusted_worker.op.registry import OP_MODE_REGISTRY
        from curator.trusted_worker.registry import OP_REGISTRY

        assert OP_MODE_REGISTRY["ocr_image"]["required_fields"] == {"sourcePath": str}
        assert "ocr_image" not in OP_REGISTRY
        OperationRegistry._ensure_initialized()
        operation_class = OperationRegistry.get_operation_class("image_ocr")
        assert operation_class is not None
        instance = OperationRegistry.create_operation("image_ocr", {"data_paths": []})
        assert instance.__class__ is operation_class

    def test_image_ocr_block_contract(self):
        block = Block(
            block_id="b-000001",
            type="paragraph",
            page=1,
            sheet=None,
            bbox=[0.1, 0.1, 0.9, 0.2],
            text="hello",
            table=None,
            kv_pair=None,
            style=None,
            ocr_confidence=0.9,
        )
        assert block.ocr_confidence == 0.9
