"""Offline OCR benchmark and ranking-writer regression tests."""
from __future__ import annotations

import json

import pytest
import yaml

pytest.importorskip("pydantic")
PIL = pytest.importorskip("PIL")

from curator.operations.loading.ocr_adapters import (
    OcrAdapter,
    OcrCapabilities,
    OcrCandidate,
    OcrImage,
    OcrRequest,
)
from curator.operations.loading.ocr_benchmark import (
    BenchmarkError,
    load_benchmark_manifest,
    run_benchmark,
    write_ranking_table,
)
from curator.operations.loading.ocr_ranking import parse_ranking_document


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    """The benchmark tests do not need a Ray cluster."""
    yield


class _BenchmarkAdapter(OcrAdapter):
    name = "mindocr"
    supported_devices = frozenset({"cpu"})
    capabilities = OcrCapabilities(
        languages=frozenset({"zh", "en", "mixed"}),
        layout=True,
        table_grid=False,
        confidence=True,
    )

    def __init__(self, candidates, *, version: str = "1.2.3", model_id="model-a"):
        self._candidates = tuple(candidates)
        self._version = version
        self._model_id = model_id
        self.close_calls = 0

    @property
    def version(self) -> str:
        return self._version

    @property
    def default_model_id(self):
        return self._model_id

    @classmethod
    def dependency_installed(cls) -> bool:
        return True

    def recognize(self, image: OcrImage, request: OcrRequest):
        return list(self._candidates)

    def close(self):
        self.close_calls += 1


def _candidate(
    text: str = "AC",
    *,
    bbox=(0.0, 0.0, 1.0, 1.0),
    order: int = 0,
) -> OcrCandidate:
    return OcrCandidate(
        page=1,
        order=order,
        block_type="paragraph",
        text=text,
        bbox=bbox,
        confidence=0.8,
    )


def _fixture(tmp_path, *, texts=("AB",), boxes=None, candidates=None):
    image_path = tmp_path / "case.png"
    PIL.Image.new("RGB", (20, 10), "white").save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    boxes = list(boxes) if boxes is not None else [[0, 0, 1, 1] for _ in texts]
    types = ["paragraph" for _ in texts]
    manifest.write_text(
        json.dumps(
            {
                "imagePath": image_path.name,
                "texts": list(texts),
                "boxes": boxes,
                "blockTypes": types,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = _BenchmarkAdapter(candidates if candidates is not None else [_candidate()])
    return manifest, adapter


def _run(manifest, adapter):
    return run_benchmark(
        manifest,
        backend="mindocr",
        device="cpu",
        lang="zh",
        layout_analysis=True,
        ranking_version="rank-test-1",
        evaluation_set_version="eval-test-1",
        adapter_factories={"mindocr": lambda: adapter},
        device_probe=lambda device: device == "cpu",
    )


class TestManifest:
    def test_loads_relative_image_and_normalizes_text(self, tmp_path):
        manifest, _ = _fixture(tmp_path, texts=("  AB  ",))
        cases = load_benchmark_manifest(manifest)
        assert cases[0].texts == ("AB",)
        assert cases[0].boxes == ((0.0, 0.0, 1.0, 1.0),)
        assert cases[0].block_types == ("paragraph",)

    def test_rejects_unknown_keys_and_mismatched_arrays(self, tmp_path):
        image_path = tmp_path / "case.png"
        PIL.Image.new("RGB", (20, 10), "white").save(image_path)
        manifest = tmp_path / "manifest.jsonl"
        base = {
            "imagePath": image_path.name,
            "texts": ["A"],
            "boxes": [[0, 0, 1, 1]],
            "blockTypes": ["paragraph"],
        }
        manifest.write_text(json.dumps({**base, "extra": 1}) + "\n", encoding="utf-8")
        with pytest.raises(BenchmarkError, match="exactly"):
            load_benchmark_manifest(manifest)
        manifest.write_text(
            json.dumps({**base, "boxes": []}) + "\n", encoding="utf-8"
        )
        with pytest.raises(BenchmarkError, match="one entry per text"):
            load_benchmark_manifest(manifest)

    def test_rejects_duplicate_images_and_outside_roots(self, tmp_path):
        manifest, _ = _fixture(tmp_path)
        line = manifest.read_text(encoding="utf-8")
        manifest.write_text(line + line, encoding="utf-8")
        with pytest.raises(BenchmarkError, match="duplicate"):
            load_benchmark_manifest(manifest)

        other = tmp_path / "other"
        other.mkdir()
        with pytest.raises(BenchmarkError, match="outside allowed roots"):
            load_benchmark_manifest(manifest, allowed_roots=[str(other)])


class TestMetrics:
    def test_cer_line_accuracy_and_layout_f1(self, tmp_path):
        manifest, adapter = _fixture(tmp_path, texts=("AB",))
        payload = _run(manifest, adapter)
        row = payload["rows"][0]
        assert row["cer"] == 0.5
        assert row["line_accuracy"] == 0.0
        assert row["layout_f1"] == 1.0
        assert row["throughput_pages_s"] > 0
        assert row["backend"] == "mindocr"
        assert row["backend_version"] == "1.2.3"
        assert row["model_id"] == "model-a"
        parse_ranking_document(payload)

    def test_exact_prediction_scores_perfectly(self, tmp_path):
        manifest, adapter = _fixture(tmp_path, texts=("AB",), candidates=[_candidate("AB")])
        row = _run(manifest, adapter)["rows"][0]
        assert row["cer"] == 0.0
        assert row["line_accuracy"] == 1.0
        assert row["layout_f1"] == 1.0

    def test_empty_prediction_does_not_fabricate_quality(self, tmp_path):
        manifest, adapter = _fixture(tmp_path, texts=("AB",), candidates=[])
        row = _run(manifest, adapter)["rows"][0]
        assert row["cer"] == 1.0
        assert row["line_accuracy"] == 0.0
        assert row["layout_f1"] == 0.0

    def test_reading_order_and_word_box_merging_are_applied(self, tmp_path):
        candidates = [
            _candidate("EF", bbox=(0.0, 0.6, 0.2, 0.8), order=0),
            _candidate("CD", bbox=(0.3, 0.1, 0.5, 0.3), order=1),
            _candidate("AB", bbox=(0.0, 0.1, 0.2, 0.3), order=2),
        ]
        manifest, adapter = _fixture(
            tmp_path,
            texts=("AB CD", "EF"),
            boxes=[[0.0, 0.1, 0.5, 0.3], [0.0, 0.6, 0.2, 0.8]],
            candidates=candidates,
        )

        row = _run(manifest, adapter)["rows"][0]

        assert row["cer"] == 0.0
        assert row["line_accuracy"] == 1.0
        assert row["layout_f1"] == 1.0

    def test_comparison_ignores_case_space_and_punctuation(self, tmp_path):
        manifest, adapter = _fixture(
            tmp_path,
            texts=("Invoice No: 42",),
            candidates=[_candidate("invoiceno42")],
        )

        row = _run(manifest, adapter)["rows"][0]

        assert row["cer"] == 0.0
        assert row["line_accuracy"] == 1.0

    def test_version_and_model_overrides_are_required_when_unknown(self, tmp_path):
        manifest, adapter = _fixture(tmp_path, candidates=[_candidate("AB")])
        adapter._version = "unknown"
        adapter._model_id = None
        with pytest.raises(BenchmarkError, match="backend version"):
            _run(manifest, adapter)


class TestWriter:
    def test_writer_emits_strict_ranking_yaml(self, tmp_path):
        manifest, adapter = _fixture(tmp_path, candidates=[_candidate("AB")])
        payload = _run(manifest, adapter)
        output = tmp_path / "ranking.yaml"
        write_ranking_table(payload, output)
        loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
        table = parse_ranking_document(loaded)
        assert table.ranking_version == "rank-test-1"
        assert table.evaluation_set_version == "eval-test-1"
        assert table.rows[0].backend == "mindocr"

    def test_writer_merges_and_replaces_backend_rows(self, tmp_path):
        manifest, adapter = _fixture(tmp_path, candidates=[_candidate("AB")])
        payload = _run(manifest, adapter)
        output = tmp_path / "ranking.yaml"
        write_ranking_table(payload, output)

        second = json.loads(json.dumps(payload))
        second["rows"][0]["backend"] = "paddleocr"
        second["rows"][0]["cer"] = 0.25
        write_ranking_table(second, output, merge=True)
        table = parse_ranking_document(yaml.safe_load(output.read_text(encoding="utf-8")))
        assert [row.backend for row in table.rows] == ["mindocr", "paddleocr"]

        replacement = json.loads(json.dumps(second))
        replacement["rows"][0]["cer"] = 0.1
        write_ranking_table(replacement, output, merge=True)
        table = parse_ranking_document(yaml.safe_load(output.read_text(encoding="utf-8")))
        assert [row.backend for row in table.rows] == ["mindocr", "paddleocr"]
        assert table.rows[1].cer == 0.1

    def test_writer_rejects_merge_with_different_versions(self, tmp_path):
        manifest, adapter = _fixture(tmp_path, candidates=[_candidate("AB")])
        payload = _run(manifest, adapter)
        output = tmp_path / "ranking.yaml"
        write_ranking_table(payload, output)
        changed = json.loads(json.dumps(payload))
        changed["ranking_version"] = "rank-test-2"
        with pytest.raises(BenchmarkError, match="ranking_version"):
            write_ranking_table(changed, output, merge=True)


def test_run_benchmark_closes_adapter_once(tmp_path):
    manifest, adapter = _fixture(tmp_path)

    _run(manifest, adapter)

    assert adapter.close_calls == 1
