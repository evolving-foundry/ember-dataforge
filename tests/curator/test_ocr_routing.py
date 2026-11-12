"""OCR ranking-table loading and auto-routing tests."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from curator.operations.loading.document_common import DocumentParseError
from curator.operations.loading.ocr_adapters import (
    OcrAdapter,
    OcrAdapterError,
    OcrCandidate,
    OcrCapabilities,
    OcrImage,
    OcrRequest,
)
from curator.operations.loading.ocr_ranking import (
    RankingTable,
    parse_ranking_document,
    resolve_ranking_table,
)
from curator.operations.loading.ocr_routing import (
    default_adapter_factories,
    more_specific_reason,
    select_ocr_routes,
)


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    yield


def _row(
    backend: str = "mindocr",
    *,
    version: str = "1.0",
    device: str = "cpu",
    model: str = "model-a",
    cer: float = 0.1,
    line_accuracy: float = 0.9,
    layout_f1: float = 0.8,
    throughput: float = 2.0,
    eligible: bool = True,
) -> dict:
    return {
        "backend": backend,
        "backend_version": version,
        "device": device,
        "model_id": model,
        "cer": cer,
        "line_accuracy": line_accuracy,
        "layout_f1": layout_f1,
        "throughput_pages_s": throughput,
        "eligible": eligible,
    }


def _table(rows: list[dict], *, version: str = "rank-v1", evaluation: str = "eval-v1"):
    return parse_ranking_document(
        {
            "schema_version": "ocr-ranking-v1",
            "ranking_version": version,
            "evaluation_set_version": evaluation,
            "primary_metric": "cer",
            "rows": rows,
        }
    )


class FakeAdapter(OcrAdapter):
    def __init__(
        self,
        *,
        name: str = "mindocr",
        version: str = "1.0",
        model_id: str | None = "model-a",
        devices: tuple[str, ...] = ("cpu",),
        languages: tuple[str, ...] = ("zh", "en", "mixed"),
        layout: bool = True,
        table_grid: bool = False,
        confidence: bool = True,
        installed: bool = True,
        error: OcrAdapterError | None = None,
        result: tuple[OcrCandidate, ...] = (),
    ) -> None:
        self.name = name
        self._version = version
        self._model_id = model_id
        self.supported_devices = frozenset(devices)
        self.capabilities = OcrCapabilities(
            languages=frozenset(languages),
            layout=layout,
            table_grid=table_grid,
            confidence=confidence,
        )
        self.installed = installed
        self.error = error
        self.result = result

    @property
    def version(self) -> str:
        return self._version

    @property
    def default_model_id(self) -> str | None:
        return self._model_id

    def dependency_installed(self) -> bool:
        return self.installed

    def recognize(self, image: OcrImage, request: OcrRequest) -> list[OcrCandidate]:
        if self.error is not None:
            raise self.error
        return list(self.result)


def _factory(adapter: FakeAdapter):
    return lambda: adapter


def _probe(*devices: str):
    return lambda device: device in devices


class TestRankingParsing:
    def test_valid_table(self):
        table = _table([_row()])
        assert isinstance(table, RankingTable)
        assert table.ranking_version == "rank-v1"
        assert table.evaluation_set_version == "eval-v1"
        assert table.rows[0].backend == "mindocr"

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {
                "schema_version": "ocr-ranking-v1",
                "ranking_version": "v1",
                "evaluation_set_version": "e1",
                "primary_metric": "wer",
                "rows": [_row()],
            },
            {
                "schema_version": "ocr-ranking-v2",
                "ranking_version": "v1",
                "evaluation_set_version": "e1",
                "primary_metric": "cer",
                "rows": [_row()],
            },
            {
                "schema_version": "ocr-ranking-v1",
                "ranking_version": "",
                "evaluation_set_version": "e1",
                "primary_metric": "cer",
                "rows": [_row()],
            },
            {
                "schema_version": "ocr-ranking-v1",
                "ranking_version": "v1",
                "evaluation_set_version": "e1",
                "primary_metric": "cer",
                "rows": [],
            },
            {
                "schema_version": "ocr-ranking-v1",
                "ranking_version": "v1",
                "evaluation_set_version": "e1",
                "primary_metric": "cer",
                "rows": [{**_row(), "unknown": 1}],
            },
            {
                "schema_version": "ocr-ranking-v1",
                "ranking_version": "v1",
                "evaluation_set_version": "e1",
                "primary_metric": "cer",
                "rows": [_row(), _row()],
            },
        ],
    )
    def test_invalid_tables(self, payload):
        with pytest.raises(DocumentParseError) as exc:
            parse_ranking_document(payload)
        assert exc.value.code == "OCR_RANKING_INVALID"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"cer": float("nan")},
            {"line_accuracy": 1.5},
            {"layout_f1": -0.1},
            {"throughput_pages_s": 0.0},
            {"device": "gpu"},
            {"eligible": 1},
            {"backend": ""},
        ],
    )
    def test_invalid_row_values(self, overrides):
        payload = {
            "schema_version": "ocr-ranking-v1",
            "ranking_version": "v1",
            "evaluation_set_version": "e1",
            "primary_metric": "cer",
            "rows": [{**_row(), **overrides}],
        }
        with pytest.raises(DocumentParseError) as exc:
            parse_ranking_document(payload)
        assert exc.value.code == "OCR_RANKING_INVALID"

    def test_tiny_positive_throughput_is_valid(self):
        table = _table([_row(throughput=1e-12)])
        assert table.rows[0].throughput_pages_s == 1e-12


def _write_table(path: Path, *, marker: str = "rank-v1", rows=None) -> Path:
    rows = rows or [_row()]
    lines = [
        "schema_version: ocr-ranking-v1",
        f"ranking_version: {marker}",
        "evaluation_set_version: eval-v1",
        "primary_metric: cer",
        "rows:",
    ]
    for row in rows:
        lines.append(f"  - backend: '{row['backend']}'")
        lines.append(f"    backend_version: '{row['backend_version']}'")
        lines.append(f"    device: '{row['device']}'")
        lines.append(f"    model_id: '{row['model_id']}'")
        lines.append(f"    cer: {row['cer']}")
        lines.append(f"    line_accuracy: {row['line_accuracy']}")
        lines.append(f"    layout_f1: {row['layout_f1']}")
        lines.append(f"    throughput_pages_s: {row['throughput_pages_s']}")
        lines.append(f"    eligible: {'true' if row['eligible'] else 'false'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestRankingSources:
    def test_parameter_wins_over_environment_and_package(self, tmp_path):
        parameter = _write_table(tmp_path / "parameter.yaml", marker="from-param")
        environment = _write_table(tmp_path / "env.yaml", marker="from-env")
        package = _write_table(tmp_path / "package.yaml", marker="from-package")
        table = resolve_ranking_table(
            str(parameter),
            environ={"CURATOR_OCR_RANKING": str(environment)},
            package_path=package,
        )
        assert table.ranking_version == "from-param"

    def test_environment_wins_over_package(self, tmp_path):
        environment = _write_table(tmp_path / "env.yaml", marker="from-env")
        package = _write_table(tmp_path / "package.yaml", marker="from-package")
        table = resolve_ranking_table(
            None,
            environ={"CURATOR_OCR_RANKING": str(environment)},
            package_path=package,
        )
        assert table.ranking_version == "from-env"

    def test_missing_package_file_means_no_ranking(self, tmp_path):
        assert (
            resolve_ranking_table(None, environ={}, package_path=tmp_path / "missing.yaml")
            is None
        )

    def test_missing_parameter_path_is_invalid_parameter(self, tmp_path):
        with pytest.raises(DocumentParseError) as exc:
            resolve_ranking_table(str(tmp_path / "missing.yaml"), environ={})
        assert exc.value.code == "INVALID_PARAMETER"

    def test_missing_environment_path_is_ranking_invalid(self, tmp_path):
        with pytest.raises(DocumentParseError) as exc:
            resolve_ranking_table(
                None,
                environ={"CURATOR_OCR_RANKING": str(tmp_path / "missing.yaml")},
            )
        assert exc.value.code == "OCR_RANKING_INVALID"

    def test_broken_environment_file_is_ranking_invalid(self, tmp_path):
        broken = tmp_path / "broken.yaml"
        broken.write_text("rows: [", encoding="utf-8")
        with pytest.raises(DocumentParseError) as exc:
            resolve_ranking_table(
                None, environ={"CURATOR_OCR_RANKING": str(broken)}
            )
        assert exc.value.code == "OCR_RANKING_INVALID"

    def test_broken_package_file_is_ranking_invalid(self, tmp_path):
        broken = tmp_path / "broken.yaml"
        broken.write_text("rows: []\n", encoding="utf-8")
        with pytest.raises(DocumentParseError) as exc:
            resolve_ranking_table(None, environ={}, package_path=broken)
        assert exc.value.code == "OCR_RANKING_INVALID"

    def test_url_parameter_is_invalid_parameter(self):
        with pytest.raises(DocumentParseError) as exc:
            resolve_ranking_table("https://example.test/ranking.yaml", environ={})
        assert exc.value.code == "INVALID_PARAMETER"

    def test_size_limit(self, tmp_path, monkeypatch):
        path = _write_table(tmp_path / "table.yaml")
        import curator.operations.loading.ocr_ranking as module

        monkeypatch.setattr(module, "MAX_RANKING_FILE_BYTES", 1)
        with pytest.raises(DocumentParseError) as exc:
            resolve_ranking_table(str(path), environ={})
        assert exc.value.code == "OCR_RANKING_INVALID"

    def test_line_limit(self, tmp_path, monkeypatch):
        path = _write_table(tmp_path / "table.yaml")
        import curator.operations.loading.ocr_ranking as module

        monkeypatch.setattr(module, "MAX_RANKING_LINES", 1)
        with pytest.raises(DocumentParseError) as exc:
            resolve_ranking_table(str(path), environ={})
        assert exc.value.code == "OCR_RANKING_INVALID"

    def test_custom_yaml_tag_is_rejected(self, tmp_path):
        path = tmp_path / "unsafe.yaml"
        path.write_text("!!python/object:os.system [echo bad]\n", encoding="utf-8")
        with pytest.raises(DocumentParseError) as exc:
            resolve_ranking_table(str(path), environ={})
        assert exc.value.code == "OCR_RANKING_INVALID"


class TestAutoRouting:
    def test_no_ranking_prefers_mindocr(self):
        mindocr = FakeAdapter(name="mindocr")
        paddle = FakeAdapter(name="paddleocr")
        routes = select_ocr_routes(
            backend="auto",
            device="auto",
            lang="zh",
            layout_analysis=True,
            adapter_factories={
                "mindocr": _factory(mindocr),
                "paddleocr": _factory(paddle),
            },
            device_probe=_probe("cpu"),
            ranking=None,
        )
        assert [route.backend for route in routes] == ["mindocr", "paddleocr"]
        assert routes[0].routing_reason == "ranking_missing"
        assert routes[0].ranking_version is None

    def test_no_ranking_uses_paddle_when_mindocr_missing(self):
        mindocr = FakeAdapter(name="mindocr", installed=False)
        paddle = FakeAdapter(name="paddleocr")
        routes = select_ocr_routes(
            backend="auto",
            device="auto",
            lang="zh",
            layout_analysis=True,
            adapter_factories={
                "mindocr": _factory(mindocr),
                "paddleocr": _factory(paddle),
            },
            device_probe=_probe("cpu"),
            ranking=None,
        )
        assert [route.backend for route in routes] == ["paddleocr"]

    def test_all_extras_missing_lists_both(self):
        with pytest.raises(DocumentParseError) as exc:
            select_ocr_routes(
                backend="auto",
                device="auto",
                lang="zh",
                layout_analysis=True,
                adapter_factories={
                    "mindocr": _factory(FakeAdapter(installed=False)),
                    "paddleocr": _factory(
                        FakeAdapter(name="paddleocr", installed=False)
                    ),
                },
                device_probe=_probe("cpu"),
                ranking=None,
            )
        assert exc.value.code == "DEPENDENCY_MISSING"
        assert "ocr-mindocr" in exc.value.message
        assert "ocr-paddleocr" in exc.value.message

    def test_capability_mismatch_is_backend_unavailable(self):
        with pytest.raises(DocumentParseError) as exc:
            select_ocr_routes(
                backend="auto",
                device="auto",
                lang="zh",
                layout_analysis=True,
                adapter_factories={
                    "mindocr": _factory(FakeAdapter(languages=("en",)))
                },
                device_probe=_probe("cpu"),
                ranking=None,
            )
        assert exc.value.code == "OCR_BACKEND_UNAVAILABLE"

    def test_device_priority_is_npu_then_cuda_then_cpu(self):
        routes = select_ocr_routes(
            backend="auto",
            device="auto",
            lang="zh",
            layout_analysis=False,
            adapter_factories={
                "mindocr": _factory(FakeAdapter(devices=("npu", "cuda", "cpu")))
            },
            device_probe=_probe("npu", "cuda", "cpu"),
            ranking=None,
        )
        assert routes[0].device == "npu"

    def test_explicit_device_must_be_available(self):
        with pytest.raises(DocumentParseError) as exc:
            select_ocr_routes(
                backend="auto",
                device="cuda",
                lang="zh",
                layout_analysis=False,
                adapter_factories={
                    "mindocr": _factory(FakeAdapter(devices=("cuda",)))
                },
                device_probe=_probe("cpu"),
                ranking=None,
            )
        assert exc.value.code == "OCR_BACKEND_UNAVAILABLE"

    def test_ranking_sorts_by_metrics(self):
        mindocr = FakeAdapter(name="mindocr")
        paddle = FakeAdapter(name="paddleocr")
        table = _table(
            [
                _row("mindocr", cer=0.2, line_accuracy=0.99),
                _row("paddleocr", cer=0.1, line_accuracy=0.5),
            ]
        )
        routes = select_ocr_routes(
            backend="auto",
            device="auto",
            lang="zh",
            layout_analysis=False,
            adapter_factories={
                "mindocr": _factory(mindocr),
                "paddleocr": _factory(paddle),
            },
            device_probe=_probe("cpu"),
            ranking=table,
        )
        assert routes[0].backend == "paddleocr"
        assert routes[0].routing_reason == "ranking"
        assert routes[0].ranked is True
        assert routes[0].ranking_version == "rank-v1"

    def test_official_default_model_ids_match_ranking_rows(self, monkeypatch):
        from curator.operations.loading.ocr_adapters import (
            MindOcrAdapter,
            PaddleOcrAdapter,
        )

        monkeypatch.setitem(sys.modules, "mindocr", types.ModuleType("mindocr"))
        monkeypatch.setitem(sys.modules, "paddleocr", types.ModuleType("paddleocr"))
        mindocr = MindOcrAdapter()
        paddleocr = PaddleOcrAdapter()
        assert mindocr.default_model_id
        assert paddleocr.default_model_id

        table = _table(
            [
                _row(
                    "mindocr",
                    version=mindocr.version,
                    model=mindocr.default_model_id,
                    cer=0.2,
                ),
                _row(
                    "paddleocr",
                    version=paddleocr.version,
                    model=paddleocr.default_model_id,
                    cer=0.1,
                ),
            ]
        )
        routes = select_ocr_routes(
            backend="auto",
            device="auto",
            lang="zh",
            layout_analysis=False,
            adapter_factories=default_adapter_factories(),
            device_probe=_probe("cpu"),
            ranking=table,
        )
        assert [route.backend for route in routes] == ["paddleocr", "mindocr"]
        assert all(route.ranked for route in routes)
        assert routes[0].model_id == paddleocr.default_model_id
        assert routes[1].model_id == mindocr.default_model_id

    def test_model_paths_route_has_null_model_id(self):
        routes = select_ocr_routes(
            backend="mindocr",
            device="cpu",
            lang="zh",
            layout_analysis=False,
            model_paths={"det": "/tmp/det", "rec": "/tmp/rec"},
            adapter_factories={"mindocr": _factory(FakeAdapter())},
            device_probe=_probe("cpu"),
        )
        assert routes[0].model_id is None
        assert routes[0].routing_reason == "explicit_backend"

    def test_version_mismatch_falls_back(self):
        mindocr = FakeAdapter(name="mindocr", version="2.0")
        table = _table([_row("mindocr", version="1.0")])
        routes = select_ocr_routes(
            backend="auto",
            device="auto",
            lang="zh",
            layout_analysis=False,
            adapter_factories={"mindocr": _factory(mindocr)},
            device_probe=_probe("cpu"),
            ranking=table,
        )
        assert routes[0].routing_reason == "backend_unavailable"
        assert routes[0].ranked is False
        assert routes[0].ranking_version == "rank-v1"

    def test_model_mismatch_falls_back(self):
        mindocr = FakeAdapter(name="mindocr", model_id="other-model")
        table = _table([_row("mindocr", model="model-a")])
        routes = select_ocr_routes(
            backend="auto",
            device="auto",
            lang="zh",
            layout_analysis=False,
            adapter_factories={"mindocr": _factory(mindocr)},
            device_probe=_probe("cpu"),
            ranking=table,
        )
        assert routes[0].routing_reason == "backend_unavailable"

    def test_device_downgrade_is_recorded(self):
        table = _table([_row("mindocr", device="cpu")])
        routes = select_ocr_routes(
            backend="auto",
            device="auto",
            lang="zh",
            layout_analysis=False,
            adapter_factories={
                "mindocr": _factory(FakeAdapter(devices=("npu", "cuda", "cpu")))
            },
            device_probe=_probe("npu", "cuda", "cpu"),
            ranking=table,
        )
        assert routes[0].device == "cpu"
        assert routes[0].routing_reason == "device_unavailable"

    def test_no_ranking_rows_is_ranking_row_unavailable(self):
        table = _table([_row("mindocr", device="cuda")])
        routes = select_ocr_routes(
            backend="auto",
            device="auto",
            lang="zh",
            layout_analysis=False,
            adapter_factories={"mindocr": _factory(FakeAdapter())},
            device_probe=_probe("cpu"),
            ranking=table,
        )
        assert routes[0].routing_reason == "ranking_row_unavailable"

    def test_ineligible_rows_are_unavailable(self):
        table = _table([_row("mindocr", eligible=False)])
        routes = select_ocr_routes(
            backend="auto",
            device="auto",
            lang="zh",
            layout_analysis=False,
            adapter_factories={"mindocr": _factory(FakeAdapter())},
            device_probe=_probe("cpu"),
            ranking=table,
        )
        assert routes[0].routing_reason == "ranking_row_unavailable"

    def test_model_paths_requires_explicit_backend(self):
        with pytest.raises(DocumentParseError) as exc:
            select_ocr_routes(
                backend="auto",
                device="cpu",
                lang="zh",
                layout_analysis=False,
                model_paths={"det": "/tmp/model"},
                adapter_factories={"mindocr": _factory(FakeAdapter())},
                device_probe=_probe("cpu"),
                ranking=None,
            )
        assert exc.value.code == "INVALID_PARAMETER"

    def test_explicit_backend_does_not_read_ranking(self):
        routes = select_ocr_routes(
            backend="mindocr",
            device="auto",
            lang="zh",
            layout_analysis=False,
            backend_ranking_path="/does/not/exist.yaml",
            adapter_factories={"mindocr": _factory(FakeAdapter())},
            device_probe=_probe("cpu"),
        )
        assert routes[0].routing_reason == "explicit_backend"
        assert routes[0].ranking_version is None

    def test_explicit_missing_dependency(self):
        with pytest.raises(DocumentParseError) as exc:
            select_ocr_routes(
                backend="mindocr",
                device="auto",
                lang="zh",
                layout_analysis=False,
                adapter_factories={
                    "mindocr": _factory(FakeAdapter(installed=False))
                },
                device_probe=_probe("cpu"),
            )
        assert exc.value.code == "DEPENDENCY_MISSING"
        assert "ocr-mindocr" in exc.value.message

    def test_explicit_backend_capability_failure(self):
        with pytest.raises(DocumentParseError) as exc:
            select_ocr_routes(
                backend="mindocr",
                device="auto",
                lang="zh",
                layout_analysis=True,
                adapter_factories={
                    "mindocr": _factory(FakeAdapter(layout=False))
                },
                device_probe=_probe("cpu"),
            )
        assert exc.value.code == "OCR_BACKEND_UNAVAILABLE"

    def test_default_device_probe_accepts_paddle_npu(self, monkeypatch):
        from curator.operations.loading import ocr_routing

        paddle = types.ModuleType("paddle")
        paddle.device = types.SimpleNamespace(
            is_compiled_with_custom_device=lambda name: name == "npu"
        )
        monkeypatch.setitem(sys.modules, "mindspore", None)
        monkeypatch.setitem(sys.modules, "paddle", paddle)
        ocr_routing.default_device_probe.cache_clear()
        try:
            assert ocr_routing.default_device_probe("npu") is True
        finally:
            ocr_routing.default_device_probe.cache_clear()

    def test_more_specific_reason_priority(self):
        assert more_specific_reason("ranking", "device_unavailable") == "device_unavailable"
        assert more_specific_reason("ranking_missing", "ranking") == "ranking_missing"
        assert more_specific_reason("ranking", "backend_unavailable") == "backend_unavailable"
