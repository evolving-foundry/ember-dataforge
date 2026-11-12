"""OCR backend/device routing for C-M3b/C-M3c."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .document_common import DocumentParseError
from .ocr_adapters import MindOcrAdapter, OcrAdapter, PaddleOcrAdapter
from .ocr_ranking import RankingTable, resolve_ranking_table

BACKEND_ORDER = ("mindocr", "paddleocr")
DEVICE_ORDER = ("npu", "cuda", "cpu")
_UNSET = object()
_MODULE_NAMES = {"mindocr": "mindocr", "paddleocr": "paddleocr"}
_EXTRA_NAMES = {"mindocr": "ocr-mindocr", "paddleocr": "ocr-paddleocr"}
_REASON_PRIORITY = {
    "explicit_backend": 0,
    "ranking_missing": 1,
    "device_unavailable": 2,
    "backend_unavailable": 3,
    "ranking_row_unavailable": 4,
    "ranking": 5,
}


@dataclass(frozen=True)
class OcrRoute:
    adapter: OcrAdapter
    backend: str
    backend_version: str
    device: str
    model_id: Optional[str]
    routing_reason: str
    ranking_version: Optional[str]
    ranking_evaluation_set: Optional[str]
    ranked: bool = False


AdapterFactory = Callable[[], OcrAdapter]
DeviceProbe = Callable[[str], bool]


def more_specific_reason(first: str, second: str) -> str:
    return first if _REASON_PRIORITY[first] <= _REASON_PRIORITY[second] else second


def default_adapter_factories(allow_model_download: bool = False) -> Dict[str, AdapterFactory]:
    return {
        "mindocr": lambda: MindOcrAdapter(
            allow_model_download=allow_model_download
        ),
        "paddleocr": lambda: PaddleOcrAdapter(
            allow_model_download=allow_model_download
        ),
    }


def _module_installed(module_name: str) -> bool:
    if module_name in sys.modules:
        return sys.modules[module_name] is not None
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _dependency_missing(backend: str) -> DocumentParseError:
    extra = _EXTRA_NAMES.get(backend, f"ocr-{backend}")
    return DocumentParseError(
        "DEPENDENCY_MISSING",
        f"OCR backend '{backend}' is not installed; install the '{extra}' extra",
        stage="validate",
    )


def _dependency_missing_all() -> DocumentParseError:
    extras = ", ".join(f"'{_EXTRA_NAMES[backend]}'" for backend in BACKEND_ORDER)
    return DocumentParseError(
        "DEPENDENCY_MISSING",
        f"no OCR backend is installed; install the {extras} extra",
        stage="validate",
    )


def _backend_unavailable(message: str) -> DocumentParseError:
    return DocumentParseError("OCR_BACKEND_UNAVAILABLE", message, stage="ocr")


@lru_cache(maxsize=8)
def default_device_probe(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False
    if device == "npu":
        try:
            import mindspore  # noqa: F401
            from mindspore import context

            target = str(context.get_context("device_target")).lower()
            if target in {"ascend", "npu"}:
                return True
        except Exception:
            pass
        try:
            import paddle

            if bool(paddle.device.is_compiled_with_custom_device("npu")):
                return True
        except Exception:
            pass
        try:
            import torch
            import torch_npu  # noqa: F401

            return bool(torch.npu.is_available())
        except Exception:
            return False
    return False


def _available_devices(probe: DeviceProbe) -> list[str]:
    devices = []
    for device in DEVICE_ORDER:
        try:
            available = bool(probe(device))
        except Exception:
            available = False
        if available:
            devices.append(device)
    return devices


def _make_adapter(factory: AdapterFactory) -> Optional[OcrAdapter]:
    try:
        adapter = factory()
    except Exception:
        return None
    return adapter if isinstance(adapter, OcrAdapter) else None


def _backend_version(adapter: OcrAdapter) -> str:
    value = adapter.version
    return value if isinstance(value, str) and value else "unknown"


def _model_id(adapter: OcrAdapter, model_paths: Mapping[str, str]) -> Optional[str]:
    if model_paths:
        return None
    value = adapter.default_model_id
    return value if isinstance(value, str) and value else None


def _capabilities_ok(adapter: OcrAdapter, lang: str, layout_analysis: bool) -> bool:
    return lang in adapter.capabilities.languages and (
        not layout_analysis or adapter.capabilities.layout
    )


def _supports_device(adapter: OcrAdapter, device: str) -> bool:
    return device in adapter.supported_devices


def _route(
    adapter: OcrAdapter,
    device: str,
    reason: str,
    table: Optional[RankingTable],
    *,
    ranked: bool = False,
    model_paths: Optional[Mapping[str, str]] = None,
) -> OcrRoute:
    return OcrRoute(
        adapter=adapter,
        backend=adapter.name,
        backend_version=_backend_version(adapter),
        device=device,
        model_id=_model_id(adapter, model_paths or {}),
        routing_reason=reason,
        ranking_version=None if table is None else table.ranking_version,
        ranking_evaluation_set=None if table is None else table.evaluation_set_version,
        ranked=ranked,
    )


def _fallback_routes(
    adapters: Mapping[str, OcrAdapter],
    device: str,
    lang: str,
    layout_analysis: bool,
    reason: str,
    table: Optional[RankingTable],
    model_paths: Mapping[str, str],
) -> list[OcrRoute]:
    routes: list[OcrRoute] = []
    for backend in BACKEND_ORDER:
        adapter = adapters.get(backend)
        if adapter is None:
            continue
        if not adapter.dependency_installed():
            continue
        if not _supports_device(adapter, device) or not _capabilities_ok(
            adapter, lang, layout_analysis
        ):
            continue
        routes.append(_route(adapter, device, reason, table, model_paths=model_paths))
    return routes


def _installed_adapters(
    factories: Mapping[str, AdapterFactory],
) -> Dict[str, OcrAdapter]:
    result: Dict[str, OcrAdapter] = {}
    for backend in BACKEND_ORDER:
        factory = factories.get(backend)
        if factory is None:
            continue
        adapter = _make_adapter(factory)
        if adapter is not None and adapter.dependency_installed():
            result[backend] = adapter
    return result


def _any_extra_installed(
    factories: Mapping[str, AdapterFactory],
    installed: Mapping[str, OcrAdapter],
) -> bool:
    if installed:
        return True
    for backend in BACKEND_ORDER:
        if factories.get(backend) is None and _module_installed(
            _MODULE_NAMES[backend]
        ):
            return True
    return False


def _explicit_route(
    backend: str,
    device: str,
    lang: str,
    layout_analysis: bool,
    model_paths: Mapping[str, str],
    factories: Mapping[str, AdapterFactory],
    probe: DeviceProbe,
) -> list[OcrRoute]:
    factory = factories.get(backend)
    if factory is None:
        if not _module_installed(_MODULE_NAMES.get(backend, backend)):
            raise _dependency_missing(backend)
        raise _backend_unavailable(f"OCR backend '{backend}' is unavailable in this build")
    adapter = _make_adapter(factory)
    if adapter is None:
        raise _backend_unavailable(f"OCR backend '{backend}' could not be initialized")
    if not adapter.dependency_installed():
        raise _dependency_missing(backend)
    if not _capabilities_ok(adapter, lang, layout_analysis):
        raise _backend_unavailable(f"OCR backend '{backend}' does not satisfy the request")
    if device == "auto":
        selected = next(
            (
                candidate
                for candidate in _available_devices(probe)
                if _supports_device(adapter, candidate)
            ),
            None,
        )
        if selected is None:
            raise _backend_unavailable(f"OCR backend '{backend}' has no available device")
    else:
        if device not in adapter.supported_devices:
            raise _backend_unavailable(f"OCR backend '{backend}' does not support {device}")
        try:
            available = bool(probe(device))
        except Exception:
            available = False
        if not available:
            raise _backend_unavailable(f"OCR device '{device}' is unavailable")
        selected = device
    return [_route(adapter, selected, "explicit_backend", None, model_paths=model_paths)]


def _device_rows(table: RankingTable, device: str) -> list[Any]:
    return [row for row in table.rows if row.device == device and row.eligible]


def _ranked_routes(
    table: RankingTable,
    device: str,
    installed: Mapping[str, OcrAdapter],
    lang: str,
    layout_analysis: bool,
    reason: str,
) -> list[OcrRoute]:
    rows = sorted(
        _device_rows(table, device),
        key=lambda row: (
            row.cer,
            -row.line_accuracy,
            -row.layout_f1,
            -row.throughput_pages_s,
            row.backend,
        ),
    )
    routes: list[OcrRoute] = []
    seen = set()
    for row in rows:
        adapter = installed.get(row.backend)
        if adapter is None or row.backend in seen:
            continue
        if _backend_version(adapter) != row.backend_version:
            continue
        if adapter.default_model_id != row.model_id:
            continue
        if not _supports_device(adapter, device):
            continue
        if not _capabilities_ok(adapter, lang, layout_analysis):
            continue
        seen.add(row.backend)
        routes.append(_route(adapter, device, reason, table, ranked=True))
    return routes


def select_ocr_routes(
    *,
    backend: str,
    device: str,
    lang: str,
    layout_analysis: bool,
    model_paths: Optional[Mapping[str, str]] = None,
    allow_model_download: bool = False,
    backend_ranking_path: Optional[str] = None,
    adapter_factories: Optional[Mapping[str, AdapterFactory]] = None,
    device_probe: Optional[DeviceProbe] = None,
    ranking: Any = _UNSET,
    environ: Optional[Mapping[str, str]] = None,
    package_ranking_path: Any = _UNSET,
) -> list[OcrRoute]:
    if backend not in {"auto", *BACKEND_ORDER}:
        raise DocumentParseError("INVALID_PARAMETER", "backend is invalid", stage="validate")
    if device not in {"auto", *DEVICE_ORDER}:
        raise DocumentParseError("INVALID_PARAMETER", "device is invalid", stage="validate")
    if lang not in {"zh", "en", "mixed"}:
        raise DocumentParseError("INVALID_PARAMETER", "lang is invalid", stage="validate")
    if not isinstance(layout_analysis, bool):
        raise DocumentParseError("INVALID_PARAMETER", "layout_analysis must be boolean", stage="validate")
    if not isinstance(allow_model_download, bool):
        raise DocumentParseError("INVALID_PARAMETER", "allow_model_download must be boolean", stage="validate")
    paths = dict(model_paths or {})
    if backend == "auto" and paths:
        raise DocumentParseError(
            "INVALID_PARAMETER", "model_paths requires an explicit backend", stage="validate"
        )
    factories = dict(
        adapter_factories or default_adapter_factories(allow_model_download)
    )
    probe = device_probe or default_device_probe
    if backend != "auto":
        return _explicit_route(
            backend, device, lang, layout_analysis, paths, factories, probe
        )

    if ranking is _UNSET:
        kwargs = {"environ": environ}
        if package_ranking_path is not _UNSET:
            kwargs["package_path"] = package_ranking_path
        table = resolve_ranking_table(backend_ranking_path, **kwargs)
    else:
        table = ranking
    if table is not None and not isinstance(table, RankingTable):
        raise DocumentParseError(
            "OCR_RANKING_INVALID", "ranking table has an invalid type", stage="validate"
        )

    installed = _installed_adapters(factories)
    if not _any_extra_installed(factories, installed):
        raise _dependency_missing_all()
    available_devices = _available_devices(probe)
    if not available_devices:
        raise _backend_unavailable("no OCR device is available")
    if device != "auto":
        if device not in available_devices:
            raise _backend_unavailable(f"OCR device '{device}' is unavailable")
        available_devices = [device]
    if table is None:
        selected = next(
            (
                candidate
                for candidate in available_devices
                if _fallback_routes(
                    installed, candidate, lang, layout_analysis, "ranking_missing", None, paths
                )
            ),
            available_devices[0],
        )
        routes = _fallback_routes(
            installed,
            selected,
            lang,
            layout_analysis,
            "ranking_missing",
            None,
            paths,
        )
        if routes:
            return routes
        raise _backend_unavailable("no installed OCR backend satisfies the request")

    devices_with_rows = [
        candidate
        for candidate in available_devices
        if any(row.backend in factories for row in _device_rows(table, candidate))
    ]
    downgraded = bool(devices_with_rows and devices_with_rows[0] != available_devices[0])
    if devices_with_rows:
        selected = devices_with_rows[0]
        base_reason = "device_unavailable" if downgraded else "ranking"
        ranked = _ranked_routes(table, selected, installed, lang, layout_analysis, base_reason)
        fallback_reason = (
            "device_unavailable"
            if downgraded
            else more_specific_reason(base_reason, "backend_unavailable")
        )
        fallback = _fallback_routes(
            installed, selected, lang, layout_analysis, fallback_reason, table, paths
        )
        routes = ranked + [
            route
            for route in fallback
            if not any(existing.backend == route.backend for existing in ranked)
        ]
        if routes:
            return routes
    else:
        selected = available_devices[0]
        fallback = _fallback_routes(
            installed,
            selected,
            lang,
            layout_analysis,
            "ranking_row_unavailable",
            table,
            paths,
        )
        if fallback:
            return fallback
    raise _backend_unavailable("no OCR backend satisfies the ranking and request")


__all__ = [
    "BACKEND_ORDER",
    "DEVICE_ORDER",
    "OcrRoute",
    "default_adapter_factories",
    "default_device_probe",
    "more_specific_reason",
    "select_ocr_routes",
]
