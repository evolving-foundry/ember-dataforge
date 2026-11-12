from __future__ import annotations

from typing import Any


def is_npu_available() -> bool:
    """Return whether an NPU is available to torch."""
    try:
        import torch_npu

        npu = getattr(torch_npu, "npu", None)
        if npu is not None and hasattr(npu, "is_available"):
            return bool(npu.is_available())
    except ImportError:
        return False
    except Exception:
        return False

    try:
        import torch

        npu = getattr(torch, "npu", None)
        return bool(npu is not None and npu.is_available())
    except Exception:
        return False


def is_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_torch_device(device_preference: str | None = "auto") -> str:
    """Resolve a torch device preference with NPU first for auto selection."""
    preference = str(device_preference or "auto").strip().lower()
    if preference in {"auto", "auto_device", "auto-device"}:
        if is_npu_available():
            return "npu"
        if is_cuda_available():
            return "cuda"
        return "cpu"

    if preference.startswith("npu"):
        if is_npu_available():
            return preference
        raise RuntimeError(f"NPU device requested but not available: {preference}")
    if preference.startswith("cuda"):
        if is_cuda_available():
            return preference
        raise RuntimeError(f"CUDA device requested but not available: {preference}")
    if preference == "cpu":
        return "cpu"

    raise ValueError(
        "Unsupported device preference: "
        f"{device_preference}. Expected auto, npu[:id], cuda[:id], or cpu."
    )


def device_kind(device: Any) -> str:
    value = str(device or "cpu").lower()
    if value.startswith("npu"):
        return "npu"
    if value.startswith("cuda"):
        return "cuda"
    return "cpu"


def is_npu_device(device: Any) -> bool:
    return device_kind(device) == "npu"


def is_cuda_device(device: Any) -> bool:
    return device_kind(device) == "cuda"
