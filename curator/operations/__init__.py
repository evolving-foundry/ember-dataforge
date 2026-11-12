"""Operation module initialization.

Submodules are imported on demand. The document parsing subprocess only needs ``operations.loading.document_parsing``;
importing the ``operations`` package must not pull in heavy dependencies such as torch/transformers.
"""
from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = (
    "loading",
    "saving",
    "cleaning",
    "filtering",
    "augmentation",
    "formatting",
    "deduplication",
    "selection",
    "aggregation",
    "grouping",
    "mapping",
    "common",
)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    # Preserve the old ``from curator.operations import SomeOperation``
    # surface without eagerly importing every optional-dependency submodule.
    if not name.endswith("Operation"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for submodule in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{submodule}")
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    *_SUBMODULES,
]
