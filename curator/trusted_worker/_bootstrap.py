"""Subprocess startup prerequisite: torch loading strategy.

``curator``'s eager import chain cascades to ``import torch``
(the augmentation operator family), but trusted_worker's operator surface (deduplication / quality assessment /
text overlap) does not depend on torch. The subprocess entry point calls
:meth:`ensure_torch_stub` before importing any Curator module:

* If the host has torch installed → load the real torch (the nltk→scipy chain calls getattr on ``torch.Tensor``,
  and an empty stub would break it; see the tokenizer backends in text_overlap);
* If torch is absent → inject an empty module with a ``Tensor`` placeholder so the import chain does not break.
"""
from __future__ import annotations

import sys
import types


def ensure_torch_stub() -> None:
    """Call before importing curator."""
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    else:
        return
    if "torch" not in sys.modules:
        stub = types.ModuleType("torch")
        # scipy.array_api_compat calls getattr(torch, 'Tensor') during import to perform
        # issubclass checks; provide a placeholder class that issubclass can use.
        stub.Tensor = type("Tensor", (), {})
        stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules["torch"] = stub
