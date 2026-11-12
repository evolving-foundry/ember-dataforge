"""PoC worker shim (RSI runtime onboarding design v0.2 §3.9.5 Step 2).

The implementation has moved to the production package ``curator.trusted_worker`` (evaluation mode,
v0.2.1 revised contract: ``plan.config.op`` / path references / exit codes 0/1/2 / failures do not write
result files). This file is retained only to:

* keep historical PoC tests launching the worker by file path;
* serve as a runtime guard that the direct-call path does not introduce data_designer.

New callers should use ``python -m curator.trusted_worker`` directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_worker_dependencies() -> None:
    """Allow running as a child process without ``pip install``-ed packages."""
    py_path = os.environ.get("PYTHONPATH", "")
    # tests/e2e_fixtures/<this file> → parents[2] = repo root (portable,
    # no machine-specific absolute path).
    needed = [str(Path(__file__).resolve().parents[3])]
    additions = [p for p in needed if p not in py_path.split(os.pathsep)]
    if additions:
        os.environ["PYTHONPATH"] = os.pathsep.join(additions + [py_path])


_resolve_worker_dependencies()

# Stub torch before any Curator import (augmentation/__init__.py cascades to torch).
if "torch" not in sys.modules:  # pragma: no cover
    import types

    sys.modules["torch"] = types.ModuleType("torch")

# Runtime proof of the zero-data_designer claim (AST tests alone are
# file-local; this covers the whole loaded import graph of the PoC entry).
if "data_designer" in sys.modules:
    print(
        "curator_quality_worker: data_designer unexpectedly in sys.modules",
        file=sys.stderr,
    )
    raise SystemExit(3)

from curator.trusted_worker.spawn import run  # noqa: E402  (re-export)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "usage: curator_quality_worker.py <input.json> <output.json>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(run(sys.argv[1], sys.argv[2]))
