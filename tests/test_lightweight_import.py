"""Lightweight import acceptance (separate process): SDK top level, api, and CLI metadata paths must not transitively load
Curator or the Torch stack—manifest/config/algorithm enumeration must work when the model stack is not installed
(design §11.1: do not conflate "dependencies installed" with "loaded at runtime")."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import json, sys
for module in {modules!r}:
    __import__(module)
loaded = [name for name in sys.modules
          if name == "curator" or name.startswith("curator.")
          or name == "torch" or name.startswith("torch.")]
print(json.dumps({{"loaded": loaded}}))
assert not loaded, f"heavyweight modules loaded: {{loaded}}"
"""


def _probe(modules: tuple[str, ...], *, cwd: Path = PROJECT_ROOT) -> None:
    process = subprocess.run(
        [sys.executable, "-c", _PROBE.format(modules=modules)],
        capture_output=True, text=True, cwd=str(cwd),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(PROJECT_ROOT),
             "HOME": str(cwd)},
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["loaded"] == []


def test_sdk_top_level_import_is_lightweight():
    _probe(("data_designer",))


def test_sdk_api_and_generation_metadata_are_lightweight():
    _probe((
        "data_designer.api",
        "data_designer.generation.algorithms.registry",
        "data_designer.config",
    ))


def test_cli_algorithms_list_is_lightweight():
    process = subprocess.run(
        [sys.executable, "-m", "data_designer.cli.main", "algorithms", "list"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(PROJECT_ROOT),
             "HOME": str(PROJECT_ROOT)},
    )
    assert process.returncode == 0, process.stderr
    entries = json.loads(process.stdout)
    assert entries, "algorithm registry listing must be non-empty"


@pytest.mark.parametrize("module", [
    "data_designer.pipelines.benchmarks.build",
    "data_designer.pipelines.benchmarks.suite",
    "data_designer.pipelines.benchmarks.reader",
])
def test_benchmark_metadata_modules_do_not_load_torch(module):
    # benchmark metadata/scoring modules may depend on curator, but must not transitively load the Torch stack.
    process = subprocess.run(
        [sys.executable, "-c", (
            "import sys; __import__(%r); "
            "assert not [m for m in sys.modules if m == 'torch' or m.startswith('torch.')]"
        ) % module],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(PROJECT_ROOT),
             "HOME": str(PROJECT_ROOT)},
    )
    assert process.returncode == 0, process.stderr
