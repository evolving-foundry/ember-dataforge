"""Regression tests for lazy augmentation operation exports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_augmentation_package_does_not_import_torch():
    code = """
import sys
import curator.operations.augmentation as augmentation

assert "torch" not in sys.modules, sorted(sys.modules)
from curator.operations.augmentation import AugmentationOperation

assert AugmentationOperation.__name__ == "AugmentationOperation"
assert "torch" not in sys.modules, sorted(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
