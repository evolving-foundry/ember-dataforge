"""Import-boundary tests for optional heavy dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_recipes_registry_does_not_import_torch():
    code = """
import sys
import data_designer.pipelines.recipes as recipes

assert "torch" not in sys.modules, sorted(sys.modules)
assert sorted(recipes.RECIPE_EXECUTOR_CLASSES) == [
    "agent_task",
    "dense_caption",
    "knowledge_reasoning_vqa",
    "video_temporal_qa",
    "workbuddy_task",
]
assert "torch" not in sys.modules, sorted(sys.modules)
from data_designer.pipelines.recipes import get_recipe_executor_class

executor = get_recipe_executor_class("agent_task")
assert executor.__name__ == "AgentTaskRecipeExecutor"
assert "torch" not in sys.modules, sorted(sys.modules)
"""
    result = _run_python(code)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_curator_augmentation_package_is_lazy():
    code = """
import sys
import curator.operations.augmentation as augmentation

assert "torch" not in sys.modules, sorted(sys.modules)
from curator.operations.augmentation import AugmentationOperation

assert AugmentationOperation.__name__ == "AugmentationOperation"
assert "torch" not in sys.modules, sorted(sys.modules)
"""
    result = _run_python(code)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr