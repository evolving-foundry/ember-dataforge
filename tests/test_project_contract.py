"""Repository-level contracts that must stay aligned with packaging metadata."""

from __future__ import annotations

import re
from pathlib import Path

import curator
from curator.core.operation_registry import OperationRegistry


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def test_generation_operator_surface_is_single_and_declared():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert re.search(
        r'^synthetic_data_generation\s*=\s*'
        r'"data_designer\.operators:SyntheticDataGenerationOperation"$',
        text,
        re.MULTILINE,
    )
    assert "data_expansion" not in text
    assert "cross_modal_generation" not in text
    assert OperationRegistry._generation_operator_names == frozenset(
        {"synthetic_data_generation"}
    )


def test_python_requirement_and_graph_extra_are_unified():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert re.search(r'^requires-python\s*=\s*">=3\.10"$', text, re.MULTILINE)
    assert "networkx>=3.2" in text
    assert "python_version < '3.10'" not in text


def test_default_curator_config_path_exists():
    assert curator.DEFAULT_CONFIG_PATH == "config/curator/global.yaml"
    assert (ROOT / curator.DEFAULT_CONFIG_PATH).is_file()


def test_default_curator_config_is_declared_for_wheel_install():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "[tool.setuptools.data-files]" in text
    assert '"config/curator" = ["config/curator/global.yaml"]' in text


def test_retired_tooling_and_rsi_surfaces_are_absent():
    # `packages` and `tests/ts` are re-admitted top-level TS surfaces (D10),
    # not retired: they must exist alongside the Python packages.
    assert (ROOT / "packages").is_dir()
    assert (ROOT / "tests" / "ts").is_dir()
    for relative in (
        "scripts/curator_tools",
        "data_designer/rsi/protocol",
        "data_designer/rsi/adapters",
        "data_designer/rsi/ops",
        "data_designer/rsi/policy",
        "data_designer/rsi/storage/lineage.py",
    ):
        assert not (ROOT / relative).exists(), relative


def test_pipelines_live_under_data_designer():
    pipelines = ROOT / "data_designer" / "pipelines"
    for name in ("benchmarks", "envsynth", "recipes"):
        assert (pipelines / name).is_dir(), name
        assert not (ROOT / "data_designer" / name).exists(), name
