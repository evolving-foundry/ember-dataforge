"""Architecture guards for the standalone environment synthesis chain."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENVSYNTH_ROOT = REPO_ROOT / "data_designer" / "pipelines" / "envsynth"
PUBLISHER = REPO_ROOT / "scripts" / "gym_publish_env.py"
ENVSYNTH_TESTS = REPO_ROOT / "tests" / "envsynth"


def _py_files(root: Path):
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_generic_layers_do_not_import_domain_modules():
    generic = [
        ENVSYNTH_ROOT / "environment_model.py",
        ENVSYNTH_ROOT / "grounding.py",
        ENVSYNTH_ROOT / "package_identity.py",
        ENVSYNTH_ROOT / "package.py",
        ENVSYNTH_ROOT / "gates.py",
        ENVSYNTH_ROOT / "domain_adapters" / "base.py",
    ] + _py_files(ENVSYNTH_ROOT / "profile_compilers")
    violations = []
    for path in generic:
        for module in _imported_modules(path):
            if module == "data_designer.pipelines.envsynth.domains" or module.startswith(
                "data_designer.pipelines.envsynth.domains."
            ):
                violations.append("{} imports {}".format(path, module))
    assert violations == []


def test_compilers_and_domain_adapters_do_not_depend_on_each_other():
    violations = []
    for path in _py_files(ENVSYNTH_ROOT / "profile_compilers"):
        for module in _imported_modules(path):
            if module == "data_designer.pipelines.envsynth.domain_adapters":
                violations.append("{} imports {}".format(path, module))
    for path in _py_files(ENVSYNTH_ROOT / "domain_adapters"):
        for module in _imported_modules(path):
            if module.startswith("data_designer.pipelines.envsynth.profile_compilers"):
                violations.append("{} imports {}".format(path, module))
    assert violations == []


def test_envsynth_source_has_no_rsi_runtime_imports():
    violations: list[str] = []
    for path in _py_files(ENVSYNTH_ROOT):
        text = path.read_text(encoding="utf-8")
        for token in ("RSIRunner", "rsi-runner"):
            if token in text:
                violations.append("{} contains {}".format(path, token))
    rsi_root = REPO_ROOT / "data_designer" / "rsi"
    assert not rsi_root.exists(), "data_designer.rsi has been removed entirely"
    assert violations == []


def test_publisher_supports_v01_store_and_file_mirror_without_runner():
    text = PUBLISHER.read_text(encoding="utf-8")
    for token in ("RSIRunner", "rsi-runner"):
        assert token not in text
    for token in ("storePath", "compileResultPath", "ArtifactCatalog"):
        assert token in text
    assert "load_release_metadata" in text


def test_legacy_v01_worker_entrypoints_exist():
    for name in ("source_import.py", "synthesis_worker.py"):
        path = REPO_ROOT / "data_designer" / "pipelines" / "envsynth" / "cli" / name
        assert path.is_file()
        assert "RSIRunner" not in path.read_text(encoding="utf-8")
