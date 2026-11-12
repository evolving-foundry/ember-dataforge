"""Global architecture guard: capability layers must not import upper layers.

Layout convention (see ``data_designer/pipelines/__init__.py``):

- Capability layers — ``contracts``, ``sources``, ``storage``, ``verifiers``,
  ``generation``, ``operators`` — are domain-agnostic building blocks.
- Vertical product pipelines live under ``pipelines/`` (``envsynth``,
  ``benchmarks``, ``recipes``) and compose the layers.
- ``integrations`` and ``cli`` are bridges / entry points; they may import
  from both sides, so they are exempt here.

Dependency direction is one-way: pipelines / bridges → capability layers.
This test fails if any capability layer imports pipelines, host adapters, or
the CLI (directly or lazily).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "data_designer"

CAPABILITY_LAYERS = (
    "contracts",
    "sources",
    "storage",
    "verifiers",
    "generation",
    "operators",
    "rubric",
)

FORBIDDEN_PREFIXES = (
    "data_designer.pipelines",
    "data_designer.integrations",
    "data_designer.cli",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_capability_layers_do_not_import_upper_layers():
    violations = []
    for layer in CAPABILITY_LAYERS:
        layer_root = PKG_ROOT / layer
        for path in sorted(layer_root.rglob("*.py")):
            if "__pycache__" in path.parts or "fixtures" in path.parts:
                continue
            for module in _imported_modules(path):
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in FORBIDDEN_PREFIXES
                ):
                    violations.append("{} imports {}".format(path, module))
    assert violations == [], "\n".join(violations)
