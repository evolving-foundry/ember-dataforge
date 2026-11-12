"""
Plugin registry - discovers, loads, registers, and unregisters third-party operators

Plugin packaging specification (plugin.json example):
{
    'plugin_id': 'my-nlp-operators',
    'version': '1.0.0',
    'author': 'community-dev',
    'description': 'Operators specialized for Chinese NLP',
    'min_framework_version': '1.0.0',
    'operators': [
        {
            'name': 'chinese_word_segmentation',
            'class': 'my_plugin.operators.nlp.ChineseWordSegmentationOperation',
            'category': 'mapping',
            'description': 'Chinese word segmentation operator based on jieba'
        }
    ]
}

Installation methods:
    registry = PluginRegistry()
    registry.install_from_path('./my_plugin/')
    registry.install_from_pypi('my-nlp-operators')
    registry.auto_discover()          # Scan standard paths such as ~/.mdp/plugins/

entry_points declaration (setup.py / pyproject.toml):
    entry_points={'mdp.operators': ['op_name = my_module:MyOpClass']}
"""

import importlib
import json
import logging
import sys
from datetime import datetime

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OperatorPluginInfo(BaseModel):
    """Metadata for an installed plugin."""
    plugin_id: str
    version: str
    author: str = "unknown"
    description: str = ""
    install_path: str = ""
    installed_at: datetime = Field(default_factory=datetime.now)
    operators: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "active"  # active | disabled | error


class PluginRegistry:
    """Plugin registry.

    Plugin lookup paths (in priority order):
      1. plugin_dirs passed to the constructor by the user
      2. ~/.mdp/plugins/
      3. .mdp_plugins/ under the current working directory
      4. Installed Python packages that declare mdp.operators entry_points

    All operators are ultimately registered through OperationRegistry.register() and remain transparent to Pipeline.
    """

    REGISTRY_FILE = ".cache/installed_plugins.json"
    DEFAULT_SCAN_DIRS = [
        str(Path.home() / ".mdp" / "plugins"),
        ".mdp_plugins",
    ]

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self._extra_dirs: List[str] = plugin_dirs or []
        self._installed: Dict[str, OperatorPluginInfo] = {}
        self._load_registry()

    # ── Install ───────────────────────────────────────────────────────────────

    def install_from_path(self, plugin_path: str) -> OperatorPluginInfo:
        """Install a plugin from a local directory.

        Workflow:
          1. Read plugin.json
          2. Add the plugin directory to sys.path
          3. Import operator classes one by one and register them in OperationRegistry
          4. Persist the installation record
        """
        path = Path(plugin_path).resolve()
        plugin_json = path / "plugin.json"
        if not plugin_json.exists():
            raise FileNotFoundError(f"plugin.json not found in {plugin_path}")

        with open(plugin_json, encoding="utf-8") as f:
            meta = json.load(f)

        self._check_framework_version(meta.get("min_framework_version"))

        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        loaded = self._load_operator_classes(meta.get("operators", []))

        info = OperatorPluginInfo(
            plugin_id=meta["plugin_id"],
            version=meta.get("version", "0.0.0"),
            author=meta.get("author", "unknown"),
            description=meta.get("description", ""),
            install_path=str(path),
            operators=loaded,
        )
        self._installed[info.plugin_id] = info
        self._save_registry()
        logger.info("[PluginRegistry] Installed plugin '%s' (%s operators)", info.plugin_id, len(loaded))
        return info

    def install_from_pypi(
        self, package_name: str, version: Optional[str] = None
    ) -> OperatorPluginInfo:
        """Install a plugin package from PyPI, then discover operators through entry_points."""
        import subprocess

        pkg_spec = f"{package_name}=={version}" if version else package_name
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg_spec, "--quiet"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pip install failed for '{pkg_spec}':\n{result.stderr}"
            )
        return self._load_from_entry_points(package_name)

    # ── Auto discovery ────────────────────────────────────────────────────────

    def auto_discover(self) -> int:
        """Scan all standard plugin directories and automatically install discovered plugins. Return the number of loaded plugins."""
        search_dirs = self._extra_dirs + self.DEFAULT_SCAN_DIRS
        loaded_count = 0

        for dir_path in search_dirs:
            p = Path(dir_path)
            if not p.exists():
                continue
            for plugin_dir in p.iterdir():
                if plugin_dir.is_dir() and (plugin_dir / "plugin.json").exists():
                    if plugin_dir.name in self._installed:
                        continue  # Already installed; skip
                    try:
                        self.install_from_path(str(plugin_dir))
                        loaded_count += 1
                    except Exception as e:
                        logger.error("[PluginRegistry] Failed to load %s: %s", plugin_dir.name, e)

        # Also scan Python entry_points
        loaded_count += self._discover_from_entry_points()
        return loaded_count

    # ── Uninstall ─────────────────────────────────────────────────────────────

    def uninstall(self, plugin_id: str) -> None:
        """Remove operators from OperationRegistry and delete the installation record."""
        if plugin_id not in self._installed:
            raise KeyError(f"Plugin '{plugin_id}' is not installed")

        from ..core.pipeline import OperationRegistry

        info = self._installed.pop(plugin_id)
        for op in info.operators:
            OperationRegistry._operations.pop(op.get("name", ""), None)

        self._save_registry()
        logger.info("[PluginRegistry] Uninstalled plugin '%s'", plugin_id)

    # ── Query ─────────────────────────────────────────────────────────────────

    def list_installed(self) -> List[OperatorPluginInfo]:
        """List all installed plugins."""
        return list(self._installed.values())

    def get_plugin(self, plugin_id: str) -> Optional[OperatorPluginInfo]:
        return self._installed.get(plugin_id)

    # ── Internal implementation ───────────────────────────────────────────────

    def _load_operator_classes(
        self, operator_defs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Import operator classes and register them in OperationRegistry; return the successfully loaded definitions."""
        from ..core.pipeline import OperationRegistry

        loaded = []
        for op_def in operator_defs:
            name = op_def.get("name", "")
            class_path = op_def.get("class", "")
            if not name or not class_path:
                continue
            try:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                op_class = getattr(module, class_name)
                OperationRegistry.register(name, op_class)
                loaded.append(op_def)
                logger.debug("[PluginRegistry] Registered operator: %s", name)
            except Exception as e:
                logger.warning("[PluginRegistry] Failed to load '%s': %s", name, e)
        return loaded

    def _load_from_entry_points(self, package_name: str) -> OperatorPluginInfo:
        """Discover and register operators through the Python entry_points mechanism."""
        from ..core.pipeline import OperationRegistry

        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="mdp.operators")
        except Exception:
            eps = []

        loaded = []
        for ep in eps:
            try:
                op_class = ep.load()
                OperationRegistry.register(ep.name, op_class)
                loaded.append({"name": ep.name, "class": f"{op_class.__module__}.{op_class.__name__}"})
                logger.debug("[PluginRegistry] Registered via entry_points: %s", ep.name)
            except Exception as e:
                logger.warning("[PluginRegistry] Failed to load entry_point '%s': %s", ep.name, e)

        info = OperatorPluginInfo(
            plugin_id=package_name,
            version="unknown",
            description=f"Loaded via entry_points: {package_name}",
            operators=loaded,
        )
        if loaded:
            self._installed[package_name] = info
            self._save_registry()
        return info

    def _discover_from_entry_points(self) -> int:
        """Globally scan all mdp.operators entry_points and return the number of newly registered operators."""
        try:
            from importlib.metadata import entry_points
            eps = list(entry_points(group="mdp.operators"))
        except Exception:
            return 0

        from ..core.pipeline import OperationRegistry

        count = 0
        for ep in eps:
            if ep.name in OperationRegistry._operations:
                continue  # Already registered
            try:
                op_class = ep.load()
                OperationRegistry.register(ep.name, op_class)
                count += 1
            except Exception as e:
                logger.warning(
                    "[PluginRegistry] Failed to register operator entry_point '%s': %s", ep.name, e
                )
        return count

    @staticmethod
    def _check_framework_version(min_version: Optional[str]) -> None:
        """Check whether the framework version satisfies the plugin's minimum requirement."""
        if not min_version:
            return
        try:
            from packaging.version import Version
            import curator
            fw_version = getattr(curator, "__version__", "0.0.0")
            if Version(fw_version) < Version(min_version):
                raise RuntimeError(
                    f"Plugin requires framework version >= {min_version}, "
                    f"but current version is {fw_version}"
                )
        except ImportError:
            pass  # Skip version checking when packaging is not installed

    def _load_registry(self) -> None:
        """Load installed-plugin records from disk."""
        registry_path = Path(self.REGISTRY_FILE)
        if not registry_path.exists():
            return
        try:
            with open(registry_path, encoding="utf-8") as f:
                data = json.load(f)
            for plugin_id, info_dict in data.items():
                try:
                    self._installed[plugin_id] = OperatorPluginInfo(**info_dict)
                except Exception as e:
                    logger.warning("[PluginRegistry] Warning: Failed to load entry '%s': %s", plugin_id, e)
        except Exception as e:
            logger.warning("[PluginRegistry] Warning: Failed to load registry file: %s", e)

    def _save_registry(self) -> None:
        """Persist installed-plugin records to disk."""
        path = Path(self.REGISTRY_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {pid: info.model_dump(mode="json") for pid, info in self._installed.items()},
                f, indent=2, default=str, ensure_ascii=False,
            )
