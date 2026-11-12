"""TB21SeedAdapter: terminal-bench-style task directory -> seed_tasks rows (terminal ingestion M2).

Input is a task root directory unpacked in the harbor native layout (containing ``dataset.toml`` and
``tasks/``) or a ``tar.gz`` archive; output rows carry the T8 dual-criterion screening result:

    expressibility ("expressible" | "excluded") + exclusion_reason

Screening (T8, trusted-side automatic pre-screen + explicit override via
``task.toml [metadata] expressibility``): acceptance that depends on container runtime state (instruction keywords
such as listening ports and service processes) -> ``excluded/runtime_state``; initial state baked into the image
(``[environment] state_in_image = true``) -> ``excluded/state_in_image``.

Boundary: ``tests/`` and ``solution/`` are read only on demand as trusted-side assets (``test_script_text``
/``gold_document``/``solution_files``); their content never enters the seed row--public items and
source manifests carry only the instruction, exported setup files, and classification markers.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workbuddy_seed import _load_toml


class TB21SeedAdapterError(RuntimeError):
    """Invalid terminal seed archive structure / missing required files."""


EXCLUDED_REASON_RUNTIME_STATE = "runtime_state"
EXCLUDED_REASON_STATE_IN_IMAGE = "state_in_image"
_EXPRESSIBILITY_OVERRIDES = (
    "expressible",
    f"excluded:{EXCLUDED_REASON_RUNTIME_STATE}",
    f"excluded:{EXCLUDED_REASON_STATE_IN_IMAGE}",
)
# Automatic pre-screen for T8 criterion 1: acceptance depends on container runtime state (port listeners, service/daemon processes, background tasks).
_RUNTIME_STATE_RE = re.compile(
    r"(listening|listen\s+on|port\s*\d+|\bserver\b|\bservice\b|daemon"
    r"|systemctl|\bssh\b|background\s+process)",
    re.IGNORECASE,
)


@dataclass
class TB21SeedManifest:
    """Seed ingestion manifest (seed_source declaration is mandatory; see module docstring)."""

    seed_source: str
    dataset_id: str
    schema: str
    task_count: int
    archive_sha256: Optional[str] = None
    upstream_commit: Optional[str] = None
    expressible_count: int = 0
    excluded: Dict[str, int] = field(default_factory=dict)
    difficulties: Dict[str, int] = field(default_factory=dict)
    categories: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class TB21SeedAdapter:
    """terminal-bench-style task root directory -> seed_tasks rows + expressible subset classification.

    Usage:
        adapter = TB21SeedAdapter(dataset_dir)          # already-unpacked root directory
        adapter = TB21SeedAdapter.from_archive(path)    # automatic tar.gz unpacking
        rows = adapter.load_seed_tasks()
    """

    def __init__(
        self,
        dataset_dir: "str | Path",
        seed_source: Optional[str] = None,
        archive_sha256: Optional[str] = None,
        upstream_commit: Optional[str] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.archive_sha256 = archive_sha256
        self.upstream_commit = upstream_commit
        dataset_toml = self.dataset_dir / "dataset.toml"
        if not dataset_toml.is_file():
            raise TB21SeedAdapterError(
                f"dataset.toml does not exist: {dataset_toml} (an extracted task root is required)")
        meta = _load_toml(dataset_toml)
        dataset = meta.get("dataset") or {}
        self.dataset_id = str(dataset.get("id") or self.dataset_dir.name)
        self.schema_name = str(dataset.get("schema") or "")
        self.declared_task_count = int(dataset.get("task_count") or 0)
        if seed_source is not None:
            self.seed_source = str(seed_source)
        elif "fixture" in self.dataset_id:
            self.seed_source = "synthetic-fixture"
        else:
            self.seed_source = "upstream-checkout"

    @classmethod
    def from_archive(
        cls,
        archive_path: "str | Path",
        extract_dir: Optional["str | Path"] = None,
        seed_source: Optional[str] = None,
        upstream_commit: Optional[str] = None,
    ) -> "TB21SeedAdapter":
        """Unpack tar.gz (default temporary directory; validate a single top-level root directory) and return the adapter."""
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            raise TB21SeedAdapterError(f"Archive does not exist: {archive_path}")
        data = archive_path.read_bytes()
        archive_sha256 = hashlib.sha256(data).hexdigest()
        target = Path(extract_dir) if extract_dir else Path(
            tempfile.mkdtemp(prefix="tb21-seeds-"))
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(target)
        roots = [p for p in target.iterdir() if (p / "dataset.toml").is_file()]
        if len(roots) != 1:
            raise TB21SeedAdapterError(
                f"Archive top level must contain exactly one root with dataset.toml; found {len(roots)}")
        return cls(
            roots[0], seed_source=seed_source,
            archive_sha256=archive_sha256, upstream_commit=upstream_commit,
        )

# ── Task directories and seed rows ────────────────────────────────────────────────

    def task_dirs(self) -> List[Path]:
        root = self.dataset_dir / "tasks"
        if not root.is_dir():
            raise TB21SeedAdapterError(f"tasks/ directory does not exist: {root}")
        return sorted(p for p in root.iterdir() if p.is_dir())

    def load_seed_tasks(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for task_dir in self.task_dirs():
            rows.append(self._load_task(task_dir))
        if self.declared_task_count and len(rows) != self.declared_task_count:
            raise TB21SeedAdapterError(
                f"task_count mismatch: dataset.toml declares {self.declared_task_count}, "
                f"but parsed {len(rows)}")
        return rows

    def _load_task(self, task_dir: Path) -> Dict[str, Any]:
        task_toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        if not task_toml_path.is_file() or not instruction_path.is_file():
            raise TB21SeedAdapterError(
                f"Task directory is missing task.toml or instruction.md: {task_dir}")
        task_meta = _load_toml(task_toml_path)
        task_sec = task_meta.get("task") or {}
        meta_sec = task_meta.get("metadata") or {}
        env_sec = task_meta.get("environment") or {}
        instruction = instruction_path.read_text(encoding="utf-8").strip()
        if not instruction:
            raise TB21SeedAdapterError(f"instruction.md is empty: {task_dir}")

        raw_name = str(task_sec.get("name") or task_dir.name)
        seed_id = raw_name.split("/", 1)[-1]

        expressibility, exclusion_reason = self._classify(
            instruction, meta_sec, env_sec)
        keywords = [str(k) for k in (task_sec.get("keywords") or [])]
        tools = [str(t) for t in (env_sec.get("tools") or ["shell"])]
        setup_files = self._export_setup_files(task_dir)
        return {
            "seed_id": seed_id,
            "task_text": instruction,
            "tools_available": tools,
            "success_criteria": (
                "Deliver the target artifacts named by the instruction as the "
                "candidate deliverable JSON; they must pass the frozen offline "
                "file assertions re-expressed from the upstream acceptance "
                "signal."
            ),
            "difficulty": str(meta_sec.get("difficulty") or "unspecified"),
            "expressibility": expressibility,
            "exclusion_reason": exclusion_reason,
            "setup_files": setup_files,
            "docker_image": str(env_sec.get("docker_image") or "unspecified"),
            "category": str(meta_sec.get("category") or "unspecified"),
            "tb_keywords": keywords,
        }

    @staticmethod
    def _classify(
        instruction: str, meta_sec: Dict[str, Any], env_sec: Dict[str, Any]
    ) -> tuple[str, Optional[str]]:
        override = str(meta_sec.get("expressibility") or "").strip()
        if override:
            if override not in _EXPRESSIBILITY_OVERRIDES:
                raise TB21SeedAdapterError(
                    f"Invalid expressibility override: {override!r} (allowed values: "
                    f"{_EXPRESSIBILITY_OVERRIDES})")
            if override == "expressible":
                return "expressible", None
            return "excluded", override.split(":", 1)[1]
        if env_sec.get("state_in_image") is True:
            return "excluded", EXCLUDED_REASON_STATE_IN_IMAGE
        if _RUNTIME_STATE_RE.search(instruction):
            return "excluded", EXCLUDED_REASON_RUNTIME_STATE
        return "expressible", None

    @staticmethod
    def _export_setup_files(task_dir: Path) -> Dict[str, str]:
        """Initial workspace files are always exported from the task directory's environment/; no image is pulled or run."""
        environment_root = task_dir / "environment"
        if not environment_root.is_dir():
            return {}
        setup_files: Dict[str, str] = {}
        for path in sorted(environment_root.rglob("*")):
            if path.is_dir():
                continue
            if path.is_symlink() or not path.is_file():
                raise TB21SeedAdapterError(
                    f"environment/ may contain only regular files: {path}")
            setup_files[path.relative_to(environment_root).as_posix()] = (
                path.read_text(encoding="utf-8")
            )
        return setup_files

# ── Trusted-side derived assets (tests/ and solution/; never enter seed rows) ────────────

    def task_dir(self, seed_id: str) -> Path:
        for task_dir in self.task_dirs():
            raw_name = str((_load_toml(task_dir / "task.toml").get("task") or {})
                           .get("name") or task_dir.name)
            if raw_name.split("/", 1)[-1] == seed_id:
                return task_dir
        raise TB21SeedAdapterError(f"Task does not exist: {seed_id}")

    def test_script_text(self, seed_id: str) -> str:
        """Trusted re-expression material from upstream tests/: concatenate all *.py files (fixture level)."""
        tests_root = self.task_dir(seed_id) / "tests"
        if not tests_root.is_dir():
            raise TB21SeedAdapterError(f"Task is missing tests/: {seed_id}")
        parts = [
            path.read_text(encoding="utf-8")
            for path in sorted(tests_root.glob("*.py"))
        ]
        return "\n\n".join(parts)

    def gold_document(self, seed_id: str) -> Dict[str, Any]:
        gold_path = self.task_dir(seed_id) / "tests" / "gold.json"
        if not gold_path.is_file():
            raise TB21SeedAdapterError(f"Task is missing tests/gold.json: {seed_id}")
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        if not isinstance(gold, dict) or not gold:
            raise TB21SeedAdapterError(f"tests/gold.json must be a non-empty object: {seed_id}")
        return gold

    def solution_files(self, seed_id: str) -> Dict[str, str]:
        """Reference-solution artifact dictionary (gold deliverable material, exported from solution/)."""
        solution_root = self.task_dir(seed_id) / "solution"
        if not solution_root.is_dir():
            raise TB21SeedAdapterError(f"Task is missing solution/: {seed_id}")
        files: Dict[str, str] = {}
        for path in sorted(solution_root.rglob("*")):
            if path.is_dir():
                continue
            if path.is_symlink() or not path.is_file():
                raise TB21SeedAdapterError(
                    f"solution/ may contain only regular files: {path}")
            files[path.relative_to(solution_root).as_posix()] = (
                path.read_text(encoding="utf-8")
            )
        if not files:
            raise TB21SeedAdapterError(f"solution/ is empty: {seed_id}")
        return files

# ── Setup snapshot archive (environment pin material; excludes tests/ and solution/) ──

    def export_setup_archive(
        self, archive_path: "str | Path", seed_id: Optional[str] = None
    ) -> str:
        """Freeze instruction + exported setup files as tar.gz and return SHA-256.

        When ``seed_id`` is omitted, package all tasks; when specified, package only that task (per-case
        environment pin). ``tests/`` and ``solution/`` never enter the archive.
        """
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tar:
            for task_dir in self.task_dirs():
                raw_name = str((_load_toml(task_dir / "task.toml").get("task") or {})
                               .get("name") or task_dir.name)
                current_id = raw_name.split("/", 1)[-1]
                if seed_id is not None and current_id != seed_id:
                    continue
                tar.add(task_dir / "instruction.md",
                        arcname=f"{current_id}/instruction.md")
                environment_root = task_dir / "environment"
                if environment_root.is_dir():
                    for path in sorted(environment_root.rglob("*")):
                        if path.is_file():
                            tar.add(path, arcname=(
                                f"{current_id}/"
                                f"{path.relative_to(environment_root).as_posix()}"
                            ))
        return hashlib.sha256(archive_path.read_bytes()).hexdigest()

    # ── manifest ─────────────────────────────────────────────────────────

    def build_manifest(self) -> Dict[str, Any]:
        rows = self.load_seed_tasks()
        manifest = TB21SeedManifest(
            seed_source=self.seed_source,
            dataset_id=self.dataset_id,
            schema=self.schema_name,
            task_count=len(rows),
            archive_sha256=self.archive_sha256,
            upstream_commit=self.upstream_commit,
        )
        for row in rows:
            if row["expressibility"] == "expressible":
                manifest.expressible_count += 1
            else:
                reason = row["exclusion_reason"]
                manifest.excluded[reason] = manifest.excluded.get(reason, 0) + 1
            manifest.difficulties[row["difficulty"]] = (
                manifest.difficulties.get(row["difficulty"], 0) + 1)
            manifest.categories[row["category"]] = (
                manifest.categories.get(row["category"], 0) + 1)
        return manifest.as_dict()


__all__ = ["TB21SeedAdapter", "TB21SeedAdapterError", "TB21SeedManifest",
           "EXCLUDED_REASON_RUNTIME_STATE", "EXCLUDED_REASON_STATE_IN_IMAGE"]
