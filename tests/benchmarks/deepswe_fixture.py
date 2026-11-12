"""Shared deepswe M1/M2 fixture: mini git repo, tree snapshot, pinned assets.

Builds the trusted-side inputs the offline derivation chain needs: a three-task
``deepswe-tree-snapshot-v1`` tree manifest over a synthetic checkout, a real
``git diff --binary`` gold diff (text + binary hunk) produced in a throwaway
git repo, base workspace zips, a runtime registry pinning the real
``runners/deepswe_offline.py``, and a deepswe offline registry admitting one
task (gold + applicable base), one excluded task (gold does not apply →
``base_unavailable``), and one with no gold at all (``gold_unavailable``).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from data_designer.pipelines.benchmarks.contracts import file_sha256
from data_designer.pipelines.benchmarks.runners.deepswe_offline import VERSION

RUNNER_PATH = Path(__file__).resolve().parents[2] / (
    "data_designer/pipelines/benchmarks/runners/deepswe_offline.py"
)

TASK_ID = "deepswe-fixture-1"
BASE_TASK_ID = "deepswe-fixture-2"
NOGOLD_TASK_ID = "deepswe-fixture-3"
RELEASE = "fixture-release-1"

BASE_COMMIT = "b" * 40
TREE_COMMIT = "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea"  # Pinned upstream commit shape (synthetic fixture)
REPOSITORY_URL = "https://fixture.invalid/deep-swe"

BINARY_BASE = bytes(range(256)) + b"\x00binary fixture tail"
BINARY_NEW = b"\x89PNG\r\n\x1a\n" + bytes(reversed(range(256)))

BASE_FILES = {
    "src/counter.py": "value = 1\n",
    "README.md": "# fixture repo\n",
    "assets/logo.bin": BINARY_BASE,
}
GOLD_NEW = {
    "src/counter.py": "value = 2\n",
    "assets/logo.bin": BINARY_NEW,
}

BASE_TWO_FILES = {
    "src/counter.py": "value = 10\n",
    "README.md": "# fixture repo\n",
}
GOLD_NEW_TWO = {"src/counter.py": "value = 11\n"}

INSTRUCTION = (
    "DeepSWE fixture: repair the counter module so the tracked state advances "
    "exactly once per update and the shipped logo asset is refreshed."
)
INSTRUCTION_TWO = (
    "DeepSWE fixture two: repair the second counter module so the tracked "
    "state advances exactly once per concurrent update."
)
INSTRUCTION_NOGOLD = (
    "DeepSWE fixture three: repair the orphan counter module; no gold diff "
    "was ever pinned for this task."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=fixture@invalid",
         "-c", "user.name=fixture", *args],
        check=True, capture_output=True,
    )


def _diff_binary(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
        check=True, capture_output=True,
    )
    return completed.stdout.decode("utf-8")


def _mini_repo(files: dict) -> Path:
    repo = Path(tempfile.mkdtemp(prefix="deepswe-fixture-repo-"))
    for relative, content in files.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(content)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _gold_patch() -> str:
    """Real ``git diff --binary`` gold diff over the mini repo (M1 validation)."""
    repo = _mini_repo(BASE_FILES)
    for relative, content in GOLD_NEW.items():
        target = repo / relative
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(content)
    return _diff_binary(repo)


def _gold_patch_two() -> str:
    # generated against a different pre-image than base-two.zip, so the hunk
    # context ("value = 100") never matches the pinned base → base_unavailable
    repo = _mini_repo(
        {"src/counter.py": "value = 100\n", "README.md": "# fixture repo\n"}
    )
    (repo / "src/counter.py").write_text(
        GOLD_NEW_TWO["src/counter.py"], encoding="utf-8"
    )
    return _diff_binary(repo)


def corrupted_patch() -> str:
    # tamper the removed line so git apply must reject the text hunk
    return _gold_patch().replace("-value = 1\n", "-value = 1 tampered\n")


def _write_zip(path: Path, files: dict) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for relative, content in files.items():
            archive.writestr(relative, content)


def prepare_area(area: Path) -> dict:
    """Write the tree checkout, tree.json, zips, diffs, and both registries."""
    area.mkdir(parents=True, exist_ok=True)

    # ── tree checkout: three tasks, each with task.toml / instruction.md / tests / env ──
    tree_root = area / "tree"
    specs = [
        (TASK_ID, "tasks/fixture-1", BASE_COMMIT, INSTRUCTION),
        (BASE_TASK_ID, "tasks/fixture-2", "c" * 40, INSTRUCTION_TWO),
        (NOGOLD_TASK_ID, "tasks/fixture-3", "d" * 40, INSTRUCTION_NOGOLD),
    ]
    task_entries = []
    for task_id, directory, base_commit, instruction in specs:
        task_dir = tree_root / directory
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)
        (task_dir / "environment").mkdir(parents=True, exist_ok=True)
        (task_dir / "task.toml").write_text(
            f'[task]\nid = "{task_id}"\n', encoding="utf-8"
        )
        (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
        (task_dir / "tests/test.sh").write_text(
            "# official grader placeholder (never executed offline)\n",
            encoding="utf-8",
        )
        (task_dir / "environment/env.txt").write_text(
            "fixture environment provenance\n", encoding="utf-8"
        )
        files = {}
        for relative in ("task.toml", "instruction.md", "tests/test.sh",
                         "environment/env.txt"):
            files[relative] = sha256_bytes((task_dir / relative).read_bytes())
        task_entries.append(
            {
                "task_id": task_id,
                "base_commit_hash": base_commit,
                "language": "python",
                "directory": directory,
                "files": files,
            }
        )
    tree_json = area / "tree.json"
    tree_json.write_text(
        json.dumps(
            {
                "schema_version": "deepswe-tree-snapshot-v1",
                "commit": TREE_COMMIT,
                "repository_url": REPOSITORY_URL,
                "root": str(tree_root),
                "tasks": task_entries,
            }
        ),
        encoding="utf-8",
    )

    # ── gold/corrupted diffs and base snapshot (real git diff --binary) ──────────
    gold = area / "gold.diff"
    gold.write_text(_gold_patch(), encoding="utf-8")
    corrupted = area / "corrupted.diff"
    corrupted.write_text(corrupted_patch(), encoding="utf-8")
    gold_two = area / "gold-two.diff"
    gold_two.write_text(_gold_patch_two(), encoding="utf-8")
    base_zip = area / "base.zip"
    _write_zip(base_zip, BASE_FILES)
    base_zip_two = area / "base-two.zip"
    _write_zip(base_zip_two, BASE_TWO_FILES)

    # ── runtime registry (pins the real deepswe_offline runner) ──────────────────
    environment = area / "environment.txt"
    environment.write_text("frozen fixture environment", encoding="utf-8")
    verifier_pin = {
        "path": str(RUNNER_PATH),
        "sha256": file_sha256(RUNNER_PATH),
        "version": VERSION,
        "command": [sys.executable, str(RUNNER_PATH)],
    }
    runtime_registry = area / "registry.json"
    runtime_registry.write_text(
        json.dumps(
            {
                "schema_version": "benchmark-runtime-registry-v1",
                "cases": {
                    task_id: {
                        "environment": {
                            "path": str(environment),
                            "sha256": file_sha256(environment),
                        },
                        "verifier": dict(verifier_pin),
                        "assets": [],
                    }
                    for task_id in (TASK_ID, BASE_TASK_ID, NOGOLD_TASK_ID)
                },
            }
        ),
        encoding="utf-8",
    )

    # ── deepswe offline registry: one expressible task, one base_unavailable ──
    offline_registry = area / "offline-bases.json"
    offline_registry.write_text(
        json.dumps(
            {
                "schema_version": "deepswe-offline-registry-v1",
                "cases": {
                    TASK_ID: {
                        "gold_diff": {
                            "path": str(gold),
                            "sha256": sha256_bytes(gold.read_bytes()),
                        },
                        "base_snapshot": {
                            "path": str(base_zip),
                            "sha256": sha256_bytes(base_zip.read_bytes()),
                        },
                    },
                    BASE_TASK_ID: {
                        # gold diff whose removal line never exists in its base
                        "gold_diff": {
                            "path": str(gold_two),
                            "sha256": sha256_bytes(gold_two.read_bytes()),
                        },
                        "base_snapshot": {
                            "path": str(base_zip_two),
                            "sha256": sha256_bytes(base_zip_two.read_bytes()),
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "tree_path": str(tree_json),
        "tree_sha256": file_sha256(tree_json),
        "tree_root": str(tree_root),
        "base_zip_path": str(base_zip),
        "base_zip_sha256": sha256_bytes(base_zip.read_bytes()),
        "base_zip_two_path": str(base_zip_two),
        "base_zip_two_sha256": sha256_bytes(base_zip_two.read_bytes()),
        "gold_path": str(gold),
        "gold_diff": gold.read_text(encoding="utf-8"),
        "gold_diff_two": gold_two.read_text(encoding="utf-8"),
        "corrupted_diff": corrupted.read_text(encoding="utf-8"),
        "environment_path": str(environment),
        "runtime_registry_path": str(runtime_registry),
        "runtime_registry_sha256": file_sha256(runtime_registry),
        "offline_registry_path": str(offline_registry),
        "offline_registry_sha256": file_sha256(offline_registry),
    }


def write_audit_roots(area: Path, pins: dict) -> tuple[str, str]:
    reference = area / "reference"
    negative = area / "negative"
    reference.mkdir(parents=True, exist_ok=True)
    negative.mkdir(parents=True, exist_ok=True)
    (reference / "patch.diff").write_text(pins["gold_diff"], encoding="utf-8")
    (negative / "patch.diff").write_text(pins["corrupted_diff"], encoding="utf-8")
    return str(reference), str(negative)


def deepswe_variant(
    task_id: str,
    *,
    variant_id: str = "v1",
    reference_root: str,
    negative_root: str,
    task_text: str = "Repair the DeepSWE fixture counter module.",
    success_criteria: str = "Post-patch tree matches the gold expectation.",
) -> dict:
    return {
        "source_task_id": task_id,
        "variant_id": variant_id,
        "operation": "regression",
        "public": {
            "task_text": task_text,
            "tools_available": ["terminal"],
            "success_criteria": success_criteria,
            "difficulty": "medium",
        },
        "deepswe_offline": {"deliverable": "patch.diff", "timeout_s": 30},
        "audit": {
            "reviewed": True,
            "reference_root": reference_root,
            "negative_root": negative_root,
        },
    }
