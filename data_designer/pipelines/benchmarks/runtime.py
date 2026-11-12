"""Per-case native runtime locks used by the builder, suite, and evaluator."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

from .contracts import (
    BenchmarkError,
    canonical_sha256,
    file_sha256,
    require_nonempty,
    require_sha,
)


def _locked_file(record: Any, name: str) -> Dict[str, str]:
    if not isinstance(record, dict):
        raise BenchmarkError(f"{name} must be an object")
    path = Path(require_nonempty(record.get("path"), f"{name}.path"))
    sha = require_sha(record.get("sha256"), f"{name}.sha256")
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != sha:
        raise BenchmarkError(f"{name} missing or SHA-256 drift: {path}")
    return {"path": str(path), "sha256": sha}


def lock_runtime(record: Any) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise BenchmarkError("case runtime must be an object")
    environment = _locked_file(record.get("environment"), "environment")
    verifier_record = record.get("verifier")
    verifier = _locked_file(verifier_record, "verifier")
    command = verifier_record.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
        or verifier["path"] not in command
    ):
        raise BenchmarkError("verifier.command must execute the locked verifier path")
# Keep the which() result without resolve: the launcher and command[0] must share the same resource identity,
# otherwise, when the frozen package deduplicates by content digest, a symlink interpreter (venv python -> base interpreter)
# launcher path may override the command binding, causing the replayed command to use the wrong interpreter.
    launcher_name = shutil.which(command[0]) or command[0]
    launcher_path = Path(launcher_name)
    if not launcher_path.is_absolute() or not launcher_path.is_file():
        raise BenchmarkError("verifier command launcher must resolve to a file")
    declared_launcher = verifier_record.get("launcher")
    if declared_launcher is None:
        launcher = {"path": str(launcher_path), "sha256": file_sha256(launcher_path)}
    else:
        launcher = _locked_file(declared_launcher, "verifier.launcher")
        if Path(launcher["path"]).resolve() != launcher_path.resolve():
            raise BenchmarkError("verifier launcher path drift")
    verifier["version"] = require_nonempty(
        verifier_record.get("version"), "verifier.version"
    )
    verifier["command"] = list(command)
    verifier["launcher"] = launcher
    assets = record.get("assets", [])
    if not isinstance(assets, list):
        raise BenchmarkError("assets must be a list")
    return {
        "environment": environment,
        "verifier": verifier,
        "assets": [_locked_file(asset, "asset") for asset in assets],
    }


def runtime_digests(
    items: List[Dict[str, Any]], runtimes: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """Digest the exact ordered task/runtime binding, including runner commands."""
    if not isinstance(runtimes, dict) or set(runtimes) != {
        item["task_id"] for item in items
    }:
        raise BenchmarkError("runtime_by_task must match the frozen task IDs")
    environments = []
    verifiers = []
    assets = []
    for item in items:
        task_id = item["task_id"]
        runtime = runtimes[task_id]
        if not isinstance(runtime, dict):
            raise BenchmarkError(f"invalid runtime for {task_id}")
        environment = runtime.get("environment")
        verifier = runtime.get("verifier")
        asset_list = runtime.get("assets")
        if (
            not isinstance(environment, dict)
            or not isinstance(verifier, dict)
            or not isinstance(asset_list, list)
        ):
            raise BenchmarkError(f"incomplete runtime for {task_id}")
        if item["benchmark"]["environment_sha256"] != require_sha(
            environment.get("sha256"), "environment.sha256"
        ) or item["benchmark"]["verifier_sha256"] != require_sha(
            verifier.get("sha256"), "verifier.sha256"
        ):
            raise BenchmarkError(f"public runtime pins disagree for {task_id}")
        environments.append({"task_id": task_id, "environment": environment})
        verifiers.append({"task_id": task_id, "verifier": verifier})
        assets.append({"task_id": task_id, "assets": asset_list})
    return {
        "environment_sha256": canonical_sha256(environments),
        "verifier_sha256": canonical_sha256(verifiers),
        "asset_manifest_sha256": canonical_sha256(assets),
    }


def verify_runtime_files(runtime: Dict[str, Any]) -> None:
    """Check every pinned file before any candidate evaluation starts."""
    _locked_file(runtime.get("environment"), "environment")
    verifier = _locked_file(runtime.get("verifier"), "verifier")
    if not isinstance(runtime.get("verifier"), dict):
        raise BenchmarkError("verifier runtime missing")
    command = runtime["verifier"].get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
        or verifier["path"] not in command
    ):
        raise BenchmarkError("verifier command drift")
    require_nonempty(runtime["verifier"].get("version"), "verifier.version")
    launcher = _locked_file(runtime["verifier"].get("launcher"), "verifier.launcher")
    launcher_name = shutil.which(command[0]) or command[0]
    if not Path(launcher_name).is_absolute() or Path(launcher_name).resolve() != Path(
        launcher["path"]
    ).resolve():
        raise BenchmarkError("verifier launcher resolution drift")
    assets = runtime.get("assets")
    if not isinstance(assets, list):
        raise BenchmarkError("runtime assets missing")
    for asset in assets:
        _locked_file(asset, "asset")
