"""Deterministic content-addressed state augmentation for local JSON snapshots."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from .contracts import BenchmarkError, canonical_sha256, file_sha256, require_sha


def _source_state(source: Dict[str, Any], key: str) -> Any:
    pin = source.get(key)
    if not isinstance(pin, dict) or not isinstance(pin.get("path"), str):
        raise BenchmarkError(f"source {key} needs an absolute path and SHA-256")
    path = Path(pin["path"])
    expected = require_sha(pin.get("sha256"), f"source {key}.sha256")
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != expected:
        raise BenchmarkError(f"source {key} snapshot missing or drifted")
    if source.get(f"{key}_sha256") != expected:
        raise BenchmarkError(f"source {key} SHA-256 fields disagree")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"source {key} is not valid JSON: {error}") from error
    return value


def _apply_patch(state: Any, operations: List[Dict[str, Any]], label: str) -> Any:
    if not isinstance(operations, list) or not operations:
        raise BenchmarkError(f"{label} patch must be a non-empty list")
    result = copy.deepcopy(state)
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or operation.get("op") not in (
            "add", "replace", "remove"
        ):
            raise BenchmarkError(f"invalid {label} patch operation {index}")
        path = operation.get("path")
        if not isinstance(path, list) or not path or any(
            not isinstance(part, (str, int)) or isinstance(part, bool) for part in path
        ):
            raise BenchmarkError(f"invalid {label} patch path {index}")
        parent = result
        for part in path[:-1]:
            try:
                parent = parent[part]
            except (KeyError, IndexError, TypeError) as error:
                raise BenchmarkError(f"missing {label} patch parent {path}") from error
        key = path[-1]
        if isinstance(parent, dict):
            if not isinstance(key, str):
                raise BenchmarkError(f"dictionary patch key must be string: {path}")
            present = key in parent
        elif isinstance(parent, list):
            if not isinstance(key, int) or key < 0 or key >= len(parent):
                raise BenchmarkError(f"array patch index out of range: {path}")
            present = True
        else:
            raise BenchmarkError(f"patch parent must be object or array: {path}")
        action = operation["op"]
        if action == "add":
            if not isinstance(parent, dict) or present or "value" not in operation:
                raise BenchmarkError(f"add patch must insert a new object key: {path}")
            parent[key] = operation["value"]
        elif action == "replace":
            if not present or "value" not in operation:
                raise BenchmarkError(f"replace patch target or value missing: {path}")
            parent[key] = operation["value"]
        else:
            if not present:
                raise BenchmarkError(f"remove patch target missing: {path}")
            del parent[key]
    return result


def _write_state(root: Path, state: Any) -> str:
    encoded = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    sha = canonical_sha256(state)
    target = root / "states" / f"{sha}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != encoded:
        raise BenchmarkError("content-addressed state collision")
    target.write_bytes(encoded)
    return sha


def materialize_json_state_variant(
    source: Dict[str, Any], variant: Dict[str, Any], trusted_root: Path
) -> Dict[str, Any]:
    """Rebuild both initial and expected state from pinned source JSON snapshots."""
    patches = variant.get("state_patch")
    if not isinstance(patches, dict) or set(patches) != {"initial", "expected"}:
        raise BenchmarkError("state_patch needs initial and expected operations")
    initial = _apply_patch(
        _source_state(source, "initial_state"), patches["initial"], "initial"
    )
    expected = _apply_patch(
        _source_state(source, "expected_state"), patches["expected"], "expected"
    )
    extras = variant.get("private", {})
    if not isinstance(extras, dict) or any(
        key in extras
        for key in (
            "initial_state_sha256", "expected_state_sha256",
            "initial_state_ref", "expected_state_ref", "state_store",
        )
    ):
        raise BenchmarkError(
            "state_patch private extras cannot override generated state"
        )
    initial_sha = _write_state(trusted_root, initial)
    expected_sha = _write_state(trusted_root, expected)
    return {
        **extras,
        "initial_state_sha256": initial_sha,
        "expected_state_sha256": expected_sha,
        "initial_state_ref": f"sha256:{initial_sha}",
        "expected_state_ref": f"sha256:{expected_sha}",
        "state_store": "relative-content-addressed-v1",
    }
