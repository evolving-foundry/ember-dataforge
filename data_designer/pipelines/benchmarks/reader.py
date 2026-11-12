"""Lightweight frozen benchmark reader. No Curator, generation or host import."""
from __future__ import annotations

import json
from pathlib import Path

from data_designer.contracts.provenance import content_digest, file_digest
from data_designer.storage.local import _member

BENCHMARK_VERSION = "dataforge-benchmark-v1"


def resource_digest(path):
    path = Path(path)
    if path.is_file():
        return file_digest(path)
    if not path.is_dir():
        raise ValueError("benchmark resource is unavailable")
    entries = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError("benchmark directory resources cannot contain symlinks")
        if child.is_file():
            entries.append({"path": child.relative_to(path).as_posix(), "digest": file_digest(child)})
    return content_digest(entries)


def read_package(uri, *, expected_digest=None, verify_resources=False):
    path = Path(uri).resolve()
    if path.is_dir():
        path /= "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != BENCHMARK_VERSION:
        raise ValueError("unsupported benchmark package protocol")
    content = manifest["content"]
    digest = content_digest(content)
    if digest != manifest.get("digest") or (expected_digest is not None and digest != expected_digest):
        raise ValueError("benchmark package content digest mismatch")
    if content.get("schema") != "dataforge-benchmark-content-v1":
        raise ValueError("unsupported benchmark content schema")
    tasks = content.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("benchmark package must contain tasks")
    ids = [task["public"]["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task id in benchmark package")
    bindings = json.loads((path.parent / "bindings.json").read_text(encoding="utf-8"))
    if set(bindings) != set(content["resources"]):
        raise ValueError("benchmark resource bindings are incomplete")
    if verify_resources:
        for key, expected in content["resources"].items():
            actual = Path(bindings[key])
            if not actual.is_absolute():
                actual = _member(path.parent, bindings[key])
            if resource_digest(actual) != expected:
                raise ValueError(f"benchmark resource drift: {key}")
    return {**manifest, "manifest_uri": str(path), "bindings": bindings}


def enumerate_public(package):
    import copy
    return [copy.deepcopy(task["public"]) for task in package["content"]["tasks"]]


def resolve_resources(value, package):
    if isinstance(value, dict):
        if set(value) == {"$resource"}:
            binding = package["bindings"][value["$resource"]]
            path = Path(binding)
            return str(path if path.is_absolute() else _member(Path(package["manifest_uri"]).parent, binding))
        return {key: resolve_resources(child, package) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_resources(child, package) for child in value]
    return value
