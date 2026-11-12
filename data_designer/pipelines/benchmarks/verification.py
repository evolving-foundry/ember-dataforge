"""One trusted offline judgement; solving, retries and aggregation belong to the host."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from data_designer.contracts.provenance import canonical_json
from .boundaries import _assert_private_boundary, _verify_manifest
from .contracts import canonical_sha256
from .native import run_case
from .reader import read_package, resolve_resources
from .runtime import verify_runtime_files


def verify_task(manifest_uri, digest, task_id, candidate_root, *, timeout=60, attempt=None):
    package = read_package(manifest_uri, expected_digest=digest, verify_resources=True)
    matches = [task for task in package["content"]["tasks"] if task["public"]["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError("task is not in the locked benchmark selection")
    task = resolve_resources(matches[0], package)
    manifest, runtime = task["private"], task["runtime"]
    root = Path(candidate_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("candidate root must be a directory")
    package_root = Path(package["manifest_uri"]).parent
    if root == package_root or root in package_root.parents or package_root in root.parents:
        raise ValueError("candidate root overlaps private benchmark package")
    verify_runtime_files(runtime)
    for binding in package["bindings"].values():
        resource = Path(binding).resolve()
        if resource == root or root in resource.parents or (resource.is_dir() and resource in root.parents):
            raise ValueError("candidate root overlaps a trusted benchmark resource")
    with tempfile.TemporaryDirectory(prefix="dataforge-verify-") as temporary:
        path = Path(temporary) / "case.json"
        path.write_text(canonical_json(manifest), encoding="utf-8")
        _verify_manifest(path, task_id, canonical_sha256(manifest), {}, root)
        public = copy.deepcopy(task["public"])
        public["benchmark"]["case_manifest_sha256"] = canonical_sha256(manifest)
        public["item_fingerprint"] = canonical_sha256({k: v for k, v in public.items() if k != "item_fingerprint"})
        return run_case(runtime["verifier"]["command"], task_id, str(path), str(root),
                        phase="evaluation", timeout=timeout, public_item=public, attempt=attempt)
