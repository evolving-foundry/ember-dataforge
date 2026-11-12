"""Freeze audited benchmark outputs without a database dependency for consumers."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

from data_designer.contracts.provenance import canonical_json, content_digest
from .contracts import canonical_sha256
from .reader import BENCHMARK_VERSION, resource_digest, read_package


def freeze_package(payload, output):
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError("benchmark output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=".benchmark-", dir=target.parent))
    bindings, resources = {}, {}

    def encode(value):
        if isinstance(value, dict):
            return {key: encode(child) for key, child in value.items()
                    if key not in {"runtime_seconds", "started_at", "finished_at", "duration_ms"}}
        if isinstance(value, (list, tuple)):
            return [encode(child) for child in value]
        if isinstance(value, str) and Path(value).is_absolute():
            path = Path(value)
            if not path.exists():
                raise ValueError("cannot freeze an unavailable absolute resource")
            digest = resource_digest(path)
            # Same bytes at multiple paths have one content identity.
            key = digest
            resources[key] = digest
    # Keep the original path without resolve: a local binding may be a symlink interpreter
    # (venv python -> base interpreter); resolving it loses the venv's site-packages.
            bindings[key] = str(path)
            return {"$resource": key}
        return value

    try:
        tasks = []
        for public in payload["taskset"]:
            case = public["benchmark"]["case_manifest_sha256"]
            case_path = Path(payload["trusted_manifest_dir"]) / f"{case}.json"
            manifest = json.loads(case_path.read_text(encoding="utf-8"))
            if canonical_sha256(manifest) != case:
                raise ValueError("trusted case drift before freeze")
            private = encode(manifest)
            runtime = encode(payload["runtime_by_task"][public["task_id"]])
            item = copy.deepcopy(public)
            item["benchmark"]["case_manifest_sha256"] = canonical_sha256(private)
            item["item_fingerprint"] = canonical_sha256({k: v for k, v in item.items() if k != "item_fingerprint"})
            attachments = []
            source_id = public["benchmark"]["source_task_id"]
            if source_id in payload.get("public_inputs_by_task", {}):
                for entry in payload["public_inputs_by_task"][source_id]:
                    relative = entry["path"]
                    if Path(relative).is_absolute() or ".." in Path(relative).parts:
                        raise ValueError("invalid public attachment path")
                    source = Path(payload["public_inputs_dir"]) / source_id / relative
                    attachments.append({"source": encode(str(source)), "target": relative})
            tasks.append({"public": item, "private": private, "runtime": runtime, "attachments": attachments})
        content = {"schema": "dataforge-benchmark-content-v1", "tasks": tasks,
                   "resources": resources, "family": payload["family"], "split": payload["split"],
                   "admission": encode(payload["build_report"]), "source_release": payload["source_release"]}
        # encode(admission) may discover more resources; keep the final mapping.
        content["resources"] = dict(resources)
        manifest = {"protocol_version": BENCHMARK_VERSION, "content": content, "digest": content_digest(content)}
        (root / "bindings.json").write_text(canonical_json(bindings), encoding="utf-8")
        (root / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
        os.rename(root, target)
    finally:
        if root.exists():
            shutil.rmtree(root)
    return read_package(target, verify_resources=True)
