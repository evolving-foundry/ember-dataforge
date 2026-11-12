from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from data_designer.contracts.datasets import DatasetPackage, DatasetRef, RecordEnvelope
from data_designer.contracts.provenance import canonical_json, content_digest, file_digest

PACKAGE_VERSION = "dataforge-dataset-v1"


def _write_json(path, value):
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def save_result(result, uri, *, overwrite=False):
    if result.status not in {"completed", "partial"}:
        raise ValueError("cannot publish a failed generation result")
    target = Path(uri).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {target}")
    if target.exists():
        validate_package(target)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    backup = None
    try:
        (stage / "data").mkdir()
        shard = stage / "data/part-00000.jsonl"
        with shard.open("w", encoding="utf-8", newline="\n") as stream:
            for record in result.records:
                stream.write(canonical_json(asdict(record)) + "\n")
        with (stage / "provenance.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for entry in result.provenance:
                stream.write(canonical_json(entry) + "\n")
        _write_json(stage / "report.json", result.report)
        schema = result.records[0].schema if result.records else result.report["output_schema"]
        content = {"schema": schema, "count": len(result.records), "shards": [
            {"uri": "data/part-00000.jsonl", "format": "jsonl", "digest": file_digest(shard)}]}
        manifest = {"protocol_version": PACKAGE_VERSION, "content": content,
                    "digest": content_digest(content),
                    "files": {name: file_digest(stage / name)
                              for name in ("provenance.jsonl", "report.json")}}
        _write_json(stage / "manifest.json", manifest)
        if target.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{target.name}-backup-", dir=target.parent))
            backup.rmdir()
            os.replace(target, backup)
        try:
            os.rename(stage, target)
        except BaseException:
            if backup is not None:
                os.replace(backup, target)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_package(target / "manifest.json")


def _member(root, uri):
    path = (root / uri).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("package reference escapes package root") from error
    return path


def validate_package(manifest_uri):
    path = Path(manifest_uri).resolve()
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PACKAGE_VERSION:
        raise ValueError("unsupported dataset package protocol version")
    content = manifest["content"]
    if content_digest(content) != manifest["digest"]:
        raise ValueError("dataset manifest content digest mismatch")
    count = 0
    ids = set()
    shards = []
    for entry in content["shards"]:
        member = _member(path.parent, entry["uri"])
        if file_digest(member) != entry["digest"]:
            raise ValueError("dataset shard digest mismatch")
        if entry["format"] != "jsonl":
            raise ValueError("unsupported package shard format")
        for record in read_records(member):
            if record.id in ids:
                raise ValueError("duplicate sample id in package")
            ids.add(record.id)
            count += 1
        shards.append(DatasetRef(entry["uri"], entry["format"], content["schema"], entry["digest"]))
    if count != content["count"]:
        raise ValueError("dataset record count mismatch")
    for name, digest in manifest["files"].items():
        if file_digest(_member(path.parent, name)) != digest:
            raise ValueError(f"package file digest mismatch: {name}")
    return DatasetPackage(str(path), manifest["digest"], tuple(shards), count, content["schema"])


def read_records(uri):
    from data_designer.sources.local import read_rows
    for value in read_rows(uri):
        yield RecordEnvelope(id=value["id"], payload=value["payload"], schema=value["schema"],
                             parents=tuple(value["parents"]), metadata=value["metadata"])
