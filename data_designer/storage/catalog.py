"""Small immutable JSON artifact catalog, independent of RSI and SQLite.

Used for benchmark build deduplication and frozen selections. A catalog is not
required to read or evaluate an exported benchmark package.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

from data_designer.contracts.provenance import canonical_json, content_digest


class ArtifactCatalog:
    def __init__(self, uri=None):
        self.uri = str(Path(uri).resolve()) if uri is not None else None
        self._artifacts = {}
        if self.uri:
            Path(self.uri).mkdir(parents=True, exist_ok=True)
            for path in Path(self.uri).glob("*.json"):
                entry = json.loads(path.read_text(encoding="utf-8"))
                if entry.get("protocol_version") != "dataforge-artifact-v1":
                    raise ValueError("unsupported artifact catalog protocol")
                identity = (entry["kind"], entry["ref_id"], entry["version"])
                if content_digest(entry["payload"]) != entry["digest"]:
                    raise ValueError("artifact content digest mismatch")
                self._artifacts[identity] = entry

    def next_artifact_version(self, kind, ref_id):
        return max((version for k, ref, version in self._artifacts if (k, ref) == (kind, ref_id)), default=0) + 1

    def put_artifact(self, kind, ref_id, version, payload, *, fingerprint=None):
        if type(version) is not int or version < 1:
            raise ValueError("artifact version must be positive")
        digest = content_digest(payload)
        key = (kind, ref_id, version)
        entry = {"protocol_version": "dataforge-artifact-v1", "kind": kind, "ref_id": ref_id,
                 "version": version, "payload": copy.deepcopy(payload), "digest": digest,
                 "fingerprint": fingerprint or digest}
        old = self._artifacts.get(key)
        if old is not None and old != entry:
            raise ValueError("immutable artifact version cannot be overwritten")
        if self.uri and old is None:
            target = Path(self.uri) / (content_digest(list(key)).split(":")[1] + ".json")
            descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.uri)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(canonical_json(entry))
                # Exclusive create prevents concurrent writers from overwriting.
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    if json.loads(target.read_text(encoding="utf-8")) != entry:
                        raise ValueError("artifact version was concurrently committed")
            finally:
                Path(temporary).unlink(missing_ok=True)
        self._artifacts[key] = entry
        return copy.deepcopy(entry)

    def get_artifact(self, kind, ref_id, version=None):
        if version is None:
            version = self.next_artifact_version(kind, ref_id) - 1
        return copy.deepcopy(self._artifacts.get((kind, ref_id, version)))

    def list_artifacts(self, kind):
        return [copy.deepcopy(entry) for key, entry in sorted(self._artifacts.items()) if key[0] == kind]

    def close(self):
        pass
