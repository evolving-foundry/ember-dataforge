from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .provenance import canonical_json


@dataclass(frozen=True)
class DatasetRef:
    uri: str
    format: str = "jsonl"
    schema: str = "record-v1"
    digest: str = ""


@dataclass(frozen=True)
class RecordEnvelope:
    id: str
    payload: Dict[str, Any]
    schema: str = "record-v1"
    parents: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("record id must be a nonempty string")
        if not isinstance(self.payload, dict):
            raise ValueError("record payload must be an object")
        if not isinstance(self.schema, str) or not self.schema.strip():
            raise ValueError("record schema must be a nonempty string")
        if any(not isinstance(p, str) or not p for p in self.parents):
            raise ValueError("parent references must be nonempty strings")
        canonical_json(self.payload)
        canonical_json(self.metadata)


@dataclass(frozen=True)
class DatasetPackage:
    manifest_uri: str
    digest: str
    shards: Tuple[DatasetRef, ...]
    count: int
    schema: str


