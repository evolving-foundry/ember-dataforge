"""Content-addressed identity helpers for Gym-native environment packages."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional


DEFAULT_MAX_STEPS = 40
GYM_FINGERPRINT_EXCLUDED = frozenset({"manifest.json", "tasks.jsonl"})


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def compute_fingerprint(
    ref_id: str, version: str | int, profile: str, files: Mapping[str, bytes]
) -> str:
    payload = {
        "ref_id": ref_id,
        "version": str(version),
        "profile": profile,
        "files": {
            path: hashlib.sha256(content).hexdigest()
            for path, content in sorted(files.items())
            if path not in GYM_FINGERPRINT_EXCLUDED
        },
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def compute_content_digest(
    ref_id: str, profile: str, files: Mapping[str, bytes]
) -> str:
    return compute_fingerprint(ref_id, "0", profile, files)


def project_manifest(
    *,
    ref_id: str,
    version: int,
    profile: str,
    fingerprint: str,
    simulator_model: Optional[str] = None,
    description: str = "",
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "ref_id": ref_id,
        "version": int(version),
        "fingerprint": fingerprint,
        "profile": profile,
    }
    if description:
        manifest["description"] = description
    if simulator_model:
        manifest["sim_model"] = {"name": str(simulator_model)}
    return manifest


__all__ = [
    "DEFAULT_MAX_STEPS",
    "GYM_FINGERPRINT_EXCLUDED",
    "canonical_json",
    "compute_content_digest",
    "compute_fingerprint",
    "project_manifest",
]
