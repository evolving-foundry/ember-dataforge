"""C-WK-1 worker: environment synthesis and artifact-store freezing.

Usage::

    python -m data_designer.pipelines.envsynth.cli.synthesis_worker <input.json> <snapshot.json>

The compile implementation remains in env_compile; this CLI is the v0.1
command surface and adds the required storePath persistence:

* env_release artifact: release record plus build report;
* taskset artifact: frozen env-task-v1 payload.

Version selection is based on the store when this worker is used, so F1
immutability and idempotency are owned by the artifact catalog just as v0.1
specifies: the same (kind, refId, version) with the same fingerprint is a
no-op, a different fingerprint is rejected. No external runner daemon is
involved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_designer.pipelines.envsynth import artifacts
from data_designer.pipelines.envsynth.cli import env_compile
from data_designer.storage.catalog import ArtifactCatalog


def _find_version(
    store: ArtifactCatalog, ref_id: str, build_fingerprint: str
) -> Optional[int]:
    for artifact in store.list_artifacts("env_release"):
        if artifact["ref_id"] != ref_id:
            continue
        if artifact["payload"].get("buildFingerprint") == build_fingerprint:
            return int(artifact["version"])
    return None


def _next_version(store: ArtifactCatalog, ref_id: str) -> int:
    return store.next_artifact_version("env_release", ref_id)


def _put_artifact(
    store: ArtifactCatalog,
    *,
    kind: str,
    ref_id: str,
    version: int,
    payload: Dict[str, Any],
    fingerprint: str,
) -> None:
    existing = store.get_artifact(kind, ref_id, version)
    if existing is not None:
        if existing["fingerprint"] == fingerprint:
            return  # Idempotent
        raise ValueError(
            f"artifact {kind}/{ref_id}/v{version} already exists with different content -"
            "Artifacts are immutable (F1): use a new version; previous versions remain readable forever"
        )
    store.put_artifact(
        kind=kind,
        ref_id=ref_id,
        version=version,
        payload=payload,
        fingerprint=fingerprint,
    )


def _persist_release(
    store: ArtifactCatalog, data: env_compile.SynthesisInput, snapshot: Dict[str, Any]
) -> None:
    release = snapshot["envRelease"]
    release_dir = Path(release["assetDir"]).parent
    release_payload, taskset_payload = artifacts.load_release_metadata(release_dir)
    release_payload = dict(release_payload)
    release_payload["assetDir"] = release["assetDir"]
    release_payload["buildReport"] = snapshot["buildReport"]

    _put_artifact(
        store,
        kind="env_release",
        ref_id=data.ref_id,
        version=int(release["version"]),
        payload=release_payload,
        fingerprint=release["fingerprint"],
    )
    taskset = snapshot["taskset"]
    _put_artifact(
        store,
        kind="taskset",
        ref_id=taskset["refId"],
        version=int(taskset["version"]),
        payload=taskset_payload,
        fingerprint=taskset["fingerprint"],
    )


def run_worker(argv: List[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: python -m data_designer.pipelines.envsynth.cli.synthesis_worker "
            "<input.json> <snapshot.json>",
            file=sys.stderr,
        )
        return env_compile.EXIT_USAGE
    try:
        data = env_compile.parse_input(argv[0], require_store=True)
    except env_compile.WorkerError as error:
        print(f"[env-synthesis-worker] Invalid input (build not started): {error}", file=sys.stderr)
        return env_compile.EXIT_USAGE

    store = ArtifactCatalog(data.store_path)
    try:
        return env_compile.run_worker(
            argv,
            data=data,
            require_store=True,
            find_version=lambda ref_id, build_fp: _find_version(
                store, ref_id, build_fp
            ),
            next_version=lambda ref_id: _next_version(store, ref_id),
            after_build=lambda input_data, snapshot: _persist_release(
                store, input_data, snapshot
            ),
        )
    finally:
        store.close()


def main(argv: Optional[List[str]] = None) -> int:
    return run_worker(sys.argv[1:] if argv is None else list(argv))


if __name__ == "__main__":
    sys.exit(main())
