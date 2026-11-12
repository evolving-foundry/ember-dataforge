"""File-based artifacts for compile-time environment synthesis.

The artifact store owns the authoritative immutable artifacts for the v0.1
worker. This module writes an auditable file mirror and never starts a runner
daemon. A release is an immutable directory tree plus two small JSON records:

``<assetRoot>/<ref_id>/v<version>/env_release.json``
    Release pins and build metadata.
``<assetRoot>/<ref_id>/v<version>/tasks.json``
    The complete Gym task rows mirrored for the publisher and the artifact store.

The package itself remains under ``<release>/package`` and its zip under
``<release>/gym-env-package-v1.zip``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from data_designer.pipelines.envsynth.serialization import json_bytes as _json_bytes


ENV_RELEASE_FILE = "env_release.json"
ENV_TASKS_FILE = "tasks.json"
ENV_RELEASE_SCHEMA = "env-release-v1"
ENV_TASKSET_SCHEMA = "env-task-v1"
COMPILE_RESULT_SCHEMA = "env-compile-result-v1"

_VERSION_RE = re.compile(r"^v(\d+)$")


class ArtifactConflictError(RuntimeError):
    """A version directory already contains different immutable metadata."""


def fingerprint_of(payload: Any) -> str:
    """Stable short fingerprint over a JSON-compatible payload."""
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file():
            raise ArtifactConflictError(
                "artifact path is not a regular file: {}".format(path)
            )
        if path.read_bytes() != data:
            raise ArtifactConflictError(
                "artifact metadata differs from existing immutable record: {}".format(path)
            )
        return
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".{}.tmp-".format(path.name)
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, str(path))
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


@contextmanager
def release_lock(asset_root: str | Path, ref_id: str) -> Iterator[None]:
    """Serialize file-mirror version allocation for one ref_id across processes."""
    lock_dir = Path(asset_root) / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "{}.lock".format(ref_id)
    with lock_path.open("a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def release_dir(asset_root: str | Path, ref_id: str, version: int) -> Path:
    return Path(asset_root) / ref_id / "v{}".format(int(version))


def iter_release_records(
    asset_root: str | Path, ref_id: str
) -> Iterator[Tuple[int, Path, Dict[str, Any]]]:
    root = Path(asset_root) / ref_id
    if not root.is_dir():
        return
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        match = _VERSION_RE.match(candidate.name)
        if not match or not candidate.is_dir():
            continue
        metadata_path = candidate / ENV_RELEASE_FILE
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(metadata, dict):
            yield int(match.group(1)), candidate, metadata


def find_release_version(
    asset_root: str | Path, ref_id: str, build_fingerprint: str
) -> Optional[int]:
    for version, _, metadata in iter_release_records(asset_root, ref_id):
        if metadata.get("buildFingerprint") == build_fingerprint:
            return version
    return None


def next_release_version(asset_root: str | Path, ref_id: str) -> int:
    versions: List[int] = [
        version for version, _, _ in iter_release_records(asset_root, ref_id)
    ]
    root = Path(asset_root) / ref_id
    if root.is_dir():
        for candidate in root.iterdir():
            match = _VERSION_RE.match(candidate.name)
            if match and candidate.is_dir():
                versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def write_release_metadata(
    release_path: str | Path,
    env_release: Dict[str, Any],
    taskset: Dict[str, Any],
) -> None:
    root = Path(release_path)
    # The release record is the commit marker: write the taskset first so an
    # interrupted write leaves an incomplete, non-loadable release rather than a
    # release that points at missing tasks.
    _atomic_write(root / ENV_TASKS_FILE, _json_bytes(taskset))
    _atomic_write(root / ENV_RELEASE_FILE, _json_bytes(env_release))


def load_release_metadata(
    release_path: str | Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    root = Path(release_path)
    release_path_obj = root / ENV_RELEASE_FILE
    tasks_path_obj = root / ENV_TASKS_FILE
    try:
        release = json.loads(release_path_obj.read_text(encoding="utf-8"))
        taskset = json.loads(tasks_path_obj.read_text(encoding="utf-8"))
    except OSError as error:
        raise ArtifactConflictError(
            "release metadata is incomplete under {}: {}".format(root, error)
        ) from error
    except ValueError as error:
        raise ArtifactConflictError(
            "release metadata is not valid JSON under {}: {}".format(root, error)
        ) from error
    if not isinstance(release, dict) or not isinstance(taskset, dict):
        raise ArtifactConflictError("release metadata must be JSON objects")
    return release, taskset


def load_compile_result(path: str | Path) -> Dict[str, Any]:
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError("compile result is unreadable: {}".format(error)) from error
    except ValueError as error:
        raise ValueError("compile result is not valid JSON: {}".format(error)) from error
    if not isinstance(result, dict):
        raise ValueError("compile result must be a JSON object")
    if result.get("schemaVersion") != COMPILE_RESULT_SCHEMA:
        raise ValueError(
            "compile result schemaVersion must be {}".format(COMPILE_RESULT_SCHEMA)
        )
    release = result.get("envRelease")
    taskset = result.get("taskset")
    if not isinstance(release, dict) or not isinstance(taskset, dict):
        raise ValueError("compile result requires envRelease and taskset objects")
    if not isinstance(release.get("assetDir"), str) or not release["assetDir"]:
        raise ValueError("compile result envRelease.assetDir is required")
    return result
