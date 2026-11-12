"""swe offline native runner: offline patch-apply scoring over the JSON protocol.

File protocol matches ``native.py:run_case`` exactly: argv is
``<request.json> <result.json>`` where the request carries
``taskId/caseManifestPath/candidateRoot/phase/publicItem``. The runner is the
trusted verifier process pinned by ``runtime["verifier"]`` — the locked file is
this script and ``verifier.command`` must execute it.

Scoring semantics (RSI_swe scoring re-expression addendum v0.1 S2–S3): the candidate protocol is unchanged (``patch.diff``,
standard unified diff, <=2MB), but the official SWE-bench harness/Docker is no longer invoked--apply
the patch offline with ``git apply`` to the pinned base snapshot and compare per-file SHA-256
for affected paths. FAIL_TO_PASS/PASS_TO_PASS are provenance only and are not executed.

Trusted ``verifier_spec``::

    {"kind": "swe_offline_patch",
     "payload": {"deliverable": "patch.diff",
                 "base_snapshot": {"path": ..., "sha256": ...},
                 "expected_files": {"relpath": "sha256", ...},
                 "deleted_paths": ["relpath", ...],
                 "timeout_s": 30.0}}

``base_snapshot`` is a zip of the pinned pre-patch workspace; ``expected_files``
/ ``deleted_paths`` describe the gold post-patch tree restricted to the paths
the gold patch touches.

Phases:
  reference/negative  audit deliverables come from the trusted audit roots; a
                      missing/unreadable deliverable is an explicit audit
                      failure (ValueError). An inapplicable audit patch is a
                      normal fail result — the suite audit asserts
                      reference==1.0/negative==0.0.
  evaluation          missing/oversized/unreadable deliverable is the
                      candidate's fault: ``invalid_candidate`` with score 0,
                      never silent. An inapplicable/unsafe/out-of-scope patch
                      scores 0.0 as a completed result.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Set

VERSION = "swe-offline-adapter-v1"
SPEC_KIND = "swe_offline_patch"
DEFAULT_DELIVERABLE = "patch.diff"
MAX_PATCH_BYTES = 2_000_000
DEFAULT_TIMEOUT_S = 30.0

_HEX = frozenset("0123456789abcdef")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def _clean_relative(relative: str) -> str:
    parts = PurePosixPath(relative).parts
    if (
        not relative.strip()
        or relative.startswith("/")
        or "\\" in relative
        or ".." in parts
        or any(part in ("", ".") for part in parts)
    ):
        raise ValueError(f"must be a clean relative path: {relative!r}")
    return relative


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_locked_file(path: Path, expected_sha: str, label: str) -> None:
    if _sha256_file(path) != expected_sha:
        raise ValueError(f"{label} sha256 mismatch for {path}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _spec_payload(request: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    if manifest.get("task_id") != request.get("taskId"):
        raise ValueError("swe task ID disagrees with trusted manifest")
    if manifest.get("family") != "swe":
        raise ValueError("trusted manifest is not a swe case")
    spec = manifest.get("private", {}).get("verifier_spec")
    if not isinstance(spec, dict):
        raise ValueError("trusted manifest lacks a verifier_spec")
    if spec.get("kind") != SPEC_KIND:
        raise ValueError(f"unsupported verifier kind: {spec.get('kind')!r}")
    payload = spec.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("verifier_spec payload missing")

    deliverable = _clean_relative(str(payload.get("deliverable") or DEFAULT_DELIVERABLE))
    snapshot = payload.get("base_snapshot")
    if (
        not isinstance(snapshot, dict)
        or not snapshot.get("path")
        or not _is_sha256(snapshot.get("sha256"))
    ):
        raise ValueError("verifier_spec payload.base_snapshot must pin path and sha256")
    expected_files = payload.get("expected_files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("verifier_spec payload.expected_files must be a non-empty mapping")
    for relative, sha in expected_files.items():
        _clean_relative(str(relative))
        if not _is_sha256(sha):
            raise ValueError(f"expected_files entry lacks a sha256: {relative!r}")
    deleted_paths = payload.get("deleted_paths") or []
    if not isinstance(deleted_paths, list):
        raise ValueError("verifier_spec payload.deleted_paths must be a list")
    for relative in deleted_paths:
        _clean_relative(str(relative))
    overlap = {str(key) for key in expected_files} & {str(item) for item in deleted_paths}
    if overlap:
        raise ValueError(f"expected_files and deleted_paths overlap: {sorted(overlap)}")
    timeout = payload.get("timeout_s", DEFAULT_TIMEOUT_S)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("verifier_spec payload.timeout_s must be a positive number")
    return {
        "deliverable": deliverable,
        "base_snapshot": {"path": str(snapshot["path"]), "sha256": snapshot["sha256"]},
        "expected_files": {str(key): value for key, value in expected_files.items()},
        "deleted_paths": [str(item) for item in deleted_paths],
        "timeout_s": float(timeout),
    }


def _patch_head_paths(patch_text: str) -> Set[str]:
    """Repo-relative paths a unified diff touches, from ---/+++ headers."""
    paths: Set[str] = set()
    for line in patch_text.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        token = line[4:].strip().split("\t", 1)[0]
        if token == "/dev/null":
            continue
        if token.startswith(("a/", "b/")):
            token = token[2:]
        paths.add(_clean_relative(token))
    return paths


def _extract_snapshot(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = member.filename
            if name.endswith("/"):
                continue
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise ValueError(f"unsafe snapshot member: {name!r}")
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"snapshot symlink member: {name!r}")
            if mode not in (0, 0o100000):
                raise ValueError(f"unsupported snapshot member type: {name!r}")
            target = dest / pure
            if not _within(target, dest):
                raise ValueError(f"snapshot member escapes workspace: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(target, "wb") as output:
                shutil.copyfileobj(source, output)


def _apply_patch(
    git: str, workdir: Path, patch_file: Path, timeout: float, check_only: bool
) -> None:
    command = [git, "apply"]
    if check_only:
        command.append("--check")
    command.append(str(patch_file.resolve()))
    completed = subprocess.run(
        command, cwd=str(workdir), capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git apply {'--check ' if check_only else ''}failed "
            f"rc={completed.returncode}: "
            f"{(completed.stderr or completed.stdout or '').strip()[:400]}"
        )


def _response(
    request: Dict[str, Any],
    status: str,
    passed: bool,
    score: float,
    metrics: Dict[str, float],
    evidence_refs: List[str],
    started: float,
) -> Dict[str, Any]:
    response = {
        "task_id": request["taskId"],
        "status": status,
        "passed": passed,
        "score": score,
        "native_metrics": metrics,
        "verifier_version": VERSION,
        "evidence_refs": evidence_refs,
        "runtime_seconds": time.monotonic() - started,
    }
    if "attempt" in request:
        response.update(
            attempt_index=request["attempt"]["index"],
            seed=request["attempt"]["seed"],
        )
    return response


def score_request(request: Dict[str, Any]) -> Dict[str, Any]:
    started = time.monotonic()
    manifest_path = Path(request["caseManifestPath"])
    manifest = _load_json(manifest_path)
    payload = _spec_payload(request, manifest)
    task_id = request["taskId"]
    phase = request.get("phase")
    relative = payload["deliverable"]

    root = Path(request["candidateRoot"]).resolve(strict=True)
    patch_file = root / relative
    size = 0.0
    try:
        if not _within(patch_file, root) or patch_file.is_symlink() or not patch_file.is_file():
            raise FileNotFoundError(f"deliverable outside candidateRoot or missing: {relative}")
        size = float(patch_file.stat().st_size)
        if size > MAX_PATCH_BYTES:
            raise ValueError(f"patch exceeds {MAX_PATCH_BYTES} bytes")
        patch_text = patch_file.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        if phase in ("reference", "negative"):
            raise ValueError(f"audit deliverable invalid for {task_id}: {error}")
        return _response(
            request, "invalid_candidate", False, 0.0,
            {"patch_bytes": 0.0, "compared_files": 0.0}, [], started,
        )

    expected_files = payload["expected_files"]
    deleted_paths = payload["deleted_paths"]
    scope = set(expected_files) | set(deleted_paths)
    metrics = {
        "patch_bytes": size,
        "compared_files": float(len(expected_files) + len(deleted_paths)),
    }
    evidence_refs = [
        f"patch-sha256:{relative}",
        f"base-snapshot-sha256:{payload['base_snapshot']['sha256']}",
    ]

    try:
        affected = _patch_head_paths(patch_text)
    except ValueError as error:
        if phase in ("reference", "negative"):
            raise ValueError(f"audit patch has unsafe paths for {task_id}: {error}")
        return _response(request, "completed", False, 0.0, metrics, evidence_refs, started)
    if not (affected & scope):
        if phase in ("reference", "negative"):
            raise ValueError(f"audit patch touches no judged path for {task_id}")
        return _response(request, "completed", False, 0.0, metrics, evidence_refs, started)

    git = shutil.which("git")
    if not git:
        raise ValueError("git binary is required for swe offline patch scoring")
    snapshot_path = Path(payload["base_snapshot"]["path"])
    if not snapshot_path.is_absolute():
        snapshot_path = (manifest_path.parent / snapshot_path).resolve()
    _require_locked_file(snapshot_path, payload["base_snapshot"]["sha256"], "base snapshot")

    passed = False
    with tempfile.TemporaryDirectory(prefix="swe-offline-") as tmp:
        workdir = Path(tmp) / "workspace"
        workdir.mkdir()
        _extract_snapshot(snapshot_path, workdir)
        patch_copy = Path(tmp) / "candidate.patch"
        patch_copy.write_text(patch_text, encoding="utf-8")
        try:
            _apply_patch(git, workdir, patch_copy, payload["timeout_s"], check_only=True)
            _apply_patch(git, workdir, patch_copy, payload["timeout_s"], check_only=False)
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            evidence_refs.append("apply-failed")
        else:
            passed = all(
                not (workdir / item).is_symlink()
                and (workdir / item).is_file()
                and _sha256_file(workdir / item) == sha
                for item, sha in expected_files.items()
            ) and all(
                not (workdir / item).is_symlink() and not (workdir / item).exists()
                for item in deleted_paths
            )
    return _response(
        request, "completed", passed, 1.0 if passed else 0.0, metrics, evidence_refs, started
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError("usage: swe_offline.py <request.json> <result.json>")
    request = _load_json(Path(sys.argv[1]))
    result = score_request(request)
    Path(sys.argv[2]).write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
