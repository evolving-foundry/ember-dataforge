"""osworld offline native runner: offline state-artifact scoring over the JSON protocol.

File protocol matches ``native.py:run_case`` exactly: argv is
``<request.json> <result.json>`` where the request carries
``taskId/caseManifestPath/candidateRoot/phase/publicItem``. The runner is the
trusted verifier process pinned by ``runtime["verifier"]`` — the locked file is
this script and ``verifier.command`` must execute it.

Scoring semantics (RSI_osworld scoring re-expression addendum v0.1 O2–O3): the candidate protocol is re-expressed from
trajectory replay into a **state artifact**--the candidate writes the final state involved in acceptance as a ``{filename: content}`` file-dictionary
JSON (or a structured JSON object for tasks with a regular artifact shape) to the candidateRoot-relative path declared by
``verifier_spec.payload.artifact.path``. After parsing that JSON, the runner dispatches through
``data_designer.verifiers.run_verifier``: prefer ``unit_test``
(a trusted assertion script performs programmatic assertions in a workspace rebuilt from the setup_files initial state + candidate state artifact),
and fall back to ``artifact_compare`` for tasks with a regular artifact shape (field-by-field comparison against gold). GUI-state and
process-state acceptance cannot be represented as files and do not use this runner (O4 exclusion).

Phases:
  reference/negative  audit deliverables come from the trusted audit roots; a
                      missing/unparseable deliverable is an explicit audit
                      failure (ValueError).
  evaluation          a missing/unparseable deliverable is the candidate's
                      fault: ``invalid_candidate`` with score 0, never silent.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

VERSION = "osworld-offline-adapter-v1"

OFFLINE_SPEC_KINDS = ("unit_test", "artifact_compare")

from data_designer.verifiers import run_verifier  # noqa: E402

DEFAULT_DELIVERABLE = "state_artifact.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def _deliverable_relative(spec: Dict[str, Any]) -> str:
    payload = spec.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("verifier_spec payload missing")
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("verifier_spec payload.artifact missing")
    relative = str(artifact.get("path") or DEFAULT_DELIVERABLE)
    parts = Path(relative).parts
    if (
        not relative.strip()
        or Path(relative).is_absolute()
        or ".." in parts
        or not all(part not in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"deliverable path must be a clean relative path: {relative!r}")
    return relative


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _artifact_size(root: Path, relative: str) -> int:
    target = root / relative
    if not _within(target, root) or target.is_symlink() or not target.is_file():
        raise FileNotFoundError(f"deliverable outside candidateRoot or missing: {relative}")
    return target.stat().st_size


def score_request(request: Dict[str, Any]) -> Dict[str, Any]:
    started = time.monotonic()
    manifest = _load_json(Path(request["caseManifestPath"]))
    if manifest.get("task_id") != request.get("taskId"):
        raise ValueError("osworld task ID disagrees with trusted manifest")
    if manifest.get("family") != "osworld":
        raise ValueError("trusted manifest is not an osworld case")
    spec = manifest.get("private", {}).get("verifier_spec")
    if not isinstance(spec, dict):
        raise ValueError("trusted manifest lacks a verifier_spec")
    if spec.get("kind") not in OFFLINE_SPEC_KINDS:
        raise ValueError(f"unsupported verifier kind: {spec.get('kind')!r}")
    relative = _deliverable_relative(spec)
    root = Path(request["candidateRoot"]).resolve(strict=True)
    phase = request.get("phase")
    task_id = request["taskId"]
    deliverable = root / relative
    try:
        size = _artifact_size(root, relative)
        artifact = _load_json(deliverable)
    except (OSError, ValueError) as error:
        if phase in ("reference", "negative"):
            raise ValueError(f"audit deliverable invalid for {task_id}: {error}")
        response = {
            "task_id": task_id, "status": "invalid_candidate",
            "passed": False, "score": 0.0,
            "native_metrics": {"artifact_bytes": 0.0},
            "verifier_version": VERSION, "evidence_refs": [],
            "runtime_seconds": time.monotonic() - started,
        }
        if "attempt" in request:
            response.update(
                attempt_index=request["attempt"]["index"],
                seed=request["attempt"]["seed"],
            )
        return response
    verdict = run_verifier(spec, artifact)
    score = float(verdict.get("score") or 0.0)
    if not 0.0 <= score <= 1.0:
        raise ValueError("verifier score outside [0,1]")
    response = {
        "task_id": task_id, "status": "completed",
        "passed": bool(verdict.get("passed")),
        "score": score,
        "native_metrics": {"artifact_bytes": float(size)},
        "verifier_version": VERSION,
        "evidence_refs": [f"deliverable-sha256:{relative}"],
        "runtime_seconds": time.monotonic() - started,
    }
    if "attempt" in request:
        response.update(
            attempt_index=request["attempt"]["index"],
            seed=request["attempt"]["seed"],
        )
    return response


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError("usage: osworld_offline.py <request.json> <result.json>")
    request = _load_json(Path(sys.argv[1]))
    result = score_request(request)
    Path(sys.argv[2]).write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
