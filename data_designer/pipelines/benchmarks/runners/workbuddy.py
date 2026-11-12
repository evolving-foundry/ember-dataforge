"""workbuddy native runner: trusted verifier_spec scoring over the JSON file protocol.

File protocol matches ``native.py:run_case`` exactly: argv is
``<request.json> <result.json>`` where the request carries
``taskId/caseManifestPath/candidateRoot/phase/publicItem``. The runner is the
trusted verifier process pinned by ``runtime["verifier"]`` — the locked file is
this script and ``verifier.command`` must execute it.

Scoring reuses the package verifier dispatch (``data_designer.verifiers
.run_verifier``), the same entry ``scripts/wb_task_eval.py`` consumes. The
candidate convention (family candidate protocol): the solver writes the
deliverable named by ``verifier_spec.payload.artifact.path`` under
``candidateRoot``. Only the relative path and the JSON content are judged.

Phases:
  reference/negative  audit deliverables come from the trusted audit roots
                      (passed in as candidateRoot by the builder); a missing or
                      unparseable deliverable is an explicit audit failure.
  evaluation          a missing or unparseable deliverable is the candidate's
                      fault: ``invalid_candidate`` with score 0, never silent.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

VERSION = "workbuddy-verifier-shim-v1"

from data_designer.verifiers import run_verifier  # noqa: E402

DEFAULT_DELIVERABLE = "output/result.json"


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
        raise ValueError("workbuddy task ID disagrees with trusted manifest")
    if manifest.get("family") != "workbuddy":
        raise ValueError("trusted manifest is not a workbuddy case")
    spec = manifest.get("private", {}).get("verifier_spec")
    if not isinstance(spec, dict):
        raise ValueError("trusted manifest lacks a verifier_spec")
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
    judge = None
    if spec.get("kind") == "llm_rubric":
        from data_designer.integrations.curator.judge import judge_from_environment
        judge = judge_from_environment()
    verdict = run_verifier(spec, artifact, judge=judge)
    if verdict["status"] != "completed":
        raise RuntimeError("domain verifier did not complete; no score is available")
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
        raise ValueError("usage: workbuddy.py <request.json> <result.json>")
    request = _load_json(Path(sys.argv[1]))
    result = score_request(request)
    Path(sys.argv[2]).write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
