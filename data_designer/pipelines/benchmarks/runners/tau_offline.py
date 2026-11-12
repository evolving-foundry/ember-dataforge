"""tau offline native runner: offline terminal-state-artifact scoring.

File protocol matches ``native.py:run_case`` exactly: argv is
``<request.json> <result.json>`` where the request carries
``taskId/caseManifestPath/candidateRoot/phase/publicItem``. The runner is the
trusted verifier process pinned by ``runtime["verifier"]`` — the locked file is
this script and ``verifier.command`` must execute it.

Scoring semantics (RSI_tau scoring re-expression addendum v0.1 TAU2–TAU3): the candidate protocol is re-expressed from
action-data replay into a **final-state artifact**--the candidate writes the final database state after interaction and the agent message record as
a ``{"db_state": {...}, "messages": [...]}`` structure to the candidateRoot-relative path declared by
``verifier_spec.payload.artifact.path``. After validating the deliverable structure, the runner dispatches through
``data_designer.verifiers.run_verifier``: ``unit_test`` writes
db_state/messages as files into a controlled workspace for trusted assertions (including required-information
assertions for the COMMUNICATE reward base), and regular DB final states fall back to ``artifact_compare`` (field-by-field
comparison against the expected gold final state). No official tool semantics are reimplemented and there is no dynamic user simulator;
how the solving arm obtains the final state is the evaluation harness's responsibility, and the scorer only examines the deliverable (red line 4).

Phases:
  reference/negative  audit deliverables come from the trusted audit roots; a
                      missing/unparseable/ill-shaped deliverable is an explicit
                      audit failure (ValueError).
  evaluation          a missing/unparseable/ill-shaped deliverable is the
                      candidate's fault: ``invalid_candidate`` with score 0,
                      never silent.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

VERSION = "tau-offline-adapter-v1"

OFFLINE_SPEC_KINDS = ("unit_test", "artifact_compare")

from data_designer.verifiers import run_verifier  # noqa: E402

DEFAULT_DELIVERABLE = "tau_offline_artifact.json"


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


def _deliverable_artifact(deliverable: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """TAU2 protocol shape: db_state object + message record, verifier-facing.

    ``unit_test`` receives a file dict so the frozen assertion script reads
    ``db_state.json`` / ``messages.json`` from the controlled workspace;
    ``artifact_compare`` compares table-wise against ``db_state`` only.
    """
    if not isinstance(deliverable, dict):
        raise ValueError("deliverable must be a JSON object")
    if set(deliverable) - {"db_state", "messages"} or not deliverable:
        raise ValueError("deliverable must be exactly {db_state, messages}")
    db_state = deliverable.get("db_state")
    messages = deliverable.get("messages")
    if not isinstance(db_state, dict) or not db_state:
        raise ValueError("deliverable db_state must be a non-empty object")
    if not isinstance(messages, list):
        raise ValueError("deliverable messages must be a list")
    if kind == "artifact_compare":
        return db_state
    return {
        "db_state.json": json.dumps(db_state, ensure_ascii=False, sort_keys=True),
        "messages.json": json.dumps(messages, ensure_ascii=False),
    }


def score_request(request: Dict[str, Any]) -> Dict[str, Any]:
    started = time.monotonic()
    manifest = _load_json(Path(request["caseManifestPath"]))
    if manifest.get("task_id") != request.get("taskId"):
        raise ValueError("tau task ID disagrees with trusted manifest")
    if manifest.get("family") != "tau":
        raise ValueError("trusted manifest is not a tau case")
    spec = manifest.get("private", {}).get("verifier_spec")
    if not isinstance(spec, dict):
        raise ValueError("trusted manifest lacks a verifier_spec")
    if spec.get("kind") not in OFFLINE_SPEC_KINDS:
        raise ValueError(f"unsupported verifier kind: {spec.get('kind')!r}")
    relative = _deliverable_relative(spec)
    root = Path(request["candidateRoot"]).resolve(strict=True)
    phase = request.get("phase")
    task_id = request["taskId"]
    deliverable_path = root / relative
    try:
        size = _artifact_size(root, relative)
        deliverable = _load_json(deliverable_path)
        artifact = _deliverable_artifact(deliverable, spec["kind"])
        message_count = len(deliverable["messages"])
    except (OSError, ValueError) as error:
        if phase in ("reference", "negative"):
            raise ValueError(f"audit deliverable invalid for {task_id}: {error}")
        response = {
            "task_id": task_id, "status": "invalid_candidate",
            "passed": False, "score": 0.0,
            "native_metrics": {"artifact_bytes": 0.0, "message_count": 0.0},
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
        "native_metrics": {
            "artifact_bytes": float(size), "message_count": float(message_count),
        },
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
        raise ValueError("usage: tau_offline.py <request.json> <result.json>")
    request = _load_json(Path(sys.argv[1]))
    result = score_request(request)
    Path(sys.argv[2]).write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
