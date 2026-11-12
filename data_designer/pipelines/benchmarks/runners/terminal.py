"""terminal native runner: trusted verifier_spec scoring over the JSON file protocol.

Structure mirrors ``runners/workbuddy.py`` (workbuddy intake, direct template):
file protocol matches ``native.py:run_case`` exactly — argv is
``<request.json> <result.json>`` where the request carries
``taskId/caseManifestPath/candidateRoot/phase/publicItem``. Zero Docker, zero
external environment: scoring happens offline through the package verifier
dispatch (``data_designer.verifiers.run_verifier``).

Candidate protocol (T6, family candidate convention): the solver writes the
deliverable — a ``{filename: content}`` file-dictionary JSON naming the target
artifacts declared by the acceptance signal — to the relative path
``verifier_spec.payload.artifact.path`` under ``candidateRoot``. The trusted
manifest's ``verifier_spec`` kind must be ``unit_test`` (frozen assertion
script executed offline in a scrubbed workspace, whitelisted interpreter,
enforced timeout) or ``artifact_compare``. The runner is family-agnostic about
kind: ``run_verifier`` dispatches; only the whitelist is enforced here.

Phases (identical audit semantics to workbuddy and the six official families):
  reference/negative  audit deliverables come from the trusted audit roots; a
                      missing or unparseable deliverable is an explicit audit
                      failure.
  evaluation          a missing or unparseable deliverable is the candidate's
                      fault: ``invalid_candidate`` with score 0, never silent.
                      Assertion timeouts/exceptions score 0 with detail via
                      the verifier itself; no environment preconditions exist.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

VERSION = "terminal-verifier-shim-v1"

SUPPORTED_SPEC_KINDS = ("unit_test", "artifact_compare")

from data_designer.verifiers import run_verifier  # noqa: E402


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
    relative = str(artifact.get("path") or "")
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
        raise ValueError("terminal task ID disagrees with trusted manifest")
    if manifest.get("family") != "terminal":
        raise ValueError("trusted manifest is not a terminal case")
    spec = manifest.get("private", {}).get("verifier_spec")
    if not isinstance(spec, dict):
        raise ValueError("trusted manifest lacks a verifier_spec")
    if spec.get("kind") not in SUPPORTED_SPEC_KINDS:
        raise ValueError(
            f"terminal verifier_spec.kind must be one of "
            f"{SUPPORTED_SPEC_KINDS}: {spec.get('kind')!r}"
        )
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
        raise ValueError("usage: terminal.py <request.json> <result.json>")
    request = _load_json(Path(sys.argv[1]))
    result = score_request(request)
    Path(sys.argv[2]).write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
