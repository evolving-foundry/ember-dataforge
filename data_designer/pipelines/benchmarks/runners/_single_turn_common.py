"""single-turn native runner: trusted verifier_spec scoring over the JSON file protocol.

Shared implementation for the four NeMo/HF single-turn families
(``reasoning_qa`` / ``knowledge_qa`` / ``code_gen`` / ``health_qa``); the four
thin entry points pin the family and delegate here. Structure mirrors
``runners/terminal.py``: file protocol matches ``native.py:run_case`` exactly —
argv is ``<request.json> <result.json>`` where the request carries
``taskId/caseManifestPath/candidateRoot/phase/publicItem``. Zero Docker, zero
browser, zero online data pull: scoring happens offline through the package
verifier dispatch (``data_designer.verifiers.run_verifier``).

Candidate protocol (D6, family candidate convention): the solver writes its
final answer text to the relative path declared by
``verifier_spec.payload.artifact.path`` (``response.txt`` by convention;
``code_gen`` tasks declare ``solution.py`` or a per-task path) under
``candidateRoot``. The trusted manifest's ``verifier_spec`` kind must be
``unit_test`` (frozen assertion script executed offline in a scrubbed
workspace, whitelisted interpreter, enforced timeout — the eight deterministic
matchers are Python re-expressions of the Evaluator ``score.ts`` semantics,
and humaneval executes candidate code inside that same trusted boundary, D8)
or ``llm_rubric`` (simpleqa / healthbench; rubric text and ``pass_threshold``
are frozen in the payload, the judge endpoint is injected by the marking side
via ``configure_llm_judge`` and is never part of the published task).

Phases (identical audit semantics to terminal and the official families):
  reference/negative  audit deliverables come from the trusted audit roots; a
                      missing or unreadable deliverable is an explicit audit
                      failure. Judge families treat ``passed`` as threshold
                      semantics (score >= ``pass_threshold``).
  evaluation          a missing or unreadable deliverable is the candidate's
                      fault: ``invalid_candidate`` with score 0, never silent.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

VERSION = "single-turn-verifier-shim-v1"

SUPPORTED_SPEC_KINDS = ("unit_test", "llm_rubric")

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


def _deliverable_size(root: Path, relative: str) -> int:
    target = root / relative
    if not _within(target, root) or target.is_symlink() or not target.is_file():
        raise FileNotFoundError(
            f"deliverable outside candidateRoot or missing: {relative}"
        )
    return target.stat().st_size


def _response(
    task_id: str, request: Dict[str, Any], **fields: Any
) -> Dict[str, Any]:
    response = {"task_id": task_id, **fields}
    if "attempt" in request:
        response.update(
            attempt_index=request["attempt"]["index"],
            seed=request["attempt"]["seed"],
        )
    return response


def score_request(request: Dict[str, Any], family: str) -> Dict[str, Any]:
    started = time.monotonic()
    manifest = _load_json(Path(request["caseManifestPath"]))
    if manifest.get("task_id") != request.get("taskId"):
        raise ValueError(f"{family} task ID disagrees with trusted manifest")
    if manifest.get("family") != family:
        raise ValueError(f"trusted manifest is not a {family} case")
    spec = manifest.get("private", {}).get("verifier_spec")
    if not isinstance(spec, dict):
        raise ValueError("trusted manifest lacks a verifier_spec")
    if spec.get("kind") not in SUPPORTED_SPEC_KINDS:
        raise ValueError(
            f"{family} verifier_spec.kind must be one of "
            f"{SUPPORTED_SPEC_KINDS}: {spec.get('kind')!r}"
        )
    relative = _deliverable_relative(spec)
    root = Path(request["candidateRoot"]).resolve(strict=True)
    phase = request.get("phase")
    task_id = request["taskId"]
    deliverable = root / relative
    try:
        size = _deliverable_size(root, relative)
        text = deliverable.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        if phase in ("reference", "negative"):
            raise ValueError(f"audit deliverable invalid for {task_id}: {error}")
        response = _response(
            task_id,
            request,
            status="invalid_candidate",
            passed=False,
            score=0.0,
            native_metrics={"deliverable_bytes": 0.0},
            verifier_version=VERSION,
            evidence_refs=[],
            runtime_seconds=time.monotonic() - started,
        )
        return response
    # unit_test expects a file dictionary (the frozen test_script asserts on
    # the deliverable contents inside its scrubbed workspace); llm_rubric
    # expects the raw response text.
    artifact: Any = (
        {relative: text} if spec["kind"] == "unit_test" else text
    )
    judge = None
    if spec.get("kind") == "llm_rubric":
        from data_designer.integrations.curator.judge import judge_from_environment
        judge = judge_from_environment()
    verdict = run_verifier(spec, artifact, judge=judge)
    if verdict["status"] != "completed":
# detail retains the verifier failure reason (such as judge not configured); red-line validation depends on it.
        raise RuntimeError(
            "domain verifier did not complete; no score is available: "
            f"{verdict.get('detail')}"
        )
    score = float(verdict.get("score") or 0.0)
    if not 0.0 <= score <= 1.0:
        raise ValueError("verifier score outside [0,1]")
    response = _response(
        task_id,
        request,
        status="completed",
        passed=bool(verdict.get("passed")),
        score=score,
        native_metrics={"deliverable_bytes": float(size)},
        verifier_version=VERSION,
        evidence_refs=[f"deliverable:{relative}:bytes:{size}"],
        runtime_seconds=time.monotonic() - started,
    )
    return response
