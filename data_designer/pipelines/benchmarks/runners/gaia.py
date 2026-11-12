"""Score a private GAIA-style answer with a pinned official scorer file.

The scorer and the expected answer live in the trusted case manifest. The
candidate directory contains only ``answer.txt``; the public task item never
contains either the answer or scorer path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


VERSION = "gaia-official-scorer-adapter-v1"
MAX_ANSWER_BYTES = 16_384


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_scorer(path: Path, expected_sha: str):
    if not path.is_absolute() or not path.is_file() or _sha256(path) != expected_sha:
        raise ValueError("GAIA scorer file is unavailable or its SHA-256 drifted")
    spec = importlib.util.spec_from_file_location("pinned_gaia_scorer", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned GAIA scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = getattr(module, "question_scorer", None)
    if not callable(scorer):
        raise ValueError("pinned GAIA scorer has no question_scorer")
    return scorer


def score_request(request: dict) -> dict:
    started = time.monotonic()
    manifest_path = Path(request["caseManifestPath"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task_id") != request.get("taskId"):
        raise ValueError("GAIA task ID disagrees with trusted manifest")
    private = manifest["private"]
    answer = private["expected_answer"]
    scorer_record = private["gaia_scorer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("trusted GAIA answer must be a nonempty string")
    answer_sha = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    if answer_sha != private.get("expected_answer_sha256"):
        raise ValueError("trusted GAIA answer SHA-256 drifted")
    scorer = _load_scorer(
        Path(scorer_record["path"]), scorer_record["sha256"]
    )

    root = Path(request["candidateRoot"]).resolve(strict=True)
    answer_path = root / "answer.txt"
    if not answer_path.is_file() or answer_path.is_symlink():
        raise ValueError("candidate answer.txt is missing or not a regular file")
    if answer_path.stat().st_size > MAX_ANSWER_BYTES:
        raise ValueError("candidate answer exceeds maximum size")
    candidate = answer_path.read_text(encoding="utf-8").strip()
    passed = bool(scorer(candidate, answer))
    result = {
        "task_id": request["taskId"],
        "status": "completed",
        "passed": passed,
        "score": float(passed),
        "native_metrics": {"official_answer_match": float(passed)},
        "verifier_version": VERSION,
        "evidence_refs": [f"gaia-scorer-sha256:{scorer_record['sha256']}"],
        "runtime_seconds": time.monotonic() - started,
    }
    if "attempt" in request:
        result["attempt_index"] = request["attempt"]["index"]
        result["seed"] = request["attempt"]["seed"]
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: gaia.py <request.json> <result.json>")
    request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    result = score_request(request)
    Path(args[1]).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
