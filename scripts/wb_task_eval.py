"""wb_task_eval.py — workbuddy_task (wb-task-v1) scoring dispatch (design §7.3).

The subprocess contract is identical to ``scripts/dsh_headless_eval.py``
(``ExternalEvalHarness``, ``rsi/evaluation.py`` unchanged):
  stdin : taskset JSON (``AgentTaskSetBuilder`` consumption shape list)
  stdout: JSON array [{"task_id", "passed", "score", "detail"}, ...]

Execution chain (per task):
  1. The solver arm produces a deliverable (JSON artifact):
     - default ``--solver dsh``: ``dsh --profile headless`` real model call
       (429/timeout backoff same as dsh_headless_eval); the prompt requires outputting only
       the JSON content of the file path specified by ``verifier_spec.payload.artifact``;
     - ``--solver-cmd <cmd...>``: external solver subprocess (stdin=task JSON,
       stdout=artifact JSON) -- an L1 boundary converter for offline e2e / swapping model arms.
  2. Dispatch scoring by ``verifier_spec.kind`` (artifact_compare already implemented;
     other kinds explicitly report verifier_dispatch_error, never silently).

Candidate arm (``--skill <file.json>``): the same instructions injection as dsh_headless_eval
(L1 boundary converter).

Usage:
  cat wb_taskset.json | python3 scripts/wb_task_eval.py
  cat wb_taskset.json | python3 scripts/wb_task_eval.py \
      --skill skill.json --timeout-per-task 240 --workers 4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_designer.verifiers import run_verifier

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Optional[Any]:
    """Extract a JSON value from the solver arm's stdout: bare JSON → code block → first/last braces."""
    raw = (text or "").strip()
    if not raw:
        return None
    for candidate in (raw, *[_m.group(1) for _m in _FENCE_RE.finditer(raw)]):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _artifact_instruction(task: Dict[str, Any]) -> str:
    spec = task.get("verifier_spec") or {}
    payload = spec.get("payload") or {}
    artifact = payload.get("artifact") or {}
    path = str(artifact.get("path") or "output/result.json")
    fmt = str(artifact.get("format") or "json")
    return (
        f"\n\nDeliverable: produce the file `{path}` (format: {fmt}). "
        "Respond with ONLY the JSON content of that file — no prose, "
        "no markdown, no code fences."
    )


def _skill_instructions(path: str) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pkg = data.get("package", data) if isinstance(data, dict) else {}
    content = pkg.get("content") or {}
    instructions = str(content.get("instructions") or "").strip()
    summary = str(pkg.get("summary") or "").strip()
    parts = [p for p in (summary, instructions) if p]
    if not parts:
        raise ValueError(f"Skill file {path} contains no injectable instructions/summary")
    return "\n\n[Operating guidelines]\n" + "\n".join(parts)


def _is_rate_limited(stderr: str) -> bool:
    low = (stderr or "").lower()
    return "rate_limit" in low or "429" in low


def _solve_dsh(
    dsh_bin: str,
    profile: str,
    timeout_s: int,
    prompt: str,
    max_retries: int,
    retry_backoff_s: int,
) -> Optional[Any]:
    for attempt in range(max_retries + 1):
        try:
            proc = subprocess.run(
                [dsh_bin, "--profile", profile, prompt],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                time.sleep(retry_backoff_s)
                continue
            return None
        if proc.returncode != 0:
            if _is_rate_limited(proc.stderr) and attempt < max_retries:
                time.sleep(retry_backoff_s)
                continue
            return None
        return _extract_json(proc.stdout)
    return None  # pragma: no cover


def _solve_cmd(
    solver_cmd: List[str],
    timeout_s: int,
    task: Dict[str, Any],
) -> Optional[Any]:
    try:
        proc = subprocess.run(
            solver_cmd,
            input=json.dumps(task, ensure_ascii=False),
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return _extract_json(proc.stdout)


def _run_one(args: argparse.Namespace, task: Dict[str, Any],
             skill_block: Optional[str]) -> Dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    spec = task.get("verifier_spec")
    if not isinstance(spec, dict) or not spec.get("kind"):
        return {"task_id": task_id, "passed": False, "score": 0.0,
                "detail": "verifier_dispatch_error: task is missing verifier_spec (configuration error)"}
    prompt = str(task.get("task_text") or "") + _artifact_instruction(task)
    if skill_block:
        prompt = skill_block + "\n\n---\n\n" + prompt
    if args.solver_cmd:
        artifact = _solve_cmd(args.solver_cmd, args.timeout_per_task, task)
    else:
        artifact = _solve_dsh(
            args.dsh_bin, args.profile, args.timeout_per_task, prompt,
            args.max_retries, args.retry_backoff_s)
    if artifact is None:
        return {"task_id": task_id, "passed": False, "score": 0.0,
                "detail": "solver did not produce a parseable JSON deliverable"}
    try:
        verdict = run_verifier(spec, artifact)
    except (NotImplementedError, RuntimeError) as exc:
        return {"task_id": task_id, "passed": False, "score": 0.0,
                "detail": f"verifier_dispatch_error: {exc}"}
    return {"task_id": task_id, "passed": bool(verdict["passed"]),
            "score": float(verdict.get("score") or 0.0),
            "detail": str(verdict.get("detail") or "")}


def main() -> int:
    ap = argparse.ArgumentParser(description="wb-task-v1 taskset scoring dispatch")
    ap.add_argument("--dsh-bin", default="dsh")
    ap.add_argument("--profile", default="headless")
    ap.add_argument("--solver-cmd", nargs="+", default=None,
                    help="External solver command (stdin=task JSON, stdout=artifact JSON)")
    ap.add_argument("--timeout-per-task", type=int, default=240)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--retry-backoff-s", type=int, default=30)
    ap.add_argument("--skill", default=None,
                    help="Exported JSON for the published skill (candidate arm injected into instructions)")
    ap.add_argument("--judge-base-url", default=None,
                    help="llm_rubric judge endpoint (configured separately from generator)")
    ap.add_argument("--judge-api-key", default=None)
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    if args.judge_base_url or args.judge_api_key or args.judge_model:
        from data_designer.verifiers import (
            configure_llm_judge,
        )
        configure_llm_judge(args.judge_base_url, args.judge_api_key,
                            args.judge_model)

    tasks = json.load(sys.stdin)
    if not isinstance(tasks, list) or not tasks:
        print("taskset is empty or not a list: E1 requires a non-empty evaluation set", file=sys.stderr)
        return 2
    skill_block = _skill_instructions(args.skill) if args.skill else None

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(
            lambda t: _run_one(args, t, skill_block), tasks))
    json.dump(rows, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
