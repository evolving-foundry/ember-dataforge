"""Real DSH headless evaluation harness (DSH-side implementation of the E1 process boundary).

Closes the untested gap in the RSI_ADF data-generation design document §5: run tasksets through the
formal subprocess boundary (``dsh --profile headless``, real model calls) instead of the injected
runner used in tests.

Contract (aligned with ``rsi/evaluation.py::ExternalEvalHarness.evaluate``):
  stdin : taskset JSON (``AgentTaskSetBuilder`` consumption shape: task_id/task_text/
          success_criteria/... list)
  stdout: JSON line array [{"task_id", "passed", "score", "detail"}, ...]

Scoring semantics (E4: programmatic assertions take precedence, deterministic, no LLM judge):
  Extract salient words from success_criteria (remove stopwords, length > 2) and count how many
  appear in the DSH final answer; passed = at least 1 hit and the answer is non-empty;
  score = hit word count / total salient word count.
  Tasks with empty success_criteria are explicitly marked failed (never silently passed).

Candidate arm (--skill <file.json>): inject the published skill's instructions into the task
prompt. Real DSH is an L0 host that does not read managed/, so prompt injection is an explicit L1
boundary converter (corresponding to the generation-side design §6.3 "multi-consumer design: boundary converters").

Usage:
  cat taskset.json | python3 scripts/dsh_headless_eval.py
  cat taskset.json | python3 scripts/dsh_headless_eval.py \
      --skill skill-v1.json --timeout-per-task 120 --workers 4
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

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "all",
    "first", "then", "its", "not", "must", "should", "contains", "contain",
    "identified", "be", "is", "of", "on", "in", "to", "a", "an",
    "was", "were", "has", "have", "had", "will", "into", "out", "one", "two",
    "after", "before", "when", "while", "than", "more", "less", "new",
}


def _salient_tokens(text: str) -> List[str]:
    """success_criteria → salient words (remove stopwords; keep words longer than 2 for both Chinese and English)."""
    words = re.findall(r"\w+", (text or "").lower(), re.UNICODE)
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


def _score(answer: str, criteria: str) -> Dict[str, Any]:
    tokens = _salient_tokens(criteria)
    if not tokens:
        return {"passed": False, "score": 0.0,
                "detail": "success_criteria is empty: the program assertion has no basis; explicitly marking failed"}
    if not (answer or "").strip():
        return {"passed": False, "score": 0.0, "detail": "DSH returned an empty answer"}
    low = answer.lower()
    hits = sorted({t for t in tokens if t in low})
    return {
        "passed": len(hits) >= 1,
        "score": round(len(hits) / len(tokens), 4),
        "detail": f"criteria tokens {len(tokens)}, hit {len(hits)}: {hits}",
    }


def _skill_instructions(path: str) -> str:
    """Read a published skill export file (published payload / frozen candidate / full package all work)."""
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


def _run_one(
    dsh_bin: str,
    profile: str,
    timeout_s: int,
    task: Dict[str, Any],
    skill_block: Optional[str],
    max_retries: int = 2,
    retry_backoff_s: int = 30,
) -> Dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    task_text = str(task.get("task_text") or "")
    criteria = str(task.get("success_criteria") or "")
    prompt = task_text + (f"\n\nDefinition of done: {criteria}" if criteria else "")
    if skill_block:
        prompt = skill_block + "\n\n---\n\n" + prompt
    # 429/rate-limit is infrastructure noise, not a task failure: bounded backoff retries (E4 scoring
    # semantics should only reflect model capability; only explicit failure after retries are exhausted counts as failed).
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
            return {"task_id": task_id, "passed": False, "score": 0.0,
                    "detail": f"dsh headless timed out (>{timeout_s}s, "
                              f"{max_retries + 1} attempts)"}
        if proc.returncode != 0:
            if _is_rate_limited(proc.stderr) and attempt < max_retries:
                time.sleep(retry_backoff_s)
                continue
            return {"task_id": task_id, "passed": False, "score": 0.0,
                    "detail": f"dsh headless exit code {proc.returncode}: "
                              f"{proc.stderr.strip()[:300]}"}
        return {"task_id": task_id, **_score(proc.stdout, criteria)}
    raise AssertionError("unreachable")  # pragma: no cover


def main() -> int:
    ap = argparse.ArgumentParser(description="Real DSH headless taskset evaluation")
    ap.add_argument("--dsh-bin", default="dsh")
    ap.add_argument("--profile", default="headless")
    ap.add_argument("--timeout-per-task", type=int, default=180)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Per-task retry count for 429/timeout (incremental 30s backoff)")
    ap.add_argument("--retry-backoff-s", type=int, default=30)
    ap.add_argument("--skill", default=None,
                    help="Exported JSON for the published skill (candidate arm injected into instructions)")
    args = ap.parse_args()

    tasks = json.load(sys.stdin)
    if not isinstance(tasks, list) or not tasks:
        print("taskset is empty or not a list: E1 requires a non-empty evaluation set", file=sys.stderr)
        return 2
    skill_block = _skill_instructions(args.skill) if args.skill else None

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(
            lambda t: _run_one(args.dsh_bin, args.profile,
                               args.timeout_per_task, t, skill_block,
                               args.max_retries, args.retry_backoff_s),
            tasks,
        ))
    json.dump(rows, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
