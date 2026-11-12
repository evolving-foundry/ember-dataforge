"""unit_test verifier: run an assertion script in a controlled temporary workspace (§7.2 Code category).

payload structure (wb-task-v1):
    {
      "setup_files": {"input/sales.csv": "region,revenue\\n..."},   # fixed task inputs
      "test_script": "assert open('output/summary.json')...",       # scoring assertion source
      "timeout_s": 30
    }

artifact is the agent-produced file dictionary ``{filename: text_content}``; the verifier writes
``setup_files`` and artifact together into a temporary workspace, then **runs ``test_script`` with python3 only**
(no shell, allowlisted interpreter, enforced timeout)--the script uses ordinary
``assert``/``open`` to make programmatic assertions about artifacts. Exit code 0 -> passed; otherwise failed,
with the tail of stderr captured in detail. score is only 0.0/1.0 (assertion scripts have no partial-credit semantics).

Security boundary: test_script is trusted code frozen on the test-setting side (generator_ref in provenance);
the agent can affect only artifact file contents and cannot change the execution entry point.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_TIMEOUT_S = 30
_MAX_STDERR = 600


def _write_tree(root: Path, files: Dict[str, Any]) -> List[str]:
    written: List[str] = []
    for name, content in files.items():
        rel = Path(str(name))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Invalid file path: {name!r}")
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        written.append(str(rel))
    return written


def run_unit_tests(artifact: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """verifier main entry: artifact={filename: content}; payload carries the frozen assertion script."""
    script = payload.get("test_script")
    if not isinstance(script, str) or not script.strip():
        return {"status": "error",
                "detail": "verifier payload is missing test_script (configuration error; fail explicitly)"}
    if not isinstance(artifact, dict) or not artifact:
        return {"passed": False, "score": 0.0,
                "detail": f"artifact is not a file dictionary: {type(artifact).__name__}"}
    timeout_s = float(payload.get("timeout_s") or DEFAULT_TIMEOUT_S)
    try:
        with tempfile.TemporaryDirectory(prefix="wb-unit-test-") as tmp:
            root = Path(tmp)
            setup = payload.get("setup_files") or {}
            if not isinstance(setup, dict):
                return {"status": "error",
                        "detail": "payload.setup_files is not a dictionary (configuration error)"}
            _write_tree(root, setup)
# artifact filenames come from candidate deliverables: path escape is the candidate's responsibility and scores 0.
            for name in artifact:
                rel = Path(str(name))
                if rel.is_absolute() or ".." in rel.parts or not all(rel.parts):
                    return {"passed": False, "score": 0.0,
                            "detail": f"Invalid file path: {name!r}"}
            _write_tree(root, artifact)
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True, text=True, timeout=timeout_s, cwd=root,
                )
            except subprocess.TimeoutExpired:
# timeout_s for the trusted assertion script is the budget set by the task-setting side for candidate execution:
# hitting the limit means the candidate did not finish within budget, scored 0 by benchmark convention (runner contract:
# assertion timeouts/exceptions score 0), not a judging-infrastructure failure.
                return {"passed": False, "score": 0.0,
                        "detail": f"test_script timed out (>{timeout_s:g}s)"}
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip()[-_MAX_STDERR:]
                return {"passed": False, "score": 0.0,
                        "detail": f"test_script exited with code {proc.returncode}: {tail}"}
            return {"passed": True, "score": 1.0,
                    "detail": "test_script exit 0"}
    except (OSError, ValueError) as exc:
        return {"status": "error",
                "detail": f"unit_test workspace execution failed: {exc}"}


__all__ = ["run_unit_tests", "DEFAULT_TIMEOUT_S"]
