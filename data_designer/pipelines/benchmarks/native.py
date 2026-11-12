"""Process boundary for an installed upstream native runner and trusted verifier."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from .contracts import BenchmarkError, NativeFailure, validate_native_result


def run_native(
    command: List[str], request: Dict[str, Any], *, timeout: float
) -> Dict[str, Any]:
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise BenchmarkError("native runner command must be a non-empty string list")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise BenchmarkError("native runner timeout must be finite and positive")
    with tempfile.TemporaryDirectory(prefix="benchmark-native-") as directory:
        input_path = Path(directory) / "request.json"
        output_path = Path(directory) / "result.json"
        input_path.write_text(
            json.dumps(request, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        try:
            proc = subprocess.run(
                [*command, str(input_path), str(output_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise NativeFailure(
                "timeout", f"native runner timed out: {error}"
            ) from error
        except OSError as error:
            raise NativeFailure(
                "runner_unavailable", f"native runner unavailable: {error}"
            ) from error
        if proc.returncode != 0:
            raise NativeFailure(
                "runner_failed",
                f"native runner failed ({proc.returncode}): {proc.stderr[-500:]}"
            )
        try:
            return json.loads(
                output_path.read_text(encoding="utf-8"),
                parse_constant=lambda name: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant: {name}")
                ),
            )
        except (OSError, ValueError) as error:
            raise NativeFailure(
                "verifier_protocol", f"native runner result invalid: {error}"
            ) from error


def run_case(
    command: List[str],
    task_id: str,
    manifest_path: str,
    candidate_root: str,
    *,
    phase: str,
    timeout: float,
    public_item: Dict[str, Any],
    attempt: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    if attempt is not None and (
        not isinstance(attempt, dict)
        or type(attempt.get("index")) is not int
        or attempt["index"] < 0
        or type(attempt.get("seed")) is not int
        or not 0 <= attempt["seed"] < 2**63
    ):
        raise BenchmarkError("native attempt index/seed invalid")
    response = run_native(
        command,
        {
            "schemaVersion": 1,
            "taskId": task_id,
            "caseManifestPath": manifest_path,
            "candidateRoot": candidate_root,
            "phase": phase,
            "publicItem": public_item,
            **({"attempt": attempt} if attempt is not None else {}),
        },
        timeout=timeout,
    )
    result = validate_native_result(response, task_id)
    if attempt is not None and (
        result.get("attempt_index") != attempt["index"]
        or result.get("seed") != attempt["seed"]
    ):
        raise NativeFailure(
            "verifier_protocol", "native runner did not acknowledge attempt seed"
        )
    return result
