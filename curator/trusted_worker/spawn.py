"""evaluation mode subprocess entry point (RSI `evaluator/command` assembly, §3.9.2 v0.2.1).

Invocation: ``python -m curator.trusted_worker <input.json> <result.json>``

Red line (unified with DataForge P0 / op mode): failures do not write the result file, exit non-zero,
and diagnostics go only to stderr; ``score == metrics[objective.metric]``.
"""
from __future__ import annotations

import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

from curator.trusted_worker._bootstrap import ensure_torch_stub

ensure_torch_stub()

from curator.trusted_worker.errors import (  # noqa: E402
    EXIT_INVALID_INPUT,
    EXIT_KNOWN_FAILURE,
    EXIT_OK,
    WorkerError,
    diagnostic_line,
)
from curator.trusted_worker.ops.dedup import execute_exact_dedup  # noqa: E402
from curator.trusted_worker.ops.quality_assess import (  # noqa: E402
    assess_rows,
    build_quality_config,
)
from curator.trusted_worker.registry import OP_REGISTRY  # noqa: E402

SCHEMA_VERSION = 2

_ENVELOPE_TOP_KEYS = {"schemaVersion", "request", "target", "output"}
_REQUEST_KEYS = {"evaluationId", "candidate", "plan"}
# targetConfigId is a required field of the RSI evaluator/command assertPlan (always present in the host envelope);
# worker does not consume it, only allows it through.
_PLAN_KEYS = {"id", "targetConfigId", "objective", "config"}
_OBJECTIVE_KEYS = {"metric", "direction"}
_CONFIG_KEYS = {
    "op",
    "trajectories_path",
    "raw_logs_path",
    "operation_name",
    "quality_config",
}


def _invalid(message: str) -> WorkerError:
    return WorkerError("ENVELOPE_INVALID", message, exit_code=EXIT_INVALID_INPUT)


def _reject_unknown(record: Dict[str, Any], allowed: set, where: str) -> None:
    unknown = set(record) - allowed
    if unknown:
        raise _invalid(f"{where} has unknown keys: {sorted(unknown)}")


def parse_envelope(raw: str) -> Dict[str, Any]:
    """Validate and return the envelope; on structural error, raise an exit-2 WorkerError."""
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _invalid(f"envelope is not valid JSON: {exc}") from exc
    if not isinstance(envelope, dict):
        raise _invalid("envelope must be a JSON object")
    if envelope.get("schemaVersion") != SCHEMA_VERSION:
        raise _invalid(f"schemaVersion must be {SCHEMA_VERSION}")
    _reject_unknown(envelope, _ENVELOPE_TOP_KEYS, "envelope")

    request = envelope.get("request")
    if not isinstance(request, dict):
        raise _invalid("request must be an object")
    _reject_unknown(request, _REQUEST_KEYS, "request")
    if not isinstance(request.get("evaluationId"), str):
        raise _invalid("request.evaluationId must be a string")
    if not isinstance(request.get("candidate"), str):
        raise _invalid("request.candidate must be a string")

    plan = request.get("plan")
    if not isinstance(plan, dict):
        raise _invalid("request.plan must be an object")
    _reject_unknown(plan, _PLAN_KEYS, "request.plan")
    if not isinstance(plan.get("id"), str):
        raise _invalid("request.plan.id must be a string")
    if "targetConfigId" in plan and not isinstance(plan["targetConfigId"], str):
        raise _invalid("request.plan.targetConfigId must be a string")

    objective = plan.get("objective")
    if not isinstance(objective, dict):
        raise _invalid("request.plan.objective must be an object")
    _reject_unknown(objective, _OBJECTIVE_KEYS, "request.plan.objective")
    if not isinstance(objective.get("metric"), str):
        raise _invalid("request.plan.objective.metric must be a string")

    config = plan.get("config")
    if not isinstance(config, dict):
        raise _invalid("request.plan.config must be an object")
    _reject_unknown(config, _CONFIG_KEYS, "request.plan.config")

    output = envelope.get("output")
    if not isinstance(output, dict):
        raise _invalid("output must be an object")
    if not isinstance(output.get("rootDir"), str):
        raise _invalid("output.rootDir must be a string")
    return envelope


def read_jsonl_rows(path: Path, where: str) -> List[Dict[str, Any]]:
    """Read the JSONL data file; if unreadable, raise exit-1; if a line is not an object, raise exit-2."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkerError(
            "DATA_UNREADABLE", f"{where} not readable: {path} ({exc})"
        ) from exc
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _invalid(f"{where} line {line_no} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise _invalid(f"{where} line {line_no} must be a JSON object")
        rows.append(row)
    if not rows:
        raise _invalid(f"{where} contains no records")
    return rows


def _assert_finite(value: Any, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise WorkerError(
            "NON_FINITE_SCORE",
            f"non-finite score for '{name}': refusing to fabricate metrics",
        )


def run_quality(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """CandidateBatchQualityAssess: DataQualityAssessor.assess path."""
    config = envelope["request"]["plan"]["config"]
    rows = read_jsonl_rows(Path(config["trajectories_path"]), "trajectories_path")
    quality_config = build_quality_config(config.get("quality_config", {}))
    report = assess_rows(
        rows,
        quality_config,
        operation_name=config.get("operation_name", "candidate_batch_quality_assess"),
    )

    score = report.overall_score
    _assert_finite(score, "overall_score")
    objective_metric = envelope["request"]["plan"]["objective"]["metric"]
    metrics: Dict[str, Any] = {objective_metric: score}
    for dim, dim_score in report.dimension_scores.items():
        key = dim.value if hasattr(dim, "value") else str(dim)
        metrics[key] = dim_score.score
    metrics["num_rows"] = float(report.num_rows)
    metrics["issues_count"] = float(len(report.issues))

    artifact_path = _write_artifact(
        envelope, "quality-report.json", report.model_dump(mode="json")
    )
    return _result(envelope, score, metrics, artifact_path)


def run_dedup(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """CandidateLogDeduplicate: ExactDuplicateRemovalOperation path."""
    config = envelope["request"]["plan"]["config"]
    rows = read_jsonl_rows(Path(config["raw_logs_path"]), "raw_logs_path")
    input_count, deduped = execute_exact_dedup(rows)
    kept = deduped.dataset.count()
    removed = max(input_count - kept, 0)
    dedup_ratio = removed / input_count if input_count > 0 else 0.0

    score = 1.0 if removed > 0 else 0.0
    objective_metric = envelope["request"]["plan"]["objective"]["metric"]
    metrics: Dict[str, Any] = {
        objective_metric: score,
        "input_count": float(input_count),
        "kept": float(kept),
        "removed": float(removed),
        "dedup_ratio": dedup_ratio,
    }
    _assert_finite(dedup_ratio, "dedup_ratio")

    artifact_path = _write_artifact(envelope, "deduped-logs.json", deduped.dataset.take_all())
    return _result(envelope, score, metrics, artifact_path)


def _write_artifact(
    envelope: Dict[str, Any], name: str, payload: Any
) -> Dict[str, str]:
    output_root = Path(envelope["output"]["rootDir"])
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / name
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    artifact = {"relativePath": name}
    scope_id = envelope["output"].get("id")
    if isinstance(scope_id, str):
        artifact = {"scopeId": scope_id, "relativePath": name}
    return artifact


def _result(
    envelope: Dict[str, Any],
    score: float,
    metrics: Dict[str, Any],
    artifact: Dict[str, str],
) -> Dict[str, Any]:
    request = envelope["request"]
    return {
        "evaluationId": request["evaluationId"],
        "candidate": request["candidate"],
        "planId": request["plan"]["id"],
        "score": score,
        "metrics": metrics,
        "artifacts": [artifact],
        "externalJobId": f"curator-py-{uuid.uuid4().hex[:8]}",
    }


_DISPATCH = {"quality": run_quality, "dedup": run_dedup}


def run(input_path: str, output_path: str) -> int:
    """evaluation mode process body: on success, write result and exit 0; on failure, no result file and exit 1/2."""
    try:
        raw = Path(input_path).read_text(encoding="utf-8")
    except OSError as exc:
        print(diagnostic_line(WorkerError("ENVELOPE_UNREADABLE", str(exc))), file=sys.stderr)
        return EXIT_INVALID_INPUT

    try:
        envelope = parse_envelope(raw)
        config = envelope["request"]["plan"]["config"]
        op_name = config.get("op")
        if not isinstance(op_name, str):
            raise _invalid("request.plan.config.op must be a string")
        if op_name not in OP_REGISTRY:
            raise WorkerError(
                "OP_NOT_FOUND",
                f"unknown op: {op_name}; supported: {sorted(OP_REGISTRY)}",
            )
        result = _DISPATCH[OP_REGISTRY[op_name]["kind"]](envelope)
    except WorkerError as error:
        print(diagnostic_line(error), file=sys.stderr)
        return error.exit_code
    except Exception as error:  # noqa: BLE001 - red line: any operator exception must not produce a result
        print(
            diagnostic_line(WorkerError("OP_EXECUTION_FAILED", str(error))),
            file=sys.stderr,
        )
        return EXIT_KNOWN_FAILURE

    Path(output_path).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return EXIT_OK
