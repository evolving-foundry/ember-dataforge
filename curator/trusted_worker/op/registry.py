"""op mode registry (Evaluator §4.6: a superset of the evaluation registry).

New operators go here by default (for example, Q2 ``CandidateTextOverlapScore``); entering the frozen evaluation
surface (``trusted_worker/registry.py``) requires the RSI §3.9 version-evolution process.
Metric key names are the contract (they enter metrics.json / report.md; renaming is a breaking change).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from curator.trusted_worker.errors import WorkerError
from curator.trusted_worker.ops.dedup import execute_exact_dedup
from curator.trusted_worker.ops.quality_assess import (
    QUALITY_CONFIG_KEYS,
    assess_rows,
    build_quality_config,
)
from curator.trusted_worker.ops.text_overlap import METRICS, score_pair
from curator.trusted_worker.ops.parse_document import run_parse_document
from curator.trusted_worker.ops.ocr_image import run_ocr_image


def _invalid(message: str) -> WorkerError:
    return WorkerError("REQUEST_INVALID", message, exit_code=2)


def _validate_common_params(op: str, params: Dict[str, Any], allowed: set) -> None:
    if not isinstance(params, dict):
        raise _invalid(f"{op}.params must be an object")
    unknown = set(params) - allowed
    if unknown:
        raise _invalid(f"{op}.params has unknown keys: {sorted(unknown)}")


def _run_dedup(
    op: str, params: Dict[str, Any], rows: List[Dict[str, Any]], output_dir: Optional[Path]
) -> Dict[str, Any]:
    _validate_common_params(op, params, set())
    input_count = len(rows)
    # Deduplicate finalText (mode collapse measurement): fold rows into {'text': ...},
    # taskId is only used to backfill artifact details (first occurrence ownership, deterministic).
    _, deduped = execute_exact_dedup([{"text": r["text"]} for r in rows])
    kept = deduped.dataset.count()
    removed = max(input_count - kept, 0)
    body: Dict[str, Any] = {
        "metrics": {
            "input_count": float(input_count),
            "kept": float(kept),
            "removed": float(removed),
            "dedup_ratio": removed / input_count,
        }
    }
    if output_dir is not None:
        seen = set()
        detail = []
        for row in rows:
            text = row["text"]
            if text not in seen:
                seen.add(text)
                detail.append({"taskId": row["taskId"], "text": text})
        _write_json(output_dir / "deduped-logs.json", detail)
        body["artifacts"] = [{"relativePath": "deduped-logs.json"}]
    return body


def _run_quality(
    op: str, params: Dict[str, Any], rows: List[Dict[str, Any]], output_dir: Optional[Path]
) -> Dict[str, Any]:
    _validate_common_params(op, params, {"quality_config"})
    quality_overrides = params.get("quality_config", {})
    if not isinstance(quality_overrides, dict):
        raise _invalid(f"{op}.params.quality_config must be an object")
    unknown = set(quality_overrides) - QUALITY_CONFIG_KEYS
    if unknown:
        raise _invalid(
            f"{op}.params.quality_config has unknown keys: {sorted(unknown)}"
        )
    report = assess_rows(rows, build_quality_config(quality_overrides))

    # Metric key names are the contract (§4.4); when a dimension is disabled by params, set it to null rather than fabricating a zero score.
    metrics: Dict[str, Any] = {"quality_score": report.overall_score}
    by_key = {
        (dim.value if hasattr(dim, "value") else str(dim)): score_obj.score
        for dim, score_obj in report.dimension_scores.items()
    }
    for dim_name in ("completeness", "consistency", "uniqueness", "text_quality"):
        metrics[dim_name] = by_key.get(dim_name)
    metrics["issues_count"] = float(len(report.issues))

    body = {"metrics": metrics}
    if output_dir is not None:
        _write_json(output_dir / "quality-report.json", report.model_dump(mode="json"))
        body["artifacts"] = [{"relativePath": "quality-report.json"}]
    return body


def _validate_overlap_params(op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    _validate_common_params(op, params, {"metric", "n"})
    metric = params.get("metric", "bleu")
    if not isinstance(metric, str) or metric not in METRICS:
        raise _invalid(f"{op}.params.metric must be one of {list(METRICS)}")
    n = params.get("n", 2 if metric == "ngram" else 1)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise _invalid(f"{op}.params.n must be a positive integer")
    return {"metric": metric, "n": n}


def _run_overlap(
    op: str, params: Dict[str, Any], rows: List[Dict[str, Any]], output_dir: Optional[Path]
) -> Dict[str, Any]:
    normalized = _validate_overlap_params(op, params)
    scores = [
        score_pair(normalized["metric"], r["candidate"], r["reference"], normalized["n"])
        for r in rows
    ]
    body: Dict[str, Any] = {
        "metrics": {
            "mean_score": round(sum(scores) / len(scores), 4),
            "count": float(len(scores)),
        }
    }
    if output_dir is not None:
        with (output_dir / "overlap-scores.jsonl").open("w", encoding="utf-8") as fh:
            for row, score in zip(rows, scores):
                fh.write(json.dumps({"taskId": row["taskId"], "score": score}) + "\n")
        body["perItemPath"] = "overlap-scores.jsonl"
    else:
        body["perItemPath"] = None
    return body


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


OP_MODE_REGISTRY = {
    "CandidateLogDeduplicate": {
        "required_fields": {"taskId": str, "text": str},
        "run": _run_dedup,
    },
    "CandidateBatchQualityAssess": {
        "required_fields": {"taskId": str, "text": str},
        "run": _run_quality,
    },
    "CandidateTextOverlapScore": {
        "required_fields": {"taskId": str, "candidate": str, "reference": str},
        "run": _run_overlap,
    },
    "parse_document": {
        "required_fields": {"sourcePath": str},
        "run_with_context": run_parse_document,
    },
    "ocr_image": {
        "required_fields": {"sourcePath": str},
        "run_with_context": run_ocr_image,
    },
}
