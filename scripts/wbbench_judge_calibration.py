"""workbuddy M4 judge calibration (design §10 M4 / §12 D1): single-round real judge measurement on 5 tasks.

Chain: M1 native ingest records → reference artifact synthesis (§7.2 gold re-serialization) → build
a wb_composite case manifest (real frozen archive pins) → ``workbuddy_composite.score_request``
phase=``evaluation`` (real sandbox + frozen grading + sidecar evidence extraction + LLM judge
rubric_binary_mean + score_merge fusion).

- endpoint: ``WB_JUDGE_*`` env; when missing, bridge at runtime from
  ``MODEL_ENDPOINT`` / ``MODEL_API_KEY`` / ``MODEL_NAME`` per the design's "reuse existing gateway token"
  (env injection only, never persisted to a database or repository). If both are missing → explicit failure (calibration must really hit the network).
- Per-task records: llm component, per-rubric binary/reason/latency, token usage, judge status;
  summarize and extrapolate single-round cost/time for 50 tasks (cost = token × --price parameter; when price is undetermined, report
  raw token count).

Usage:
  PYTHONPATH=. python3 scripts/wbbench_judge_calibration.py \
      --seed-archive <workbuddy-seed.tar.gz> --work-dir <directory outside the repo>

Exit code: 0 = 5-task calibration complete; 2 = precondition failure (explicit failure, never silent).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_designer.pipelines.benchmarks.runners import (
    workbuddy_composite as runner,
)
from data_designer.pipelines.benchmarks.runners import workbuddy_judge as wb_judge
from data_designer.pipelines.benchmarks.families.workbuddy import (
    derive_release,
)
from data_designer.pipelines.benchmarks.families.workbuddy_audit import (
    synthesize_reference_artifacts,
)
from data_designer.sources.workbuddy_seed import WBBenchSeedAdapter


# The same 5-task subset used in the M2 trial run (JSON deliverables; reference artifacts = gold re-serialization).
CALIBRATION_TASKS = [
    "handbook-notice-insert-L3L4-030",
    "effective-sop-reconstruction-L5-034",
    "effective-control-state-L5-036",
    "device-incident-attribution-L5-037",
    "inspection-notice-provenance-L4-032",
]


def _write_json(path: Path, value: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(path)


def _bridge_judge_env() -> Dict[str, str]:
    """Bridge from the existing gateway token when WB_JUDGE_* is missing (env only, effective at runtime)."""
    source = "wb_env"
    if not wb_judge.backend_config():
        endpoint = os.environ.get("MODEL_ENDPOINT") or ""
        api_key = os.environ.get("MODEL_API_KEY") or ""
        model = os.environ.get("MODEL_NAME") or ""
        if endpoint and api_key and model:
            os.environ["WB_JUDGE_BASE_URL"] = endpoint
            os.environ["WB_JUDGE_API_KEY"] = api_key
            os.environ["WB_JUDGE_MODEL"] = model
            source = "model_env_bridge"
    config = wb_judge.backend_config()
    if config is None:
        raise RuntimeError(
            "judge backend is missing: neither WB_JUDGE_* nor MODEL_* environment variables are available;"
            "calibration must make real network calls (section 10, M4); silent skipping is not allowed"
        )
    return {"model": config["model"], "source": source}


def _case_payload(
    record: Dict[str, Any], archive: str, archive_sha: str
) -> Dict[str, Any]:
    task_id = str(record["seed_id"])
    judge_cfg = record["judge"]
    return {
        "artifacts": judge_cfg["artifacts"],
        "rubrics": judge_cfg["llm_judge"]["rubrics"],
        "llm_judge_method": judge_cfg["llm_judge"].get("method"),
        "score_merge": judge_cfg["score_merge"],
        "grading_dir_sha256": record["grading_dir_sha256"],
        "grading_source": {
            "path": archive,
            "sha256": archive_sha,
            "member_prefix": f"tasks/{task_id}/tests",
        },
        "env_pin": dict(record["workspace"]),
        "gold": record["gold"],
        "timeout_sec": record["verifier_timeout_sec"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="WorkBuddy M4 judge calibration: measured single-round cost and latency for 5 real tasks"
    )
    ap.add_argument(
        "--seed-archive",
        default="/tmp/wb-real-source-freeze/workbuddy-seed.tar.gz",
        help="workbuddy-seed.tar.gz (single top-level directory)",
    )
    ap.add_argument(
        "--work-dir", default=None,
        help="Working directory (synthetic references, cache, report; should be outside the repository)",
    )
    ap.add_argument(
        "--input-price-per-mtok", type=float, default=0.0,
        help="Judge model input price (USD per million tokens); 0 = report token counts only",
    )
    ap.add_argument(
        "--output-price-per-mtok", type=float, default=0.0,
        help="Judge model output price (USD per million tokens); 0 = report token counts only",
    )
    ap.add_argument("--max-concurrency", type=int, default=4)
    args = ap.parse_args(argv)
    work = Path(
        args.work_dir or tempfile.mkdtemp(prefix="wb-m4-calibration-")
    )
    work.mkdir(parents=True, exist_ok=True)
    archive_str = str(Path(args.seed_archive).resolve())
    os.environ["WB_JUDGE_MAX_CONCURRENCY"] = str(args.max_concurrency)
    os.environ.setdefault(
        "WB_JUDGE_CACHE_DIR", str(work / "judge_cache")
    )
    backend = _bridge_judge_env()
    print(f"[calibration] judge backend: model={backend['model']} "
          f"source={backend['source']}", flush=True)

    adapter = WBBenchSeedAdapter.from_archive(
        archive_str, seed_source="real-hf-subset"
    )
    release = derive_release(
        "real-hf-subset", adapter.load_seed_tasks(), adapter.archive_sha256
    )
    records = {
        str(record["seed_id"]): record
        for record in adapter.load_native_task_records()
    }
    missing = [t for t in CALIBRATION_TASKS if t not in records]
    if missing:
        raise RuntimeError(f"calibration tasks absent from archive: {missing}")

    per_task: List[Dict[str, Any]] = []
    for task_id in CALIBRATION_TASKS:
        record = records[task_id]
        synthesis = synthesize_reference_artifacts(
            record, str(work / "reference")
        )
        if synthesis["outcome"] != "synthesized":
            raise RuntimeError(
                f"{task_id}: failed to synthesize reference artifacts: {synthesis.get('unavailable')}"
            )
        payload = _case_payload(
            record, archive_str, adapter.archive_sha256 or ""
        )
        manifest = {
            "schema_version": "benchmark-case-v1",
            "task_id": task_id,
            "family": "workbuddy",
            "private": {"verifier_spec": {
                "kind": "wb_composite",
                "generator_ref": f"wb_bench_native_ingest@{task_id}",
                "payload": payload,
            }},
        }
        manifest_path = Path(
            _write_json(work / "manifests" / f"{task_id}.json", manifest)
        )
        request = {
            "schemaVersion": 1,
            "taskId": task_id,
            "caseManifestPath": str(manifest_path),
            "candidateRoot": synthesis["root"],
            "phase": "evaluation",
            "publicItem": {"task_id": task_id},
        }
        started = time.monotonic()
        result = runner.score_request(request)
        wall = round(time.monotonic() - started, 3)
        diag = result.get("rule_diagnostics", {}).get("judge") or {}
        per_task.append(
            {
                "task_id": task_id,
                "rubric_count": len(payload["rubrics"]),
                "llm_judge_status": result["llm_judge_status"],
                "rule_pass_rate": result["native_metrics"]["rule_pass_rate"],
                "merged_score": result["score"],
                "passed": result["passed"],
                "llm_score": diag.get("llm_score"),
                "merge_source": diag.get("merge_source"),
                "llm_missing": diag.get("llm_missing"),
                "judge_errors": diag.get("judge_errors") or [],
                "call_count": diag.get("judge_call_count", 0),
                "cached_count": diag.get("judge_cached_count", 0),
                "prompt_tokens": diag.get("judge_prompt_tokens", 0),
                "completion_tokens": diag.get("judge_completion_tokens", 0),
                "judge_seconds": diag.get("judge_total_latency_seconds"),
                "wall_seconds": wall,
                "per_rubric": [
                    {
                        "rubric_id": item.get("rubric_id"),
                        "binary": item.get("binary"),
                        "cached": item.get("cached"),
                        "latency_seconds": item.get("latency_seconds"),
                        "reason": (item.get("reason") or "")[:200],
                    }
                    for item in diag.get("judge_rubric_results") or []
                ],
            }
        )
        print(
            f"[calibration] {task_id}: judge={result['llm_judge_status']} "
            f"llm={diag.get('llm_score')} merged={result['score']} "
            f"tokens={diag.get('judge_prompt_tokens', 0)}+"
            f"{diag.get('judge_completion_tokens', 0)} "
            f"judge_s={diag.get('judge_total_latency_seconds')}",
            flush=True,
        )
        if result["llm_judge_status"] == "judge_error":
            raise RuntimeError(
                f"{task_id}: judge_error (explicit failure): "
                f"{diag.get('judge_errors')}"
            )

    total_prompt = sum(item["prompt_tokens"] for item in per_task)
    total_completion = sum(item["completion_tokens"] for item in per_task)
    total_rubrics = sum(item["rubric_count"] for item in per_task)
    total_calls = sum(item["call_count"] for item in per_task)
    latencies = [
        item["judge_seconds"] for item in per_task
        if isinstance(item["judge_seconds"], (int, float))
    ]
    cost_usd = (
        total_prompt / 1e6 * args.input_price_per_mtok
        + total_completion / 1e6 * args.output_price_per_mtok
    )
    fifty_scale = {
        "rubrics_all_50": 504,
        "extrapolated_prompt_tokens": round(
            total_prompt / total_rubrics * 504
        ) if total_rubrics else None,
        "extrapolated_completion_tokens": round(
            total_completion / total_rubrics * 504
        ) if total_rubrics else None,
        "extrapolated_cost_usd": round(
            cost_usd / total_rubrics * 504, 4
        ) if total_rubrics and cost_usd else None,
        "extrapolated_wall_seconds_serial": (
            round(statistics.fmean(latencies) / total_calls * 504, 1)
            if latencies and total_calls else None
        ),
        "extrapolated_wall_seconds_at_concurrency": (
            round(statistics.fmean(latencies) / total_calls * 504
                  / args.max_concurrency, 1)
            if latencies and total_calls else None
        ),
        "note": (
            "Extrapolation uses the mean per rubric x 504 rubrics; concurrency scaling assumes"
            f" WB_JUDGE_MAX_CONCURRENCY={args.max_concurrency} linear estimation"
            " (the actual ceiling is constrained by gateway throughput)"
        ),
    }
    report = {
        "schema_version": "wb-m4-judge-calibration-v1",
        "acceptance": {
            "design": "RSI_workbuddy真实子集原生收题设计_v0.1 section 10 M4 / section 12 D1",
            "criteria": (
                "Measured single-round cost and latency for 5 real judge calls; the full budget is set after calibration is recorded"
            ),
        },
        "backend": {
            **backend,
            "temperature": wb_judge.backend_config()["temperature"],
            "thinking_level": wb_judge.backend_config()["thinking_level"],
            "prompt_version": wb_judge.JUDGE_PROMPT_VERSION,
            "max_concurrency": args.max_concurrency,
            "fingerprint": wb_judge.judge_fingerprint(
                wb_judge.backend_config()
            ),
        },
        "tasks": per_task,
        "totals": {
            "task_count": len(per_task),
            "rubric_count": total_rubrics,
            "judge_calls": total_calls,
            "cached_calls": sum(item["cached_count"] for item in per_task),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "judge_seconds_mean": (
                round(statistics.fmean(latencies), 3) if latencies else None
            ),
            "judge_seconds_max": (
                round(max(latencies), 3) if latencies else None
            ),
            "cost_usd": round(cost_usd, 6),
            "price_assumption": {
                "input_usd_per_mtok": args.input_price_per_mtok,
                "output_usd_per_mtok": args.output_price_per_mtok,
            },
        },
        "fifty_task_extrapolation": fifty_scale,
        "cache_dir": os.environ["WB_JUDGE_CACHE_DIR"],
        "source_release": release,
        "work_dir": str(work),
    }
    report_path = Path(
        _write_json(work / "m4_judge_calibration_report.json", report)
    )
    print(json.dumps(
        {
            "calibration_report": str(report_path),
            "task_count": len(per_task),
            "rubric_count": total_rubrics,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "judge_seconds_mean": report["totals"]["judge_seconds_mean"],
            "cost_usd": report["totals"]["cost_usd"],
            "fifty_task_extrapolation": fifty_scale,
        },
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(json.dumps({"error": str(error), "exit_code": 2},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
