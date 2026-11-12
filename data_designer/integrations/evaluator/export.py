from __future__ import annotations

import json
from pathlib import Path

from data_designer.pipelines.benchmarks.reader import read_package
from data_designer.contracts.provenance import canonical_json


def export_evaluator(manifest, output, *, digest=None, requirements=None, python_executable="python"):
    package = read_package(manifest, expected_digest=digest, verify_resources=True)
    requirements = requirements or {}
    allowed = {"attempts", "seed", "family_weights", "aggregation", "task_ids"}
    if set(requirements) - allowed:
        raise ValueError("unknown Evaluator export requirements")
    if requirements.get("attempts", 1) != 1:
        raise ValueError("Evaluator 0.1.0 has no repeated-attempt plan mapping")
    if requirements.get("seed") is not None:
        raise ValueError("Evaluator 0.1.0 sampling seed does not pin solver decoding")
    if requirements.get("family_weights") or requirements.get("aggregation", "micro") != "micro":
        raise ValueError("Evaluator 0.1.0 has no family-weighted/macro metric mapping")
    task_ids = requirements.get("task_ids")
    ids = {task["public"]["task_id"] for task in package["content"]["tasks"]}
    if task_ids is not None and (not isinstance(task_ids, list) or len(task_ids) != len(set(task_ids)) or not set(task_ids) <= ids):
        raise ValueError("task selection must contain unique locked task ids")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = {"benchmark": {"name": "dataforge", "config": {
        "manifest": package["manifest_uri"], "digest": package["digest"],
        "pythonExecutable": python_executable}},
        "run": {"taskIds": task_ids}}
    (output / "benchmark.json").write_text(canonical_json(config), encoding="utf-8")
    (output / "capabilities.json").write_text(canonical_json({
        "host": "ember-evaluator", "version": "0.1.0", "source_digest": package["digest"],
        "attempts": 1, "aggregation": "micro", "coverage_denominator": "explicit selected task ids",
        "usage_units": {"inputTokens": "tokens", "outputTokens": "tokens", "durationMs": "milliseconds"},
        "unsupported": ["repeated attempts", "solver seed", "family weights", "macro aggregation"]}), encoding="utf-8")
    return config
