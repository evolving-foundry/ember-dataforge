"""Text dataset/profile export for the installed Gym eval_service contract."""
from __future__ import annotations

import json
from pathlib import Path

from data_designer.pipelines.benchmarks.reader import read_package
from data_designer.contracts.provenance import canonical_json
from data_designer.storage.writers import write_jsonl


def export_gym(manifest, output, *, digest=None, config_paths=None):
    package = read_package(manifest, expected_digest=digest, verify_resources=True)
    paths = [Path(path).resolve() for path in config_paths or []]
    if not paths or not all(path.is_file() for path in paths):
        raise ValueError("Gym export requires existing, fully configured host config_paths; no placeholder resource server is generated")
    rows = []
    for task in package["content"]["tasks"]:
        family = task["public"]["benchmark"]["family"]
        private = task["private"]["private"]
        if family not in {"reasoning_qa", "knowledge_qa", "health_qa"} or not isinstance(private.get("gold_deliverable"), str):
            raise ValueError(f"Gym text profile cannot express {family} artifact/verifier semantics")
        rows.append({"task_id": task["public"]["task_id"], "question": task["public"]["task_text"],
                     "expected_answer": private["gold_deliverable"], "reference_solution": private["gold_deliverable"],
                     "info": {"source_digest": package["digest"], "family": family}})
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "data.jsonl", rows)
    profile = {"profile_id": "dataforge", "benchmarks": {"dataforge": {
        "dataset_fpath": str(output / "data.jsonl"), "dataset_version": package["digest"],
        "task_id_field": "task_id", "config_paths": [str(path) for path in paths]}}}
    # JSON is also valid YAML for Gym's safe_load profile reader.
    (output / "profile.yaml").write_text(canonical_json(profile), encoding="utf-8")
    return profile
