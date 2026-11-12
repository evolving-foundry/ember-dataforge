"""Workbuddy generation independent of Curator, taskset builders and host stores."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from data_designer.contracts.algorithms import AlgorithmResult
from data_designer.contracts.datasets import RecordEnvelope
from data_designer.contracts.provenance import content_digest
from .descriptors import WORKBUDDY
from .workbuddy_mutation import WorkbuddyMutationEngine


class WorkbuddyAlgorithm:
    def describe(self):
        return WORKBUDDY

    def run(self, request, context):
        from data_designer.verifiers import run_verifier
        cfg = request.config
        count = min(request.target_new_records if request.target_new_records is not None else cfg["expand_count"],
                    request.max_attempts)
        profiles = [{**r.payload, "seed_id": r.id} for r in request.records]
        strategy = cfg["strategy"]
        if strategy == "mutation":
            tasks = WorkbuddyMutationEngine(seed=context.seed).generate(profiles or None, count)
        elif strategy == "taxonomy":
            from .workbuddy_taxonomy import WorkbuddyTaxonomyEngine
            grid = {}
            for p in profiles:
                category, difficulty = str(p.get("category", "unspecified")), str(p.get("difficulty", "unspecified"))
                grid.setdefault(category, {})[difficulty] = grid.get(category, {}).get(difficulty, 0) + 1
            tasks = WorkbuddyTaxonomyEngine(seed=context.seed).generate(grid, count)
        else:
            from .workbuddy_failure_probe import WorkbuddyFailureProbeEngine
            clusters = cfg["failure_clusters"]
            if not clusters:
                raise ValueError("failure_probe requires failure_clusters")
            tasks = WorkbuddyFailureProbeEngine(seed=context.seed).generate(clusters, per_cluster=cfg["per_cluster"])[:count]
        records, rejected = [], Counter()
        for index, task in enumerate(tasks):
            context.check_cancelled()
            positive = run_verifier(task.verifier_spec, task.gold_solution)
            negative = run_verifier(task.verifier_spec, task.negative_solution)
            if positive.get("status", "completed") != "completed" or negative.get("status", "completed") != "completed":
                raise RuntimeError("workbuddy reference/negative verifier did not complete")
            if not positive["passed"] or negative["passed"]:
                rejected["reference_negative_check"] += 1
                continue
            item = {"seed_id": task.seed_id, "task_text": task.task_text,
                    "success_criteria": task.success_criteria, "difficulty": task.difficulty,
                    "category": task.category, "family": task.family,
                    "verifier_spec": task.verifier_spec, "provenance_extra": task.provenance_extra,
                    "reference_check": positive, "negative_check": negative}
            digest = content_digest(item)
            item["item_fingerprint"] = digest
            if digest in cfg["exclude_item_fingerprints"]:
                rejected["excluded_content"] += 1
                continue
            item_id = f"workbuddy:{index}:{task.seed_id}"
            item["task_id"] = item_id
            if cfg["contamination_seed_texts"]:
                from .workbuddy_contamination import contamination_report
                report = contamination_report([{**item, "task_id": item_id}], cfg["contamination_seed_texts"],
                                              threshold=cfg["contamination_threshold"])
                if report["flagged"]:
                    rejected["contamination"] += 1
                    continue
            source = task.provenance_extra.get("seed_task_id")
            ids = {r.id for r in request.records}
            if source in ids:
                parents = (source,)
            elif strategy == "taxonomy" and request.records:
                parents = tuple(r.id for r in request.records)
            else:
                parents = request.source_refs
            records.append(RecordEnvelope(item_id, item, WORKBUDDY.output_schema, parents))
        return AlgorithmResult(records, dict(rejected), {"strategy": strategy, "attempts": len(tasks)})
