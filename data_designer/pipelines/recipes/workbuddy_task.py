"""workbuddy_task recipe: the fifth synthesis recipe--an evaluation task production line with real expansion capability.

Parallel to ``agent_task`` (pure normalization passthrough; its docstring explicitly says it does not expand); this recipe dispatches three production lines according to
``recipe_config.strategy`` (design §6):

  mutation       S1 parameterized mutation (``workbuddy_mutation.WorkbuddyMutationEngine``,
                 pure Python, optional LLM polishing, M1)
  taxonomy       S2 taxonomy-driven template generation (``workbuddy_taxonomy``, coverage-gap
                 filling, M2)
  failure_probe  S3 targeted failure-cluster probes (``workbuddy_failure_probe``, M2)

Release gates (§7.1/§8.2, carried on existing audit semantics with zero new concepts):
  - Hard rule: synthetic tasks without ``verifier_spec`` may not be released;
  - Trial-run dual validation: the verifier must pass on the constructed positive solution and fail on the empty/negative solution;
    any mismatch -> row-level ``audit_status=auto_fail`` (existing base.py semantics);
  - Seed contamination (§8.2-1): when ``contamination_seed_texts`` provides real seed text,
    rows whose n-gram similarity exceeds the threshold are marked ``derivative=true`` and quarantined by default;
  - Cross-round exclusion set (Gym §9.3): candidates that hit ``exclude_item_fingerprints`` (fingerprints released last round)
    are removed directly, with an e2e regression assertion.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from base.dataset import MultimodalDataset
from data_designer.verifiers import run_verifier
from data_designer.generation.recipes.context import BaseSyntheticRecipeExecutor
from data_designer.generation.algorithms.builtin.workbuddy_mutation import WorkbuddyMutationEngine

WB_TASK_SCHEMA_VERSION = "wb-task-v1"

_OUTPUT_FIELDS = (
    "task_id",
    "task_text",
    "tools_available",
    "success_criteria",
    "ground_truth_action_sequence",
    "scoring_rubric",
    "difficulty",
    "item_fingerprint",
    "verifier_spec",
)

STRATEGY_MUTATION = "mutation"
STRATEGY_TAXONOMY = "taxonomy"
STRATEGY_FAILURE_PROBE = "failure_probe"
STRATEGIES_IMPLEMENTED = (STRATEGY_MUTATION, STRATEGY_TAXONOMY,
                          STRATEGY_FAILURE_PROBE)


def _published_fingerprint_of_task(task) -> str:
    """Precompute item_fingerprint using fields isomorphic to the recipe release row (the key for exclusion-set checks)."""
    probe = {
        "seed_id": task.seed_id,
        "task_text": task.task_text,
        "tools_available": [],
        "success_criteria": task.success_criteria,
        "ground_truth_action_sequence": [],
        "scoring_rubric":
            f"verifier={task.verifier_spec['kind']}; "
            f"generator_ref={task.verifier_spec['generator_ref']}",
        "difficulty": task.difficulty,
        "verifier_spec": task.verifier_spec,
    }
    return wb_item_fingerprint(probe)


def wb_item_fingerprint(item: Dict[str, Any]) -> str:
    """wb-task-v1 item fingerprint: agent_task content + the full verifier_spec participate in the hash."""
    fingerprint_input = {
        "seed_id": item["seed_id"],
        "task_text": item["task_text"],
        "tools_available": item["tools_available"],
        "success_criteria": item["success_criteria"],
        "ground_truth_action_sequence": item["ground_truth_action_sequence"],
        "scoring_rubric": item["scoring_rubric"],
        "difficulty": item["difficulty"],
        "verifier_spec": item["verifier_spec"],
    }
    return hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _llm_polish(task_text: str, params: Dict[str, Any]) -> str:
    """Optional polishing: an OpenAI-compatible endpoint configured by COMMON_LLM_PARAM_KEYS.

    Failures are raised explicitly (no silent fallback)--the caller handles them with on_error/mark_failed semantics.
    """
    base_url = params.get("llm_base_url")
    api_key = params.get("llm_api_key")
    model = params.get("llm_model")
    if not (base_url and api_key and model):
        raise ValueError(
            "llm_polish_enabled=true but llm_base_url/llm_api_key/llm_model is missing")
    from curator.utils.llm_api_config import (
        OpenAICompatibleChatClient,
        OpenAICompatibleConfig,
    )
    client = OpenAICompatibleChatClient(OpenAICompatibleConfig(
        base_url=str(base_url), api_key=str(api_key), model=str(model)))
    resp = client.chat(messages=[
        {"role": "system", "content":
            "You are an evaluation-task rewriter. Keep every number, constraint, definition, and output contract unchanged while "
            "polishing only the natural-language expression of the task description. Output the rewritten task in full."},
        {"role": "user", "content": task_text},
    ], temperature=float(params.get("temperature") or 0.2))
    polished = str(resp.choices[0].message.content or "").strip()
    if not polished:
        raise ValueError("llm_polish returned empty text")
    return polished


class WorkbuddyTaskRecipeExecutor(BaseSyntheticRecipeExecutor):
    """workbuddy_task recipe executor: seed profiles -> expansion candidates (with verifier_spec)."""

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        if dataset is None:
            raise ValueError("WorkbuddyTaskRecipeExecutor requires an input dataset")

        cfg = self.recipe_config or {}
        strategy = str(cfg.get("strategy") or STRATEGY_MUTATION).lower()
        if strategy not in STRATEGIES_IMPLEMENTED:
            raise NotImplementedError(
                f"workbuddy_task strategy={strategy!r} is not implemented"
                f" (currently implemented: {STRATEGIES_IMPLEMENTED})")

        profiles = [
            dict(row) for row in dataset.dataset.take_all() if isinstance(row, dict)
        ]
    # Inline expand hints are not profile fields
        per_row_expand = any("expand_count" in p for p in profiles)
        profiles = [
            {k: v for k, v in p.items() if k != "expand_count"} for p in profiles
        ]

        exclude_fps = {str(fp) for fp in (cfg.get("exclude_item_fingerprints") or [])}

        if strategy == STRATEGY_TAXONOMY:
            from data_designer.generation.algorithms.builtin.workbuddy_taxonomy import WorkbuddyTaxonomyEngine
            seed_grid: Dict[str, Dict[str, int]] = {}
            for p in profiles:
                cat = str(p.get("category") or "unspecified")
                diff = str(p.get("difficulty") or "unspecified")
                seed_grid.setdefault(cat, {})
                seed_grid[cat][diff] = seed_grid[cat].get(diff, 0) + 1
            engine = WorkbuddyTaxonomyEngine(seed=int(cfg.get("seed") or 0))
            tasks = engine.generate(
                seed_grid, int(cfg.get("expand_count") or 0) or 10)
        elif strategy == STRATEGY_FAILURE_PROBE:
            from data_designer.generation.algorithms.builtin.workbuddy_failure_probe import WorkbuddyFailureProbeEngine
            clusters = cfg.get("failure_clusters")
            if not isinstance(clusters, list) or not clusters:
                raise ValueError(
                    "strategy=failure_probe requires recipe_config.failure_clusters"
                    " ([{cluster_id, error_category, count?}]; fail explicitly rather than silently)")
            engine = WorkbuddyFailureProbeEngine(seed=int(cfg.get("seed") or 0))
            tasks = engine.generate(
                clusters, per_cluster=int(cfg.get("per_cluster") or 1))
        else:
            engine = WorkbuddyMutationEngine(seed=int(cfg.get("seed") or 0))
            if per_row_expand:
                tasks = []
                for idx, profile in enumerate(profiles):
                    tasks.extend(engine.generate(
                        [profile], int(profile.get("expand_count") or 1)))
            else:
                expand_count = int(cfg.get("expand_count") or 0) or len(profiles) or 10
                tasks = engine.generate(profiles or None, expand_count)

        polish_enabled = self.step_enabled("llm_polish", False)
        keep_rows: List[Dict[str, Any]] = []
        for task in tasks:
            if _published_fingerprint_of_task(task) in exclude_fps:
                continue  # Cross-round exclusion set (Gym §9.3): last round's fingerprints do not enter this round's item set
            task_text = task.task_text
            if polish_enabled:
                task_text = _llm_polish(task_text, self.params)

            # Trial-run dual validation (§8.2): positive solutions must pass and negative examples must fail
            gold_run = run_verifier(task.verifier_spec, task.gold_solution)
            negative_run = run_verifier(task.verifier_spec, task.negative_solution)
            trial_ok = bool(gold_run["passed"]) and not bool(negative_run["passed"])

            item = {
                "seed_id": task.seed_id,
                "sample_id": task.seed_id,
                "task_text": task_text,
                "tools_available": [],
                "success_criteria": task.success_criteria,
                "ground_truth_action_sequence": [],
                "scoring_rubric":
                    f"verifier={task.verifier_spec['kind']}; "
                    f"generator_ref={task.verifier_spec['generator_ref']}",
                "difficulty": task.difficulty,
                "verifier_spec": task.verifier_spec,
                "provenance_extra": dict(
                    task.provenance_extra,
                    family=task.family,
                    trial_run={
                        "gold_passed": bool(gold_run["passed"]),
                        "negative_failed": not bool(negative_run["passed"]),
                        "gold_detail": gold_run["detail"],
                        "negative_detail": negative_run["detail"],
                    },
                ),
            }
            item["item_fingerprint"] = wb_item_fingerprint(item)
            item["task_id"] = f"ts-{item['item_fingerprint']}"
            if not trial_ok:
                # base.py audit_status() prefers the existing inline value -> auto_fail semantics are carried with zero changes
                item[self.audit_status_key] = "auto_fail"
            keep_rows.append(item)

        # Seed contamination detection (§8.2-1): recipe_config may optionally carry real seed text.
        contamination_seed_items = cfg.get("contamination_seed_texts") or []
        if contamination_seed_items:
            from data_designer.generation.algorithms.builtin.workbuddy_contamination import (
                DEFAULT_CONTAMINATION_THRESHOLD,
                contamination_report,
            )
            report = contamination_report(
                keep_rows, contamination_seed_items,
                threshold=float(cfg.get("contamination_threshold")
                                or DEFAULT_CONTAMINATION_THRESHOLD))
            flagged = {f["task_id"]: f for f in report["flagged"]}
            for item in keep_rows:
                flag = flagged.get(item["task_id"])
                if flag:
                    item["provenance_extra"] = dict(
                        item["provenance_extra"],
                        derivative=True,
                        contamination={
                            "max_similarity": flag["max_similarity"],
                            "matched_seed_id": flag["matched_seed_id"],
                        },
                    )
                    item[self.audit_status_key] = "auto_fail"  # Quarantined by default

        if not keep_rows:
            return dataset

        import ray
        regenerated_ds = MultimodalDataset(
            ray.data.from_items(keep_rows),
            modalities=["text"],
        )
        return self.annotate_dataset(
            regenerated_ds,
            output_keys=list(_OUTPUT_FIELDS),
            modality=["text"],
            generator_model=self.resolve_generator_model(self.recipe_config or {}),
        )


def build_workbuddy_profile_dataset(
    profiles: List[Dict[str, Any]],
) -> MultimodalDataset:
    """Seed profiles -> Ray dataset (the wb entry point for builder stage 1; empty table -> default profiles)."""
    from data_designer.generation.algorithms.builtin.workbuddy_mutation import DEFAULT_PROFILES

    rows = [dict(p) for p in (profiles or []) if isinstance(p, dict)] or \
        [dict(p) for p in DEFAULT_PROFILES]
    import ray
    return MultimodalDataset(
        ray.data.from_items(rows),
        modalities=["text"],
    )
