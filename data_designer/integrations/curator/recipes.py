"""Concrete Curator capabilities injected into otherwise host-independent algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RecipeContext:
    recipe: str
    params: Dict[str, Any]
    recipe_config: Dict[str, Any]
    schema_version: str = "synthetic-data-v1"
    generator_prompt_version: str = "v1"
    provenance_key: str = "provenance"
    quality_scores_key: str = "quality_scores"
    evidence_refs_key: str = "evidence_refs"
    audit_status_key: str = "audit_status"
    on_error: str = "raise"
    max_retries: int = 0
    caption_cache: dict = field(default_factory=dict)


class CuratorRecipeProvider:
    def __init__(self, providers=None):
        self.providers = providers or {}

    def execute(self, recipe, records, params):
        import ray
        from base.dataset import MultimodalDataset
        from data_designer.contracts.datasets import RecordEnvelope
        from data_designer.generation.algorithms.registry import load_symbol
        executor = load_symbol({
            "dense_caption": "data_designer.pipelines.recipes.dense_caption:DenseCaptionRecipeExecutor",
            "video_temporal_qa": "data_designer.pipelines.recipes.video_temporal_qa:VideoTemporalQaRecipeExecutor",
        }[recipe])
        config = dict(params)
        steps = config.pop("steps", {})
        captioner_name = config.pop("captioner_provider", None)
        if captioner_name:
            try:
                config["captioner"] = self.providers[captioner_name]
            except KeyError as error:
                raise ValueError(f"captioner provider not configured: {captioner_name}") from error
        context = RecipeContext(recipe, config, steps,
                                schema_version=config.get("annotation_schema_version", "synthetic-data-v1"),
                                generator_prompt_version=config.get("generator_prompt_version", "v1"),
                                on_error=config.get("on_error", "raise"))
        rows = [{**record.payload, "_dataforge_parent_id": record.id} for record in records]
        dataset = MultimodalDataset(ray.data.from_items(rows), modalities=["text", "image", "video"])
        result = executor(context).execute(dataset)
        for index, row in enumerate(result.dataset.iter_rows()):
            row = dict(row)
            parent = row.pop("_dataforge_parent_id", None)
            if parent is None:
                raise ValueError("Curator recipe lost parent identity")
        # Failed rows must land with _synthetic_generation_error / auto_fail;
        # they must not be dropped silently (error visibility is a data-correctness red line).
            yield RecordEnvelope(f"{recipe}:{parent}:{index}", row, f"{recipe}-v1", (parent,))
