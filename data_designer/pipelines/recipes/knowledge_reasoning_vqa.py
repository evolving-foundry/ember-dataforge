from __future__ import annotations

from pathlib import Path

import ray

from base.dataset import MultimodalDataset
from data_designer.generation.recipes.context import BaseSyntheticRecipeExecutor
from data_designer.storage.reasonvqa import write_reasonvqa_exports
from data_designer.generation.algorithms.builtin.reasonvqa_paths import (
    generate_reasonvqa_rows_from_wikidata_paths,
)


class KnowledgeReasoningVqaRecipeExecutor(BaseSyntheticRecipeExecutor):
    """Dataset-configurable ReasoningVQA synthesis recipe."""

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        config = self.section_config("knowledge_reasoning_vqa")
        rows = dataset.dataset.take_all()
        generated_rows = generate_reasonvqa_rows_from_wikidata_paths(
            path_records=[dict(row) for row in rows],
            config=config,
            recipe=self.recipe,
            generator_prompt_version=self.generator_prompt_version,
            schema_version=self.schema_version,
            params=self.params,
        )

        if bool(config.get("export_enabled", True)):
            export_dir = config.get("export_dir")
            if not export_dir:
                raise ValueError(
                    "recipe_config.export_dir is required when export is enabled."
                )
            write_reasonvqa_exports(
                rows=generated_rows,
                export_dir=Path(export_dir),
                split=str(config.get("split") or "train"),
                categories=list(config.get("categories") or []),
            )

        return MultimodalDataset(
            ray.data.from_items(generated_rows),
            modalities=dataset.modalities,
            validation_config=dataset.validation_config,
            **dataset.kwargs,
        )


__all__ = [
    "KnowledgeReasoningVqaRecipeExecutor",
]
