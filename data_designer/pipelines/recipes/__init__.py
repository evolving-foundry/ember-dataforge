"""Recipe registry with lazy imports for optional heavy dependencies."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module

_RECIPE_EXPORTS = {
    "AgentTaskRecipeExecutor": (
        "data_designer.pipelines.recipes.agent_task",
        "AgentTaskRecipeExecutor",
    ),
    "BaseSyntheticRecipeExecutor": (
        "data_designer.generation.recipes.context",
        "BaseSyntheticRecipeExecutor",
    ),
    "DenseCaptionRecipeExecutor": (
        "data_designer.pipelines.recipes.dense_caption",
        "DenseCaptionRecipeExecutor",
    ),
    "KnowledgeReasoningVqaRecipeExecutor": (
        "data_designer.pipelines.recipes.knowledge_reasoning_vqa",
        "KnowledgeReasoningVqaRecipeExecutor",
    ),
    "VideoTemporalQaRecipeExecutor": (
        "data_designer.pipelines.recipes.video_temporal_qa",
        "VideoTemporalQaRecipeExecutor",
    ),
    "WB_TASK_SCHEMA_VERSION": (
        "data_designer.pipelines.recipes.workbuddy_task",
        "WB_TASK_SCHEMA_VERSION",
    ),
    "WorkbuddyTaskRecipeExecutor": (
        "data_designer.pipelines.recipes.workbuddy_task",
        "WorkbuddyTaskRecipeExecutor",
    ),
    "build_agent_task_dataset": (
        "data_designer.pipelines.recipes.agent_task",
        "build_agent_task_dataset",
    ),
    "build_workbuddy_profile_dataset": (
        "data_designer.pipelines.recipes.workbuddy_task",
        "build_workbuddy_profile_dataset",
    ),
}

_RECIPE_CLASSES = {
    "agent_task": (
        "data_designer.pipelines.recipes.agent_task",
        "AgentTaskRecipeExecutor",
    ),
    "dense_caption": (
        "data_designer.pipelines.recipes.dense_caption",
        "DenseCaptionRecipeExecutor",
    ),
    "knowledge_reasoning_vqa": (
        "data_designer.pipelines.recipes.knowledge_reasoning_vqa",
        "KnowledgeReasoningVqaRecipeExecutor",
    ),
    "video_temporal_qa": (
        "data_designer.pipelines.recipes.video_temporal_qa",
        "VideoTemporalQaRecipeExecutor",
    ),
    "workbuddy_task": (
        "data_designer.pipelines.recipes.workbuddy_task",
        "WorkbuddyTaskRecipeExecutor",
    ),
}


def __getattr__(name: str):
    target = _RECIPE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def get_recipe_executor_class(recipe: str) -> type:
    try:
        module_name, class_name = _RECIPE_CLASSES[recipe]
    except KeyError as exc:
        supported = ", ".join(sorted(_RECIPE_CLASSES))
        raise ValueError(
            f"Unsupported synthetic data recipe: {recipe}. Supported recipes: {supported}"
        ) from exc
    return getattr(import_module(module_name), class_name)


class _RecipeClassRegistry(Mapping[str, type]):
    """Backward-compatible mapping that resolves classes on access."""

    def __getitem__(self, recipe: str) -> type:
        return get_recipe_executor_class(recipe)

    def __iter__(self) -> Iterator[str]:
        return iter(_RECIPE_CLASSES)

    def __len__(self) -> int:
        return len(_RECIPE_CLASSES)


RECIPE_EXECUTOR_CLASSES = _RecipeClassRegistry()


__all__ = [
    "AgentTaskRecipeExecutor",
    "BaseSyntheticRecipeExecutor",
    "DenseCaptionRecipeExecutor",
    "KnowledgeReasoningVqaRecipeExecutor",
    "VideoTemporalQaRecipeExecutor",
    "WB_TASK_SCHEMA_VERSION",
    "WorkbuddyTaskRecipeExecutor",
    "build_agent_task_dataset",
    "build_workbuddy_profile_dataset",
    "get_recipe_executor_class",
]
