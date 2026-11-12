from data_designer.config import object_schema
from data_designer.contracts.algorithms import AlgorithmDescriptor, AlgorithmSpec
from .rubric_criteria import CRITERIA_EXTRACT, CRITERIA_EXTRACT_ERROR, CRITERIA_GENERATE


AGENT_TASK = AlgorithmSpec(
    id="native.agent_task", implementation_version="1",
    config_schema=object_schema({}), output_schema="agent-task-v1",
    capabilities=("transform", "filter"), cardinality="filter",
    resources={"full_input_memory": False},
    upstream={"method": "DataForge deterministic seed task normalization"},
)

DAG_FUSION = AlgorithmSpec(
    id="deepfusion.dag_fusion", implementation_version="1",
    config_schema=object_schema({
        "strategy": {"enum": ["bridge", "chain", "both"], "default": "both"},
        "max_pair_occurrences": {"type": "integer", "minimum": 1, "default": 5},
        "sample_b_per_answer": {"type": "integer", "minimum": 1, "default": 5},
        "require_answer_terminal": {"type": "boolean", "default": True},
        "gold_graph_ids": {"type": "array", "items": {"type": "string"}, "default": []},
        "max_input_records": {"type": "integer", "minimum": 1, "default": 10000},
    }), input_schema="multihop-dag-v1", output_schema="multihop-fusion-v1",
    scope="dataset", capabilities=("expand", "quota"), cardinality="many-to-many",
    resources={"full_input_memory": True, "max_input_records": 10000, "dependencies": ["networkx"]},
    determinism="seeded",
    upstream={"method": "DeepFusion safe_bridge_union / strict_answer_to_start",
              "source": "DeepFusion/data_generation/step*_iter*_Fusion",
              "differences": "Deterministic seeded sampling; no model calls; N+1 hop convention",
              "revision": "workspace-port; upstream revision not pinned"},
)


_COMMON_RECIPE_PARAMS = {
    "annotation_schema_version": {"type": "string", "default": "synthetic-data-v1"},
    "generator_prompt_version": {"type": "string", "default": "v1"},
    "generator_model": {"type": "string"},
    "filter_versions": {"type": "array", "items": {"type": "string"}},
    "intended_split": {"type": "string"},
    "input_key": {"type": "string"}, "output_key": {"type": "string"},
    "on_error": {"enum": ["raise", "mark_failed"], "default": "raise"},
}

_DENSE_PARAMS = {
    **_COMMON_RECIPE_PARAMS,
    "captioner_provider": {"type": "string"},
    "caption_key": {"type": "string"}, "caption_text_key": {"type": "string"},
    "dense_caption_schema_version": {"type": "string"},
    "model_path": {"type": "string"}, "model_type": {"type": "string"},
    "device_preference": {"type": "string"}, "prompt_text": {"type": "string"},
    "max_new_tokens": {"type": "integer", "minimum": 1},
    "local_model_roots": {"type": "array", "items": {"type": "string"}},
    "local_files_only": {"type": "boolean"},
    "fallback_image_keys": {"type": "array", "items": {"type": "string"}},
    "steps": {"type": "object", "default": {}},
}

CURATOR_RECIPES = {
    name: AlgorithmSpec(id=f"native.{name}", implementation_version="1",
                        config_schema=object_schema(properties), scope="dataset",
                        output_schema=f"{name}-v1", capabilities=("transform",),
                        resources={"full_input_memory": True, "providers": ["curator.recipe"]},
                        determinism="external",
                        upstream={"method": f"DataForge {name}, Curator public operations"})
    for name, properties in (("dense_caption", _DENSE_PARAMS),
                             ("video_temporal_qa", {**_COMMON_RECIPE_PARAMS,
                              "steps": {"type": "object", "default": {}}}))
}

WORKBUDDY = AlgorithmSpec(
    id="native.workbuddy", implementation_version="1", scope="dataset",
    output_schema="wb-task-v1", capabilities=("expand", "quota"), cardinality="many-to-many",
    config_schema=object_schema({
        "strategy": {"enum": ["mutation", "taxonomy", "failure_probe"], "default": "mutation"},
        "expand_count": {"type": "integer", "minimum": 0, "default": 10},
        "failure_clusters": {"type": "array", "items": {"type": "object"}, "default": []},
        "per_cluster": {"type": "integer", "minimum": 1, "default": 1},
        "exclude_item_fingerprints": {"type": "array", "items": {"type": "string"}, "default": []},
        "contamination_seed_texts": {"type": "array", "default": []},
        "contamination_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": .8},
    }), resources={"full_input_memory": True}, determinism="seeded",
    upstream={"method": "DataForge workbuddy mutation, taxonomy and failure probes"})

REASONVQA = AlgorithmSpec(
    id="native.reasonvqa", implementation_version="1", scope="dataset",
    output_schema="reasonvqa-v1", capabilities=("transform",),
    config_schema=object_schema({
        "category_ids_by_property": {"type": "object", "default": {}},
        "categories": {"type": "array", "items": {"type": "object"}, "default": []},
        "difficulty_policy": {"type": "object", "default": {}},
        "question_templates": {"type": "object", "default": {}},
        "route_question_templates": {"type": "object", "default": {}},
        "filter_template_usable_paths": {"type": "boolean", "default": True},
        "image_dir": {"type": "string"}, "image_source": {"type": "string"},
        "export_dir": {"type": "string"},
        "export_enabled": {"type": "boolean", "default": True},
        "root_entity_domain": {"type": "string"}, "root_entity_type": {"type": "string"},
        "template_version": {"type": "string"}, "visual_anchor_template": {"type": "string"},
    }), resources={"full_input_memory": True},
    upstream={"method": "DataForge Wikidata-path ReasonVQA with typed distractors"})


TRAJECTORY_CLEAN = AlgorithmSpec(
    id="native.trajectory_clean", implementation_version="1", output_schema="trajectory-v1",
    config_schema=object_schema({
        "max_tool_chars": {"type": "integer", "minimum": 1, "default": 300},
        "require_core_meta": {"type": "boolean", "default": True},
    }), upstream={"method": "Reasoning stripping, PII/secret scrubbing, receipt truncation and degeneration evidence"})

TRAJECTORY_LESSON = AlgorithmSpec(
    id="native.trajectory_lesson", implementation_version="1", output_schema="trajectory-lesson-v1",
    config_schema=object_schema({"max_quote_chars": {"type": "integer", "minimum": 1, "default": 120}}),
    upstream={"method": "Deterministic, source-grounded failure lesson extraction"})


def descriptors():
    return (
        AlgorithmDescriptor(AGENT_TASK, "data_designer.generation.algorithms.builtin.agent_task:AgentTaskAlgorithm"),
        AlgorithmDescriptor(DAG_FUSION, "data_designer.generation.algorithms.builtin.multihop:DagFusionAlgorithm"),
        AlgorithmDescriptor(CURATOR_RECIPES["dense_caption"], "data_designer.generation.algorithms.builtin.curator_recipe:DenseCaptionAlgorithm"),
        AlgorithmDescriptor(CURATOR_RECIPES["video_temporal_qa"], "data_designer.generation.algorithms.builtin.curator_recipe:VideoTemporalQaAlgorithm"),
        AlgorithmDescriptor(WORKBUDDY, "data_designer.generation.algorithms.builtin.workbuddy:WorkbuddyAlgorithm"),
        AlgorithmDescriptor(REASONVQA, "data_designer.generation.algorithms.builtin.reasonvqa:ReasonVqaAlgorithm"),
        AlgorithmDescriptor(TRAJECTORY_CLEAN, "data_designer.generation.algorithms.builtin.trajectory:TrajectoryCleanAlgorithm"),
        AlgorithmDescriptor(TRAJECTORY_LESSON, "data_designer.generation.algorithms.builtin.trajectory:TrajectoryLessonAlgorithm"),
        AlgorithmDescriptor(CRITERIA_EXTRACT, "data_designer.generation.algorithms.builtin.rubric_criteria:CriteriaExtractAlgorithm"),
        AlgorithmDescriptor(CRITERIA_EXTRACT_ERROR, "data_designer.generation.algorithms.builtin.rubric_criteria:CriteriaExtractErrorAlgorithm"),
        AlgorithmDescriptor(CRITERIA_GENERATE, "data_designer.generation.algorithms.builtin.rubric_criteria:CriteriaGenerateAlgorithm"),
    )
