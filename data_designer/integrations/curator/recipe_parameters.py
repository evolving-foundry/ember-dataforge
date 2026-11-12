from __future__ import annotations

COMMON_LLM_PARAM_KEYS = (
    "llm_config_path",
    "llm_base_url",
    "llm_api_key",
    "llm_model",
    "temperature",
    "use_llm_client",
    "require_llm_client",
)

LOCAL_VLM_PARAM_KEYS = (
    "use_local_vlm_client",
    "vlm_model_path",
    "local_model_roots",
    "vlm_local_files_only",
    "vlm_device_preference",
    "vlm_device_map",
    "vlm_torch_dtype",
    "vlm_min_pixels",
    "vlm_max_pixels",
    "vlm_max_new_tokens",
    "max_evidence_images",
    "ray_num_gpus",
    "ray_num_npus",
)

QUESTION_PLAN_PARAM_KEYS = (
    "question_plan_profile",
    "question_plans",
    "question_plan_config_path",
    "question_plan_config",
    "generic_question_prefixes",
)

DENSE_CAPTION_OPERATION_PARAM_KEYS = (
    "model_path",
    "model_type",
    "device_preference",
    "prompt_text",
    "max_new_tokens",
    "local_model_roots",
    "local_files_only",
    "fallback_image_keys",
)

VIDEO_DESCRIPTION_PARAM_KEYS = (
    "model_name",
    "model_path",
    "local_model_roots",
    "local_files_only",
    "max_new_tokens",
    "frame_path_key",
    "device_preference",
    "vlm_model_path",
    "vlm_local_files_only",
    "vlm_device_preference",
    "vlm_device_map",
    "vlm_torch_dtype",
    "vlm_min_pixels",
    "vlm_max_pixels",
    "vlm_max_new_tokens",
)

DEFAULT_DENSE_CAPTION_FALLBACK_IMAGE_KEYS = ("image_path", "image")

VIDEO_DESCRIPTION_MODES = (
    "FRAME_DESCRIPTION",
    "EVENT_DESCRIPTION",
    "SEGMENT_DESCRIPTION",
    "CONTEXT_DESCRIPTION",
)
PRE_TEMPORAL_VIDEO_DESCRIPTION_MODES = {"FRAME_DESCRIPTION", "SEGMENT_DESCRIPTION"}

