from __future__ import annotations

from typing import Any

from base.dataset import MultimodalDataset
from curator.operations.augmentation.video_consistency_revision_operation import (
    VideoConsistencyRevisionOperation,
)
from curator.operations.augmentation.video_multigranularity_description_operation import (
    VideoMultigranularityDescriptionOperation,
)
from curator.operations.augmentation.video_partition_operation import VideoPartitionOperation
from curator.operations.augmentation.video_qae_materialization_operation import (
    VideoQaeMaterializationOperation,
)
from curator.operations.augmentation.video_sampling_operation import VideoSamplingOperation
from curator.operations.augmentation.video_scaffold_construction_operation import (
    VideoScaffoldConstructionOperation,
)
from curator.operations.augmentation.video_temporal_structure_operation import (
    VideoTemporalStructureOperation,
)
from curator.operations.filtering.video_quality_screening_operation import VideoQualityScreeningOperation
from data_designer.integrations.curator.recipe_parameters import (
    COMMON_LLM_PARAM_KEYS,
    LOCAL_VLM_PARAM_KEYS,
    PRE_TEMPORAL_VIDEO_DESCRIPTION_MODES,
    QUESTION_PLAN_PARAM_KEYS,
    VIDEO_DESCRIPTION_MODES,
    VIDEO_DESCRIPTION_PARAM_KEYS,
)
from data_designer.generation.recipes.context import RecipeExecutor


class VideoTemporalQaRecipeExecutor(RecipeExecutor):
    """video_temporal_qa recipe executor."""

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        materialization_params = self._video_step_params(
            "materialization_params",
            fallback_key="materialization",
        )
        revision_params = self._video_step_params("revision_params", fallback_key="revision")
        quality_params = self._video_step_params("quality_params", fallback_key="quality")

        if self.step_enabled("raw_video_prepare", default=True):
            dataset = self._prepare_raw_video_temporal_qa_inputs(
                dataset=dataset,
                partition_cls=VideoPartitionOperation,
                sampling_cls=VideoSamplingOperation,
                temporal_cls=VideoTemporalStructureOperation,
                description_cls=VideoMultigranularityDescriptionOperation,
                scaffold_cls=VideoScaffoldConstructionOperation,
            )
        dataset = self._validate_video_temporal_qa_preconditions(dataset)

        dataset = VideoQaeMaterializationOperation(materialization_params).execute(dataset)

        if self.step_enabled("revision", default=True):
            dataset = VideoConsistencyRevisionOperation(revision_params).execute(dataset)
        if self.step_enabled("quality", default=True):
            dataset = VideoQualityScreeningOperation(quality_params).execute(dataset)

        return self.annotate_dataset(
            dataset,
            output_keys=["qae_triplets", "quality_records"],
            modality=["video", "text"],
            generator_model=self.resolve_generator_model(materialization_params),
        )

    def _video_step_params(self, primary_key: str, fallback_key: str) -> dict[str, Any]:
        raw = self.recipe_config.get(primary_key)
        if raw is None:
            raw = self.recipe_config.get(fallback_key)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"recipe_config.{primary_key} must be a dict.")
        params = dict(raw)
        for key in COMMON_LLM_PARAM_KEYS + LOCAL_VLM_PARAM_KEYS + QUESTION_PLAN_PARAM_KEYS:
            if key in self.recipe_config and key not in params:
                params[key] = self.recipe_config[key]
            if key in self.params and key not in params:
                params[key] = self.params[key]
        return params

    def _prepare_raw_video_temporal_qa_inputs(
        self,
        dataset: MultimodalDataset,
        partition_cls,
        sampling_cls,
        temporal_cls,
        description_cls,
        scaffold_cls,
    ) -> MultimodalDataset:
        raw_config = self.section_config("raw_video_prepare")
        video_key = raw_config.get("video_key") or self.params.get("video_key") or "video_path"
        partition_params = self._raw_video_step_params(raw_config, "partition_params")
        sampling_params = self._raw_video_step_params(raw_config, "sampling_params")
        temporal_params = self._raw_video_step_params(raw_config, "temporal_params")
        description_enabled = self._raw_video_description_enabled(raw_config)
        description_params = self._raw_video_description_params(raw_config)
        scaffold_params = self._raw_video_step_params(raw_config, "scaffold_params")
        requested_description_modes = self._normalize_video_description_modes(
            description_params.get("description_modes")
            or description_params.get("description_mode")
            or list(VIDEO_DESCRIPTION_MODES)
        )
        description_params["description_modes"] = requested_description_modes

        partition_params.setdefault("video_key", video_key)
        partition_params.setdefault("partition_mode", "FIXED")
        partition_params.setdefault("chunk_duration_sec", raw_config.get("chunk_duration_sec", 5.0))
        partition_params.setdefault("min_segment_duration_sec", 0.1)
        sampling_params.setdefault("video_key", video_key)
        sampling_params.setdefault("sampling_mode", "CENTER")
        sampling_params.setdefault(
            "save_sample_frames",
            self._local_vlm_enabled() or description_enabled,
        )
        temporal_params.setdefault("temporal_mode", "EVENT_WINDOW")
        description_params.setdefault("output_key", "description_units")
        description_params.setdefault("result_key", "description_result")
        description_params.setdefault("append_to_existing", True)
        scaffold_params.setdefault("scaffold_mode", "NARRATION_CHUNK")

        partition_op = partition_cls(partition_params)
        sampling_op = sampling_cls(sampling_params)
        temporal_op = temporal_cls(temporal_params)
        description_op = description_cls(description_params) if description_enabled else None
        scaffold_op = scaffold_cls(scaffold_params)

        def prepare_row(row):
            row = dict(row)
            if row.get("scaffold_units"):
                return row
            video_path = row.get(video_key) or row.get("video_path")
            if not video_path:
                return row

            self._ensure_video_metadata(row=row, video_path=video_path)
            if not row.get("segments"):
                partition_result = partition_op.partition(
                    video_path=video_path,
                    video_duration_sec=row.get("video_duration_sec"),
                    semantic_embeddings=row.get("semantic_embeddings"),
                    shot_scores=row.get("shot_scores"),
                    shot_fps=row.get("shot_fps"),
                )
                row["segments"] = partition_result["segments"]
                row["partition_result"] = partition_result
                row["partition_metrics"] = partition_result["global_metrics"]
            if not row.get("samples"):
                sampling_result = sampling_op.sample(
                    video_path=video_path,
                    segments=row.get("segments") or [],
                    semantic_frame_features=row.get("semantic_frame_features"),
                    video_fps=row.get("video_fps") or row.get("fps") or row.get("shot_fps"),
                )
                row["samples"] = sampling_result["samples"]
                row["sampling_result"] = sampling_result
                row["sampling_metrics"] = sampling_result["global_metrics"]

            if description_op is not None:
                pre_temporal_modes = [
                    mode
                    for mode in requested_description_modes
                    if mode in PRE_TEMPORAL_VIDEO_DESCRIPTION_MODES
                ]
                missing_pre_modes = self._missing_video_description_modes(
                    row,
                    pre_temporal_modes,
                )
                if missing_pre_modes:
                    description_result = description_op.describe_modes(
                        segments=row.get("segments") or [],
                        samples=row.get("samples") or [],
                        temporal_structures=None,
                        description_modes=missing_pre_modes,
                        id_start=len(row.get("description_units") or []) + 1,
                    )
                    self._merge_video_description_result(row, description_result)

            if not row.get("temporal_structures"):
                temporal_result = temporal_op.extract(
                    segments=row.get("segments") or [],
                    samples=row.get("samples") or [],
                    description_units=row.get("description_units") or [],
                    text_signals=row.get("text_signals") or [],
                )
                row["temporal_structures"] = temporal_result["temporal_structures"]
                row["temporal_result"] = temporal_result
                row["temporal_mode"] = temporal_result["temporal_mode"]
                row["temporal_metrics"] = temporal_result["global_metrics"]

            if description_op is not None:
                post_temporal_modes = [
                    mode
                    for mode in requested_description_modes
                    if mode not in PRE_TEMPORAL_VIDEO_DESCRIPTION_MODES
                ]
                missing_post_modes = self._missing_video_description_modes(
                    row,
                    post_temporal_modes,
                )
                if missing_post_modes:
                    row["segments"] = description_op._segments_for_modes(
                        row.get("segments") or [],
                        missing_post_modes,
                    )
                    description_result = description_op.describe_modes(
                        segments=row.get("segments") or [],
                        samples=row.get("samples") or [],
                        temporal_structures=row.get("temporal_structures") or [],
                        description_modes=missing_post_modes,
                        id_start=len(row.get("description_units") or []) + 1,
                    )
                    self._merge_video_description_result(row, description_result)

            if not row.get("scaffold_units"):
                scaffold_result = scaffold_op.construct(
                    description_units=row.get("description_units") or [],
                    temporal_structures=row.get("temporal_structures") or [],
                    text_signals=row.get("text_signals") or [],
                    samples=row.get("samples") or [],
                )
                row["scaffold_units"] = scaffold_result["scaffold_units"]
                row["scaffold_result"] = scaffold_result
                row["scaffold_mode"] = scaffold_result["scaffold_mode"]
                row["scaffold_metrics"] = scaffold_result["global_metrics"]
            return row

        return dataset.map(prepare_row)

    def _raw_video_step_params(self, raw_config: dict[str, Any], key: str) -> dict[str, Any]:
        value = raw_config.get(key) or {}
        if not isinstance(value, dict):
            raise ValueError(f"recipe_config.raw_video_prepare.{key} must be a dict.")
        return dict(value)

    def _raw_video_description_enabled(self, raw_config: dict[str, Any]) -> bool:
        if "description_enabled" in raw_config:
            return bool(raw_config["description_enabled"])
        if "description_enabled" in self.recipe_config:
            return bool(self.recipe_config["description_enabled"])
        return self.step_enabled("description", default=True)

    def _raw_video_description_params(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in ("description_params", "description"):
            value = self.recipe_config.get(key)
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"recipe_config.{key} must be a dict.")
                params.update(value)

        raw_value = raw_config.get("description_params") or raw_config.get("description") or {}
        if not isinstance(raw_value, dict):
            raise ValueError("recipe_config.raw_video_prepare.description_params must be a dict.")
        params.update(raw_value)

        for key in VIDEO_DESCRIPTION_PARAM_KEYS:
            if key in self.recipe_config and key not in params:
                params[key] = self.recipe_config[key]
            if key in self.params and key not in params:
                params[key] = self.params[key]

        if "vlm_local_files_only" in params and "local_files_only" not in params:
            params["local_files_only"] = params["vlm_local_files_only"]
        if "vlm_device_preference" in params and "device_preference" not in params:
            params["device_preference"] = params["vlm_device_preference"]
        if "vlm_max_new_tokens" in params and "max_new_tokens" not in params:
            params["max_new_tokens"] = params["vlm_max_new_tokens"]

        params.setdefault("description_modes", list(VIDEO_DESCRIPTION_MODES))
        return params

    @staticmethod
    def _normalize_video_description_modes(raw_modes: Any) -> list[str]:
        if raw_modes is None:
            values = list(VIDEO_DESCRIPTION_MODES)
        elif isinstance(raw_modes, str):
            values = [part.strip() for part in raw_modes.split(",")]
        else:
            values = list(raw_modes)
        normalized = []
        supported = set(VIDEO_DESCRIPTION_MODES)
        for value in values:
            mode = str(value or "").strip().upper()
            if not mode:
                continue
            if mode not in supported:
                raise ValueError(f"Unsupported video description mode: {mode}")
            if mode not in normalized:
                normalized.append(mode)
        if not normalized:
            raise ValueError("video_temporal_qa description_modes must not be empty.")
        return normalized

    @staticmethod
    def _missing_video_description_modes(
        row: dict[str, Any],
        requested_modes: list[str],
    ) -> list[str]:
        existing_modes = {
            str(unit.get("description_mode") or "").strip().upper()
            for unit in row.get("description_units") or []
            if isinstance(unit, dict)
        }
        return [mode for mode in requested_modes if mode not in existing_modes]

    def _merge_video_description_result(
        self,
        row: dict[str, Any],
        description_result: dict[str, Any],
    ) -> None:
        existing_units = list(row.get("description_units") or [])
        generated_units = list(description_result.get("description_units") or [])
        row["description_units"] = existing_units + generated_units

        results = list(row.get("description_results") or [])
        results.extend(description_result.get("description_results") or [])
        row["description_results"] = results
        row["description_result"] = {
            "description_mode": "MULTI_GRANULARITY",
            "description_modes": [
                item["description_mode"]
                for item in results
                if isinstance(item, dict) and item.get("description_mode")
            ],
            "description_results": results,
            "global_metrics": {
                "total_description_units": len(row["description_units"]),
                "description_units_by_mode": self._count_description_units_by_mode(
                    row["description_units"]
                ),
            },
        }
        row["description_mode"] = "MULTI_GRANULARITY"
        row["description_modes"] = row["description_result"]["description_modes"]
        row["description_metrics"] = row["description_result"]["global_metrics"]

    @staticmethod
    def _count_description_units_by_mode(
        description_units: list[dict[str, Any]],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for unit in description_units or []:
            if not isinstance(unit, dict):
                continue
            mode = str(unit.get("description_mode") or "UNKNOWN").strip().upper()
            counts[mode] = counts.get(mode, 0) + 1
        return counts

    @staticmethod
    def _ensure_video_metadata(row: dict[str, Any], video_path: Any) -> None:
        if row.get("video_duration_sec") and row.get("video_fps"):
            return
        try:
            from curator.utils.lazy_imports import get_cv2_backend

            cv2_backend = get_cv2_backend()
            capture = cv2_backend.VideoCapture(str(video_path))
            if capture is None or not capture.isOpened():
                if capture is not None:
                    capture.release()
                return
            try:
                fps = float(capture.get(cv2_backend.CAP_PROP_FPS) or 0.0)
                frame_count = int(capture.get(cv2_backend.CAP_PROP_FRAME_COUNT) or 0)
            finally:
                capture.release()
        except Exception:
            return
        if fps > 0 and not row.get("video_fps"):
            row["video_fps"] = fps
        if fps > 0 and frame_count > 0 and not row.get("video_duration_sec"):
            row["video_duration_sec"] = frame_count / fps

    def _validate_video_temporal_qa_preconditions(
        self,
        dataset: MultimodalDataset,
    ) -> MultimodalDataset:
        def validate_row(row):
            row = dict(row)
            if row.get("scaffold_units"):
                return row
            raise ValueError(
                "video_temporal_qa requires scaffold_units or a readable raw video_path "
                "that can be prepared into scaffold_units."
            )

        return dataset.map(validate_row)

    def _local_vlm_enabled(self) -> bool:
        if bool(self.params.get("use_local_vlm_client", False)):
            return True
        if bool(self.recipe_config.get("use_local_vlm_client", False)):
            return True
        materialization = self.recipe_config.get("materialization_params")
        if isinstance(materialization, dict) and bool(materialization.get("use_local_vlm_client", False)):
            return True
        materialization = self.recipe_config.get("materialization")
        return isinstance(materialization, dict) and bool(materialization.get("use_local_vlm_client", False))
