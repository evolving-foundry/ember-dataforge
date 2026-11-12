from typing import Any, Dict, Optional

from .video_augmentation_operator import VideoAugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.video_ops_utils import (
    compute_video_semantic_embeddings,
    compute_video_shot_scores,
    cosine_distance,
    sec_to_frame,
    _resolve_torch_device,
)


SEMANTIC_MODEL_LIST = {
    "cosmos_embed1_224p": "nvidia/Cosmos-Embed1-224p",
}

SHOT_MODEL_LIST = {
    "transnetv2": "magnusdtd/TransNetV2",
}


class VideoPartitionOperation(VideoAugmentationOperation):
    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Segment videos by fixed windows, semantic changes, or shot changes",
        "params": {
            "video_key": {
                "type": "string",
                "required": False,
                "default": "video_path",
                "description": "Input video path field name"
            },
            "partition_mode": {
                "type": "string",
                "required": False,
                "default": "FIXED",
                "description": "Segmentation mode: FIXED, SEMANTIC, SHOT"
            },
            "chunk_duration_sec": {
                "type": "number",
                "required": False,
                "default": 5.0,
                "description": "Window length for fixed-window segmentation (seconds)"
            },
            "video_duration_sec": {
                "type": "number",
                "required": False,
                "default": None,
                "description": "Total video duration for fixed-window segmentation (seconds)"
            },
            "semantic_model_name": {
                "type": "string",
                "required": False,
                "default": "cosmos_embed1_224p",
                "description": "Embedding model name used for semantic segmentation"
            },
            "semantic_distance_threshold": {
                "type": "number",
                "required": False,
                "default": 0.35,
                "description": "Semantic segmentation boundary threshold"
            },
            "semantic_window_sec": {
                "type": "number",
                "required": False,
                "default": 4.0,
                "description": "Sliding-window duration for semantic embeddings (seconds)"
            },
            "shot_model_name": {
                "type": "string",
                "required": False,
                "default": "transnetv2",
                "description": "Model name used for shot segmentation"
            },
            "shot_threshold": {
                "type": "number",
                "required": False,
                "default": 0.55,
                "description": "Shot-change threshold"
            },
            "shot_fps": {
                "type": "number",
                "required": False,
                "default": 1.0,
                "description": "Sampling rate corresponding to shot-segmentation scores"
            },
            "min_segment_duration_sec": {
                "type": "number",
                "required": False,
                "default": 2.0,
                "description": "Minimum segment duration (seconds)"
            },
            "device_preference": {
                "type": "string",
                "required": False,
                "default": "auto",
                "description": "Device selection for SEMANTIC mode: auto (prefers NPU > CUDA > CPU), npu, cuda, cpu"
            },
            "semantic_model_path": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Override the semantic embedding model path; resolved from semantic_model_name when empty"
            }
        },
        "example": {
            "video_key": "video_path",
            "partition_mode": "FIXED",
            "chunk_duration_sec": 5.0,
            "video_duration_sec": 12.0,
            "min_segment_duration_sec": 2.0
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        partition_mode = str(self.params.get("partition_mode", "FIXED")).upper()
        semantic_model_name = self.params.get("semantic_model_name", "cosmos_embed1_224p")
        shot_model_name = self.params.get("shot_model_name", "transnetv2")

        if partition_mode not in {"FIXED", "SEMANTIC", "SHOT"}:
            raise ValueError("Unsupported partition mode")
        if semantic_model_name not in SEMANTIC_MODEL_LIST:
            raise ValueError(f"Unsupported semantic model name: {semantic_model_name}")
        if shot_model_name not in SHOT_MODEL_LIST:
            raise ValueError(f"Unsupported shot model name: {shot_model_name}")

        self.partition_mode = partition_mode
        self.chunk_duration_sec = float(self.params.get("chunk_duration_sec", 5.0))
        self.video_duration_sec = self.params.get("video_duration_sec")
        self.semantic_model_name = semantic_model_name
        self.semantic_model_path = SEMANTIC_MODEL_LIST[semantic_model_name]
        self.semantic_distance_threshold = float(self.params.get("semantic_distance_threshold", 0.35))
        self.shot_model_name = shot_model_name
        self.shot_model_path = SHOT_MODEL_LIST[shot_model_name]
        self.shot_threshold = float(self.params.get("shot_threshold", 0.55))
        self.shot_fps = float(self.params.get("shot_fps", 1.0))
        self.semantic_window_sec = float(self.params.get("semantic_window_sec", 4.0))
        self.min_segment_duration_sec = float(self.params.get("min_segment_duration_sec", 2.0))
        self.video_key = self.params.get("video_key", "video_path")
        self.device_preference = self.params.get("device_preference", "auto")
        # Allow the user to override the semantic model path
        override_model_path = self.params.get("semantic_model_path")
        if override_model_path:
            self.semantic_model_path = override_model_path

    def _resolve_device(self) -> str:
        """Resolve the actual device based on device_preference"""
        pref = self.device_preference
        if pref == "auto":
            return _resolve_torch_device(None)
        return _resolve_torch_device(pref)

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        def partition_row(row):
            video_path = row.get(self.video_key)
            if not video_path:
                raise ValueError("Missing video path.")

            semantic_embeddings = row.get("semantic_embeddings")
            shot_scores = row.get("shot_scores")
            shot_fps = row.get("shot_fps", self.shot_fps)

            if self.partition_mode == "SEMANTIC" and semantic_embeddings is None:
                device = self._resolve_device()
                semantic_embeddings = compute_video_semantic_embeddings(
                    video_path=video_path,
                    model_repo_or_path=self.semantic_model_path,
                    window_sec=self.semantic_window_sec,
                    device=device,
                )

            if self.partition_mode == "SHOT" and shot_scores is None:
                shot_outputs = compute_video_shot_scores(
                    video_path=video_path,
                    model_repo_or_path=self.shot_model_path,
                )
                shot_scores = shot_outputs["shot_scores"]
                shot_fps = shot_outputs.get("shot_fps", shot_fps)

            partition_result = self.partition(
                video_path=video_path,
                video_duration_sec=row.get("video_duration_sec", self.video_duration_sec),
                semantic_embeddings=semantic_embeddings,
                shot_scores=shot_scores,
                shot_fps=shot_fps,
            )

            row["segments"] = partition_result["segments"]
            row["partition_result"] = partition_result
            row["partition_metrics"] = partition_result["global_metrics"]
            row["shot_fps"] = shot_fps
            return row

        return dataset.map(partition_row)

    def partition(self, video_path, video_duration_sec=None, semantic_embeddings=None, shot_scores=None, shot_fps=None):
        if self.partition_mode == "FIXED":
            boundaries, partition_details = self._partition_fixed(video_duration_sec)
            boundary_type = "FIXED_WINDOW"
        elif self.partition_mode == "SEMANTIC":
            if semantic_embeddings is None:
                semantic_embeddings = compute_video_semantic_embeddings(
                    video_path=video_path,
                    model_repo_or_path=self.semantic_model_path,
                    window_sec=self.semantic_window_sec,
                    device=self._resolve_device(),
                )
            boundaries, partition_details = self._partition_semantic(semantic_embeddings)
            boundary_type = "SEMANTIC_SHIFT"
        elif self.partition_mode == "SHOT":
            boundaries, partition_details = self._partition_shot(shot_scores, shot_fps=shot_fps)
            boundary_type = "SHOT_CHANGE"
        else:
            raise ValueError(f"Unsupported partition mode: {self.partition_mode}")

        segments = self._build_segments(boundaries, boundary_type, shot_fps=shot_fps)

        return {
            "video_path": video_path,
            "partition_mode": self.partition_mode,
            "global_metrics": {
                "total_segments": len(segments),
            },
            "partition_details": partition_details,
            "segments": segments,
        }

    def _partition_fixed(self, video_duration_sec=None):
        if self.chunk_duration_sec <= 0:
            raise ValueError("chunk_duration_sec must be positive.")
        if video_duration_sec is None:
            raise ValueError("FIXED mode requires 'video_duration_sec'.")

        total_duration = float(video_duration_sec)
        if total_duration <= 0:
            raise ValueError("video_duration_sec must be positive.")

        boundaries = [0.0]
        cursor = 0.0
        while cursor < total_duration:
            cursor = min(total_duration, cursor + self.chunk_duration_sec)
            rounded_cursor = round(cursor, 3)
            if rounded_cursor != boundaries[-1]:
                boundaries.append(rounded_cursor)

        return boundaries, {
            "chunk_duration_sec": self.chunk_duration_sec,
            "video_duration_sec": total_duration,
        }

    def _partition_semantic(self, semantic_embeddings=None):
        if not semantic_embeddings:
            raise ValueError("SEMANTIC mode requires 'semantic_embeddings'.")

        if len(semantic_embeddings) == 1:
            return [0.0, round(float(semantic_embeddings[0]["end_sec"]), 3)], {
                "semantic_model": self.semantic_model_path,
                "window_count": 1,
                "distance_scores": [],
            }

        boundaries = [0.0]
        distance_scores = []

        for index in range(1, len(semantic_embeddings)):
            previous_window = semantic_embeddings[index - 1]
            current_window = semantic_embeddings[index]
            boundary_sec = float(current_window["start_sec"])
            current_distance = cosine_distance(
                previous_window["embedding"],
                current_window["embedding"],
            )
            distance_scores.append(
                {
                    "boundary_sec": round(boundary_sec, 3),
                    "cosine_distance": round(current_distance, 4),
                }
            )

            if (
                current_distance >= self.semantic_distance_threshold
                and boundary_sec - boundaries[-1] >= self.min_segment_duration_sec
            ):
                boundaries.append(round(boundary_sec, 3))

        video_end = round(float(semantic_embeddings[-1].get("end_sec", semantic_embeddings[-1]["start_sec"])), 3)
        if boundaries[-1] != video_end:
            boundaries.append(video_end)

        return boundaries, {
            "semantic_model": self.semantic_model_path,
            "semantic_distance_threshold": self.semantic_distance_threshold,
            "window_count": len(semantic_embeddings),
            "distance_scores": distance_scores,
        }

    def _partition_shot(self, shot_scores=None, shot_fps=None):
        if shot_scores is None:
            raise ValueError("SHOT mode requires 'shot_scores'.")
        effective_shot_fps = float(self.shot_fps if shot_fps is None else shot_fps)
        if effective_shot_fps <= 0:
            raise ValueError("shot_fps must be positive.")

        boundaries = [0.0]
        detected_boundaries = []

        for frame_index, score in enumerate(shot_scores):
            timestamp_sec = frame_index / effective_shot_fps
            if score >= self.shot_threshold and timestamp_sec - boundaries[-1] >= self.min_segment_duration_sec:
                boundaries.append(round(timestamp_sec, 3))
                detected_boundaries.append(
                    {
                        "frame_index": int(frame_index),
                        "boundary_sec": round(timestamp_sec, 3),
                        "shot_score": round(float(score), 4),
                    }
                )

        video_end = round(len(shot_scores) / effective_shot_fps, 3)
        if boundaries[-1] != video_end:
            boundaries.append(video_end)

        return boundaries, {
            "shot_model": self.shot_model_path,
            "shot_threshold": self.shot_threshold,
            "score_count": int(len(shot_scores)),
            "detected_boundaries": detected_boundaries,
            "fps": effective_shot_fps,
        }

    def _build_segments(self, boundaries, boundary_type, shot_fps=None):
        effective_shot_fps = float(self.shot_fps if shot_fps is None else shot_fps)
        ordered_boundaries = sorted(set(float(boundary) for boundary in boundaries))
        if not ordered_boundaries or ordered_boundaries[0] != 0.0:
            ordered_boundaries.insert(0, 0.0)

        segments = []
        for index in range(len(ordered_boundaries) - 1):
            start_sec = round(ordered_boundaries[index], 3)
            end_sec = round(ordered_boundaries[index + 1], 3)
            segment_id = f"seg_{index + 1:04d}"

            segments.append(
                {
                    "id": segment_id,
                    "segment_id": segment_id,
                    "timestamp_range": {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                    },
                    "frame_range": {
                        "start_frame": sec_to_frame(start_sec, effective_shot_fps),
                        "end_frame": sec_to_frame(end_sec, effective_shot_fps),
                    },
                    "duration_sec": round(end_sec - start_sec, 3),
                    "partition_mode": self.partition_mode,
                    "boundary_type": "START" if index == 0 else boundary_type,
                    "status": "active",
                }
            )

        return segments
