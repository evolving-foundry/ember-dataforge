from typing import Any, Dict, Optional

import hashlib
import math
import os

from .video_augmentation_operator import VideoAugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_cv2_backend
from ...utils.video_ops_utils import (
    resolve_frame_count,
    cosine_distance,
    unique_preserve_order
)

SUPPORTED_VIDEO_SAMPLING_MODES = {
    "CENTER",
    "UNIFORM",
    "HEAD_TAIL",
    "STATE",
    "SEMANTIC_AWARE",
    "MOTION_AWARE",
}

SEMANTIC_MODEL_LIST = {
    "clip_vit_base_patch32": "openai/clip-vit-base-patch32",
}


class VideoSamplingOperation(VideoAugmentationOperation):
    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Sample videos based on existing video segment results",
        "params": {
            "video_key": {
                "type": "string",
                "required": False,
                "default": "video_path",
                "description": "Input video path field name"
            },
            "segments_key": {
                "type": "string",
                "required": False,
                "default": "segments",
                "description": "Input segment-result field name"
            },
            "sampling_mode": {
                "type": "string",
                "required": False,
                "default": "CENTER",
                "description": "Sampling mode: CENTER, UNIFORM, HEAD_TAIL, STATE, SEMANTIC_AWARE, MOTION_AWARE"
            },
            "num_frames": {
                "type": "integer",
                "required": False,
                "default": 4,
                "description": "Maximum sampled frames per segment"
            },
            "padding_frames": {
                "type": "integer",
                "required": False,
                "default": 2,
                "description": "Boundary inset frame count for STATE mode"
            },
            "target_fps": {
                "type": "number",
                "required": False,
                "default": None,
                "description": "Target sampling rate for UNIFORM mode"
            },
            "max_frames_per_segment": {
                "type": "integer",
                "required": False,
                "default": None,
                "description": "Maximum output frames per segment"
            },
            "semantic_model_name": {
                "type": "string",
                "required": False,
                "default": "clip_vit_base_patch32",
                "description": "Model name used for semantic sampling"
            },
            "semantic_min_distance_threshold": {
                "type": "number",
                "required": False,
                "default": 0.12,
                "description": "Minimum semantic difference threshold"
            },
            "save_sample_frames": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Write sampled frames to disk and update frame_path"
            },
            "frame_output_dir": {
                "type": "string",
                "required": False,
                "default": "artifacts/sample_frames",
                "description": "Sampled-frame output directory; overridable via SAMPLE_FRAME_OUTPUT_DIR"
            },
            "frame_path_key": {
                "type": "string",
                "required": False,
                "default": "frame_path",
                "description": "Frame path field name in sampling results"
            },
            "frame_image_format": {
                "type": "string",
                "required": False,
                "default": "jpg",
                "description": "Sampled-frame image format"
            },
            "overwrite_sample_frames": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Overwrite existing sampled frames"
            }
        },
        "example": {
            "video_key": "video_path",
            "segments_key": "segments",
            "sampling_mode": "CENTER",
            "num_frames": 4,
            "padding_frames": 2,
            "save_sample_frames": True
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        sampling_mode = str(self.params.get("sampling_mode", "CENTER")).upper()
        semantic_model_name = self.params.get("semantic_model_name", "clip_vit_base_patch32")

        if sampling_mode not in SUPPORTED_VIDEO_SAMPLING_MODES:
            raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
        if semantic_model_name not in SEMANTIC_MODEL_LIST:
            raise ValueError(f"Unsupported semantic model name: {semantic_model_name}")

        self.video_key = self.params.get("video_key", "video_path")
        self.segments_key = self.params.get("segments_key", "segments")
        self.sampling_mode = sampling_mode
        self.num_frames = int(self.params.get("num_frames", 4))
        self.padding_frames = int(self.params.get("padding_frames", 2))
        self.target_fps = self.params.get("target_fps")
        self.target_fps = None if self.target_fps is None else float(self.target_fps)
        self.max_frames_per_segment = self.params.get("max_frames_per_segment")
        self.max_frames_per_segment = None if self.max_frames_per_segment is None else int(self.max_frames_per_segment)
        self.semantic_model_name = semantic_model_name
        self.semantic_model_path = SEMANTIC_MODEL_LIST[semantic_model_name]
        self.semantic_min_distance_threshold = float(self.params.get("semantic_min_distance_threshold", 0.12))
        self.save_sample_frames = bool(self.params.get("save_sample_frames", False))
        self.frame_output_dir = self.params.get(
            "frame_output_dir",
            os.environ.get("SAMPLE_FRAME_OUTPUT_DIR", "artifacts/sample_frames"),
        )
        self.frame_path_key = self.params.get("frame_path_key", "frame_path")
        self.frame_image_format = str(self.params.get("frame_image_format", "jpg")).strip().lstrip(".") or "jpg"
        self.overwrite_sample_frames = bool(self.params.get("overwrite_sample_frames", False))

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        def sample_row(row):
            video_path = row.get(self.video_key)
            if not video_path:
                raise ValueError("Missing video path.")

            segments = row.get(self.segments_key)
            if not segments:
                raise ValueError("Missing segments.")

            sampling_result = self.sample(
                video_path=video_path,
                segments=segments,
                semantic_frame_features=row.get("semantic_frame_features"),
                video_fps=row.get("video_fps") or row.get("fps") or row.get("shot_fps"),
            )

            row["samples"] = sampling_result["samples"]
            row["sampling_result"] = sampling_result
            row["sampling_metrics"] = sampling_result["global_metrics"]
            return row

        return dataset.map(sample_row)

    def sample(self, video_path, segments, semantic_frame_features=None, video_fps=None):
        samples = []
        resolved_fps = self._resolve_video_fps(
            segments=segments,
            explicit_fps=video_fps,
            video_path=video_path,
            read_video_metadata=self.save_sample_frames,
        )

        if self.sampling_mode in {"SEMANTIC_AWARE", "MOTION_AWARE"} and semantic_frame_features is None:
            raise ValueError(f"{self.sampling_mode} mode requires 'semantic_frame_features'.")

        for segment in segments:
            frame_indices, sample_types = self._select_frame_indices(
                segment=segment,
                fps=resolved_fps,
                semantic_frame_features=semantic_frame_features,
            )
            segment_id = str(segment["segment_id"])

            for frame_index, sample_type in zip(frame_indices, sample_types):
                samples.append(
                    {
                        "segment_id": segment_id,
                        "timestamp_sec": round(int(frame_index) / resolved_fps, 3),
                        "frame_index": int(frame_index),
                        "sample_type": sample_type,
                    }
                )

        for sample_index, sample in enumerate(samples, start=1):
            sample_id = f"smp_{sample_index:06d}"
            sample["id"] = sample_id
            sample["sample_id"] = sample_id

        if self.save_sample_frames:
            self._save_sample_frames(video_path=video_path, samples=samples)

        return {
            "video_path": video_path,
            "sampling_mode": self.sampling_mode,
            "global_metrics": {
                "total_segments": len(segments),
                "total_samples": len(samples),
                "fps": resolved_fps,
                "sample_frames_saved": bool(self.save_sample_frames),
            },
            "samples": samples,
        }

    @staticmethod
    def _resolve_video_fps(segments, explicit_fps=None, video_path=None, read_video_metadata=False):
        if explicit_fps is not None:
            try:
                explicit_fps = float(explicit_fps)
            except (TypeError, ValueError):
                explicit_fps = 0.0
            if explicit_fps > 0:
                return explicit_fps

        for segment in segments or []:
            frame_range = segment.get("frame_range") or {}
            timestamp_range = segment.get("timestamp_range") or {}
            values = (
                frame_range.get("start_frame"),
                frame_range.get("end_frame"),
                timestamp_range.get("start_sec"),
                timestamp_range.get("end_sec"),
            )
            if any(value is None for value in values):
                continue

            try:
                start_frame, end_frame, start_sec, end_sec = values
                frame_count = int(end_frame) - int(start_frame)
                duration_sec = float(end_sec) - float(start_sec)
            except (TypeError, ValueError):
                continue

            if frame_count > 0 and duration_sec > 0:
                return float(frame_count) / duration_sec

        if read_video_metadata:
            cv2_backend = get_cv2_backend()
            cv2 = cv2_backend.get_cv2()
            if cv2 is None:
                raise ImportError("OpenCV is required to resolve video fps from metadata.")

            capture = cv2_backend.VideoCapture(str(video_path))
            if capture is None or not capture.isOpened():
                if capture is not None:
                    capture.release()
                raise ValueError(f"Cannot open video to resolve fps: {video_path}")
            try:
                fps = float(capture.get(cv2_backend.CAP_PROP_FPS) or 0.0)
            finally:
                capture.release()
            if fps > 0:
                return fps
            raise ValueError(f"Cannot resolve positive fps from video metadata: {video_path}")

        raise ValueError(
            "Cannot resolve video fps. Provide row video_fps/fps/shot_fps "
            "or segment frame_range plus timestamp_range."
        )

    def _sample_frame_path(self, video_path, sample):
        normalized_video_path = os.path.abspath(str(video_path))
        video_hash = hashlib.sha1(normalized_video_path.encode("utf-8")).hexdigest()[:16]
        output_dir = os.path.join(os.path.abspath(self.frame_output_dir), video_hash)
        filename = f"{sample['sample_id']}_{int(sample['frame_index'])}.{self.frame_image_format}"
        return os.path.join(output_dir, filename)

    def _save_sample_frames(self, video_path, samples):
        cv2_backend = get_cv2_backend()
        cv2 = cv2_backend.get_cv2()
        if cv2 is None:
            raise ImportError("OpenCV is required when save_sample_frames=True.")

        # Path writing and existing-frame skipping logic match the original implementation
        pending = []
        for sample in samples:
            output_path = self._sample_frame_path(video_path, sample)
            sample[self.frame_path_key] = output_path
            if os.path.exists(output_path) and not self.overwrite_sample_frames:
                continue
            pending.append(sample)

        if not pending:
            return

        # Batch frame extraction: when the mm acceleration backend is available, decoding is done by the backend (per-item fallback for non-MP4),
        # Otherwise the default backend uses the cv2 implementation internally; output is uniformly RGB, converted back to BGR before writing (exact channel swap, bytes unchanged)
        from ...utils.media_backend import get_media_backend
        backend = get_media_backend()
        frame_indices = sorted({int(sample["frame_index"]) for sample in pending})
        frames = backend.decode_video_frames(str(video_path), frame_indices=set(frame_indices))
        frame_map = dict(zip(frame_indices, frames))

        for sample in pending:
            frame_index = int(sample["frame_index"])
            output_path = sample[self.frame_path_key]
            frame_rgb = frame_map.get(frame_index)
            if frame_rgb is None:
                raise ValueError(
                    f"Failed to read sample frame for video_path={video_path}, "
                    f"sample_id={sample['sample_id']}, frame_index={frame_index}"
                )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(output_path, frame_bgr):
                raise ValueError(
                    f"Failed to write sample frame for video_path={video_path}, "
                    f"sample_id={sample['sample_id']}, frame_index={frame_index}, output_path={output_path}"
                )

    def _select_frame_indices(self, segment, fps, semantic_frame_features=None):
        start_frame, end_frame = resolve_segment_frame_range(segment, fps)
        last_frame = max(start_frame, end_frame - 1)

        if self.sampling_mode == "CENTER":
            return [start_frame + (last_frame - start_frame) // 2], ["CENTER"]

        if self.sampling_mode == "UNIFORM":
            frame_indices = select_uniform_frame_indices(
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                num_frames=self.num_frames,
                target_fps=self.target_fps,
            )
            return frame_indices, ["UNIFORM"] * len(frame_indices)

        if self.sampling_mode == "HEAD_TAIL":
            if start_frame == last_frame:
                return [start_frame], ["HEAD"]
            return [start_frame, last_frame], ["HEAD", "TAIL"]

        if self.sampling_mode == "STATE":
            return select_state_frame_indices(
                start_frame=start_frame,
                end_frame=end_frame,
                padding_frames=self.padding_frames,
            )

        candidate_indices = list(range(start_frame, last_frame + 1))
        frame_count = resolve_frame_count(
            start_frame=start_frame,
            end_frame=end_frame,
            num_frames=self.num_frames,
            max_frames_per_segment=self.max_frames_per_segment,
        )

        if self.sampling_mode == "SEMANTIC_AWARE":
            frame_indices = select_semantic_diverse_indices(
                candidate_indices=candidate_indices,
                frame_count=frame_count,
                semantic_frame_features=semantic_frame_features,
                min_distance_threshold=self.semantic_min_distance_threshold,
            )
            if not frame_indices:
                frame_indices = [start_frame]
            return frame_indices, ["SEMANTIC_AWARE"] * len(frame_indices)

        frame_indices = select_motion_diverse_indices(
            candidate_indices=candidate_indices,
            frame_count=frame_count,
            semantic_frame_features=semantic_frame_features,
            min_distance_threshold=self.semantic_min_distance_threshold,
        )
        if not frame_indices:
            frame_indices = [start_frame]
        return frame_indices, ["MOTION_AWARE"] * len(frame_indices)


def resolve_segment_frame_range(segment, fps):
    timestamp_range = segment.get("timestamp_range") or {}
    try:
        start_sec = float(timestamp_range["start_sec"])
        end_sec = float(timestamp_range["end_sec"])
        fps = float(fps)
    except (KeyError, TypeError, ValueError):
        start_sec = end_sec = fps = 0.0

    if fps > 0 and end_sec > start_sec:
        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        if end_frame > start_frame:
            return start_frame, end_frame

    frame_range = segment["frame_range"]
    return int(frame_range["start_frame"]), int(frame_range["end_frame"])


def select_uniform_frame_indices(
    start_frame,
    end_frame,
    fps,
    num_frames,
    target_fps=None,
):
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    last_frame = max(start_frame, end_frame - 1)
    available_frames = max(1, last_frame - start_frame + 1)
    frame_count = int(num_frames)
    if target_fps is not None:
        frame_count = int(math.ceil((end_frame - start_frame) / float(fps) * float(target_fps)))
    frame_count = max(1, min(frame_count, available_frames))
    if frame_count == 1:
        return [start_frame + (last_frame - start_frame) // 2]
    step = (last_frame - start_frame) / max(1, frame_count - 1)
    return unique_preserve_order(
        [
            int(round(start_frame + index * step))
            for index in range(frame_count)
        ]
    )


def select_state_frame_indices(start_frame, end_frame, padding_frames):
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    last_frame = max(start_frame, end_frame - 1)
    offset = min(int(padding_frames), max(0, (last_frame - start_frame) // 2))
    head = start_frame + offset
    tail = last_frame - offset
    if head == tail:
        return [head], ["STATE_HEAD"]
    return [head, tail], ["STATE_HEAD", "STATE_TAIL"]


def select_semantic_diverse_indices(
    candidate_indices,
    frame_count,
    semantic_frame_features,
    min_distance_threshold,
):
    valid_candidates = [
        int(candidate)
        for candidate in candidate_indices
        if int(candidate) in semantic_frame_features
    ]
    if not valid_candidates:
        return []

    selected_indices = [valid_candidates[0]]
    remaining_indices = valid_candidates[1:]

    while remaining_indices and len(selected_indices) < int(frame_count):
        best_index = None
        best_score = -1.0
        for candidate in remaining_indices:
            score = min(
                cosine_distance(
                    semantic_frame_features[candidate],
                    semantic_frame_features[selected],
                )
                for selected in selected_indices
            )
            if score > best_score:
                best_score = score
                best_index = candidate
        if best_index is None or best_score < float(min_distance_threshold):
            break
        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return selected_indices[: int(frame_count)]


def select_motion_diverse_indices(
    candidate_indices,
    frame_count,
    semantic_frame_features,
    min_distance_threshold,
):
    selected_indices = []
    for candidate in candidate_indices:
        candidate = int(candidate)
        if candidate not in semantic_frame_features:
            continue
        if not selected_indices:
            selected_indices.append(candidate)
            if len(selected_indices) >= int(frame_count):
                break
            continue

        min_distance = min(
            cosine_distance(
                semantic_frame_features[candidate],
                semantic_frame_features[selected],
            )
            for selected in selected_indices
        )
        if min_distance < float(min_distance_threshold):
            continue
        selected_indices.append(candidate)
        if len(selected_indices) >= int(frame_count):
            break

    return selected_indices
