import json
import math
import os
import sys
import importlib.util
import importlib
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


SUPPORTED_TEMPORAL_MODES = {"EVENT_WINDOW", "EVENT_CHAIN"}
SUPPORTED_VIDEO_DESCRIPTION_MODES = {
    "FRAME_DESCRIPTION",
    "EVENT_DESCRIPTION",
    "SEGMENT_DESCRIPTION",
    "CONTEXT_DESCRIPTION",
}
VIDEO_DESCRIPTION_PROMPTS = {
    "FRAME_DESCRIPTION": "a picture of",
    "EVENT_DESCRIPTION": "an event showing",
    "SEGMENT_DESCRIPTION": "a video segment showing",
    "CONTEXT_DESCRIPTION": "a broader video context showing",
}


def cosine_distance(embedding_a, embedding_b):
    vector_a = np.asarray(embedding_a, dtype=np.float32)
    vector_b = np.asarray(embedding_b, dtype=np.float32)
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    cosine_similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    return 1.0 - cosine_similarity


def sec_to_frame(sec, fps):
    return max(0, int(math.floor(float(sec) * float(fps))))


def unique_preserve_order(values):
    ordered_values = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered_values.append(value)
    return ordered_values


def resolve_frame_count(start_frame, end_frame, num_frames, max_frames_per_segment=None):
    last_frame = max(int(start_frame), int(end_frame) - 1)
    available_frames = max(1, last_frame - int(start_frame) + 1)
    frame_count = max(1, min(int(num_frames), available_frames))
    if max_frames_per_segment is not None:
        frame_count = min(frame_count, max(1, int(max_frames_per_segment)))
    return frame_count


def _require_cv2():
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise ImportError("OpenCV is required. Please install opencv-python>=4.5.0.") from exc
    return cv2


def _open_video_capture(video_path: str):
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {video_path}")
    return cv2, capture


def _read_video_metadata(capture, cv2, video_path: str) -> Tuple[float, int]:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        raise ValueError(f"Invalid video metadata for {video_path}: fps={fps}, total_frames={total_frames}")
    return fps, total_frames


def _resolve_repo_dir(model_repo_or_path: str, cache_dir: Optional[str] = None) -> str:
    if os.path.isdir(model_repo_or_path):
        return os.path.abspath(model_repo_or_path)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for remote model download. "
            "Please install huggingface_hub."
        ) from exc

    if cache_dir:
        return snapshot_download(repo_id=model_repo_or_path, local_dir=cache_dir)
    return snapshot_download(repo_id=model_repo_or_path)


def _get_video_fps(video_path: str) -> float:
    cv2, capture = _open_video_capture(video_path)
    fps, _ = _read_video_metadata(capture, cv2, video_path)
    return fps


@lru_cache(maxsize=2)
def _load_transnet_runner(model_repo_or_path: str, cache_dir: Optional[str] = None):
    repo_dir = _resolve_repo_dir(model_repo_or_path, cache_dir=cache_dir)
    main_file = os.path.join(repo_dir, "main.py")
    if not os.path.isfile(main_file):
        raise FileNotFoundError(f"Cannot find TransNetV2 main.py in {repo_dir}")

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    module_name = f"transnetv2_main_{abs(hash(repo_dir))}"
    spec = importlib.util.spec_from_file_location(module_name, main_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import TransNetV2 entry from {main_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    weights_file = os.path.join(repo_dir, "transnetv2-pytorch-weights.pth")
    runner = module.TransNetV2Torch(model_path=weights_file if os.path.isfile(weights_file) else None)
    return runner


def compute_video_shot_scores(
    video_path: str,
    model_repo_or_path: str = "magnusdtd/TransNetV2",
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute frame-level shot boundary scores for a video using TransNetV2.

    Returns:
        dict with keys:
          - shot_scores: List[float], one score per frame
          - shot_fps: float, inferred source video fps
    """
    runner = _load_transnet_runner(model_repo_or_path, cache_dir=cache_dir)
    _, single_frame_predictions, _ = runner.predict_video(video_path)

    return {
        "shot_scores": np.asarray(single_frame_predictions, dtype=np.float32).tolist(),
        "shot_fps": _get_video_fps(video_path),
    }


def _resolve_torch_device(device: Optional[str] = None) -> str:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required. Please install torch>=2.0.0.") from exc

    if device:
        return device
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return "npu"
    except ImportError:
        pass
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _load_semantic_encoder(model_repo_or_path: str, device: str):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for semantic embedding extraction. "
            "Please install transformers>=4.40.0 and torch>=2.0.0."
        ) from exc

    processor = None
    try:
        processor = AutoImageProcessor.from_pretrained(model_repo_or_path, trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_repo_or_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(model_repo_or_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    return processor, model, torch


def _extract_embeddings_from_images(
    frames_rgb: List[np.ndarray],
    model_repo_or_path: str,
    device: Optional[str],
) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required. Please install Pillow>=8.0.0.") from exc

    resolved_device = _resolve_torch_device(device)
    processor, model, torch = _load_semantic_encoder(model_repo_or_path, resolved_device)

    images = [Image.fromarray(frame) for frame in frames_rgb]
    inputs = processor(images=images, return_tensors="pt")
    inputs = {
        key: value.to(resolved_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        if hasattr(model, "get_image_features") and "pixel_values" in inputs:
            outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
        else:
            outputs = model(**inputs)

    embeddings = None
    if hasattr(outputs, "detach"):
        embeddings = outputs
    elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        embeddings = outputs.image_embeds
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embeddings = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        embeddings = outputs.last_hidden_state.mean(dim=1)
    elif isinstance(outputs, (tuple, list)) and outputs:
        first_output = outputs[0]
        if getattr(first_output, "ndim", 0) >= 3:
            embeddings = first_output.mean(dim=1)
        else:
            embeddings = first_output

    if embeddings is None:
        raise RuntimeError("Cannot infer embedding tensor from model outputs.")

    return embeddings.detach().cpu().numpy().astype(np.float32)


def _sample_window_frames(
    video_path: str,
    window_sec: float,
    max_windows: Optional[int] = None,
) -> Tuple[List[Dict[str, float]], List[np.ndarray]]:
    cv2, capture = _open_video_capture(video_path)
    try:
        fps, total_frames = _read_video_metadata(capture, cv2, video_path)
        duration_sec = float(total_frames) / fps
        windows: List[Dict[str, float]] = []
        frame_indices: List[int] = []

        start_sec = 0.0
        while start_sec < duration_sec:
            if max_windows is not None and len(windows) >= max_windows:
                break

            end_sec = min(duration_sec, start_sec + window_sec)
            center_sec = (start_sec + end_sec) / 2.0
            frame_indices.append(min(total_frames - 1, max(0, int(round(center_sec * fps)))))
            windows.append(
                {
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                }
            )
            start_sec = end_sec

        frames_rgb: List[np.ndarray] = []
        next_frame_index = 0
        last_frame_index = None
        last_frame_rgb = None

        for target_index in frame_indices:
            if target_index == last_frame_index and last_frame_rgb is not None:
                frames_rgb.append(last_frame_rgb.copy())
                continue

            while next_frame_index <= target_index:
                ok = capture.grab()
                if not ok:
                    break

                current_index = next_frame_index
                next_frame_index += 1
                if current_index == target_index:
                    ok, frame_bgr = capture.retrieve()
                    if not ok:
                        break
                    last_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    last_frame_index = target_index
                    frames_rgb.append(last_frame_rgb.copy())
                    break

            if len(frames_rgb) == 0 or last_frame_index != target_index:
                break
    finally:
        capture.release()

    if len(frames_rgb) != len(windows):
        windows = windows[:len(frames_rgb)]

    if not windows:
        raise ValueError(f"No frames sampled from video: {video_path}")

    return windows, frames_rgb


def compute_video_semantic_embeddings(
    video_path: str,
    model_repo_or_path: str = "nvidia/Cosmos-Embed1-224p",
    window_sec: float = 4.0,
    batch_size: int = 8,
    max_windows: Optional[int] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute window-level semantic embeddings for a video.

    Returns:
        List[dict], each dict contains:
          - start_sec: float
          - end_sec: float
          - embedding: List[float]
    """
    if window_sec <= 0:
        raise ValueError("window_sec must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    windows, frames_rgb = _sample_window_frames(video_path, window_sec=window_sec, max_windows=max_windows)

    all_embeddings: List[np.ndarray] = []
    for start in range(0, len(frames_rgb), batch_size):
        frame_batch = frames_rgb[start:start + batch_size]
        all_embeddings.append(
            _extract_embeddings_from_images(
                frame_batch,
                model_repo_or_path=model_repo_or_path,
                device=device,
            )
        )

    embedding_matrix = np.concatenate(all_embeddings, axis=0)

    semantic_embeddings: List[Dict[str, Any]] = []
    for index, window in enumerate(windows):
        semantic_embeddings.append(
            {
                "start_sec": window["start_sec"],
                "end_sec": window["end_sec"],
                "embedding": embedding_matrix[index].tolist(),
            }
        )

    return semantic_embeddings


def merge_description_captions(captions):
    merged_captions = []
    for caption in captions:
        if caption and caption not in merged_captions:
            merged_captions.append(caption)
    return " ".join(merged_captions)


def load_text_entries(text_uri):
    if not text_uri:
        raise ValueError("Missing text source uri.")

    local_path = os.path.abspath(text_uri)
    if not os.path.exists(local_path):
        raise ValueError(f"Text source does not exist: {local_path}")

    file_ext = os.path.splitext(local_path)[1].lower()
    if file_ext == ".json":
        with open(local_path, "r", encoding="utf-8") as file_obj:
            entries = json.load(file_obj)
        if isinstance(entries, dict):
            return entries["entries"]
        return entries

    if file_ext == ".jsonl":
        entries = []
        with open(local_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    raise ValueError(f"Unsupported text source format: {file_ext}")


def merge_text_signals_by_segment(text_signals):
    signal_map = {}
    for text_signal in text_signals:
        merge_key = (
            text_signal["segment_id"],
            text_signal["text_type"],
            text_signal["language"],
        )
        if merge_key not in signal_map:
            signal_map[merge_key] = {
                "segment_id": text_signal["segment_id"],
                "timestamp_range": dict(text_signal["timestamp_range"]),
                "text_type": text_signal["text_type"],
                "text": text_signal["text"],
                "language": text_signal["language"],
            }
            continue

        merged_signal = signal_map[merge_key]
        merged_signal["timestamp_range"]["start_sec"] = min(
            float(merged_signal["timestamp_range"]["start_sec"]),
            float(text_signal["timestamp_range"]["start_sec"]),
        )
        merged_signal["timestamp_range"]["end_sec"] = max(
            float(merged_signal["timestamp_range"]["end_sec"]),
            float(text_signal["timestamp_range"]["end_sec"]),
        )
        if text_signal["text"] not in merged_signal["text"]:
            merged_signal["text"] = merged_signal["text"] + " " + text_signal["text"]

    return list(signal_map.values())



def build_scaffold_description_map(description_units):
    description_map = {}
    for unit in description_units or []:
        segment_ids = description_unit_segment_ids(unit)
        for segment_id in segment_ids:
            bucket = description_map.setdefault(segment_id, {"ids": [], "units": []})
            if unit.get("description_id"):
                bucket["ids"].append(str(unit["description_id"]))
            bucket["units"].append(unit)
    return description_map


def build_scaffold_text_map(text_signals):
    text_map = {}
    for signal in text_signals or []:
        segment_id = str(signal["segment_id"])
        text_map.setdefault(segment_id, []).append(signal)
    return text_map


def build_segment_text(segment_id, description_map, text_map, sample_map):
    description_texts = description_map.get(segment_id) or []
    signal_texts = text_map.get(segment_id) or []
    sample_entries = sample_map.get(segment_id) or []
    if description_texts:
        part = description_texts[0]
    elif signal_texts:
        part = "text: " + " / ".join(signal_texts[:2])
    elif sample_entries:
        part = f"{segment_id} with {len(sample_entries)} samples"
    else:
        raise Warning(f"No description or text found for segment {segment_id}")
    if description_texts and signal_texts:
        part += "; text: " + " / ".join(signal_texts[:2])
    return part


def build_temporal_structure_item(segment_ids, structure_type, summary, segment_lookup):
    return {
        "segment_id": segment_ids[-1],
        "segment_ids": segment_ids,
        "timestamp_range": {
            "start_sec": float(segment_lookup[segment_ids[0]]["timestamp_range"]["start_sec"]),
            "end_sec": float(segment_lookup[segment_ids[-1]]["timestamp_range"]["end_sec"]),
        },
        "structure_type": structure_type,
        "summary": summary,
        "precedence_refs": segment_ids[:-1] if len(segment_ids) > 1 else [],
    }


def build_event_chain_prompt(window_segments, description_map, text_map, sample_map):
    lines = []
    for index, segment in enumerate(window_segments, start=1):
        segment_id = str(segment["segment_id"])
        lines.append(
            f"{index}. {segment_id}: {build_segment_text(segment_id, description_map, text_map, sample_map)}"
        )
    return (
        "Summarize these consecutive video segments as one coherent event chain.\n"
        "Write one short sentence that captures the main action flow or higher-level event.\n\n"
        "Segments:\n" + "\n".join(lines)
    )



def build_temporal_candidate_text(candidates):
    lines = []
    for candidate in candidates or []:
        candidate_id = str(candidate["candidate_id"])
        segment_ids = ", ".join(str(segment_id) for segment_id in candidate.get("segment_ids") or [])
        summary = str(candidate.get("summary") or "").strip()
        lines.append(f"- {candidate_id}: segments=[{segment_ids}] summary={summary}")
    return "\n".join(lines)


def parse_temporal_evidence_refs(response_text, candidates, max_evidence_refs=3):
    candidate_ids = {
        str(candidate["candidate_id"]): candidate
        for candidate in candidates or []
    }
    for raw_line in str(response_text or "").splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("evidence_ids:"):
            continue
        selected_ids = []
        for part in line.split(":", 1)[1].split(","):
            candidate_id = part.strip()
            if candidate_id in candidate_ids and candidate_id not in selected_ids:
                selected_ids.append(candidate_id)
        return selected_ids[:max_evidence_refs]
    return []


def build_temporal_grounding_candidates(temporal_structures, description_units, text_signals):
    description_by_id = {
        str(unit["description_unit_id"]): str(unit.get("description_text") or "").strip()
        for unit in description_units or []
        if unit.get("description_unit_id")
    }
    text_by_id = {
        str(signal["signal_id"]): str(signal.get("text") or signal.get("content") or "").strip()
        for signal in text_signals or []
        if signal.get("signal_id")
    }
    description_map = build_temporal_description_map(
        [unit for unit in (description_units or []) if unit.get("segment_id") is not None]
    )
    text_map = build_temporal_text_map(
        [signal for signal in (text_signals or []) if signal.get("segment_id") is not None]
    )
    candidates = []
    for temporal_structure in temporal_structures or []:
        temporal_unit_id = str(
            temporal_structure.get("temporal_unit_id")
            or temporal_structure.get("temporal_id")
            or temporal_structure.get("id")
        )
        if not temporal_unit_id:
            continue
        segment_ids = [str(segment_id) for segment_id in temporal_structure.get("segment_ids") or []]
        summary_parts = []
        summary = str(
            temporal_structure.get("summary")
            or temporal_structure.get("temporal_label")
            or ""
        ).strip()
        if summary:
            summary_parts.append(summary)
        for segment_id in segment_ids:
            summary_parts.extend(description_map.get(segment_id) or [])
            summary_parts.extend(text_map.get(segment_id) or [])
        for description_id in temporal_structure.get("source_description_ids") or []:
            text = description_by_id.get(str(description_id))
            if text:
                summary_parts.append(text)
        for signal_id in temporal_structure.get("source_text_signal_ids") or []:
            text = text_by_id.get(str(signal_id))
            if text:
                summary_parts.append(text)
        candidates.append(
            {
                "candidate_id": temporal_unit_id,
                "temporal_unit_id": temporal_unit_id,
                "segment_ids": segment_ids,
                "source_description_ids": [
                    str(description_id)
                    for description_id in temporal_structure.get("source_description_ids") or []
                ],
                "source_text_signal_ids": [
                    str(signal_id)
                    for signal_id in temporal_structure.get("source_text_signal_ids") or []
                ],
                "source_evidence_unit_ids": [
                    str(evidence_id)
                    for evidence_id in temporal_structure.get("source_evidence_unit_ids") or []
                ],
                "summary": " / ".join(unique_preserve_order([part for part in summary_parts if part])),
            }
        )
    return candidates


def build_temporal_description_map(description_units):
    description_map = {}
    for unit in description_units or []:
        text_value = str(unit.get("description_text") or "").strip()
        if text_value:
            for segment_id in description_unit_segment_ids(unit):
                description_map.setdefault(segment_id, []).append(text_value)
    return description_map


def description_unit_segment_ids(unit):
    segment_ids = []
    if unit.get("segment_id") is not None:
        segment_ids.append(str(unit["segment_id"]))
    if unit.get("segment_ids"):
        segment_ids.extend(str(segment_id) for segment_id in unit.get("segment_ids") or [])
    anchor_refs = unit.get("anchor_refs") or {}
    if isinstance(anchor_refs, dict):
        segment_ids.extend(
            str(segment_id)
            for segment_id in anchor_refs.get("segment_ids") or []
        )
    return unique_preserve_order([segment_id for segment_id in segment_ids if segment_id])


def build_temporal_text_map(text_signals):
    text_map = {}
    for signal in text_signals or []:
        segment_id = str(signal["segment_id"])
        text_value = str(signal.get("text") or signal.get("content") or "").strip()
        if text_value:
            text_map.setdefault(segment_id, []).append(text_value)
    return text_map


def build_temporal_sample_map(samples):
    sample_map = {}
    for sample in samples or []:
        sample_map.setdefault(str(sample["segment_id"]), []).append(sample)
    return sample_map

def merge_visual_actions(visual_actions):
    action_groups = []
    for visual_action in visual_actions or []:
        label = str(visual_action.get("label") or "").strip()
        if not label:
            continue
        normalized_label = label.lower().replace("_", " ")
        if action_groups and action_groups[-1]["normalized_label"] == normalized_label:
            group = action_groups[-1]
        else:
            group = {
                "label": normalized_label,
                "normalized_label": normalized_label,
                "scores": [],
                "segment_ids": [],
                "sample_ids": [],
                "frame_paths": [],
                "visual_evidence": [],
            }
            action_groups.append(group)

        group["scores"].append(float(visual_action.get("score") or 0.0))
        if visual_action.get("segment_id"):
            group["segment_ids"].append(str(visual_action["segment_id"]))
        if visual_action.get("sample_id"):
            group["sample_ids"].append(str(visual_action["sample_id"]))
        if visual_action.get("frame_path"):
            group["frame_paths"].append(str(visual_action["frame_path"]))
        group["visual_evidence"].append(dict(visual_action))

    for group in action_groups:
        group["score"] = round(sum(group["scores"]) / max(len(group["scores"]), 1), 6)
        group["segment_ids"] = unique_preserve_order(group["segment_ids"])
        group["sample_ids"] = unique_preserve_order(group["sample_ids"])
        group["frame_paths"] = unique_preserve_order(group["frame_paths"])
        del group["scores"]
        del group["normalized_label"]
    return action_groups

@lru_cache(maxsize=2)
def load_action_scene_pipeline(model_path, device):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for ACTION_GRAPH frame action recognition. "
            "Please install transformers and torch."
        ) from exc

    pipeline_kwargs = {}
    resolved_device = device or _resolve_torch_device(device)
    if resolved_device:
        pipeline_kwargs["device"] = resolved_device
    return pipeline("image-classification", model=model_path, **pipeline_kwargs)


def attach_context_windows(*args, **kwargs):
    from ..operations.filtering.video_context_orchestration_operation import (
        attach_context_windows as _attach_context_windows,
    )

    return _attach_context_windows(*args, **kwargs)


def build_filter_hints(*args, **kwargs):
    from ..operations.filtering.video_context_orchestration_operation import (
        build_filter_hints as _build_filter_hints,
    )

    return _build_filter_hints(*args, **kwargs)


def merge_stitch_segments(*args, **kwargs):
    from ..operations.filtering.video_context_orchestration_operation import (
        merge_stitch_segments as _merge_stitch_segments,
    )

    return _merge_stitch_segments(*args, **kwargs)


def select_uniform_frame_indices(*args, **kwargs):
    from ..operations.augmentation.video_sampling_operation import (
        select_uniform_frame_indices as _select_uniform_frame_indices,
    )

    return _select_uniform_frame_indices(*args, **kwargs)


def select_state_frame_indices(*args, **kwargs):
    from ..operations.augmentation.video_sampling_operation import (
        select_state_frame_indices as _select_state_frame_indices,
    )

    return _select_state_frame_indices(*args, **kwargs)


def select_semantic_diverse_indices(*args, **kwargs):
    from ..operations.augmentation.video_sampling_operation import (
        select_semantic_diverse_indices as _select_semantic_diverse_indices,
    )

    return _select_semantic_diverse_indices(*args, **kwargs)


def select_motion_diverse_indices(*args, **kwargs):
    from ..operations.augmentation.video_sampling_operation import (
        select_motion_diverse_indices as _select_motion_diverse_indices,
    )

    return _select_motion_diverse_indices(*args, **kwargs)


def build_event_window_structures(*args, **kwargs):
    from ..operations.augmentation.video_temporal_structure_operation import (
        build_event_window_structures as _build_event_window_structures,
    )

    return _build_event_window_structures(*args, **kwargs)


def build_event_chain_structures(*args, **kwargs):
    from ..operations.augmentation.video_temporal_structure_operation import (
        build_event_chain_structures as _build_event_chain_structures,
    )

    return _build_event_chain_structures(*args, **kwargs)


def build_qae_materialization_evidence_map(*args, **kwargs):
    from ..operations.augmentation.video_scaffold_construction_operation import (
        build_qae_materialization_evidence_map as _build_qae_materialization_evidence_map,
    )

    return _build_qae_materialization_evidence_map(*args, **kwargs)


def materialize_qae_from_scaffold(*args, **kwargs):
    from ..operations.augmentation.video_qae_materialization_operation import (
        materialize_qae_from_scaffold as _materialize_qae_from_scaffold,
    )

    return _materialize_qae_from_scaffold(*args, **kwargs)


def materialize_qae_with_temporal_grounding(*args, **kwargs):
    from ..operations.augmentation.video_qae_materialization_operation import (
        materialize_qae_with_temporal_grounding as _materialize_qae_with_temporal_grounding,
    )

    return _materialize_qae_with_temporal_grounding(*args, **kwargs)


def evidence_ref_count(*args, **kwargs):
    from ..operations.augmentation.video_qae_materialization_operation import (
        evidence_ref_count as _evidence_ref_count,
    )

    return _evidence_ref_count(*args, **kwargs)


def evidence_refs_have_values(*args, **kwargs):
    from ..operations.augmentation.video_qae_materialization_operation import (
        evidence_refs_have_values as _evidence_refs_have_values,
    )

    return _evidence_refs_have_values(*args, **kwargs)


def evidence_text_ref_ids(*args, **kwargs):
    from ..operations.augmentation.video_qae_materialization_operation import (
        evidence_text_ref_ids as _evidence_text_ref_ids,
    )

    return _evidence_text_ref_ids(*args, **kwargs)


def evidence_image_lookup_ids(*args, **kwargs):
    from ..operations.augmentation.video_qae_materialization_operation import (
        evidence_image_lookup_ids as _evidence_image_lookup_ids,
    )

    return _evidence_image_lookup_ids(*args, **kwargs)


def evidence_image_frame_paths(*args, **kwargs):
    from ..operations.augmentation.video_qae_materialization_operation import (
        evidence_image_frame_paths as _evidence_image_frame_paths,
    )

    return _evidence_image_frame_paths(*args, **kwargs)


def evidence_temporal_ref_ids(*args, **kwargs):
    from ..operations.augmentation.video_qae_materialization_operation import (
        evidence_temporal_ref_ids as _evidence_temporal_ref_ids,
    )

    return _evidence_temporal_ref_ids(*args, **kwargs)


def build_revision_records(*args, **kwargs):
    from ..operations.augmentation.video_consistency_revision_operation import (
        build_revision_records as _build_revision_records,
    )

    return _build_revision_records(*args, **kwargs)


def build_quality_records(*args, **kwargs):
    from ..operations.filtering.video_quality_screening_operation import (
        build_quality_records as _build_quality_records,
    )

    allowed_kwargs = {"quality_mode", "qae_triplets", "revisions", "required_dataset_tier"}
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in allowed_kwargs}
    return _build_quality_records(*args, **filtered_kwargs)
