#!/usr/bin/env python3
"""Generate a real image-video alignment trend chart for image operations.

This script follows ``text-image-alignment-trend.py`` but focuses on the
``video_image_alignment`` metric under a cumulative image-operation pipeline.
It uses the repository Megamind fixture, but does not use toy vectors:

- it cuts real segment clips from ``data/videos/Megamind.avi`` into
  ``data/videos/megamind_segments``;
- it samples multiple original frames from each segment and averages their real
  image embeddings into the segment anchor;
- it applies image operations cumulatively to those frames; and
- it scores every stage through ``VideoImageAlignmentAssessor``.

Run in the local ML environment, for example:

    python test/image-video-alignment-trend.py

No network downloads are intended.  The default image embedding model is a local
CLIP snapshot when present; override it with ``VIDEO_IMAGE_ALIGNMENT_MODEL_PATH``.
The script also attempts real video-segment embedding on the extracted clips
when ``VIDEO_SEGMENT_EMBEDDING_MODEL_PATH`` is available, and records that
status in ``image-video-alignment-segments.csv``.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import sys
import types
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
FIXTURE_RELATIVE_PATH = Path("data") / "ops_quality_megamind_fixture.json"
SEGMENT_OUTPUT_DIR = Path("data") / "videos" / "megamind_segments"
DEFAULT_SEED = int(os.environ.get("IMAGE_VIDEO_ALIGNMENT_SEED", "7"))
DEVICE = os.environ.get("VIDEO_IMAGE_ALIGNMENT_DEVICE") or None
ANCHOR_SAMPLE_FRACTIONS = tuple(
    float(item)
    for item in os.environ.get("IMAGE_VIDEO_ALIGNMENT_SAMPLE_FRACTIONS", "0.15,0.50,0.85").split(",")
    if item.strip()
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")


def find_repo_root(start: Path = SCRIPT_PATH) -> Path:
    """Find the DataForge repository root from this script location."""
    for candidate in [start.parent, *start.parents]:
        if (candidate / "data").is_dir() and (candidate / "curator").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def default_embedding_model_path() -> str:
    configured = os.environ.get("VIDEO_IMAGE_ALIGNMENT_MODEL_PATH")
    if configured:
        return configured

    hf_hub_root = Path(
        os.environ.get(
            "HF_HUB_ROOT",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        )
    )
    preferred = [
        hf_hub_root
        / "models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K"
        / "snapshots"
        / "1a25a446712ba5ee05982a381eed697ef9b435cf"
    ]
    preferred += [
        hf_hub_root
        / "models--openai--clip-vit-base-patch32"
        / "snapshots"
        / "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    ]
    for path in preferred:
        if (path / "config.json").is_file():
            return str(path)
    return "openai/clip-vit-base-patch32"


IMAGE_EMBEDDING_MODEL_PATH = default_embedding_model_path()


def default_video_segment_embedding_model_path() -> str:
    configured = os.environ.get("VIDEO_SEGMENT_EMBEDDING_MODEL_PATH")
    if configured:
        return configured

    hf_hub_root = Path(
        os.environ.get(
            "HF_HUB_ROOT",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        )
    )
    preferred = (
        hf_hub_root
        / "models--nvidia--Cosmos-Embed1-224p"
        / "snapshots"
        / "787e0b996f5260a71ad474a283c90539a2e12986"
    )
    if (preferred / "config.json").is_file():
        return str(preferred)
    return "nvidia/Cosmos-Embed1-224p"


VIDEO_SEGMENT_EMBEDDING_MODEL_PATH = default_video_segment_embedding_model_path()


class _LocalSchema:
    def __init__(self, rows: list[dict[str, Any]]):
        names: list[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
        self.names = names


class _LocalDataset:
    """Small Ray Dataset stand-in for deterministic in-process script runs."""

    def __init__(self, rows: Iterable[dict[str, Any]]):
        self._rows = [dict(row) for row in rows]

    def map(self, func):
        return _LocalDataset(func(dict(row)) for row in self._rows)

    def filter(self, func):
        return _LocalDataset(row for row in self._rows if func(dict(row)))

    def limit(self, n: int):
        return _LocalDataset(self._rows[:n])

    def take_all(self):
        return [dict(row) for row in self._rows]

    def take(self, n: int):
        return [dict(row) for row in self._rows[:n]]

    def count(self):
        return len(self._rows)

    def schema(self):
        return _LocalSchema(self._rows)


def _install_ray_stub_if_needed() -> bool:
    """Install a minimal ray/ray.data module before project imports if needed."""
    if importlib.util.find_spec("ray") is not None:
        return False

    ray_module = types.ModuleType("ray")
    ray_data_module = types.ModuleType("ray.data")
    ray_module.__spec__ = ModuleSpec("ray", loader=None)
    ray_data_module.__spec__ = ModuleSpec("ray.data", loader=None)
    ray_data_module.Dataset = _LocalDataset
    ray_data_module.from_items = lambda rows: _LocalDataset(rows)
    ray_data_module.read_images = lambda *args, **kwargs: _LocalDataset([])
    ray_data_module.read_audio = lambda *args, **kwargs: _LocalDataset([])
    ray_data_module.read_videos = lambda *args, **kwargs: _LocalDataset([])
    ray_module.data = ray_data_module
    ray_module.init = lambda *args, **kwargs: None
    ray_module.shutdown = lambda: None
    ray_module.is_initialized = lambda: False
    sys.modules["ray"] = ray_module
    sys.modules["ray.data"] = ray_data_module
    return True


USING_RAY_STUB = _install_ray_stub_if_needed()


def _install_operations_namespace_stubs() -> None:
    """Avoid executing operations/__init__.py, which imports optional heavy deps."""
    operations_dir = REPO_ROOT / "curator" / "operations"
    augmentation_dir = operations_dir / "augmentation"

    operations_module = types.ModuleType("curator.operations")
    operations_module.__path__ = [str(operations_dir)]
    operations_module.__package__ = "curator"
    operations_module.__spec__ = ModuleSpec(
        "curator.operations",
        loader=None,
        is_package=True,
    )

    augmentation_module = types.ModuleType("curator.operations.augmentation")
    augmentation_module.__path__ = [str(augmentation_dir)]
    augmentation_module.__package__ = "curator.operations"
    augmentation_module.__spec__ = ModuleSpec(
        "curator.operations.augmentation",
        loader=None,
        is_package=True,
    )

    sys.modules.setdefault("curator.operations", operations_module)
    sys.modules.setdefault("curator.operations.augmentation", augmentation_module)


_install_operations_namespace_stubs()

from base.dataset import MultimodalDataset  # noqa: E402
from curator.operations.augmentation.image_canny_operation import (  # noqa: E402
    ImageCannyAugmentationOperation,
)
from curator.operations.augmentation.image_random_crop_operation import (  # noqa: E402
    ImageRandomCropOperation,
)
from curator.operations.augmentation.image_random_flip_operation import (  # noqa: E402
    ImageRandomFlipOperation,
)
from curator.quality.ops_quality_assessor.config import (  # noqa: E402
    OpsQualityAssessmentConfig,
)
from curator.quality.ops_quality_assessor.context import (  # noqa: E402
    EvaluationContext,
)
from curator.quality.ops_quality_assessor.metrics.common import (  # noqa: E402
    image_embeddings,
    video_segment_embeddings,
)
from curator.quality.ops_quality_assessor.metrics.video_image_alignment import (  # noqa: E402
    VideoImageAlignmentAssessor,
)
from curator.quality.ops_quality_assessor.results import MetricResult  # noqa: E402


@dataclass(frozen=True)
class TrendPoint:
    branch: str
    operation: str
    alignment: float
    rows: int
    metric_status: str
    temporal_coverage: float | None
    anchor_source: str


@dataclass(frozen=True)
class SegmentAsset:
    segment_id: str
    segment: dict[str, Any]
    clip_path: Path
    samples: list[dict[str, Any]]


@dataclass(frozen=True)
class SegmentEmbeddingReport:
    segment_id: str
    clip_path: Path
    anchor_sample_frames: list[Path]
    anchor_embedding_dim: int
    video_embedding_status: str
    video_embedding_dim: int | None
    video_embedding_error: str


def build_ray_dataset(rows: list[dict[str, Any]]):
    # Keep this script deterministic and independent of Ray worker import paths.
    return _LocalDataset(rows)


def rows_of(dataset: MultimodalDataset) -> list[dict[str, Any]]:
    return dataset.dataset.take_all()


def first_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def load_fixture() -> dict[str, Any]:
    fixture_path = REPO_ROOT / FIXTURE_RELATIVE_PATH
    if not fixture_path.is_file():
        raise RuntimeError(f"Missing fixture: {fixture_path}")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def segment_time_range(segment: dict[str, Any], fallback_fps: float | None) -> tuple[float, float]:
    timestamp_range = segment.get("timestamp_range") or {}
    start = timestamp_range.get("start_sec")
    end = timestamp_range.get("end_sec")
    if start is not None and end is not None:
        return float(start), float(end)

    frame_range = segment.get("frame_range") or {}
    start_frame = frame_range.get("start_frame")
    end_frame = frame_range.get("end_frame")
    if start_frame is not None and end_frame is not None and fallback_fps:
        return float(start_frame) / fallback_fps, float(end_frame) / fallback_fps

    raise RuntimeError(f"Segment does not contain a usable time range: {segment}")


def sample_frame_indices(start_frame: int, end_frame: int) -> list[tuple[str, int]]:
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame
    span = max(0, end_frame - start_frame)
    indices: list[tuple[str, int]] = []
    seen: set[int] = set()
    for fraction in ANCHOR_SAMPLE_FRACTIONS:
        clipped = max(0.0, min(1.0, fraction))
        frame_index = start_frame + int(round(span * clipped))
        if frame_index in seen:
            continue
        seen.add(frame_index)
        label = f"f{int(clipped * 100):02d}"
        indices.append((label, frame_index))
    if not indices:
        indices.append(("f50", start_frame + span // 2))
    return indices


def extract_segment_assets(fixture: dict[str, Any]) -> list[SegmentAsset]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to cut Megamind video segments") from exc

    video_path = Path(str(fixture.get("video_path") or REPO_ROOT / "data" / "videos" / "Megamind.avi"))
    if not video_path.is_file():
        raise RuntimeError(f"Missing source video: {video_path}")

    output_dir = REPO_ROOT / SEGMENT_OUTPUT_DIR
    frame_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or fixture.get("video_fps") or 0.0)
        if fps <= 0.0:
            raise RuntimeError(f"Could not resolve positive FPS for {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if total_frames <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Could not read video metadata for {video_path}")

        assets: list[SegmentAsset] = []
        for raw_segment in fixture.get("segments", []):
            if not isinstance(raw_segment, dict):
                continue
            segment = dict(raw_segment)
            segment_id = str(segment.get("segment_id") or segment.get("id") or f"seg_{len(assets) + 1:04d}")
            start_sec, end_sec = segment_time_range(segment, fps)
            if end_sec < start_sec:
                start_sec, end_sec = end_sec, start_sec
            start_frame = max(0, min(total_frames - 1, int(math.floor(start_sec * fps))))
            end_frame = max(start_frame, min(total_frames - 1, int(math.ceil(end_sec * fps)) - 1))

            clip_path = output_dir / f"{segment_id}.avi"
            writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create segment clip: {clip_path}")

            requested_samples = dict(sample_frame_indices(start_frame, end_frame))
            samples: list[dict[str, Any]] = []
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            try:
                for frame_index in range(start_frame, end_frame + 1):
                    ok, frame_bgr = capture.read()
                    if not ok:
                        break
                    writer.write(frame_bgr)
                    for sample_label, sample_frame in requested_samples.items():
                        if frame_index != sample_frame:
                            continue
                        sample_dir = frame_dir / segment_id
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        frame_path = sample_dir / f"{segment_id}_{sample_label}_{frame_index:06d}.jpg"
                        if not cv2.imwrite(str(frame_path), frame_bgr):
                            raise RuntimeError(f"Could not write sample frame: {frame_path}")
                        samples.append(
                            {
                                "segment_id": segment_id,
                                "timestamp_sec": round(frame_index / fps, 6),
                                "frame_index": frame_index,
                                "sample_type": f"ANCHOR_{sample_label}",
                                "id": f"{segment_id}_{sample_label}",
                                "sample_id": f"{segment_id}_{sample_label}",
                                "frame_path": str(frame_path),
                            }
                        )
            finally:
                writer.release()

            if not clip_path.is_file() or clip_path.stat().st_size == 0:
                raise RuntimeError(f"Segment clip was not created: {clip_path}")
            if not samples:
                raise RuntimeError(f"No sample frames were created for segment {segment_id}")

            segment["segment_id"] = segment_id
            segment["video_segment"] = str(clip_path)
            assets.append(SegmentAsset(segment_id, segment, clip_path, samples))
    finally:
        capture.release()

    if not assets:
        raise RuntimeError("Fixture did not contain usable video segments")
    return assets


def build_rows(assets: list[SegmentAsset], fixture: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for asset in assets:
        anchor_sample_paths = [str(Path(sample["frame_path"])) for sample in asset.samples]
        for sample in asset.samples:
            frame_path = Path(str(sample["frame_path"]))
            if not frame_path.is_file():
                raise RuntimeError(f"Missing generated sample frame: {frame_path}")
            rows.append(
                {
                    "id": f"{asset.segment_id}:{sample['sample_id']}",
                    "video_path": fixture.get("video_path"),
                    "video_duration_sec": fixture.get("video_duration_sec"),
                    "video_fps": fixture.get("video_fps"),
                    "segment_id": asset.segment_id,
                    "segment_clip_path": str(asset.clip_path),
                    "segments": [dict(asset.segment)],
                    "samples": [dict(sample)],
                    "anchor_sample_paths": anchor_sample_paths,
                    "anchor_sample_count": len(anchor_sample_paths),
                    "image": str(frame_path),
                    "image_path": str(frame_path),
                    "original_frame_path": str(frame_path),
                    "alignment_image": str(frame_path),
                }
            )
    if not rows:
        raise RuntimeError("Generated segment assets did not contain usable sample frames")
    return rows


def build_dataset(rows: list[dict[str, Any]]) -> MultimodalDataset:
    return MultimodalDataset(build_ray_dataset(rows), modalities=["image", "video"])


def normalize_image_value(value: Any, color_order: str | None) -> Any:
    value = first_value(value)
    if value is None or color_order != "bgr":
        return value

    try:
        import numpy as np
    except ImportError:
        return value

    array = np.asarray(value)
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3][:, :, ::-1]
    return value


def with_alignment_image(
    dataset: MultimodalDataset,
    *,
    image_field: str,
    image_color_order: str | None = None,
) -> MultimodalDataset:
    def bind(row: dict[str, Any]) -> dict[str, Any]:
        image_value = first_value(row.get(image_field))
        if image_value is None:
            image_value = row.get("image") or row.get("image_path")
        row["alignment_image"] = image_value
        if image_color_order:
            row["alignment_image_color_order"] = image_color_order
        else:
            row.pop("alignment_image_color_order", None)
        return row

    return dataset.map(bind)


def promote_alignment_image(dataset: MultimodalDataset) -> MultimodalDataset:
    """Make the latest image-operation output the next image-operator input."""

    def promote(row: dict[str, Any]) -> dict[str, Any]:
        image_value = row.get("alignment_image")
        if image_value is None:
            image_value = row.get("image") or row.get("image_path")
        row["image"] = image_value
        return row

    return dataset.map(promote)


def mean_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise RuntimeError("Cannot average an empty embedding list")
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise RuntimeError("Cannot average embeddings with inconsistent dimensions")
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(width)
    ]


def compute_video_segment_embeddings(
    assets: list[SegmentAsset],
    *,
    model_path: str,
    device: str | None,
) -> tuple[dict[str, list[float] | None], dict[str, str]]:
    clip_paths = [str(asset.clip_path) for asset in assets]
    try:
        vectors = video_segment_embeddings(
            clip_paths,
            model_repo_or_path=model_path,
            device=device,
        )
    except Exception as exc:
        return {asset.segment_id: None for asset in assets}, {
            asset.segment_id: repr(exc) for asset in assets
        }

    embeddings: dict[str, list[float] | None] = {}
    errors: dict[str, str] = {}
    for asset, vector in zip(assets, vectors):
        embeddings[asset.segment_id] = vector
        errors[asset.segment_id] = "" if vector is not None else "video embedding backend returned None"
    return embeddings, errors


def attach_segment_anchor_embeddings(
    dataset: MultimodalDataset,
    assets: list[SegmentAsset],
    *,
    image_model_path: str,
    video_model_path: str,
    device: str | None,
) -> tuple[MultimodalDataset, list[SegmentEmbeddingReport]]:
    rows = rows_of(dataset)
    anchor_paths_by_segment: dict[str, list[str]] = {}
    for row in rows:
        segment_id = str(row.get("segment_id"))
        paths = [str(path) for path in row.get("anchor_sample_paths", [])]
        if not paths:
            raise RuntimeError(f"Missing anchor sample paths for segment {segment_id}")
        anchor_paths_by_segment.setdefault(segment_id, paths)

    anchor_embeddings_by_segment: dict[str, list[float]] = {}
    for segment_id, paths in anchor_paths_by_segment.items():
        frame_embeddings = image_embeddings(paths, model_repo_or_path=image_model_path, device=device)
        valid_embeddings = [embedding for embedding in frame_embeddings if embedding is not None]
        if len(valid_embeddings) != len(paths):
            raise RuntimeError(f"Could not compute all anchor frame embeddings for segment {segment_id}")
        anchor_embeddings_by_segment[segment_id] = mean_vectors(valid_embeddings)

    video_embeddings_by_segment, video_embedding_errors = compute_video_segment_embeddings(
        assets,
        model_path=video_model_path,
        device=device,
    )

    reports: list[SegmentEmbeddingReport] = []
    for asset in assets:
        segment_anchor = anchor_embeddings_by_segment.get(asset.segment_id)
        if segment_anchor is None:
            raise RuntimeError(f"Missing averaged anchor embedding for segment {asset.segment_id}")
        video_embedding = video_embeddings_by_segment.get(asset.segment_id)
        video_error = video_embedding_errors.get(asset.segment_id, "")
        reports.append(
            SegmentEmbeddingReport(
                segment_id=asset.segment_id,
                clip_path=asset.clip_path,
                anchor_sample_frames=[Path(str(sample["frame_path"])) for sample in asset.samples],
                anchor_embedding_dim=len(segment_anchor),
                video_embedding_status="computed" if video_embedding is not None else "unavailable",
                video_embedding_dim=None if video_embedding is None else len(video_embedding),
                video_embedding_error=video_error,
            )
        )

    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        segment_id = str(row.get("segment_id"))
        updated = dict(row)
        updated["video_segment_embedding"] = anchor_embeddings_by_segment[segment_id]
        updated["video_segment_anchor_source"] = "sample_frame_mean_embedding"
        updated["video_segment_embedding_model"] = image_model_path
        video_embedding = video_embeddings_by_segment.get(segment_id)
        if video_embedding is not None:
            updated["real_video_segment_embedding"] = video_embedding
            updated["real_video_segment_embedding_model"] = video_model_path
        updated_rows.append(updated)

    return build_dataset(updated_rows), reports


def assessment_context(
    rows: list[dict[str, Any]],
    *,
    model_path: str,
    video_model_path: str,
) -> EvaluationContext:
    dataset = build_dataset(rows)
    schema_fields = {field_name for row in rows for field_name in row}
    data = {
        field_name: [row.get(field_name) for row in rows]
        for field_name in schema_fields
    }
    config = OpsQualityAssessmentConfig(
        sample_size=len(rows),
        enabled_metrics=["video_image_alignment"],
        metric_configs={
            "video_image_alignment": {
                "image_embedding_model_path": model_path,
                "video_embedding_model_path": video_model_path,
                "device": DEVICE,
            }
        },
    )
    return EvaluationContext(
        dataset=dataset,
        rows=rows,
        schema_fields=schema_fields,
        data=data,
        row_count=len(rows),
        config=config,
    )


def score_alignment(
    dataset: MultimodalDataset,
    *,
    model_path: str,
    video_model_path: str,
    device: str | None,
) -> tuple[float, MetricResult]:
    rows = rows_of(dataset)
    current_images = [
        normalize_image_value(row.get("alignment_image"), row.get("alignment_image_color_order"))
        for row in rows
    ]
    current_embeddings = image_embeddings(current_images, model_repo_or_path=model_path, device=device)

    assessment_rows: list[dict[str, Any]] = []
    for row, current_embedding in zip(rows, current_embeddings):
        anchor_embedding = row.get("video_segment_embedding")
        if anchor_embedding is None:
            raise RuntimeError(f"Missing video anchor embedding for row {row.get('id')}")
        if current_embedding is None:
            raise RuntimeError(f"Could not compute current image embedding for row {row.get('id')}")

        segment = dict(first_value(row.get("segments")) or {})
        sample = dict(first_value(row.get("samples")) or {})
        segment["embedding"] = anchor_embedding
        sample["embedding"] = current_embedding
        sample.setdefault("segment_id", row.get("segment_id"))

        assessment_rows.append(
            {
                "id": row.get("id"),
                "video_path": row.get("video_path"),
                "segments": [segment],
                "samples": [sample],
            }
        )

    metric = VideoImageAlignmentAssessor().assess(
        assessment_context(
            assessment_rows,
            model_path=model_path,
            video_model_path=video_model_path,
        )
    )
    if metric.status not in {"passed", "warning"}:
        raise RuntimeError(f"video_image_alignment failed: {metric.notices or metric.skipped_scores}")
    score = metric.scores.get("segment_frame_alignment_mean")
    if score is None:
        raise RuntimeError(f"video_image_alignment did not produce a score: {metric.notices or metric.skipped_scores}")
    return round(float(score), 6), metric


def collect_trend_points(
    base_dataset: MultimodalDataset,
    *,
    model_path: str,
    video_model_path: str,
    device: str | None,
) -> list[TrendPoint]:
    row_count = base_dataset.dataset.count()
    points: list[TrendPoint] = []

    current_dataset = with_alignment_image(base_dataset, image_field="image")
    baseline_score, baseline_metric = score_alignment(
        current_dataset,
        model_path=model_path,
        video_model_path=video_model_path,
        device=device,
    )
    points.append(
        TrendPoint(
            "pipeline",
            "input",
            baseline_score,
            row_count,
            baseline_metric.status,
            baseline_metric.scores.get("temporal_coverage_rate"),
            "sample_frame_mean_embedding",
        )
    )

    image_ops: list[tuple[str, Any, str, str | None]] = [
        (
            "image_random_flip",
            ImageRandomFlipOperation(
                {
                    "image_key": "image",
                    "flip_horizontal_prob": 1.0,
                    "flip_vertical_prob": 0.0,
                    "rotation_prob": 0.0,
                    "num_augmentations": 1,
                    "output_key": "flip_image_data",
                }
            ),
            "flip_image_data",
            "bgr",
        ),
        (
            "image_random_crop",
            ImageRandomCropOperation(
                {
                    "image_key": "image",
                    "min_scale": 0.62,
                    "max_scale": 0.62,
                    "aspect_ratio_range": [0.9, 1.1],
                    "num_crops": 1,
                    "output_key": "crop_image_data",
                }
            ),
            "crop_image_data",
            "bgr",
        ),
        (
            "image_canny",
            ImageCannyAugmentationOperation(
                {
                    "image_key": "image",
                    "low_threshold": 30,
                    "high_threshold": 100,
                    "output_key": "canny_image_data",
                }
            ),
            "canny_image_data",
            None,
        ),
    ]

    for operation_name, operation, image_field, color_order in image_ops:
        current_dataset = operation.execute(current_dataset)
        current_dataset = with_alignment_image(
            current_dataset,
            image_field=image_field,
            image_color_order=color_order,
        )
        score, metric = score_alignment(
            current_dataset,
            model_path=model_path,
            video_model_path=video_model_path,
            device=device,
        )
        points.append(
            TrendPoint(
                "pipeline",
                operation_name,
                score,
                row_count,
                metric.status,
                metric.scores.get("temporal_coverage_rate"),
                "sample_frame_mean_embedding",
            )
        )
        current_dataset = promote_alignment_image(current_dataset)

    return points


def write_csv(points: list[TrendPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "branch",
                "operation",
                "alignment_score",
                "rows",
                "metric_status",
                "temporal_coverage_rate",
                "anchor_source",
                "image_embedding_model",
                "video_segment_embedding_model",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.branch,
                    point.operation,
                    f"{point.alignment:.6f}",
                    point.rows,
                    point.metric_status,
                    "" if point.temporal_coverage is None else f"{point.temporal_coverage:.6f}",
                    point.anchor_source,
                    IMAGE_EMBEDDING_MODEL_PATH,
                    VIDEO_SEGMENT_EMBEDDING_MODEL_PATH,
                ]
            )


def write_segment_report_csv(reports: list[SegmentEmbeddingReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "segment_id",
                "segment_clip_path",
                "anchor_sample_frame_count",
                "anchor_sample_frames",
                "segment_anchor_source",
                "segment_anchor_status",
                "segment_anchor_embedding_dim",
                "image_embedding_model",
                "native_video_embedding_status",
                "native_video_embedding_dim",
                "native_video_embedding_model",
                "native_video_embedding_error",
            ]
        )
        for report in reports:
            writer.writerow(
                [
                    report.segment_id,
                    str(report.clip_path),
                    len(report.anchor_sample_frames),
                    ";".join(str(path) for path in report.anchor_sample_frames),
                    "sample_frame_mean_embedding",
                    "computed",
                    report.anchor_embedding_dim,
                    IMAGE_EMBEDDING_MODEL_PATH,
                    report.video_embedding_status,
                    "" if report.video_embedding_dim is None else report.video_embedding_dim,
                    VIDEO_SEGMENT_EMBEDDING_MODEL_PATH,
                    report.video_embedding_error,
                ]
            )


def svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(points: list[TrendPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1040, 600
    left, right, top, bottom = 88, 48, 66, 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.0, 1.03

    def x_coord(index: int) -> float:
        if len(points) == 1:
            return left + plot_w / 2
        return left + index * plot_w / (len(points) - 1)

    def y_coord(score: float) -> float:
        score = max(y_min, min(y_max, score))
        return top + (y_max - score) * plot_h / (y_max - y_min)

    line_color = "#1f77b4"
    path_data = " ".join(
        f'{"M" if index == 0 else "L"} {x_coord(index):.1f},{y_coord(point.alignment):.1f}'
        for index, point in enumerate(points)
    )
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.small{font-size:13px}.label{font-size:15px}.title{font-size:22px;font-weight:700}</style>',
        f'<text class="title" x="{width/2:.1f}" y="34" text-anchor="middle">Real image-video alignment trend under cumulative image operations</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
    ]
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = y_coord(tick)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e2e2"/>')
        lines.append(f'<text class="small" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{tick:.1f}</text>')

    for index, point in enumerate(points):
        x = x_coord(index)
        lines.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+6}" stroke="#333"/>')
        lines.append(
            f'<text class="small" transform="translate({x-8:.1f},{top+plot_h+20}) rotate(34)" text-anchor="start">{svg_escape(point.operation)}</text>'
        )

    lines.append(
        f'<path d="{path_data}" fill="none" stroke="{line_color}" stroke-width="3" stroke-linejoin="round"/>'
    )
    for index, point in enumerate(points):
        x = x_coord(index)
        y = y_coord(point.alignment)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{line_color}" stroke="white" stroke-width="2"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="{y-11:.1f}" text-anchor="middle">{point.alignment:.3f}</text>')

    lines.extend(
        [
            f'<text class="label" x="{left+plot_w/2:.1f}" y="{height-20}" text-anchor="middle">operation</text>',
            f'<text class="label" transform="translate(22,{top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle">real video-image alignment score</text>',
            f'<rect x="{width-298}" y="{top+4}" width="238" height="52" rx="8" fill="#f8f8f8" stroke="#d0d0d0"/>',
            f'<circle cx="{width-273}" cy="{top+31}" r="6" fill="{line_color}"/><text class="small" x="{width-257}" y="{top+36}">cumulative image pipeline</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(points: list[TrendPoint], output_dir: Path) -> Path:
    svg_path = output_dir / "image-video-alignment-trend.svg"
    write_svg(points, svg_path)
    return svg_path


def main() -> int:
    random.seed(DEFAULT_SEED)
    fixture = load_fixture()
    assets = extract_segment_assets(fixture)
    rows = build_rows(assets, fixture)
    dataset = build_dataset(rows)

    output_dir = SCRIPT_PATH.parent / "outputs"
    anchored_dataset, segment_reports = attach_segment_anchor_embeddings(
        dataset,
        assets,
        image_model_path=IMAGE_EMBEDDING_MODEL_PATH,
        video_model_path=VIDEO_SEGMENT_EMBEDDING_MODEL_PATH,
        device=DEVICE,
    )
    points = collect_trend_points(
        anchored_dataset,
        model_path=IMAGE_EMBEDDING_MODEL_PATH,
        video_model_path=VIDEO_SEGMENT_EMBEDDING_MODEL_PATH,
        device=DEVICE,
    )

    csv_path = output_dir / "image-video-alignment-trend.csv"
    segment_csv_path = output_dir / "image-video-alignment-segments.csv"
    write_csv(points, csv_path)
    write_segment_report_csv(segment_reports, segment_csv_path)
    plot_path = write_plot(points, output_dir)

    print("\nDataset backend:", "local in-process dataset" if not USING_RAY_STUB else "local ray stub")
    print(f"Image embedding model: {IMAGE_EMBEDDING_MODEL_PATH}")
    print(f"Video segment embedding model: {VIDEO_SEGMENT_EMBEDDING_MODEL_PATH}")
    print(f"Device: {DEVICE or 'auto'}")
    print("Video fixture:", FIXTURE_RELATIVE_PATH)
    print("Extracted segment clips:")
    for asset in assets:
        print(f"  - {asset.clip_path.relative_to(REPO_ROOT)}")
    print("Generated anchor sample frames:")
    for asset in assets:
        sample_paths = ", ".join(
            str(Path(sample["frame_path"]).relative_to(REPO_ROOT))
            for sample in asset.samples
        )
        print(f"  - {asset.segment_id}: {sample_paths}")
    print("Segment anchor status:")
    for report in segment_reports:
        print(
            f"  - {report.segment_id}: computed from "
            f"{len(report.anchor_sample_frames)} frames dim={report.anchor_embedding_dim}"
        )
    print("Native video encoder status:")
    for report in segment_reports:
        detail = report.video_embedding_status
        if report.video_embedding_dim is not None:
            detail += f" dim={report.video_embedding_dim}"
        elif report.video_embedding_error:
            detail += " error=backend unavailable; see segment metadata CSV"
        print(f"  - {report.segment_id}: {detail}")
    print("\nReal cumulative image-video alignment trend:")
    for point in points:
        coverage = "" if point.temporal_coverage is None else f" temporal={point.temporal_coverage:.6f}"
        print(
            f"  {point.operation:24s} | {point.alignment:.6f} | "
            f"anchor={point.anchor_source} | status={point.metric_status}{coverage}"
        )
    print(f"\nCSV:      {csv_path.relative_to(REPO_ROOT)}")
    print(f"Segments: {segment_csv_path.relative_to(REPO_ROOT)}")
    print(f"Plot:     {plot_path.relative_to(REPO_ROOT)}")

    for path in (csv_path, segment_csv_path, plot_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"output was not created: {path}")
    for asset in assets:
        if not asset.clip_path.exists() or asset.clip_path.stat().st_size == 0:
            raise RuntimeError(f"segment clip was not created: {asset.clip_path}")
    if len(points) < 4 or not all(math.isfinite(point.alignment) for point in points):
        raise RuntimeError("trend points are incomplete or non-finite")
    if not all(0.0 <= point.alignment <= 1.0 for point in points):
        raise RuntimeError("alignment scores must be in [0, 1]")
    if {point.branch for point in points} != {"pipeline"}:
        raise RuntimeError("trend must be a single cumulative pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
