from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import ray

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
pythonpath_entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
if str(REPO_ROOT) not in pythonpath_entries:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *[entry for entry in pythonpath_entries if entry]]
    )

from base.dataset import MultimodalDataset
from curator.operations.augmentation import VideoSamplingOperation
from curator.quality.ops_quality_assessor import (
    OpsQualityAssessmentConfig,
    OpsQualityAssessor,
)

MEGAMIND_FIXTURE_PATH = REPO_ROOT / "data" / "ops_quality_megamind_fixture.json"
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/data/llm_models"))
HF_HUB_ROOT = Path(
    os.environ.get(
        "HF_HUB_ROOT",
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    )
)
LOCAL_BGE_M3_MODEL_PATH = MODEL_ROOT / "BAAI"
LOCAL_CLIP_MODEL_PATH = Path(
    os.environ.get(
        "CLIP_MODEL_PATH",
        str(
            HF_HUB_ROOT
            / "models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K"
            / "snapshots"
            / "1a25a446712ba5ee05982a381eed697ef9b435cf"
        ),
    )
)


def _preferred_text_value(row: dict[str, Any]) -> Any:
    if row.get("output_text") is not None:
        return row.get("output_text")
    return row.get("text")


def _build_assessment_rows(
    initial_dataset: MultimodalDataset,
    augmented_dataset: MultimodalDataset,
    sample_size: int,
) -> list[dict[str, Any]]:
    before_rows = initial_dataset.dataset.limit(sample_size).take_all()
    after_rows = augmented_dataset.dataset.limit(sample_size).take_all()
    rows = []
    for before_row, after_row in zip(before_rows, after_rows):
        row = dict(after_row)
        row.setdefault("before_text", _preferred_text_value(before_row))
        row.setdefault("after_text", _preferred_text_value(after_row))
        rows.append(row)
    return rows


def test_offline_quality_assessor():
    initial_row = json.loads(MEGAMIND_FIXTURE_PATH.read_text(encoding="utf-8"))
    initial_row["text"] = "Megamind protects Metro City after learning to become a hero."
    augmented_row = dict(initial_row)
    augmented_row["text"] = "Megamind becomes a hero and protects Metro City."

    initial_dataset = MultimodalDataset(
        ray.data.from_items([initial_row]),
        modalities=["text", "video"],
    )
    augmented_dataset = MultimodalDataset(
        ray.data.from_items([augmented_row]),
        modalities=["text", "video"],
    )
    augmented_dataset = VideoSamplingOperation(
        {
            "video_key": "video_path",
            "segments_key": "segments",
            "sampling_mode": "CENTER",
            "num_frames": 1,
            "save_sample_frames": True,
            "frame_output_dir": str(
                REPO_ROOT / "data" / "ops_quality_sample_frames" / "standalone"
            ),
            "overwrite_sample_frames": True,
        }
    ).execute(augmented_dataset)
    assessment_rows = _build_assessment_rows(
        initial_dataset=initial_dataset,
        augmented_dataset=augmented_dataset,
        sample_size=1,
    )

    result = OpsQualityAssessor(
        OpsQualityAssessmentConfig(
            sample_size=1,
            enabled_metrics=[
                "text_semantic_preservation",
                "video_image_alignment",
            ],
            metric_configs={
                "text_semantic_preservation": {
                    "embedding_model_path": str(LOCAL_BGE_M3_MODEL_PATH),
                    "device": "cpu",
                    "drift_threshold": 0.8,
                },
                "video_image_alignment": {
                    "image_embedding_model_path": str(LOCAL_CLIP_MODEL_PATH),
                    "device": "cpu",
                },
            },
        )
    ).evaluate_rows(augmented_dataset, assessment_rows)

    semantic_metric = result.metric_results["text_semantic_preservation"]
    if semantic_metric.status == "failed":
        pytest.fail(str(semantic_metric.notices or semantic_metric.skipped_scores))
    assert semantic_metric.status == "passed"
    assert semantic_metric.metadata["before_text_key"] == "before_text"
    assert semantic_metric.metadata["after_text_key"] == "after_text"
    assert semantic_metric.scores["semantic_preservation_score"] >= 0.8
    assert semantic_metric.scores["semantic_drift_rate"] == 0.0

    video_metric = result.metric_results["video_image_alignment"]
    if video_metric.status == "failed":
        pytest.fail(str(video_metric.notices or video_metric.skipped_scores))
    assert video_metric.status == "passed"
    assert video_metric.metadata["segment_count"] == 2
    assert video_metric.metadata["used_embeddings"] is True
    assert video_metric.scores["segment_frame_alignment_mean"] >= 0.8
    assert video_metric.scores["temporal_coverage_rate"] == 1.0
