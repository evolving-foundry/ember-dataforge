from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if importlib.util.find_spec("ray") is None:
    ray_module = types.ModuleType("ray")
    ray_data_module = types.ModuleType("ray.data")

    class _Dataset:
        pass

    def _read_unsupported_media(*args, **kwargs):
        return None

    ray_data_module.Dataset = _Dataset
    ray_data_module.read_images = _read_unsupported_media
    ray_data_module.read_audio = _read_unsupported_media
    ray_data_module.read_videos = _read_unsupported_media
    ray_module.data = ray_data_module
    sys.modules["ray"] = ray_module
    sys.modules["ray.data"] = ray_data_module

from curator.quality.ops_quality_assessor.config import (
    OpsQualityAssessmentConfig,
)
from curator.quality.ops_quality_assessor.context import (
    EvaluationContext,
)
from curator.quality.ops_quality_assessor.metrics.video_image_alignment import (
    ROW_IMAGE_ALIGNMENT_NOTICE,
    VideoImageAlignmentAssessor,
)


def _context(rows: list[dict[str, Any]]) -> EvaluationContext:
    schema_fields = {field_name for row in rows for field_name in row}
    data = {
        field_name: [row.get(field_name) for row in rows]
        for field_name in schema_fields
    }
    return EvaluationContext(
        dataset=SimpleNamespace(modalities=["image", "video"]),
        rows=rows,
        schema_fields=schema_fields,
        data=data,
        row_count=len(rows),
        config=OpsQualityAssessmentConfig(
            sample_size=len(rows),
            enabled_metrics=["video_image_alignment"],
        ),
    )


def test_video_image_alignment_keeps_sampled_frame_temporal_coverage() -> None:
    metric = VideoImageAlignmentAssessor().assess(
        _context(
            [
                {
                    "segments": [
                        {
                            "segment_id": "segment-1",
                            "embedding": [1.0, 0.0],
                            "timestamp_range": {
                                "start_sec": 0.0,
                                "end_sec": 2.0,
                            },
                        },
                    ],
                    "samples": [
                        {
                            "segment_id": "segment-1",
                            "embedding": [1.0, 0.0],
                            "timestamp_sec": 1.0,
                        },
                    ],
                },
            ]
        )
    )

    assert metric.status == "passed"
    assert metric.scores["segment_frame_alignment_mean"] == 1.0
    assert metric.scores["temporal_coverage_rate"] == 1.0
    assert metric.metadata["used_provided_image_alignment"] is False


def test_video_image_alignment_accepts_row_image_and_video_segment() -> None:
    metric = VideoImageAlignmentAssessor().assess(
        _context(
            [
                {
                    "image": {"embedding": [1.0, 0.0]},
                    "video_segment": {"embedding": [1.0, 0.0]},
                },
            ]
        )
    )

    assert metric.status == "passed"
    assert metric.scores["segment_frame_alignment_mean"] == 1.0
    assert "temporal_coverage_rate" not in metric.scores
    assert "temporal_coverage_rate" not in metric.skipped_scores
    assert ROW_IMAGE_ALIGNMENT_NOTICE in metric.notices
    assert metric.metadata["used_provided_image_alignment"] is True
