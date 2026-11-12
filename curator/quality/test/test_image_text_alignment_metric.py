from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
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
    ray_module.__spec__ = ModuleSpec("ray", loader=None)
    ray_data_module.__spec__ = ModuleSpec("ray.data", loader=None)

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
from curator.quality.ops_quality_assessor.metrics import (
    image_text_alignment,
)
from curator.quality.ops_quality_assessor.metrics.image_text_alignment import (
    ImageTextAlignmentAssessor,
    ImageTextAlignmentConfig,
)


def _context(
    rows: list[dict[str, Any]],
    metric_config: dict[str, Any] | None = None,
) -> EvaluationContext:
    schema_fields = {field_name for row in rows for field_name in row}
    data = {
        field_name: [row.get(field_name) for row in rows]
        for field_name in schema_fields
    }
    return EvaluationContext(
        dataset=SimpleNamespace(modalities=["image", "text"]),
        rows=rows,
        schema_fields=schema_fields,
        data=data,
        row_count=len(rows),
        config=OpsQualityAssessmentConfig(
            sample_size=len(rows),
            enabled_metrics=["image_text_alignment"],
            metric_configs={"image_text_alignment": metric_config or {}},
        ),
    )


def _stub_embeddings(monkeypatch, observed_texts: list[str]) -> None:
    monkeypatch.setattr(
        image_text_alignment,
        "image_embeddings",
        lambda values, **kwargs: [[1.0, 0.0] for _ in values],
    )

    def _text_embeddings(values, **kwargs):
        observed_texts.extend(values)
        return [[1.0, 0.0] for _ in values]

    monkeypatch.setattr(image_text_alignment, "text_embeddings", _text_embeddings)


def test_image_text_alignment_caption_source_uses_caption_only(monkeypatch) -> None:
    observed_texts: list[str] = []
    _stub_embeddings(monkeypatch, observed_texts)

    metric = ImageTextAlignmentAssessor().assess(
        _context(
            [
                {
                    "image": "image-a.png",
                    "caption": "a red car",
                    "text": "plain text should not be used in caption mode",
                }
            ]
        )
    )

    assert metric.status == "passed"
    assert metric.scores["clip_score_mean"] == 1.0
    assert observed_texts == ["a red car"]
    assert metric.metadata["text_source"] == "caption"


def test_image_text_alignment_any_text_source_accepts_plain_text(monkeypatch) -> None:
    observed_texts: list[str] = []
    _stub_embeddings(monkeypatch, observed_texts)

    metric = ImageTextAlignmentAssessor().assess(
        _context(
            [
                {
                    "image": "image-a.png",
                    "text": "a person crossing a street",
                }
            ],
            {"text_source": "any_text"},
        )
    )

    assert metric.status == "passed"
    assert metric.scores["clip_score_mean"] == 1.0
    assert observed_texts == ["a person crossing a street"]
    assert metric.metadata["text_source"] == "any_text"


def test_image_text_alignment_any_text_source_can_be_field_scoped() -> None:
    pairs = ImageTextAlignmentAssessor._build_pairs(
        [
            {
                "image": "image-a.png",
                "caption": "caption text",
                "text": "generic text",
                "output_text": "augmented text",
            }
        ],
        ImageTextAlignmentConfig(
            text_source="any_text",
            any_text_fields=("output_text",),
        ),
    )

    assert pairs == [("image-a.png", "augmented text")]
