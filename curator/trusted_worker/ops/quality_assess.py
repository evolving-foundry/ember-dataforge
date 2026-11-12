"""Quality assessment op: shared core (used by both evaluation mode and op mode)."""
from __future__ import annotations

from typing import Any, Dict, List

from curator.quality import (
    DataQualityAssessor,
    QualityAssessmentConfig,
    QualityDimension,
)


QUALITY_CONFIG_KEYS = {
    "sample_size",
    "null_threshold",
    "enabled_dimensions",
    "dimension_weights",
    "text_fields",
}


def build_quality_config(overrides: Dict[str, Any]) -> QualityAssessmentConfig:
    """Translate host-provided overrides into ``QualityAssessmentConfig``.

    Allowed keys: sample_size / null_threshold / enabled_dimensions /
    dimension_weights / text_fields. Defaults are the public API (Evaluator §7-4:
    quality-gate defaults feed both Q1 metrics and taskset construction, so changes require regression on both hosts).
    """
    kwargs: Dict[str, Any] = {}
    if "sample_size" in overrides:
        kwargs["sample_size"] = int(overrides["sample_size"])
    if "null_threshold" in overrides:
        kwargs["null_threshold"] = float(overrides["null_threshold"])
    if "enabled_dimensions" in overrides:
        kwargs["enabled_dimensions"] = [
            QualityDimension(d) for d in overrides["enabled_dimensions"]
        ]
    if "dimension_weights" in overrides:
        kwargs["dimension_weights"] = {
            str(k): float(v) for k, v in overrides["dimension_weights"].items()
        }
    if "text_fields" in overrides:
        kwargs["text_fields"] = [str(f) for f in overrides["text_fields"]]
    return QualityAssessmentConfig(**kwargs)


def assess_rows(
    rows: List[Dict[str, Any]],
    config: QualityAssessmentConfig,
    operation_name: str = "candidate_batch_quality_assess",
):
    """Shared core: rows → MultimodalDataset → DataQualityAssessor.assess."""
    import ray

    from base.dataset import MultimodalDataset

    dataset = MultimodalDataset(dataset=ray.data.from_items(rows))
    return DataQualityAssessor(config).assess(dataset, operation_name=operation_name)
