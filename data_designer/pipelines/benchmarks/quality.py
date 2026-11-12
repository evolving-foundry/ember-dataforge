"""Curator deduplication and quality gate for audited benchmark items."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List

from .contracts import BenchmarkError


def assess_taskset(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score every public case after native validation and before publication.

    Ray and Curator are imported only on the build path. Runtime evaluation has
    no dependency on their operator implementations.
    """
    if not items:
        raise BenchmarkError("quality gate requires non-empty audited items")
    try:
        import ray

        from base.dataset import MultimodalDataset
        from curator.operations import deduplication as dedup_ops
        from curator.quality.quality_assessor import DataQualityAssessor
        from curator.quality.quality_metrics import (
            QualityAssessmentConfig,
            QualityDimension,
        )
    except ImportError as error:
        raise BenchmarkError(f"Curator quality gate unavailable: {error}") from error

    rows = [
        {
            "task_id": item["task_id"],
            "task_text": item["task_text"],
            "success_criteria": item["success_criteria"],
        }
        for item in items
    ]
    try:
        dataset = MultimodalDataset(ray.data.from_items(rows), modalities=["text"])
        deduped = dedup_ops.ExactDuplicateRemovalOperation().execute(dataset)
        dedup_count = int(deduped.dataset.count())
        if dedup_count != len(rows):
            raise BenchmarkError(
                f"Curator exact dedup removed {len(rows) - dedup_count} audited items"
            )
        curator = DataQualityAssessor(
            QualityAssessmentConfig(
                sample_size=len(rows),
                enabled_dimensions=list(QualityDimension),
                text_fields=["task_text", "success_criteria"],
            )
        ).assess(deduped, operation_name="benchmark_case_build")
    except BenchmarkError:
        raise
    except Exception as error:
        raise BenchmarkError(f"Curator quality gate failed: {error}") from error

    # A directional floor supplements Curator's structural dimensions. Its
    # thresholds are explicit and deterministic; a one-word task cannot receive
    # a perfect score merely because all fields are non-null and unique.
    task_clarity = statistics.mean(
        min(len(item["task_text"].strip()) / 32, 1.0) for item in items
    )
    criteria_clarity = statistics.mean(
        min(len(item["success_criteria"].strip()) / 24, 1.0) for item in items
    )
    domain_score = round((task_clarity + criteria_clarity) / 2, 4)
    score = min(float(curator.overall_score), domain_score)
    if not 0 <= score <= 1:
        raise BenchmarkError("quality gate produced an invalid score")
    return {
        "quality_overall_score": score,
        "curator_score": float(curator.overall_score),
        "domain_clarity_score": domain_score,
        "quality_dimension_scores": {
            (key.value if hasattr(key, "value") else str(key)): value.score
            for key, value in curator.dimension_scores.items()
        },
        "quality_issues": list(curator.issues),
        "dedup_removed": len(rows) - dedup_count,
        "dedup_kept": dedup_count,
    }
