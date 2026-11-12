from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Type

from base.dataset import MultimodalDataset
from .base import BaseMetricAssessor
from .config import OpsQualityAssessmentConfig
from .context import EvaluationContext, EvaluationContextBuilder
from .metrics import (
    ChairObjectHallucinationAssessor,
    CoherenceScoreAssessor,
    ImageTextAlignmentAssessor,
    QaeGroundingAlignmentAssessor,
    TextNoiseContaminationAssessor,
    TextNormalizationValidityAssessor,
    TextSemanticPreservationAssessor,
    VideoImageAlignmentAssessor,
    VisualRobustnessAssessor,
    VisualTransformConsistencyAssessor,
)
from .results import MetricResult, OpsQualityAssessmentResult



DEFAULT_METRIC_ASSESSORS: tuple[Type[BaseMetricAssessor], ...] = (
    TextNoiseContaminationAssessor,
    TextNormalizationValidityAssessor,
    ImageTextAlignmentAssessor,
    VideoImageAlignmentAssessor,
    ChairObjectHallucinationAssessor,
    TextSemanticPreservationAssessor,
    VisualTransformConsistencyAssessor,
    VisualRobustnessAssessor,
    QaeGroundingAlignmentAssessor,
    CoherenceScoreAssessor,
)


class OpsQualityAssessor:
    """Pipeline-friendly assessor for MultimodalDataset snapshots."""

    def __init__(
        self,
        config: Optional[OpsQualityAssessmentConfig] = None,
        metric_assessors: Sequence[Type[BaseMetricAssessor] | BaseMetricAssessor] | None = None,
    ):
        self.config = config or OpsQualityAssessmentConfig()
        self.metric_assessors = self._build_metric_assessors(
            metric_assessors or DEFAULT_METRIC_ASSESSORS
        )

    def evaluate(self, dataset: MultimodalDataset) -> OpsQualityAssessmentResult: 
        context = EvaluationContextBuilder(self.config).build(dataset)
        return self.evaluate_context(context)

    def evaluate_rows(
        self,
        dataset: MultimodalDataset,
        rows: list[dict],
    ) -> OpsQualityAssessmentResult:
        context = EvaluationContextBuilder(self.config).build_from_rows(dataset, rows)
        return self.evaluate_context(context)

    def evaluate_context(self, context: EvaluationContext) -> OpsQualityAssessmentResult:
        metric_results: dict[str, MetricResult] = {}

        for metric_name in self.config.enabled_metrics:
            try:
                assessor = self.metric_assessors[metric_name]
                metric_results[metric_name] = assessor.assess(context)
            except KeyError as exc:
                metric_results[metric_name] = MetricResult(
                    metric_name=metric_name,
                    status="failed",
                    scores={},
                    notices=[str(exc)],
                )

        return OpsQualityAssessmentResult(
            evaluated_at=datetime.now(),
            row_count=context.row_count,
            sample_size=len(context.rows),
            metric_results=metric_results,
        )

    @staticmethod
    def _build_metric_assessors(
        metric_assessors: Sequence[Type[BaseMetricAssessor] | BaseMetricAssessor],
    ) -> dict[str, BaseMetricAssessor]:
        result: dict[str, BaseMetricAssessor] = {}
        for item in metric_assessors:
            assessor = item() if isinstance(item, type) else item
            result[assessor.metric_name] = assessor
        return result
