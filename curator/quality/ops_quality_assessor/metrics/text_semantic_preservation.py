from __future__ import annotations

from typing import Optional, cast

from pydantic import BaseModel

from ..base import BaseMetricAssessor, InvalidMetricInputError
from ..context import EvaluationContext
from ..results import MetricResult
from .common import (
    cosine_similarity,
    mean,
    round_score,
    text_embeddings,
)


class TextSemanticPreservationConfig(BaseModel):
    before_text_key: Optional[str] = None
    after_text_key: Optional[str] = None
    embedding_model_path: str = "BAAI/bge-m3"
    device: str | None = None
    drift_threshold: float = 0.8


class TextSemanticPreservationAssessor(BaseMetricAssessor):
    metric_name = "text_semantic_preservation"
    output_scores = ("semantic_preservation_score", "semantic_drift_rate")
    config_model = TextSemanticPreservationConfig

    def _assess(self, context: EvaluationContext) -> MetricResult:
        metric_config = cast(
            TextSemanticPreservationConfig,
            self.resolve_metric_config(context) or TextSemanticPreservationConfig(),
        )

        before_field = self._resolve_field(
            context,
            metric_config.before_text_key or "before_text",
            "before",
        )
        after_field = self._resolve_field(
            context,
            metric_config.after_text_key or "after_text",
            "after",
        )

        scores = []
        pairs = list(zip(context.values(before_field), context.values(after_field)))
        try:
            before_vectors = text_embeddings(
                [str(before or "") for before, _after in pairs],
                model_repo_or_path=metric_config.embedding_model_path,
                device=metric_config.device,
            )
            after_vectors = text_embeddings(
                [str(after or "") for _before, after in pairs],
                model_repo_or_path=metric_config.embedding_model_path,
                device=metric_config.device,
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            raise InvalidMetricInputError(str(exc)) from exc

        for left, right in zip(before_vectors, after_vectors):
            if left is not None and right is not None:
                scores.append(cosine_similarity(left, right))

        if not scores:
            raise InvalidMetricInputError("all before/after embedding pairs are missing or invalid")

        drift_threshold = float(metric_config.drift_threshold)
        drift_rate = sum(1 for score in scores if score < drift_threshold) / len(scores)

        return MetricResult(
            metric_name=self.metric_name,
            status="passed",
            scores={
                "semantic_preservation_score": mean(scores),
                "semantic_drift_rate": round_score(drift_rate),
            },
            metadata={
                "before_text_key": before_field,
                "after_text_key": after_field,
                "embedding_model_path": metric_config.embedding_model_path,
                "evaluated_pair_count": len(scores),
                "drift_threshold": drift_threshold,
            },
        )

    @staticmethod
    def _resolve_field(
        context: EvaluationContext,
        field_name: str,
        role: str,
    ) -> str:
        if context.has_field(field_name):
            return field_name
        raise InvalidMetricInputError(f"configured {role} text field is missing: {field_name}")
