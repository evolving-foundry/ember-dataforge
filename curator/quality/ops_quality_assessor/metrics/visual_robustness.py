from __future__ import annotations

from typing import Any, List, cast

from pydantic import BaseModel, Field

from ..base import BaseMetricAssessor, InvalidMetricInputError
from ..context import EvaluationContext
from ..results import MetricResult
from .common import (
    as_list,
    cosine_similarity,
    has_modality,
    image_embeddings,
    mean,
    structural_similarity_score,
)


class VisualRobustnessConfig(BaseModel):
    image_field: str = "image"
    transform_fields: List[str] = Field(
        default_factory=lambda: ["crop_image_data_list", "flip_image_data_list", "canny_image_data"]
    )
    embedding_model_path: str = "openai/clip-vit-base-patch32"
    device: str | None = None


class VisualRobustnessAssessor(BaseMetricAssessor):
    metric_name = "visual_robustness"
    output_scores = ("embedding_drift_mean", "structural_similarity_mean")
    config_model = VisualRobustnessConfig

    def _assess(self, context: EvaluationContext) -> MetricResult:
        metric_config = cast(
            VisualRobustnessConfig,
            self.resolve_metric_config(context) or VisualRobustnessConfig(),
        )

        if not has_modality(context, "image") and not context.has_field(metric_config.image_field):
            return MetricResult(
                metric_name=self.metric_name,
                status="not_applicable",
                scores={},
                skipped_scores={score: "dataset does not contain image fields" for score in self.output_scores},
                notices=["dataset does not contain image fields"],
            )
        if not context.has_field(metric_config.image_field):
            raise InvalidMetricInputError("missing original image field")
        if not any(context.has_field(field) for field in metric_config.transform_fields):
            raise InvalidMetricInputError("missing transformed image fields")

        drift_scores = []
        structural_scores = []
        pair_count = 0
        for row in context.rows:
            original_value = row.get(metric_config.image_field)
            original_vector = self._image_embedding(original_value, metric_config)
            for transformed in self._transformed_values(row, metric_config.transform_fields):
                pair_count += 1
                transformed_value = self._image_value(transformed)
                transformed_vector = self._image_embedding(transformed_value, metric_config)
                if original_vector is not None and transformed_vector is not None:
                    drift_scores.append(1.0 - cosine_similarity(original_vector, transformed_vector))
                structural_score = structural_similarity_score(original_value, transformed_value)
                if structural_score is not None:
                    structural_scores.append(structural_score)

        if pair_count == 0:
            raise InvalidMetricInputError("missing before/after image pairs")

        notices = []
        skipped_scores = {}
        scores = {}
        if not drift_scores:
            skipped_scores["embedding_drift_mean"] = (
                "CLIP embeddings could not be computed"
            )
            notices.append("CLIP embeddings could not be computed")
        else:
            scores["embedding_drift_mean"] = mean(drift_scores)
        if not structural_scores:
            skipped_scores["structural_similarity_mean"] = (
                "SSIM scores could not be computed"
            )
            notices.append("SSIM scores could not be computed")
        else:
            scores["structural_similarity_mean"] = mean(structural_scores)

        status = "passed"
        if skipped_scores:
            status = "warning" if scores else "failed_precondition"

        return MetricResult(
            metric_name=self.metric_name,
            status=status,
            scores=scores,
            skipped_scores=skipped_scores,
            notices=notices,
            metadata={
                "pair_count": pair_count,
                "drift_pair_count": len(drift_scores),
                "structural_pair_count": len(structural_scores),
                "embedding_model_path": metric_config.embedding_model_path,
            },
        )

    @staticmethod
    def _transformed_values(row: dict[str, Any], fields: list[str]) -> list[Any]:
        values: list[Any] = []
        for field_name in fields:
            if field_name in row:
                values.extend(as_list(row.get(field_name)))
        return [value for value in values if value is not None]

    @staticmethod
    def _image_value(value: Any) -> Any:
        if isinstance(value, dict):
            for key in ("image", "image_path"):
                if key in value:
                    return value.get(key)
        return value

    @staticmethod
    def _image_embedding(
        value: Any,
        metric_config: VisualRobustnessConfig,
    ) -> list[float] | None:
        try:
            return image_embeddings(
                [value],
                model_repo_or_path=metric_config.embedding_model_path,
                device=metric_config.device,
            )[0]
        except (ImportError, RuntimeError, ValueError):
            return None
