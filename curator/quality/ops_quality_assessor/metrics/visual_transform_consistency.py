from __future__ import annotations

from typing import Any, List, Optional, cast

from pydantic import BaseModel, Field

from ..base import BaseMetricAssessor, InvalidMetricInputError
from ..context import EvaluationContext
from ..results import MetricResult
from .common import (
    as_list,
    cosine_similarity,
    has_modality,
    image_embeddings,
    load_image_array,
    mean,
    round_score,
)


class VisualTransformConsistencyConfig(BaseModel):
    image_field: str = "image"
    transform_fields: List[str] = Field(
        default_factory=lambda: ["crop_image_data_list", "flip_image_data_list", "canny_image_data"]
    )
    embedding_model_path: str = "openai/clip-vit-base-patch32"
    device: str | None = None


class VisualTransformConsistencyAssessor(BaseMetricAssessor):
    metric_name = "visual_transform_consistency"
    output_scores = ("transform_output_success_rate", "semantic_similarity_before_after")
    config_model = VisualTransformConsistencyConfig

    def _assess(self, context: EvaluationContext) -> MetricResult:
        metric_config = cast(
            VisualTransformConsistencyConfig,
            self.resolve_metric_config(context) or VisualTransformConsistencyConfig(),
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
            raise InvalidMetricInputError("missing visual transform output fields")

        expected_count = 0
        success_count = 0
        similarities = []
        for row in context.rows:
            original_vector = self._image_embedding(row.get(metric_config.image_field), metric_config)
            for transformed in self._transformed_values(row, metric_config.transform_fields):
                expected_count += 1
                if not self._is_valid_transform_output(transformed):
                    continue
                success_count += 1
                transformed_vector = self._image_embedding(
                    self._image_value(transformed),
                    metric_config,
                )
                if original_vector is not None and transformed_vector is not None:
                    similarities.append(cosine_similarity(original_vector, transformed_vector))

        if expected_count == 0:
            raise InvalidMetricInputError("visual transform output fields are empty")

        success_rate = success_count / expected_count
        notices = []
        status = "passed"
        skipped_scores = {}
        scores = {
            "transform_output_success_rate": round_score(success_rate),
        }
        if not similarities:
            status = "warning"
            skipped_scores["semantic_similarity_before_after"] = (
                "CLIP embeddings could not be computed"
            )
            notices.append("CLIP embeddings could not be computed")
        else:
            scores["semantic_similarity_before_after"] = mean(similarities)

        return MetricResult(
            metric_name=self.metric_name,
            status=status,
            scores=scores,
            skipped_scores=skipped_scores,
            notices=notices,
            metadata={
                "expected_output_count": expected_count,
                "decoded_output_count": success_count,
                "semantic_similarity_pair_count": len(similarities),
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
    def _is_valid_transform_output(value: Any) -> bool:
        if isinstance(value, dict):
            for key in ("image", "image_path"):
                if key in value:
                    return load_image_array(value[key]) is not None
            return bool(value)
        return load_image_array(value) is not None

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
        metric_config: VisualTransformConsistencyConfig,
    ) -> list[float] | None:
        try:
            return image_embeddings(
                [value],
                model_repo_or_path=metric_config.embedding_model_path,
                device=metric_config.device,
            )[0]
        except (ImportError, RuntimeError, ValueError):
            return None
