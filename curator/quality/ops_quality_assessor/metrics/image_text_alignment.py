from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel

from ..base import BaseMetricAssessor, InvalidMetricInputError
from ..context import EvaluationContext
from ..results import MetricResult
from .common import (
    as_list,
    cosine_similarity,
    get_path_values,
    has_modality,
    image_embeddings,
    mean,
    non_empty_text,
    text_embeddings,
)


class ImageTextAlignmentConfig(BaseModel):
    image_field: str = "image"
    text_field: str = "caption"
    text_source: Literal["caption", "any_text"] = "caption"
    any_text_fields: tuple[str, ...] = (
        "text",
        "input_text",
        "output_text",
        "caption",
        "before_text",
        "after_text",
        "description_units[].description_text",
        "text_signals[].text",
    )
    embedding_model_path: str = "openai/clip-vit-base-patch32"
    device: str | None = None


class ImageTextAlignmentAssessor(BaseMetricAssessor):
    metric_name = "image_text_alignment"
    output_scores = ("clip_score_mean",)
    config_model = ImageTextAlignmentConfig

    def _assess(self, context: EvaluationContext) -> MetricResult:
        metric_config = cast(
            ImageTextAlignmentConfig,
            self.resolve_metric_config(context) or ImageTextAlignmentConfig(),
        )

        if not has_modality(context, "image") and not (
            context.has_field(metric_config.image_field)
            or context.has_field("samples")
        ):
            return MetricResult(
                metric_name=self.metric_name,
                status="not_applicable",
                scores={},
                skipped_scores={
                    score: "dataset does not contain image fields"
                    for score in self.output_scores
                },
                notices=["dataset does not contain image fields"],
            )

        pairs = self._build_pairs(context.rows, metric_config)

        if not pairs:
            raise InvalidMetricInputError("missing image-text pairs")

        scores = []
        for image_value, text_value in pairs:
            score = self._score_from_embeddings(
                image_value,
                text_value,
                metric_config,
            )
            if score is not None:
                scores.append(score)

        if not scores:
            return MetricResult(
                metric_name=self.metric_name,
                status="failed_precondition",
                scores={},
                skipped_scores={
                    score: "CLIP scores or embeddings could not be computed"
                    for score in self.output_scores
                },
                notices=["CLIP scores or embeddings could not be computed"],
                metadata={"pair_count": len(pairs)},
            )

        notices = []
        status = "passed"
        if len(scores) < len(pairs):
            status = "warning"
            notices.append(
                f"{len(pairs) - len(scores)} image-text pairs could not be scored by CLIP"
            )

        return MetricResult(
            metric_name=self.metric_name,
            status=status,
            scores={
                "clip_score_mean": mean(scores),
            },
            notices=notices,
            metadata={
                "pair_count": len(pairs),
                "scored_pair_count": len(scores),
                "text_source": metric_config.text_source,
                "embedding_model_path": metric_config.embedding_model_path,
            },
        )

    @classmethod
    def _build_pairs(
        cls,
        rows: list[dict[str, Any]],
        metric_config: ImageTextAlignmentConfig,
    ) -> list[tuple[Any, str]]:
        pairs: list[tuple[Any, str]] = []

        for row in rows:
            image_values = as_list(row.get(metric_config.image_field))
            image_values.extend(get_path_values(row, "samples[].frame_path"))
            image_values = [value for value in image_values if value is not None]

            text_values = cls._row_text_values(row, metric_config)

            if not image_values or not text_values:
                continue

            for image_value in image_values:
                for text_value in text_values:
                    pairs.append((image_value, text_value))

        return pairs

    @classmethod
    def _row_text_values(
        cls,
        row: dict[str, Any],
        metric_config: ImageTextAlignmentConfig,
    ) -> list[str]:
        fields = (
            (metric_config.text_field,)
            if metric_config.text_source == "caption"
            else metric_config.any_text_fields
        )

        values: list[str] = []
        seen: set[str] = set()
        for field_name in fields:
            raw_values = cls._row_field_values(row, field_name)
            for value in raw_values:
                text_value = non_empty_text(value)
                if not text_value or text_value in seen:
                    continue
                values.append(text_value)
                seen.add(text_value)
        return values

    @staticmethod
    def _row_field_values(row: dict[str, Any], field_name: str) -> list[Any]:
        if "." in field_name or "[]" in field_name:
            return get_path_values(row, field_name)
        return as_list(row.get(field_name))

    @staticmethod
    def _score_from_embeddings(
        image_value: Any,
        text_value: str,
        metric_config: ImageTextAlignmentConfig,
    ) -> float | None:
        try:
            image_vector = image_embeddings(
                [image_value],
                model_repo_or_path=metric_config.embedding_model_path,
                device=metric_config.device,
            )[0]

            text_vector = text_embeddings(
                [text_value],
                model_repo_or_path=metric_config.embedding_model_path,
                device=metric_config.device,
            )[0]

        except (ImportError, RuntimeError, ValueError):
            return None

        if image_vector is None or text_vector is None:
            return None

        return cosine_similarity(image_vector, text_vector)
