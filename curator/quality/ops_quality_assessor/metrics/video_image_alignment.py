from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from ..base import BaseMetricAssessor, InvalidMetricInputError
from ..context import EvaluationContext
from ..results import MetricResult
from .common import (
    as_float,
    as_list,
    as_vector,
    cosine_similarity,
    has_modality,
    image_embeddings,
    mean,
    round_score,
    video_segment_embeddings,
)

ROW_ALIGNMENT_SEGMENT_ID = "__row_alignment__"
ROW_IMAGE_ALIGNMENT_NOTICE = "Using alignment for provided image and video currently."


class VideoImageAlignmentConfig(BaseModel):
    segment_embedding_key: str = "embedding"
    sample_embedding_key: str = "embedding"
    frame_path_key: str = "frame_path"
    image_path_key: str = "image_path"
    image_segment_id_key: str = "segment_id"
    # Local semantic alignment should use a retrieval-oriented vision backend.
    # BLIP2 vision-only over `/data/llm_models/blip2-opt-2.7b` separated positive
    # and negative probes better than InternVL2 hidden-state cosine. Validate
    # the chosen backend before wiring it into `common.image_embeddings`.
    image_embedding_model_path: str = "nvidia/Cosmos-Embed1-224p"
    video_embedding_model_path: str = "videoclip"
    uses_provided_image_alignment: bool = False
    device: str | None = None


class VideoImageAlignmentAssessor(BaseMetricAssessor):
    metric_name = "video_image_alignment"
    output_scores = ("segment_frame_alignment_mean", "temporal_coverage_rate")
    config_model = VideoImageAlignmentConfig

    def _assess(self, context: EvaluationContext) -> MetricResult:
        metric_config = cast(
            VideoImageAlignmentConfig,
            self.resolve_metric_config(context) or VideoImageAlignmentConfig(),
        )

        if not has_modality(context, "video") and not (
            context.has_field("video_path")
            or context.has_field("video")
            or context.has_field("video_segment")
            or context.has_field("segments")
            or context.has_field("samples")
        ):
            return MetricResult(
                metric_name=self.metric_name,
                status="not_applicable",
                scores={},
                skipped_scores={score: "dataset does not contain video fields" for score in self.output_scores},
                notices=["dataset does not contain video fields"],
            )
        if not (
            context.has_field("segments")
            or context.has_field("video")
            or context.has_field("video_segment")
            or context.has_field("video_path")
        ):
            raise InvalidMetricInputError("missing video segments or video")
        if not (
            context.has_field("samples")
            or context.has_field("image")
            or context.has_field(metric_config.image_path_key)
        ):
            raise InvalidMetricInputError("missing alignment samples or image")

        alignment_scores = []
        temporal_hits = 0
        segment_count = 0
        missing_sample_segments = 0
        used_embeddings = False
        uses_provided_image_alignment = metric_config.uses_provided_image_alignment

        for row in context.rows:
            segments = self._collect_alignment_segments(row)
            samples = self._collect_alignment_samples(row, metric_config)
            if (
                not uses_provided_image_alignment
                and segments
                and samples
                and any(
                    str(segment.get("segment_id")) == ROW_ALIGNMENT_SEGMENT_ID
                    for segment in segments
                )
            ):
                uses_provided_image_alignment = True
            self._ensure_embeddings(segments, samples, metric_config)
            samples_by_segment: dict[str, list[dict[str, Any]]] = {}
            unlabeled_samples: list[dict[str, Any]] = []
            for sample in samples:
                segment_id = sample.get(metric_config.image_segment_id_key)
                if segment_id is None:
                    unlabeled_samples.append(sample)
                    continue
                samples_by_segment.setdefault(str(segment_id), []).append(sample)

            for segment in segments:
                segment_id = str(
                    segment.get("segment_id")
                    or segment.get("id")
                    or segment.get("segment_idx")
                    or ""
                )
                if not segment_id:
                    continue
                segment_count += 1
                segment_samples = samples_by_segment.get(segment_id, [])
                if not segment_samples and len(segments) == 1:
                    segment_samples = unlabeled_samples
                if not segment_samples:
                    missing_sample_segments += 1
                    continue
                if (
                    not uses_provided_image_alignment
                    and self._segment_temporally_covered(segment, segment_samples)
                ):
                    temporal_hits += 1
                score = self._segment_alignment_score(segment, segment_samples, metric_config)
                if score is not None:
                    alignment_scores.append(score)
                    used_embeddings = True

        if segment_count == 0:
            raise InvalidMetricInputError("segments are empty")

        notices = []
        status = "passed"
        skipped_scores = {}
        if uses_provided_image_alignment:
            notices.append(ROW_IMAGE_ALIGNMENT_NOTICE)
        if missing_sample_segments:
            status = "warning"
            notices.append(f"{missing_sample_segments} segments have no sampling frames")
        if not alignment_scores:
            status = "warning" if status == "passed" else status
            notices.append(
                "Cosmos/video-image embeddings could not be computed"
            )
            skipped_scores["segment_frame_alignment_mean"] = (
                "Cosmos/video-image embeddings could not be computed"
            )

        scores = {}
        if not uses_provided_image_alignment:
            scores["temporal_coverage_rate"] = round_score(temporal_hits / segment_count)
        if alignment_scores:
            scores["segment_frame_alignment_mean"] = mean(alignment_scores)

        return MetricResult(
            metric_name=self.metric_name,
            status=status,
            scores=scores,
            skipped_scores=skipped_scores,
            notices=notices,
            metadata={
                "segment_count": segment_count,
                "missing_sample_segment_count": missing_sample_segments,
                "used_embeddings": used_embeddings,
                "used_provided_image_alignment": uses_provided_image_alignment,
            },
        )

    @classmethod
    def _collect_alignment_segments(
        cls,
        row: dict[str, Any]
    ) -> list[dict[str, Any]]:
        segments = [
            dict(item)
            for item in as_list(row.get("segments"))
            if isinstance(item, dict)
        ]
        if segments:
            return segments
        video_segment = row.get("video_segment")
        if video_segment is None:
            return []
        if isinstance(video_segment, dict):
            segment = dict(video_segment)
        else:
            segment = {"video_segment": video_segment}
        segment.setdefault("segment_id", ROW_ALIGNMENT_SEGMENT_ID)
        return [segment]

    @classmethod
    def _collect_alignment_samples(
        cls,
        row: dict[str, Any],
        metric_config: VideoImageAlignmentConfig,
    ) -> list[dict[str, Any]]:
        if not metric_config.uses_provided_image_alignment:
            samples = [
                dict(item)
                for item in as_list(row.get("samples"))
                if isinstance(item, dict)
            ]
            if samples:
                return samples

        image_values = as_list(row.get("image"))
        if row.get(metric_config.image_path_key) is not None:
            image_values.extend(as_list(row.get(metric_config.image_path_key)))

        collected_samples = []
        for image_value in image_values:
            if image_value is None:
                continue
            sample = cls._image_value_to_sample(image_value, metric_config)
            sample.setdefault(metric_config.image_segment_id_key, ROW_ALIGNMENT_SEGMENT_ID)
            collected_samples.append(sample)

        return collected_samples

    @staticmethod
    def _image_value_to_sample(
        image_value: Any,
        metric_config: VideoImageAlignmentConfig,
    ) -> dict[str, Any]:
        if not isinstance(image_value, dict):
            return {metric_config.frame_path_key: image_value}

        sample = {}
        if image_value.get(metric_config.sample_embedding_key) is not None:
            sample[metric_config.sample_embedding_key] = image_value[
                metric_config.sample_embedding_key
            ]

        if metric_config.frame_path_key in image_value:
            sample[metric_config.frame_path_key] = image_value[metric_config.frame_path_key]

        if metric_config.frame_path_key not in sample:
            for field_name in (
                metric_config.image_path_key,
                "image",
                "image_path",
            ):
                if field_name in image_value and image_value[field_name] is not None:
                    sample[metric_config.frame_path_key] = image_value[field_name]
                    break
        return sample

    @staticmethod
    def _segment_temporally_covered(
        segment: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> bool:
        start = _segment_start_sec(segment)
        end = _segment_end_sec(segment)

        if start is None or end is None:
            return bool(samples)

        if end < start:
            start, end = end, start

        return any(
            timestamp is not None and start <= timestamp <= end
            for timestamp in (as_float(sample.get("timestamp_sec")) for sample in samples)
        )

    @staticmethod
    def _segment_alignment_score(
        segment: dict[str, Any],
        samples: list[dict[str, Any]],
        metric_config: VideoImageAlignmentConfig,
    ) -> float | None:
        segment_vector = as_vector(segment.get(metric_config.segment_embedding_key))
        sample_vectors = [
            as_vector(sample.get(metric_config.sample_embedding_key))
            for sample in samples
        ]
        sample_vectors = [vector for vector in sample_vectors if vector is not None]
        if segment_vector is None or not sample_vectors:
            return None
        pooled = [
            sum(vector[index] for vector in sample_vectors) / len(sample_vectors)
            for index in range(len(segment_vector))
        ]
        return cosine_similarity(segment_vector, pooled)

    @classmethod
    def _ensure_embeddings(
        cls,
        segments: list[dict[str, Any]],
        samples: list[dict[str, Any]],
        metric_config: VideoImageAlignmentConfig,
    ) -> None:

        segment_inputs = []
        segment_targets = []
        embedding_inputs: list[Any] = []
        sample_targets: list[tuple[dict[str, Any], str, int]] = []

        for segment in segments:
            if as_vector(segment.get(metric_config.segment_embedding_key)) is not None:
                continue
            segment_input = segment.get("video_segment")
            if segment_input is None:
                continue
            segment_inputs.append(segment_input)
            segment_targets.append(segment)

        if segment_inputs:
            try:
                segment_vectors = video_segment_embeddings(
                    segment_inputs,
                    model_repo_or_path=metric_config.video_embedding_model_path,
                    device=metric_config.device,
                )

                for segment, vector in zip(segment_targets, segment_vectors):
                    if vector is not None:
                        segment[metric_config.segment_embedding_key] = vector
            except (ImportError, RuntimeError, ValueError):
                pass

        for sample in samples:
            if as_vector(sample.get(metric_config.sample_embedding_key)) is not None:
                continue
            frame_value = sample.get(metric_config.frame_path_key)
            if frame_value is None:
                continue
            sample_targets.append(
                (sample, metric_config.sample_embedding_key, len(embedding_inputs))
            )
            embedding_inputs.append(frame_value)

        if not embedding_inputs:
            return

        try:
            vectors = image_embeddings(
                embedding_inputs,
                model_repo_or_path=metric_config.image_embedding_model_path,
                device=metric_config.device,
            )
        except (ImportError, RuntimeError, ValueError):
            return

        for target, target_key, index in sample_targets:
            if index < len(vectors) and vectors[index] is not None:
                target[target_key] = vectors[index]

    @staticmethod
    def _sample_frame_index(sample: dict[str, Any]) -> int | None:
        value = as_float(sample.get("frame_index"))
        return int(value) if value is not None else None


def _segment_start_sec(segment: dict[str, Any]) -> float | None:
    timestamp_range = segment.get("timestamp_range") or {}
    return as_float(timestamp_range.get("start_sec") or segment.get("start_sec"))


def _segment_end_sec(segment: dict[str, Any]) -> float | None:
    timestamp_range = segment.get("timestamp_range") or {}
    return as_float(timestamp_range.get("end_sec") or segment.get("end_sec"))
