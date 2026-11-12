from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, Field

from ..base import BaseMetricAssessor, InvalidMetricInputError
from ..context import EvaluationContext
from ..results import MetricResult
from .common import (
    as_list,
    cosine_similarity,
    non_empty_text,
    round_score,
    text_embeddings,
    tokenize,
)
from ....utils.video_ops_utils import load_text_entries


class CoherenceScoreConfig(BaseModel):
    duplicate_similarity_threshold: float = 0.9
    duplicate_token_overlap_threshold: float = 0.6
    embedding_model_path: str = "BAAI/bge-m3"
    device: str | None = None
    text_source_priority: list[str] = Field(
        default_factory=lambda: [
            "subtitle_entries",
            "text_signals",
            "description_units",
        ]
    )


class CoherenceScoreAssessor(BaseMetricAssessor):
    metric_name = "coherence_score"
    output_scores = ("coherence_score", "duplicate_description_rate")
    config_model = CoherenceScoreConfig

    def _assess(self, context: EvaluationContext) -> MetricResult:
        metric_config = cast(
            CoherenceScoreConfig,
            self.resolve_metric_config(context) or CoherenceScoreConfig(),
        )

        if not context.has_field("segments"):
            raise InvalidMetricInputError("missing segments")
        if not context.has_any_field(
            ("video", "subtitle_uri", "text_signals", "description_units")
        ):
            raise InvalidMetricInputError(
                "missing video.subtitle_uri, text_signals, or description_units"
            )

        pair_count = 0
        embedding_pair_count = 0
        duplicate_count = 0
        semantic_candidate_count = 0
        source_counts: dict[str, int] = {}
        for row in context.rows:
            text_units, text_source = self._ordered_text_units(row, metric_config)
            if text_source:
                source_counts[text_source] = source_counts.get(text_source, 0) + 1
            unit_texts = self._unit_texts(text_units)
            vectors = self._text_embeddings(text_units, metric_config)
            if len(text_units) < 2:
                continue
            for left_text, right_text, left_vector, right_vector in zip(
                unit_texts,
                unit_texts[1:],
                vectors,
                vectors[1:],
            ):
                pair_count += 1
                if left_vector is None or right_vector is None:
                    continue
                embedding_pair_count += 1
                semantic_similarity = cosine_similarity(left_vector, right_vector)
                token_overlap = self._token_overlap(left_text, right_text)
                if semantic_similarity >= metric_config.duplicate_similarity_threshold:
                    semantic_candidate_count += 1
                if (
                    semantic_similarity >= metric_config.duplicate_similarity_threshold
                    and token_overlap >= metric_config.duplicate_token_overlap_threshold
                ):
                    duplicate_count += 1

        if pair_count == 0:
            return MetricResult(
                metric_name=self.metric_name,
                status="not_applicable",
                scores={},
                skipped_scores={
                    score: "not enough adjacent text units"
                    for score in self.output_scores
                },
                notices=["not enough adjacent text units"],
            )
        if embedding_pair_count == 0:
            return MetricResult(
                metric_name=self.metric_name,
                status="failed_precondition",
                scores={},
                skipped_scores={
                    score: "text embeddings could not be computed"
                    for score in self.output_scores
                },
                notices=["text embeddings could not be computed"],
                metadata={
                    "adjacent_pair_count": pair_count,
                    "text_source_counts": source_counts,
                },
            )

        duplicate_rate = round_score(duplicate_count / embedding_pair_count)
        notices = []
        status = "passed"
        if embedding_pair_count < pair_count:
            status = "warning"
            notices.append(
                f"{pair_count - embedding_pair_count} adjacent text pairs were missing embeddings"
            )
        return MetricResult(
            metric_name=self.metric_name,
            status=status,
            scores={
                "coherence_score": round_score(1.0 - duplicate_rate),
                "duplicate_description_rate": duplicate_rate,
            },
            notices=notices,
            metadata={
                "adjacent_pair_count": pair_count,
                "embedding_pair_count": embedding_pair_count,
                "semantic_candidate_count": semantic_candidate_count,
                "duplicate_similarity_threshold": metric_config.duplicate_similarity_threshold,
                "duplicate_token_overlap_threshold": (
                    metric_config.duplicate_token_overlap_threshold
                ),
                "embedding_model_path": metric_config.embedding_model_path,
                "text_source_priority": list(metric_config.text_source_priority),
                "text_source_counts": source_counts,
            },
        )

    @classmethod
    def _ordered_text_units(
        cls,
        row: dict[str, Any],
        metric_config: CoherenceScoreConfig,
    ) -> tuple[list[dict[str, Any]], str | None]:
        source_builders = {
            "subtitle_entries": cls._ordered_subtitle_entries,
            "text_signals": cls._ordered_text_signals,
            "description_units": cls._ordered_description_units,
        }
        priority = list(metric_config.text_source_priority) or [
            "subtitle_entries",
            "text_signals",
            "description_units",
        ]
        for source_name in priority:
            builder = source_builders.get(source_name)
            if builder is None:
                continue
            units = [
                unit
                for unit in builder(row)
                if non_empty_text(cls._unit_text(unit, source_name))
            ]
            if units:
                return units, source_name
        return [], None

    @classmethod
    def _ordered_subtitle_entries(cls, row: dict[str, Any]) -> list[dict[str, Any]]:
        video = row.get("video") if isinstance(row.get("video"), dict) else {}
        subtitle_uri = row.get("subtitle_uri") or video.get("subtitle_uri")
        if not subtitle_uri:
            return []
        try:
            entries = load_text_entries(subtitle_uri)
        except (OSError, KeyError, TypeError, ValueError):
            return []

        subtitle_entries = []
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue
            text = non_empty_text(entry.get("text") or entry.get("subtitle"))
            if not text:
                continue
            start_sec = entry.get("start_sec")
            end_sec = entry.get("end_sec", start_sec)
            subtitle_entries.append(
                {
                    "text_signal_id": entry.get("text_signal_id")
                    or entry.get("id")
                    or f"subtitle_{index:06d}",
                    "timestamp_range": {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                    },
                    "text": text,
                }
            )
        subtitle_entries.sort(
            key=lambda unit: (
                cls._unit_start_sec(unit),
                str(unit.get("text_signal_id") or ""),
            )
        )
        return subtitle_entries

    @classmethod
    def _ordered_text_signals(cls, row: dict[str, Any]) -> list[dict[str, Any]]:
        segment_order = cls._segment_order(row)
        text_signals = [
            unit for unit in as_list(row.get("text_signals")) if isinstance(unit, dict)
        ]
        text_signals.sort(
            key=lambda unit: (
                cls._unit_start_sec(unit),
                segment_order.get(str(unit.get("segment_id")), len(segment_order)),
                str(unit.get("text_signal_id") or unit.get("signal_id") or ""),
            )
        )
        return text_signals

    @classmethod
    def _ordered_description_units(cls, row: dict[str, Any]) -> list[dict[str, Any]]:
        segment_order = cls._segment_order(row)
        description_units = [
            unit for unit in as_list(row.get("description_units")) if isinstance(unit, dict)
        ]
        description_units.sort(
            key=lambda unit: (
                segment_order.get(str(unit.get("segment_id")), len(segment_order)),
                str(unit.get("description_unit_id") or unit.get("description_id") or ""),
            )
        )
        return description_units

    @classmethod
    def _text_embeddings(
        cls,
        text_units: list[dict[str, Any]],
        metric_config: CoherenceScoreConfig,
    ) -> list[list[float] | None]:
        if not text_units:
            return []
        texts = cls._unit_texts(text_units)
        try:
            return text_embeddings(
                texts,
                model_repo_or_path=metric_config.embedding_model_path,
                device=metric_config.device,
            )
        except (ImportError, RuntimeError, ValueError):
            return [None for _unit in text_units]

    @classmethod
    def _unit_texts(cls, text_units: list[dict[str, Any]]) -> list[str]:
        return [non_empty_text(cls._unit_text(unit)) for unit in text_units]

    @staticmethod
    def _token_overlap(left_text: str, right_text: str) -> float:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "been",
            "but",
            "can",
            "for",
            "had",
            "has",
            "have",
            "i",
            "in",
            "is",
            "it",
            "m",
            "of",
            "on",
            "or",
            "re",
            "s",
            "so",
            "that",
            "the",
            "this",
            "to",
            "ve",
            "was",
            "were",
            "with",
            "you",
            "your",
        }
        left_tokens = {token for token in tokenize(left_text) if token not in stopwords}
        right_tokens = {token for token in tokenize(right_text) if token not in stopwords}
        if not left_tokens or not right_tokens:
            return 0.0
        return round_score(
            len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
        )

    @staticmethod
    def _unit_text(unit: dict[str, Any], source_name: str | None = None) -> Any:
        if source_name == "text_signals":
            return unit.get("text") or unit.get("summary")
        if source_name == "description_units":
            return unit.get("description_text") or unit.get("summary")
        return unit.get("text") or unit.get("description_text") or unit.get("summary")

    @staticmethod
    def _unit_start_sec(unit: dict[str, Any]) -> float:
        timestamp_range = unit.get("timestamp_range") or {}
        try:
            return float(timestamp_range.get("start_sec") or unit.get("start_sec") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _segment_order(row: dict[str, Any]) -> dict[str, int]:
        segments = [
            segment for segment in as_list(row.get("segments")) if isinstance(segment, dict)
        ]

        def start_sec(segment: dict[str, Any]) -> float:
            return CoherenceScoreAssessor._unit_start_sec(segment)

        ordered = sorted(segments, key=start_sec)
        result = {}
        for index, segment in enumerate(ordered):
            segment_id = segment.get("segment_id") or segment.get("id")
            if segment_id is not None:
                result[str(segment_id)] = index
        return result
