from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .quality_metrics import (
    DatasetQualityReport,
    DimensionScore,
    FieldQualityMetrics,
    QualityAssessmentConfig,
    QualityDimension,
    TextFieldMetrics,
)


class DataQualityAssessor:
    """Data quality assessor.

    Compute multi-dimensional quality scores for a MultimodalDataset:
      - Completeness: field fill rate
      - Consistency: type consistency rate
      - Uniqueness: unique-value ratio
      - TextQuality: non-empty text rate
    """

    def __init__(self, config: Optional[QualityAssessmentConfig] = None):
        self.config = config or QualityAssessmentConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def assess(self, dataset, operation_name: str = "unknown") -> DatasetQualityReport:
        """Assess dataset quality and return a DatasetQualityReport.

        Args:
            dataset: MultimodalDataset instance
            operation_name: Name of the current operation (written to the report)
        """
        rows = self._get_sample(dataset)
        if not rows:
            return self._empty_report(operation_name)

        fields = list(rows[0].keys()) if rows else [] # The dataset keys here are fixed by default
        text_fields = self._resolve_text_fields(fields, rows)

        field_metrics = self._compute_field_metrics(rows, fields)
        text_metrics = self._compute_text_metrics(rows, text_fields)
        dim_scores = self._compute_dimension_scores(field_metrics, text_metrics)
        overall = self._compute_overall_score(dim_scores)
        issues = self._detect_issues(field_metrics, dim_scores)
        recommendations = self._generate_recommendations(issues, dim_scores)

        return DatasetQualityReport(
            assessed_at=datetime.now(),
            operation_name=operation_name,
            num_rows=dataset.dataset.count(),
            num_columns=len(fields),
            field_metrics=field_metrics,
            text_field_metrics=text_metrics,
            dimension_scores=dim_scores,
            overall_score=overall,
            issues=issues,
            recommendations=recommendations,
        )

    def compare(
        self,
        before: DatasetQualityReport,
        after: DatasetQualityReport,
    ) -> Dict[str, Any]:
        """Compare two quality reports and return a summary of the differences."""
        result: Dict[str, Any] = {
            "overall_score_delta": after.overall_score - before.overall_score,
            "row_count_delta": after.num_rows - before.num_rows,
            "dimension_deltas": {},
        }
        all_dims = set(before.dimension_scores) | set(after.dimension_scores)
        for dim in all_dims:
            before_score = before.dimension_scores.get(dim)
            after_score = after.dimension_scores.get(dim)
            b = before_score.score if before_score else 0.0
            a = after_score.score if after_score else 0.0
            result["dimension_deltas"][dim] = round(a - b, 4)
        result["new_issues"] = [i for i in after.issues if i not in before.issues]
        result["resolved_issues"] = [i for i in before.issues if i not in after.issues]
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_empty(value: Any) -> bool:
        """Robust emptiness check: covers None / str / dict / list / tuple / numpy ndarray.

        Ray ``Dataset.map_batches`` serializes list/str columns as numpy ``ndarray``,
        and directly evaluating ``v == ''`` triggers numpy's multi-element truth-value ambiguity. This helper uses an explicit
        ``len`` check instead, avoiding boolean evaluation of numpy arrays.
        """
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        try:
            return len(value) == 0
        except TypeError:
            return False

    def _get_sample(self, dataset) -> List[Dict[str, Any]]:
        """Sample rows from the dataset, up to sample_size rows."""
        try:
            limited = dataset.dataset.limit(self.config.sample_size)
            return limited.take_all()
        except Exception:
            return []

    def _resolve_text_fields(
        self, fields: List[str], rows: List[Dict[str, Any]]
    ) -> List[str]:
        """Determine the list of text fields (user-specified first, otherwise auto-detect)."""
        if self.config.text_fields:
            return [f for f in self.config.text_fields if f in fields]
        # Auto-detection: field values are mostly of type str
        text_fields: List[str] = []
        for field in fields:
            values = [r.get(field) for r in rows if r.get(field) is not None]
            if not values:
                continue
            str_count = sum(1 for v in values if isinstance(v, str))
            if str_count / len(values) >= 0.5:
                text_fields.append(field)
        return text_fields

    def _compute_field_metrics(
        self, rows: List[Dict[str, Any]], fields: List[str]
    ) -> List[FieldQualityMetrics]:
        metrics = []
        total = len(rows)
        for field in fields:
            values = [r.get(field) for r in rows]
            null_count = sum(1 for v in values if self._is_empty(v))
            null_rate = null_count / total if total > 0 else 0.0

            non_null = [v for v in values if not self._is_empty(v)]
            unique_rate = (
                len(set(str(v) for v in non_null)) / len(non_null)
                if non_null
                else 0.0
            )

            # Type consistency rate: proportion of the most common type
            type_counts: Dict[str, int] = {}
            for v in non_null:
                t = type(v).__name__
                type_counts[t] = type_counts.get(t, 0) + 1
            if type_counts:
                dominant_count = max(type_counts.values())
                type_consistency_rate = dominant_count / len(non_null)
            else:
                type_consistency_rate = 1.0

            metrics.append(
                FieldQualityMetrics(
                    field_name=field,
                    null_rate=round(null_rate, 4),
                    unique_rate=round(min(unique_rate, 1.0), 4),
                    type_consistency_rate=round(type_consistency_rate, 4),
                    sample_invalid_values=[
                        v for v in values if self._is_empty(v)
                    ][:5],
                )
            )
        return metrics

    def _compute_text_metrics(
        self, rows: List[Dict[str, Any]], text_fields: List[str]
    ) -> List[TextFieldMetrics]:
        metrics = []
        for field in text_fields:
            values = [r.get(field) for r in rows]
            texts = [v for v in values if isinstance(v, str)]
            if not texts:
                continue
            lengths = [len(t) for t in texts]
            empty_count = sum(1 for t in texts if t.strip() == "")
            sorted_lengths = sorted(lengths)
            n = len(sorted_lengths)

            def percentile(p: float) -> float:
                idx = int(n * p / 100)
                idx = min(idx, n - 1)
                return float(sorted_lengths[idx])

            metrics.append(
                TextFieldMetrics(
                    field_name=field,
                    avg_length=round(statistics.mean(lengths), 2),
                    min_length=min(lengths),
                    max_length=max(lengths),
                    length_percentiles={
                        "p50": percentile(50),
                        "p90": percentile(90),
                        "p99": percentile(99),
                    },
                    empty_rate=round(empty_count / len(texts), 4),
                )
            )
        return metrics

    def _compute_dimension_scores(
        self,
        field_metrics: List[FieldQualityMetrics],
        text_metrics: List[TextFieldMetrics],
    ) -> Dict[str, DimensionScore]:
        scores: Dict[str, DimensionScore] = {}
        enabled = self.config.enabled_dimensions

        if QualityDimension.COMPLETENESS in enabled and field_metrics:
            score = statistics.mean(1 - m.null_rate for m in field_metrics)
            scores[QualityDimension.COMPLETENESS] = DimensionScore(
                dimension=QualityDimension.COMPLETENESS,
                score=round(score, 4),
                contributing_fields=[m.field_name for m in field_metrics],
            )

        if QualityDimension.CONSISTENCY in enabled and field_metrics:
            score = statistics.mean(m.type_consistency_rate for m in field_metrics)
            scores[QualityDimension.CONSISTENCY] = DimensionScore(
                dimension=QualityDimension.CONSISTENCY,
                score=round(score, 4),
                contributing_fields=[m.field_name for m in field_metrics],
            )

        if QualityDimension.UNIQUENESS in enabled and field_metrics:
            score = statistics.mean(m.unique_rate for m in field_metrics)
            scores[QualityDimension.UNIQUENESS] = DimensionScore(
                dimension=QualityDimension.UNIQUENESS,
                score=round(score, 4),
                contributing_fields=[m.field_name for m in field_metrics],
            )

        if QualityDimension.TEXT_QUALITY in enabled and text_metrics:
            score = statistics.mean(1 - m.empty_rate for m in text_metrics)
            scores[QualityDimension.TEXT_QUALITY] = DimensionScore(
                dimension=QualityDimension.TEXT_QUALITY,
                score=round(score, 4),
                contributing_fields=[m.field_name for m in text_metrics],
            )

        return scores

    def _compute_overall_score(self, dim_scores: Dict[str, DimensionScore]) -> float:
        """Weighted aggregation of scores across dimensions."""
        weights = self.config.dimension_weights
        total_weight = 0.0
        weighted_sum = 0.0
        for dim_key, score_obj in dim_scores.items():
            # dim_key may be an enum instance or a string
            key = dim_key.value if hasattr(dim_key, "value") else str(dim_key)
            w = weights.get(key, 0.0)
            weighted_sum += score_obj.score * w
            total_weight += w
        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 4)

    def _detect_issues(
        self,
        field_metrics: List[FieldQualityMetrics],
        dim_scores: Dict[str, DimensionScore],
    ) -> List[str]:
        issues: List[str] = []
        threshold = self.config.null_threshold
        for m in field_metrics:
            if m.null_rate > threshold:
                issues.append(
                    f"Field '{m.field_name}' has a high null rate: {m.null_rate:.1%} (threshold {threshold:.1%})"
                )
            if m.type_consistency_rate < 0.9:
                issues.append(
                    f"Field '{m.field_name}' type mismatch rate: {1 - m.type_consistency_rate:.1%}"
                )
        for dim_key, score_obj in dim_scores.items():
            if score_obj.score < 0.5:
                key = dim_key.value if hasattr(dim_key, "value") else str(dim_key)
                issues.append(f"Dimension '{key}' score is too low: {score_obj.score:.2f}")
        return issues

    def _generate_recommendations(
        self,
        issues: List[str],
        dim_scores: Dict[str, DimensionScore],
    ) -> List[str]:
        recommendations: List[str] = []
        if any("Null rate" in i for i in issues):
            recommendations.append("Use the invalid_data_removal operation to remove null records")
        if any("Type mismatch" in i for i in issues):
            recommendations.append("Use the format_standardization operation to normalize field types")
        # Uniqueness too low suggests deduplication
        for dim_key, score_obj in dim_scores.items():
            key = dim_key.value if hasattr(dim_key, "value") else str(dim_key)
            if key == "uniqueness" and score_obj.score < 0.7:
                recommendations.append(
                    "Use the exact_duplicate_removal operation to remove duplicate records"
                )
        if not recommendations:
            recommendations.append("Data quality is good; no additional processing is required")
        return recommendations

    def _empty_report(self, operation_name: str) -> DatasetQualityReport:
        return DatasetQualityReport(
            assessed_at=datetime.now(),
            operation_name=operation_name,
            num_rows=0,
            num_columns=0,
            overall_score=0.0,
            issues=["The dataset is empty and cannot be evaluated"],
            recommendations=[],
        )
