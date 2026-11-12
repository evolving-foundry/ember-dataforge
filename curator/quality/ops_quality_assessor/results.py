from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


MetricStatus = Literal[
    "passed",
    "warning",
    "failed",
    "not_applicable",
    "failed_precondition",
]


class MetricResult(BaseModel):
    """Structured in-memory result for one ops-quality metric."""

    metric_name: str
    status: MetricStatus
    scores: Dict[str, float] = Field(default_factory=dict)
    skipped_scores: Dict[str, str] = Field(default_factory=dict)
    notices: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def report_score(self) -> float | Dict[str, float] | None:
        """Return a report-friendly score field without inventing a fake aggregate."""
        if not self.scores:
            return None
        if len(self.scores) == 1:
            return next(iter(self.scores.values()))
        return {name: round(float(value), 4) for name, value in self.scores.items()}

    def build_analysis(self) -> str:
        """Build a concise human-readable explanation for report items."""
        parts: list[str] = []

        if self.scores:
            if len(self.scores) == 1:
                score_name, score_value = next(iter(self.scores.items()))
                parts.append(f"Primary component {score_name}={score_value:.4f}")
            else:
                score_text = ", ".join(
                    f"{score_name}={score_value:.4f}"
                    for score_name, score_value in self.scores.items()
                )
                parts.append(f"Score details: {score_text}")

        if self.skipped_scores:
            skipped_text = ", ".join(
                f"{score_name}: {reason}"
                for score_name, reason in self.skipped_scores.items()
            )
            parts.append(f"Skipped subcomponents: {skipped_text}")

        if self.notices:
            parts.append("Hint: " + " | ".join(self.notices))

        if not parts:
            parts.append("No numeric subcomponents are available; inspect the raw metric_results.")

        if self.status != "passed":
            parts.insert(0, f"status={self.status}")

        return ";".join(parts)

    def to_report_item(
        self,
        *,
        related_operations: list[str] | None = None,
    ) -> Dict[str, Any]:
        return {
            "test_metric": self.metric_name,
            "related_operations": [
                operation_name
                for operation_name in (related_operations or [])
                if operation_name
            ],
            "score": self.report_score(),
            "analysis": self.build_analysis(),
            "status": self.status,
        }


class OpsQualityAssessmentResult(BaseModel):
    """Top-level in-memory assessment result.
    """

    evaluated_at: datetime
    row_count: int
    sample_size: int
    metric_results: Dict[str, MetricResult] = Field(default_factory=dict)

    def to_report_items(
        self,
        *,
        related_operations: list[str] | None = None,
    ) -> list[Dict[str, Any]]:
        return [
            metric_result.to_report_item(related_operations=related_operations)
            for metric_result in self.metric_results.values()
        ]


MetricResult.model_rebuild()
OpsQualityAssessmentResult.model_rebuild()
