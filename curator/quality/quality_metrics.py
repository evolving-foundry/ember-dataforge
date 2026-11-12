from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QualityDimension(str, Enum):
    COMPLETENESS = "completeness"   # Field fill rate
    CONSISTENCY = "consistency"     # Type consistency rate
    UNIQUENESS = "uniqueness"       # Unique-value ratio
    TEXT_QUALITY = "text_quality"   # Text length / empty rate


class FieldQualityMetrics(BaseModel):
    field_name: str
    null_rate: float                          # [0, 1]
    unique_rate: float                        # [0, 1]
    type_consistency_rate: float              # [0, 1]
    sample_invalid_values: List[Any] = Field(default_factory=list)  # At most 5 invalid-value samples


class TextFieldMetrics(BaseModel):
    field_name: str
    avg_length: float
    min_length: int
    max_length: int
    length_percentiles: Dict[str, float] = Field(default_factory=dict)  # p50/p90/p99
    empty_rate: float


class DimensionScore(BaseModel):
    dimension: QualityDimension
    score: float                              # [0, 1]
    contributing_fields: List[str] = Field(default_factory=list)


class DatasetQualityReport(BaseModel):
    assessed_at: datetime
    operation_name: str
    num_rows: int
    num_columns: int
    field_metrics: List[FieldQualityMetrics] = Field(default_factory=list)
    text_field_metrics: List[TextFieldMetrics] = Field(default_factory=list)
    dimension_scores: Dict[str, DimensionScore] = Field(default_factory=dict)
    overall_score: float                      # Weighted total score [0, 1]
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class QualityAssessmentConfig(BaseModel):
    sample_size: int = 1000
    enabled_dimensions: List[QualityDimension] = Field(
        default_factory=lambda: list(QualityDimension)
    )
    text_fields: List[str] = Field(default_factory=list)  # Empty = auto-detect
    null_threshold: float = 0.1
    dimension_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "completeness": 0.30,
            "consistency": 0.25,
            "uniqueness": 0.15,
            "text_quality": 0.10,
        }
    )
