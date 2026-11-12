from __future__ import annotations

import re
from typing import Any, List, Optional, cast

from pydantic import BaseModel, Field

from ..base import BaseMetricAssessor, InvalidMetricInputError
from ..context import EvaluationContext
from ..results import MetricResult
from .common import first_available_field, non_empty_text, round_score


class NormalizationPatternRule(BaseModel):
    name: str
    before_regex: str


class TextNormalizationValidityConfig(BaseModel):
    before_text_key: Optional[str] = None
    after_text_key: Optional[str] = None
    pattern_rules: List[NormalizationPatternRule] = Field(default_factory=list)
    use_default_rules: bool = True


DEFAULT_PATTERN_RULES = [
    NormalizationPatternRule(
        name="slash_date",
        before_regex=r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    ),
    NormalizationPatternRule(
        name="repeated_whitespace",
        before_regex=r"[ \t]{2,}",
    ),
    NormalizationPatternRule(
        name="repeated_punctuation",
        before_regex=r"([!?.,;:])\1{1,}",
    ),
    NormalizationPatternRule(
        name="currency_symbol",
        before_regex=r"(?:[$￥¥€£]\s*\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s*(?:usd|eur|rmb|cny|gbp))",
    ),
]


class TextNormalizationValidityAssessor(BaseMetricAssessor):
    metric_name = "text_normalization_validity"
    output_scores = (
        "residual_unstandardized_pattern_rate",
    )
    config_model = TextNormalizationValidityConfig

    def _assess(self, context: EvaluationContext) -> MetricResult:
        metric_config = cast(
            TextNormalizationValidityConfig,
            self.resolve_metric_config(context) or TextNormalizationValidityConfig(),
        )

        before_field = self._resolve_field(
            context,
            metric_config.before_text_key,
            ("source_text", "raw_text", "original_text", "before_text", "input_text"),
            "before",
        )
        after_field = self._resolve_field(
            context,
            metric_config.after_text_key,
            ("normalized_text", "cleaned_text", "after_text", "text"),
            "after",
        )
        rules = list(metric_config.pattern_rules)
        if metric_config.use_default_rules:
            rules.extend(DEFAULT_PATTERN_RULES)
        if not rules:
            raise InvalidMetricInputError("pattern rules are not configured")

        rows = list(zip(context.values(before_field), context.values(after_field)))
        before_pattern_count = 0
        residual_pattern_count = 0

        for before_value, after_value in rows:
            before_text = non_empty_text(before_value)
            after_text = non_empty_text(after_value)
            for rule in rules:
                before_matches = list(re.finditer(rule.before_regex, before_text, re.IGNORECASE))
                if not before_matches:
                    continue
                before_pattern_count += len(before_matches)
                residual_pattern_count += len(
                    list(re.finditer(rule.before_regex, after_text, re.IGNORECASE))
                )

        if before_pattern_count == 0:
            return MetricResult(
                metric_name=self.metric_name,
                status="not_applicable",
                scores={},
                skipped_scores={
                    score: "sample contains no normalizable pattern"
                    for score in self.output_scores
                },
                notices=["sample contains no normalizable pattern"],
                metadata={"before_text_key": before_field, "after_text_key": after_field},
            )

        residual_rate = residual_pattern_count / before_pattern_count
        return MetricResult(
            metric_name=self.metric_name,
            status="passed",
            scores={
                "residual_unstandardized_pattern_rate": round_score(residual_rate),
            },
            metadata={
                "before_text_key": before_field,
                "after_text_key": after_field,
                "normalizable_pattern_count": before_pattern_count,
            },
        )

    @staticmethod
    def _resolve_field(
        context: EvaluationContext,
        configured: str | None,
        candidates: tuple[str, ...],
        role: str,
    ) -> str:
        if configured:
            if context.has_field(configured):
                return configured
            raise InvalidMetricInputError(f"configured {role} text field is missing: {configured}")
        resolved = first_available_field(context, candidates)
        if resolved:
            return resolved
        raise InvalidMetricInputError(f"missing {role} text field")
