import re
from typing import List, Optional, cast

from pydantic import BaseModel, Field

from ....utils.lazy_imports import simple_ngrams, simple_tokenize
from ..base import BaseMetricAssessor, InvalidMetricInputError, ScoreSpec
from ..context import EvaluationContext
from ..results import MetricResult


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


class TextNoiseContaminationConfig(BaseModel):
    text_field: Optional[str] = None
    blocklist: List[str] = Field(default_factory=list)
    n_gram: int = 2


class TextNoiseContaminationAssessor(BaseMetricAssessor):
    metric_name = "text_noise_contamination"
    output_scores = ("ngram_repetition_rate_mean", "blocklist_hit_rate")
    score_specs = (
        ScoreSpec(name="ngram_repetition_rate_mean"),
        ScoreSpec(name="blocklist_hit_rate", required_config=("blocklist",)),
    )
    config_model = TextNoiseContaminationConfig

    def _assess(self, context: EvaluationContext) -> MetricResult:
        metric_config = cast(
            TextNoiseContaminationConfig,
            self.resolve_metric_config(context) or TextNoiseContaminationConfig(),
        )
        text_field = self._resolve_text_field(context, metric_config)
        raw_texts = context.values(text_field)
        texts = ["" if value is None else str(value) for value in raw_texts]
        non_empty_texts = [text for text in texts if text.strip()]
        if not non_empty_texts:
            raise InvalidMetricInputError("all target texts are empty")

        scores, skipped_scores = self.compute_declared_scores(
            context,
            calculators={
                "ngram_repetition_rate_mean": lambda: self._mean_ngram_repetition_rate(
                    non_empty_texts,
                    n_gram=metric_config.n_gram,
                ),
                "blocklist_hit_rate": lambda: self._blocklist_hit_rate(
                    texts,
                    metric_config.blocklist,
                ),
            },
        )
        notices = self._notices_from_skipped_scores(skipped_scores)

        if scores and skipped_scores:
            status = "warning"
        elif scores:
            status = "passed"
        else:
            status = "failed_precondition"

        return MetricResult(
            metric_name=self.metric_name,
            status=status,
            scores=scores,
            skipped_scores=skipped_scores,
            notices=notices,
            metadata={
                "text_field": text_field,
                "evaluated_text_count": len(non_empty_texts),
                "total_sampled_text_count": len(texts),
            },
        )

    @staticmethod
    def _notices_from_skipped_scores(skipped_scores: dict[str, str]) -> list[str]:
        if "blocklist_hit_rate" not in skipped_scores:
            return []
        return [
            "blocklist_filter was not detected or blocklist was not configured; "
            "only n-gram repetition score was computed"
        ]

    def _resolve_text_field(
        self, context: EvaluationContext, metric_config: TextNoiseContaminationConfig
    ) -> str:
        configured = metric_config.text_field
        if configured:
            if context.has_field(str(configured)):
                return str(configured)
            raise InvalidMetricInputError(f"configured text field is missing: {configured}")

        for candidate in (
            "text",
            "input_text",
            "generated_text",
            "dense_caption_text",
            "caption",
            "dense_caption",
            "after_text",
            "output_text",
        ):
            if context.has_field(candidate):
                return candidate

        raise InvalidMetricInputError(
            "missing target text field; expected one of: text, input_text, after_text, "
            "generated_text, dense_caption_text, caption, dense_caption, output_text"
        )

    @staticmethod
    def _mean_ngram_repetition_rate(texts: list[str], n_gram: int) -> float:
        n = max(1, int(n_gram))
        rates = []
        for text in texts:
            tokens = _TOKEN_RE.findall(text.lower()) or simple_tokenize(text)
            ngrams = simple_ngrams(tokens, n)
            if not ngrams:
                rates.append(0.0)
                continue
            total = len(ngrams)
            unique = len(set(ngrams))
            rates.append((total - unique) / total)
        return round(sum(rates) / len(rates), 6) if rates else 0.0

    @staticmethod
    def _blocklist_hit_rate(texts: list[str], blocklist: list[str]) -> float:
        normalized_blocklist = [item.lower() for item in blocklist if item]
        if not normalized_blocklist:
            return 0.0
        hit_count = 0
        for text in texts:
            normalized_text = text.lower()
            if any(item in normalized_text for item in normalized_blocklist):
                hit_count += 1
        return round(hit_count / len(texts), 6) if texts else 0.0
