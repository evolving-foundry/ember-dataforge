from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple, Type

from pydantic import BaseModel

from .context import EvaluationContext
from .results import MetricResult


class ScoreSpec(BaseModel):
    name: str
    required_fields: Tuple[str, ...] = ()
    required_config: Tuple[str, ...] = ()
    optional_upstream_ops: Tuple[str, ...] = ()


class InvalidMetricInputError(Exception):
    """Raised when current dataset fields/content cannot satisfy a metric."""


class BaseMetricAssessor(ABC):
    metric_name: str
    required_fields: Tuple[str, ...] = ()
    optional_fields: Tuple[str, ...] = ()
    output_scores: Tuple[str, ...] = ()
    score_specs: Tuple[ScoreSpec, ...] = ()
    config_model: Type[BaseModel] | None = None

    def assess(self, context: EvaluationContext) -> MetricResult:
        try:
            self.assert_required_fields(context)
            result = self._assess(context)
        except InvalidMetricInputError as exc:
            return self._status_result("failed_precondition", str(exc))
        except Exception as exc:
            return self._status_result("failed", str(exc))
        return result

    def assert_required_fields(self, context: EvaluationContext) -> None:
        missing = context.missing_fields(self.required_fields)
        if missing:
            raise InvalidMetricInputError(
                "missing required fields: " + ", ".join(missing)
            )

    def resolve_metric_config(self, context: EvaluationContext) -> BaseModel | None:
        if self.config_model is None:
            return None
        return self.config_model(**context.config.get_metric_config(self.metric_name))

    def get_config_value(
        self, context: EvaluationContext, key: str, default: Any = None
    ) -> Any:
        metric_config = self.resolve_metric_config(context)
        if metric_config is not None and key in type(metric_config).model_fields:
            return getattr(metric_config, key)
        return context.config.get_metric_value(self.metric_name, key, default)

    def has_config_value(self, context: EvaluationContext, key: str) -> bool:
        value = self.get_config_value(context, key, None)
        if value is None:
            return False
        if isinstance(value, (list, tuple, set, dict, str)) and len(value) == 0:
            return False
        return True

    def missing_score_dependencies(
        self, context: EvaluationContext, spec: ScoreSpec
    ) -> Dict[str, str]:
        missing: Dict[str, str] = {}
        missing_fields = context.missing_fields(spec.required_fields)
        if missing_fields:
            missing[spec.name] = "missing required fields: " + ", ".join(missing_fields)
            return missing

        missing_config = [
            key
            for key in spec.required_config
            if not self.has_config_value(context, key)
        ]
        if missing_config:
            missing[spec.name] = "missing required config: " + ", ".join(missing_config)
        return missing

    def skipped_score_dependencies(self, context: EvaluationContext) -> Dict[str, str]:
        skipped: Dict[str, str] = {}
        for spec in self.score_specs:
            skipped.update(self.missing_score_dependencies(context, spec))
        return skipped

    def compute_declared_scores(
        self,
        context: EvaluationContext,
        calculators: Dict[str, Callable[[], float]],
    ) -> tuple[Dict[str, float], Dict[str, str]]:
        skipped_scores = self.skipped_score_dependencies(context)
        scores: Dict[str, float] = {}
        for spec in self.score_specs:
            if spec.name in skipped_scores:
                continue
            if spec.name not in calculators:
                raise RuntimeError(f"missing score calculator: {spec.name}")
            scores[spec.name] = calculators[spec.name]()
        return scores, skipped_scores

    def _status_result(self, status: str, notice: str) -> MetricResult:
        return MetricResult(
            metric_name=self.metric_name,
            status=status,  # type: ignore[arg-type]
            scores={},
            skipped_scores={score: notice for score in self.output_scores},
            notices=[notice] if notice else [],
        )

    @abstractmethod
    def _assess(self, context: EvaluationContext) -> MetricResult:
        pass
