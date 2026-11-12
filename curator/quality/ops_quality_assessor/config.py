from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class OpsQualityAssessmentConfig(BaseModel):
    """Configuration for plugin-based ops-output quality assessment."""

    sample_size: int = 1000
    enabled_metrics: List[str] = Field(
        default_factory=lambda: ["text_noise_contamination"]
    )
    executed_operations: List[str] = Field(default_factory=list)
    metric_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def get_metric_config(self, metric_name: str) -> Dict[str, Any]:
        return dict(self.metric_configs.get(metric_name, {}))

    def get_metric_value(
        self, metric_name: str, key: str, default: Any = None
    ) -> Any:
        metric_config = self.get_metric_config(metric_name)
        if key in metric_config:
            return metric_config[key]
        return default

    def has_config_value(self, metric_name: str, key: str) -> bool:
        value = self.get_metric_value(metric_name, key, None)
        if value is None:
            return False
        if isinstance(value, (list, tuple, set, dict, str)) and len(value) == 0:
            return False
        return True
