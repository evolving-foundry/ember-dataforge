from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from base.dataset import MultimodalDataset
from .config import OpsQualityAssessmentConfig


@dataclass
class EvaluationContext:
    """Sampled view over the current MultimodalDataset.

    The context records what is actually present. Metric assessors decide which
    scores can be computed from these fields.
    """
    
    dataset: MultimodalDataset
    rows: List[Dict[str, Any]]
    schema_fields: Set[str]
    data: Dict[str, List[Any]]
    row_count: int
    config: OpsQualityAssessmentConfig
    executed_operations: Set[str] = field(default_factory=set)

    def has_field(self, field_name: str) -> bool:
        return field_name in self.schema_fields

    def has_any_field(self, field_names: list[str] | tuple[str, ...]) -> bool:
        return any(self.has_field(field_name) for field_name in field_names)

    def missing_fields(self, field_names: tuple[str, ...]) -> list[str]:
        return [field_name for field_name in field_names if not self.has_field(field_name)]

    def values(self, field_name: str) -> list[Any]:
        return self.data.get(field_name, [])


class EvaluationContextBuilder:
    """Build EvaluationContext from a sampled MultimodalDataset snapshot."""

    def __init__(self, config: OpsQualityAssessmentConfig):
        self.config = config

    def build(self, dataset: MultimodalDataset) -> EvaluationContext:
        if not isinstance(dataset, MultimodalDataset):
            raise TypeError("OpsQualityAssessor.evaluate expects a MultimodalDataset")

        sample_size = max(0, int(self.config.sample_size))
        rows = dataset.dataset.limit(sample_size).take_all() if sample_size else []
        return self.build_from_rows(dataset, rows)

    def build_from_rows(
        self,
        dataset: MultimodalDataset,
        rows: List[Dict[str, Any]],
    ) -> EvaluationContext:
        schema_fields = self._schema_fields(dataset, rows)
        data: Dict[str, List[Any]] = {field_name: [] for field_name in schema_fields}
        for row in rows:
            for field_name in schema_fields:
                data.setdefault(field_name, []).append(row.get(field_name))

        row_count = len(rows)

        return EvaluationContext(
            dataset=dataset,
            rows=rows,
            schema_fields=schema_fields,
            data=data,
            row_count=row_count,
            config=self.config,
            executed_operations=set(self.config.executed_operations or []),
        )

    @staticmethod
    def _schema_fields(dataset: MultimodalDataset, rows: List[Dict[str, Any]]) -> Set[str]:
        fields: Set[str] = set()
        try:
            schema = dataset.dataset.schema()
            names = getattr(schema, "names", None)
            if names:
                fields.update(str(name) for name in names)
        except Exception:
            pass

        for row in rows:
            fields.update(str(key) for key in row.keys())

        return fields
