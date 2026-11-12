from __future__ import annotations

from data_designer.config import load_config, operator_schema
from data_designer.api import generate
from data_designer.operators.dataset_bridge import rows_from_dataset, wrap_result
from .generation_operator import GenerationOperation


class SyntheticDataGenerationOperation(GenerationOperation):
    """The sole Curator entry point, delegating to the independent generation use case."""

    PARAMS_SCHEMA = operator_schema()

    def __init__(self, params=None):
        super().__init__(params)
        self.config = load_config(self.params)

    def execute(self, dataset):
        if dataset is None:
            raise ValueError("SyntheticDataGenerationOperation requires an input dataset")
        self.result = generate(rows_from_dataset(dataset), self.config)
        return wrap_result(self.result, dataset)
