from typing import Any, Dict, Optional
from .aggregation_operator import AggregationOperation
from base.dataset import MultimodalDataset


class EntityAttributeAggregationOperation(AggregationOperation):
    """Entity attribute aggregation operation"""


    PARAMS_SCHEMA = {
        "category":    "aggregation",
        "description": "Entity-attribute aggregation",
        "params":      {},
        "example":     {},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        # TODO: Implement entity attribute aggregation logic
        raise NotImplementedError("EntityAttributeAggregationOperation.execute() not implemented")