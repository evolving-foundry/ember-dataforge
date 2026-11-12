from abc import abstractmethod
from typing import Any, Dict, Optional
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class AggregationOperation(Operation):
    """Abstract data aggregation operation

    Provides a base class for data aggregation operations; subclasses must implement the concrete aggregation logic.
    """

    operation_type = OperationType.AGGREGATION

    PARAMS_SCHEMA = {
        "category": "aggregation",
        "description": "Abstract base class for aggregation operations; subclasses must implement aggregation logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute a data aggregation operation

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Aggregated MultimodalDataset
        """
        pass