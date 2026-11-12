from abc import abstractmethod
from typing import Any, Dict, Optional
import ray
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class FilteringOperation(Operation):
    """Abstract intelligent filtering operation

    Provides a base class for data filtering operations; subclasses must implement the concrete filtering logic.
    """

    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Abstract base class for filtering operations; subclasses must implement filtering logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute intelligent filtering

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Filtered MultimodalDataset
        """
        pass

    @abstractmethod
    def filter(self, dataset):
        """
        Concrete filtering logic

        Args:
            dataset: Input dataset

        Returns:
            Filtered dataset
        """
        pass
