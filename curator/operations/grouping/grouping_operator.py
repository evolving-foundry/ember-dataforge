from abc import abstractmethod
from typing import Any, Dict, Optional
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class GroupingOperation(Operation):
    """Abstract data grouping operation

    Provides a base class for data grouping operations; subclasses must implement the concrete grouping logic.
    """

    operation_type = OperationType.GROUPING

    PARAMS_SCHEMA = {
        "category": "grouping",
        "description": "Abstract base class for grouping operations; subclasses must implement grouping logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute data grouping

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Grouped MultimodalDataset
        """
        pass