from abc import abstractmethod
from typing import Any, Dict, Optional
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class SelectionOperation(Operation):
    """Abstract precise selection operation

    Provides a base class for data selection operations; subclasses must implement the concrete selection logic.
    """

    operation_type = OperationType.SELECTION

    PARAMS_SCHEMA = {
        "category": "selection",
        "description": "Abstract base class for data selection operations; subclasses must implement selection logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute precise selection

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Selected MultimodalDataset
        """
        pass