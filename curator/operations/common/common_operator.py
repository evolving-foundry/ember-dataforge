from abc import abstractmethod
from typing import Any, Dict, Optional
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class CommonOperation(Operation):
    """Abstract common operation

    Provides a base class for common/utility operations; subclasses must implement the concrete processing logic.
    """

    operation_type = OperationType.UTILITY

    PARAMS_SCHEMA = {
        "category": "common",
        "description": "Abstract base class for common operations; subclasses must implement their logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute a common operation

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Processed MultimodalDataset
        """
        pass