from abc import abstractmethod
from typing import Any, Dict, Optional, Union
import ray
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class SavingOperation(Operation):
    """Abstract data saving operation.

    Save operations have side effects (writing to disk), so Pipeline skips Checkpointing for them (they are not idempotent).
    """

    operation_type = OperationType.SAVING
    IS_IO_OPERATION = True

    PARAMS_SCHEMA = {
        "category": "saving",
        "description": "Abstract base class for saving operations; subclasses must implement saving logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute a data saving operation

        Args:
            dataset: Input MultimodalDataset

        Returns:
            The input MultimodalDataset returned unchanged (save operations do not modify data)
        """
        pass