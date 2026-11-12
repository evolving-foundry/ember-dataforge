from abc import abstractmethod
from typing import Any, Dict, Optional, Union
import ray
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class CleaningOperation(Operation):
    """Abstract data cleaning operation

    Provides a base class for data cleaning operations; subclasses must implement the concrete cleaning logic.
    """

    operation_type = OperationType.CLEANING

    PARAMS_SCHEMA = {
        "category": "cleaning",
        "description": "Abstract base class for cleaning operations; subclasses must implement cleaning logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute a data cleaning operation

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Cleaned MultimodalDataset
        """
        pass