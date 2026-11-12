from abc import abstractmethod
from typing import Any, Dict, Optional
from curator.core.operation import Operation
from base.dataset import MultimodalDataset
from curator.tracking.tracking import OperationType


class GenerationOperation(Operation):
    """Abstract data generation operation.

    Base class for data generation operations; subclasses must implement the concrete generation logic.
    """

    operation_type = OperationType.GENERATION

    PARAMS_SCHEMA = {
        "category": "generation",
        "description": "Abstract base class for data-generation operations; subclasses must implement specific generation logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute a data generation operation.

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Generated MultimodalDataset
        """
        pass