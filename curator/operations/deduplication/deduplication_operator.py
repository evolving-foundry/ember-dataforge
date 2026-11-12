from abc import abstractmethod
from typing import Any, Dict, Optional
import ray
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class DeduplicationOperation(Operation):
    """Abstract efficient deduplication operation

    Provides a base class for data deduplication operations; subclasses must implement the concrete deduplication logic.
    """

    operation_type = OperationType.DEDUPLICATION

    PARAMS_SCHEMA = {
        "category": "deduplication",
        "description": "Abstract base class for deduplication operations; subclasses must implement deduplication logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute efficient deduplication

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Deduplicated MultimodalDataset
        """
        pass

    @abstractmethod
    def deduplicate(self, dataset):
        """
        Concrete deduplication logic

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Deduplicated MultimodalDataset
        """
        pass