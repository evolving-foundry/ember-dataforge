from abc import abstractmethod
from typing import Any, Dict, Optional
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class AugmentationOperation(Operation):
    """Abstract data augmentation operation

    Provides a base class for data augmentation operations; subclasses must implement the concrete augmentation logic.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Abstract base class for augmentation operations; subclasses must implement augmentation logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute a data augmentation operation

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Augmented MultimodalDataset
        """
        pass