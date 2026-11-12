from abc import abstractmethod
from typing import Any, Dict, Optional
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class MappingOperation(Operation):
    """Abstract data mapping operation

    Provides a base class for data mapping operations; subclasses must implement the concrete mapping logic.
    """

    operation_type = OperationType.MAPPING

    PARAMS_SCHEMA = {
        "category": "mapping",
        "description": "Abstract base class for mapping operations; subclasses must implement mapping logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute a data mapping operation

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Mapped MultimodalDataset
        """
        pass