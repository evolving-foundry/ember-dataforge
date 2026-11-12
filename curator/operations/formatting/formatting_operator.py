from abc import abstractmethod
from typing import Any, Dict, Optional
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class FormattingOperation(Operation):
    """Abstract standardized formatting operation

    Provides a base class for data formatting operations; subclasses must implement the concrete formatting logic.
    """

    operation_type = OperationType.FORMATTING

    PARAMS_SCHEMA = {
        "category": "formatting",
        "description": "Abstract base class for formatting operations; subclasses must implement formatting logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute standardized formatting

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Formatted MultimodalDataset
        """
        pass