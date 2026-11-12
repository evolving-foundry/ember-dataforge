from abc import abstractmethod
from typing import Any, Dict, Optional, Union
import ray
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class LoadingOperation(Operation):
    """Abstract data loading operation.

    When used as the first operator in a Pipeline: load data externally;
    when placed in the middle of a Pipeline: pass through the previous operator's output.
    """

    operation_type = OperationType.LOADING
    IS_IO_OPERATION = True

    PARAMS_SCHEMA = {
        "category": "loading",
        "description": "Abstract base class for loading operations; subclasses must implement loading logic",
        "params": {},
        "example": {},
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)

    @abstractmethod
    def execute(self, dataset: Optional[MultimodalDataset] = None) -> MultimodalDataset:
        """
        Execute a data loading operation

        Args:
            dataset: Input dataset; usually None for loading operations

        Returns:
            Loaded MultimodalDataset instance
        """
        pass