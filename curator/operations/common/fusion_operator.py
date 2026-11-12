"""Fusion operator that combines multiple operations into one atomic operation to improve performance and consistency"""

import logging
from typing import List, Any, Dict, Optional

logger = logging.getLogger(__name__)
import ray
from ray.data import Dataset
from ...core.operation import Operation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class FusionOperation(Operation):
    """Fusion operation that combines multiple operations into one atomic operation

    This operation allows a series of operations to be combined into a single atomic operation, thereby:
    1. Reducing intermediate data transfer overhead
    2. Improving execution efficiency
    3. Guaranteeing atomicity
    """

    # Set the operation type to UTILITY
    operation_type = OperationType.UTILITY


    PARAMS_SCHEMA = {
        "category":    "common",
        "description": "Fusion operation: merge multiple Operation instances into one atomic operation to reduce intermediate transfer overhead. Operation instances must be provided through the Python API; string-only configuration is not supported",
        "params":      {},
        "example":     {},
    }

    def __init__(self, operations: List[Operation], params: Optional[Dict[str, Any]] = None):
        """
        Initialize the fusion operation

        Args:
            operations: List of operations to fuse
            params: Extra parameters
        """
        super().__init__(params or {})
        if not operations:
            raise ValueError("At least one operation is required for fusion")

        self.operations = operations
        self.name = f"FusionOperation({len(self.operations)} ops)"

        # Collect information about the fused operations
        self.operation_names = [op.name for op in operations]

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute the fusion operation

        Args:
            dataset: Input MultimodalDataset

        Returns:
            Processed MultimodalDataset
        """
        result = dataset

        # Execute all fused operations in order
        for i, operation in enumerate(self.operations):
            logger.info("Executing fused operation %s/%s: %s", i+1, len(self.operations), operation.name)
            result = operation.execute(result)

        return result

    def get_fused_operations(self) -> List[Operation]:
        """Get the list of fused operations"""
        return self.operations.copy()
        
    def get_operation_names(self) -> List[str]:
        """Get the name list of fused operations"""
        return self.operation_names.copy()

    def __repr__(self) -> str:
        return f"{self.name} -> {[op.name for op in self.operations]}"