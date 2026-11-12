from typing import Any, Dict, Optional
from .common_operator import CommonOperation
from base.dataset import MultimodalDataset


class HelperFunctionOperation(CommonOperation):
    """Helper function operation"""


    PARAMS_SCHEMA = {
        "category":    "common",
        "description": "Helper-function operation",
        "params":      {},
        "example":     {},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        # TODO: Implement helper function logic
        raise NotImplementedError("HelperFunctionOperation.execute() not implemented")