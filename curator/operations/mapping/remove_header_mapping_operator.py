from typing import Any, Dict, Optional
from .mapping_operator import MappingOperation
from base.dataset import MultimodalDataset


class RemoveHeaderMappingOperation(MappingOperation):
    """Remove header mapping operation"""


    PARAMS_SCHEMA = {
        "category":    "mapping",
        "description": "Remove text header information",
        "params":      {},
        "example":     {},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        # TODO: Implement remove header mapping logic
        raise NotImplementedError("RemoveHeaderMappingOperation.execute() not implemented")