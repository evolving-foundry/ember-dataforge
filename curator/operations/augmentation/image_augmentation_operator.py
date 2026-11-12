from typing import Any, Dict, Optional
from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset


class ImageAugmentationOperation(AugmentationOperation):
    """Image data augmentation operation"""


    PARAMS_SCHEMA = {
        "category":    "augmentation",
        "description": "Image data augmentation",
        "params":      {},
        "example":     {},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        # TODO: Implement image data augmentation logic
        raise NotImplementedError("ImageAugmentationOperation.execute() not implemented")