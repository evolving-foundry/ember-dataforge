from typing import Any, Dict, Optional
from .saving_operator import SavingOperation
from ...utils.io.data_saver_factory import DataSaverFactory
from base.dataset import MultimodalDataset


class ImageSavingOperation(SavingOperation):
    """Image data saving operation"""


    PARAMS_SCHEMA = {
        "category":    "saving",
        "description": "Save an image dataset to the specified path",
        "params": {
            "output_path": {
                "type":        "string",
                "required":    True,
                "description": "Output file or directory path",
            },
            "format": {
                "type":        "string",
                "required":    False,
                "default":     "parquet",
                "description": "Output format (images usually use parquet)",
            },
        },
        "example": {"output_path": "./data/output_images"},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """Save image data"""
        output_path = self.params.get("output_path")
        format_type = self.params.get("format", "parquet")  # Images are usually stored as parquet
        
        if not output_path:
            raise ValueError("ImageSavingOperation requires 'output_path' parameter")
        
        # Use DataSaverFactory to save
        ray_dataset = dataset.dataset if isinstance(dataset, MultimodalDataset) else dataset
        DataSaverFactory.save_dataset(ray_dataset, output_path, format_type)
        
        # Return the original MultimodalDataset (saving does not modify data)
        return dataset  # Save operations usually return the original dataset