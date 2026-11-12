from typing import Any, Dict, Optional
from .saving_operator import SavingOperation
from ...utils.io.data_saver_factory import DataSaverFactory
from base.dataset import MultimodalDataset


class MultiModalSavingOperation(SavingOperation):
    """Multimodal data saving operation"""


    PARAMS_SCHEMA = {
        "category":    "saving",
        "description": "Save a multimodal dataset to the specified path",
        "params": {
            "output_path": {
                "type":        "string",
                "required":    True,
                "description": "Output file or directory path",
            },
            "format": {
                "type":        "string",
                "required":    False,
                "description": "Output format (json / parquet / csv, etc.)",
            },
        },
        "example": {"output_path": "./data/processed", "format": "json"},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """Save multimodal data"""
        output_path = self.params.get("output_path")
        format_type = self.params.get("format")
        
        if not output_path:
            raise ValueError("MultiModalSavingOperation requires 'output_path' parameter")
        
        # Use DataSaverFactory to save
        ray_dataset = dataset.dataset if isinstance(dataset, MultimodalDataset) else dataset
        DataSaverFactory.save_dataset(ray_dataset, output_path, format_type)
        
        # Return the original MultimodalDataset (saving does not modify data)
        return dataset
