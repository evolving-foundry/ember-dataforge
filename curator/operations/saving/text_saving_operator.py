from typing import Any, Dict, Optional
from .saving_operator import SavingOperation
from ...utils.io.data_saver_factory import DataSaverFactory
from base.dataset import MultimodalDataset


class TextSavingOperation(SavingOperation):
    """Text data saving operation"""


    PARAMS_SCHEMA = {
        "category":    "saving",
        "description": "Save a text dataset to the specified path",
        "params": {
            "output_path": {
                "type":        "string",
                "required":    True,
                "description": "Output file or directory path",
            },
            "format": {
                "type":        "string",
                "required":    False,
                "enum":        ["json", "jsonl", "csv", "parquet"],
                "default":     "json",
                "description": "Output format",
            },
        },
        "example": {"output_path": "./data/output", "format": "json"},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """Save text data"""
        output_path = self.params.get("output_path")
        format_type = self.params.get("format", "json")
        
        if not output_path:
            raise ValueError("TextSavingOperation requires 'output_path' parameter")
        
        # Use DataSaverFactory to save
        ray_dataset = dataset.dataset if isinstance(dataset, MultimodalDataset) else dataset
        DataSaverFactory.save_dataset(ray_dataset, output_path, format_type)
        
        # Return the original MultimodalDataset (saving does not modify data)
        return dataset
