from typing import Any, Dict, Optional
from .saving_operator import SavingOperation
from ...utils.io.data_saver_factory import DataSaverFactory
from base.dataset import MultimodalDataset


class VideoSavingOperation(SavingOperation):
    """Video data saving operation"""


    PARAMS_SCHEMA = {
        "category":    "saving",
        "description": "Save a video dataset to the specified path",
        "params": {
            "output_path": {
                "type":        "string",
                "required":    True,
                "description": "Output file or directory path",
            },
            "format": {
                "type":        "string",
                "required":    False,
                "description": "Output format",
            },
        },
        "example": {"output_path": "./data/output_videos"},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """Save video data"""
        output_path = self.params.get("output_path")
        format_type = self.params.get("format", "parquet")  # Videos are usually stored as parquet
        
        if not output_path:
            raise ValueError("VideoSavingOperation requires 'output_path' parameter")
        
        # Use DataSaverFactory to save
        ray_dataset = dataset.dataset if isinstance(dataset, MultimodalDataset) else dataset
        DataSaverFactory.save_dataset(ray_dataset, output_path, format_type)
        
        # Return the original MultimodalDataset (saving does not modify data)
        return dataset
