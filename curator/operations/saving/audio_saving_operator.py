from typing import Any, Dict, Optional
from .saving_operator import SavingOperation
from ...utils.io.data_saver_factory import DataSaverFactory
from base.dataset import MultimodalDataset


class AudioSavingOperation(SavingOperation):
    """Audio data saving operation"""


    PARAMS_SCHEMA = {
        "category":    "saving",
        "description": "Save an audio dataset to the specified path",
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
        "example": {"output_path": "./data/output_audios"},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """Save audio data."""
        output_path = self.params.get("output_path")
        format_type = self.params.get("format", "parquet")

        if not output_path:
            raise ValueError("AudioSavingOperation requires 'output_path' parameter")

        ray_dataset = dataset.dataset if isinstance(dataset, MultimodalDataset) else dataset
        DataSaverFactory.save_dataset(ray_dataset, output_path, format_type)

        # Saving returns the original dataset (unchanged)
        return dataset