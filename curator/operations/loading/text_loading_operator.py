from typing import Any, Dict, Optional, List, Union
from .loading_operator import LoadingOperation
from ...utils.io.data_loader_factory import DataLoaderFactory
from base.dataset import MultimodalDataset


class TextLoadingOperation(LoadingOperation):
    """Text data loading operation"""


    PARAMS_SCHEMA = {
        "category":    "loading",
        "description": "Load text data from local paths; supports .txt, .json, .jsonl, and .csv",
        "params": {
            "data_paths": {
                "type":        "array",
                "items":       {"type": "string"},
                "required":    True,
                "description": "Text file path list; supports .txt, .json, .jsonl, and .csv",
            },
        },
        "example": {"data_paths": ["./data/sample.json"]},
    }

    def execute(self, dataset: Optional[MultimodalDataset] = None) -> MultimodalDataset:
        """Load text data and return a MultimodalDataset"""
        if dataset is not None:
            # If a dataset already exists, return it
            return dataset

        data_paths = self.params.get("data_paths", [])
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if not data_paths:
            raise ValueError("TextLoadingOperation requires 'data_paths' parameter")

        # Filter out text files
        text_paths = [path for path in data_paths
                     if any(path.lower().endswith(ext) for ext in ['.txt', '.json', '.jsonl', '.csv'])]

        if not text_paths:
            raise ValueError("No text files found in provided paths")

        # Use DataLoaderFactory to load the Ray Dataset
        ray_dataset = DataLoaderFactory.load_dataset(text_paths)

        # Wrap it in a MultimodalDataset and return it
        return MultimodalDataset(
            dataset=ray_dataset,
            modalities=["text"]
        )