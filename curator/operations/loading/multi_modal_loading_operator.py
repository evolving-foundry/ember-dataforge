from typing import Any, Dict, Optional, List, Union
from .loading_operator import LoadingOperation
from ...utils.io.data_loader_factory import DataLoaderFactory
from base.dataset import MultimodalDataset


class MultiModalLoadingOperation(LoadingOperation):
    """Multimodal data loading operation"""


    PARAMS_SCHEMA = {
        "category":    "loading",
        "description": "Load multimodal data, auto-detect file types, or filter by specified modality",
        "params": {
            "data_paths": {
                "type":        "array",
                "items":       {"type": "string"},
                "required":    True,
                "description": "Data file path list",
            },
            "modalities": {
                "type":        "array",
                "items":       {"type": "string", "enum": ["text", "image", "audio", "video"]},
                "required":    False,
                "description": "Modality types to load; auto-detects when empty",
            },
        },
        "example": {"data_paths": ["./data/sample.json"], "modalities": ["text"]},
    }

    def execute(self, dataset: Optional[MultimodalDataset] = None) -> MultimodalDataset:
        """Load multimodal data"""
        if dataset is not None:
            # If a dataset already exists, return it
            return dataset
            
        data_paths = self.params.get("data_paths", [])
        modalities = self.params.get("modalities", [])
        
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
        if not data_paths:
            raise ValueError("MultiModalLoadingOperation requires 'data_paths' parameter")
        
        # If no modalities are specified, use DataLoaderFactory for automatic detection
        if not modalities:
            ray_dataset = DataLoaderFactory.load_dataset(data_paths)
            return MultimodalDataset(ray_dataset)
        else:
            # If modalities are specified, filter paths by modality
            filtered_paths = []
            modality_extensions = {
                "text": [".txt", ".json", ".jsonl", ".csv"],
                "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif", ".svg"],
                "video": [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"],
                "audio": [".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma"]
            }
            
            for modality in modalities:
                if modality in modality_extensions:
                    extensions = modality_extensions[modality]
                    modality_paths = [path for path in data_paths 
                                     if any(path.lower().endswith(ext) for ext in extensions)]
                    filtered_paths.extend(modality_paths)
            
            if not filtered_paths:
                raise ValueError("No valid data paths found for specified modalities")

            ray_dataset = DataLoaderFactory.load_dataset(filtered_paths)
            # If videos are loaded, add a video_path column to preserve the original file path
            if "video" in modalities:
                ray_dataset = ray_dataset.add_column(
                    "video_path",
                    lambda _: filtered_paths
                )
            return MultimodalDataset(ray_dataset)