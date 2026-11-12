from typing import Any, Dict, Optional, List, Union
from .loading_operator import LoadingOperation
from ...utils.io.data_loader_factory import DataLoaderFactory
from ...utils.media_backend import get_media_backend
from base.dataset import MultimodalDataset


class ImageLoadingOperation(LoadingOperation):
    """Image data loading operation"""


    PARAMS_SCHEMA = {
        "category":    "loading",
        "description": "Load image data from local paths; supports .jpg, .png, .bmp, .tiff, .webp, .gif, and .svg",
        "params": {
            "data_paths": {
                "type":        "array",
                "items":       {"type": "string"},
                "required":    True,
                "description": "Image file path list",
            },
        },
        "example": {"data_paths": ["./data/images/"]},
    }

    def execute(self, dataset: Optional[MultimodalDataset] = None) -> MultimodalDataset:
        """Load image data.

        Directory expansion is handled uniformly by DataLoaderFactory;
        this method only calls it and wraps the result in a MultimodalDataset.
        """
        if dataset is not None:
            return dataset

        data_paths = self.params.get("data_paths", [])
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if not data_paths:
            raise ValueError("ImageLoadingOperation requires 'data_paths' parameter")

        # Use the batch-decoding path when the mm acceleration backend is available; otherwise keep the original Ray reader path
        backend = get_media_backend()
        if backend.name == "mm_sdk":
            ray_dataset = self._load_with_mm_backend(data_paths, backend)
        else:
            # Directory expansion and modality detection are handled uniformly by DataLoaderFactory
            ray_dataset = DataLoaderFactory.load_dataset(data_paths)
        return MultimodalDataset(dataset=ray_dataset, modalities=["image"])

    @staticmethod
    def _load_with_mm_backend(data_paths: List[str], backend) -> Any:
        """mm acceleration path: map_batches batch-decodes images.

        Row schema matches ray.data.read_images: each row is {'image': RGB ndarray}.
        Bad-file semantics: if mm decoding fails, fall back per item to the default implementation; only raise and terminate if the default implementation also fails.
        """
        import ray

        flat_paths = ImageLoadingOperation._collect_image_paths(data_paths)
        if not flat_paths:
            raise ValueError("No image files found in provided paths")

        dataset = ray.data.from_items([{"image_path": path} for path in flat_paths])

        def decode_batch(batch: Dict[str, Any]) -> Dict[str, List[Any]]:
            return {
                "image": [backend.decode_image(path) for path in batch["image_path"]]
            }

        return dataset.map_batches(decode_batch, batch_size=64, batch_format="numpy")

    @staticmethod
    def _collect_image_paths(data_paths: List[str]) -> List[str]:
        """Expand the directory and collect image file paths (consistent with DataLoaderFactory's expansion logic)."""
        import os

        flat_paths: List[str] = []
        for path in data_paths:
            if os.path.isdir(path):
                flat_paths.extend(DataLoaderFactory._expand_directory(path))
            else:
                flat_paths.append(path)
        return [
            path for path in flat_paths
            if DataLoaderFactory.detect_modality(path) == "image"
        ]
