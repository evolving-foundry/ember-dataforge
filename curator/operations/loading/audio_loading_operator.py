from typing import Any, Dict, Optional, List, Tuple, Union
from .loading_operator import LoadingOperation
from ...utils.io.data_loader_factory import DataLoaderFactory
from ...utils.media_backend import get_media_backend
from base.dataset import MultimodalDataset


class AudioLoadingOperation(LoadingOperation):
    """Audio data loading operation"""


    PARAMS_SCHEMA = {
        "category":    "loading",
        "description": "Load audio data from local paths; supports .mp3, .wav, .aac, .flac, .ogg, and similar",
        "params": {
            "data_paths": {
                "type":        "array",
                "items":       {"type": "string"},
                "required":    True,
                "description": "Audio file path list",
            },
        },
        "example": {"data_paths": ["./data/audios/"]},
    }

    def execute(self, dataset: Optional[MultimodalDataset] = None) -> MultimodalDataset:
        """Load audio data"""
        if dataset is not None:
            # If a dataset already exists, return it
            return dataset

        data_paths = self.params.get("data_paths", [])
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if not data_paths:
            raise ValueError("AudioLoadingOperation requires 'data_paths' parameter")

        # Collect all audio files
        import os
        audio_extensions = ['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma']
        audio_paths = []

        for path in data_paths:
            if os.path.isfile(path) and any(path.lower().endswith(ext) for ext in audio_extensions):
                audio_paths.append(path)
            elif os.path.isdir(path):
                for filename in os.listdir(path):
                    if any(filename.lower().endswith(ext) for ext in audio_extensions):
                        audio_paths.append(os.path.join(path, filename))

        if not audio_paths:
            raise ValueError("No audio files found in provided paths")

        # Use the batch-loading path when the mm acceleration backend is available; otherwise keep the original Ray reader path
        backend = get_media_backend()
        if backend.name == "mm_sdk":
            ray_dataset = self._load_with_mm_backend(audio_paths, backend)
        else:
            # Use DataLoaderFactory to handle reading
            ray_dataset = DataLoaderFactory.load_dataset(audio_paths)

        # Add a path column to preserve the original file path (ray.data.read_audio does not retain path information)
        ray_dataset = ray_dataset.add_column(
            "path",
            lambda _: audio_paths
        )
        return MultimodalDataset(dataset=ray_dataset, modalities=["audio"])

    @staticmethod
    def _load_with_mm_backend(audio_paths: List[str], backend) -> Any:
        """mm acceleration path: map_batches batch-loads audio (at most 128 files per batch).

        Row schema matches ray.data.read_audio:
        each row is {'amplitude': float32 (channels, samples), 'sample_rate': int}.
        Bad-file semantics: if mm batch loading fails, fall back per item to the default implementation; only raise and terminate if the default implementation also fails.
        """
        import ray

        dataset = ray.data.from_items([{"audio_path": path} for path in audio_paths])

        def load_batch(batch: Dict[str, Any]) -> Dict[str, List[Any]]:
            paths = [str(path) for path in batch["audio_path"]]
            results: List[Tuple[Any, int]] = backend.load_audio_batch(paths, sr=None)
            return {
                "amplitude": [amplitude for amplitude, _ in results],
                "sample_rate": [int(sample_rate) for _, sample_rate in results],
            }

        return dataset.map_batches(load_batch, batch_size=128, batch_format="numpy")
