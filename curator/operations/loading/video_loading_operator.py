from typing import Any, Dict, Optional, List, Union
from .loading_operator import LoadingOperation
from ...utils.io.data_loader_factory import DataLoaderFactory
from ...utils.media_backend import get_media_backend
from base.dataset import MultimodalDataset


class VideoLoadingOperation(LoadingOperation):
    """Video data loading operation"""


    PARAMS_SCHEMA = {
        "category":    "loading",
        "description": "Load video data from local paths; supports .mp4, .avi, .mov, .mkv, .webm, and similar",
        "params": {
            "data_paths": {
                "type":        "array",
                "items":       {"type": "string"},
                "required":    True,
                "description": "Video file path list",
            },
        },
        "example": {"data_paths": ["./data/videos/"]},
    }

    def execute(self, dataset: Optional[MultimodalDataset] = None) -> MultimodalDataset:
        """Load video data"""
        if dataset is not None:
            # If a dataset already exists, return it
            return dataset

        data_paths = self.params.get("data_paths", [])
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if not data_paths:
            raise ValueError("VideoLoadingOperation requires 'data_paths' parameter")

        # Collect all video files
        import os
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
        video_paths = []

        for path in data_paths:
            if os.path.isfile(path) and any(path.lower().endswith(ext) for ext in video_extensions):
                video_paths.append(path)
            elif os.path.isdir(path):
                for filename in os.listdir(path):
                    if any(filename.lower().endswith(ext) for ext in video_extensions):
                        video_paths.append(os.path.join(path, filename))

        if not video_paths:
            raise ValueError("No video files found in provided paths")

        # Use the batch frame-extraction path when the mm acceleration backend is available; otherwise keep the original Ray reader path
        backend = get_media_backend()
        if backend.name == "mm_sdk":
            ray_dataset = self._load_with_mm_backend(video_paths, backend)
        else:
            # Use DataLoaderFactory to handle reading
            ray_dataset = DataLoaderFactory.load_dataset(video_paths)
            # Add a video_path column to preserve the original file path (ray.data.read_videos does not retain path information)
            ray_dataset = ray_dataset.add_column(
                "video_path",
                lambda _: video_paths
            )
        return MultimodalDataset(dataset=ray_dataset, modalities=["video"])

    @staticmethod
    def _load_with_mm_backend(video_paths: List[str], backend) -> Any:
        """mm acceleration path: map_batches batch frame extraction and decoding.

        Row schema matches ray.data.read_videos: each video is emitted frame by frame as
        {'frame': RGB ndarray, 'frame_index': int}, with video_path attached to each row.
        Entries outside the mm support matrix, such as non-MP4 files, fall back per item to the default implementation.
        """
        import ray

        dataset = ray.data.from_items([{"video_path": path} for path in video_paths])

        def decode_batch(batch: Dict[str, Any]) -> Dict[str, List[Any]]:
            frames: List[Any] = []
            frame_indices: List[int] = []
            row_paths: List[str] = []
            for path in batch["video_path"]:
                decoded = backend.decode_video_frames(str(path))
                frames.extend(decoded)
                frame_indices.extend(range(len(decoded)))
                row_paths.extend([str(path)] * len(decoded))
            return {"frame": frames, "frame_index": frame_indices, "video_path": row_paths}

        return dataset.map_batches(decode_batch, batch_size=4, batch_format="numpy")
