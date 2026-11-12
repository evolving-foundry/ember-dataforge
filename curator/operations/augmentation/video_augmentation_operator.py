import os
import random

from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...utils.lazy_imports import get_cv2_backend


class VideoAugmentationOperation(AugmentationOperation):
    """Video data augmentation operation"""


    PARAMS_SCHEMA = {
        "category":    "augmentation",
        "description": "Video data augmentation",
        "params":      {
            "video_key": {
                "type": "string",
                "required": False,
                "default": "video_path",
                "description": "Input video field name; supports video paths or frame sequences"
            },
            "flip_horizontal_prob": {
                "type": "number",
                "required": False,
                "default": 0.3,
                "description": "Horizontal flip probability"
            },
            "flip_vertical_prob": {
                "type": "number",
                "required": False,
                "default": 0.1,
                "description": "Vertical flip probability"
            },
            "rotation_prob": {
                "type": "number",
                "required": False,
                "default": 0.2,
                "description": "Rotation probability"
            },
            "num_augmentations": {
                "type": "integer",
                "required": False,
                "default": 1,
                "description": "Number of augmented samples generated per video"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "augmented_video_data",
                "description": "Output augmented-video field name"
            }
        },
        "example":     {
            "video_key": "video_path",
            "flip_horizontal_prob": 0.3,
            "rotation_prob": 0.2,
            "num_augmentations": 1,
            "output_key": "augmented_video_data"
        },
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        video_key = self.params.get("video_key", "video_path")
        flip_horizontal_prob = self.params.get("flip_horizontal_prob", 0.3)
        flip_vertical_prob = self.params.get("flip_vertical_prob", 0.1)
        rotation_prob = self.params.get("rotation_prob", 0.2)
        num_augmentations = max(1, int(self.params.get("num_augmentations", 1)))
        output_key = self.params.get("output_key", "augmented_video_data")

        cv2_backend = get_cv2_backend()
        cv2 = cv2_backend.get_cv2()

        def load_frames(video_data):
            if video_data is None:
                return []

            if isinstance(video_data, str):
                if not cv2 or not os.path.exists(video_data):
                    return []

                cap = cv2_backend.VideoCapture(video_data)
                if not cap or not cap.isOpened():
                    return []

                frames = []
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                finally:
                    cap.release()
                return frames

            if hasattr(video_data, "ndim") and getattr(video_data, "ndim", 0) >= 4:
                return list(video_data)

            if isinstance(video_data, (list, tuple)):
                return list(video_data)

            return []

        def augment_frames(frames):
            if not frames:
                return []

            apply_hflip = random.random() < flip_horizontal_prob
            apply_vflip = random.random() < flip_vertical_prob
            apply_rotation = bool(cv2) and random.random() < rotation_prob
            angle = random.randint(1, 359) if apply_rotation else 0

            augmented_frames = []
            for frame in frames:
                augmented_frame = frame

                if apply_hflip:
                    augmented_frame = cv2_backend.flip(augmented_frame, 1)

                if apply_vflip:
                    augmented_frame = cv2_backend.flip(augmented_frame, 0)

                if apply_rotation:
                    height, width = augmented_frame.shape[:2]
                    center = (width // 2, height // 2)
                    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    augmented_frame = cv2.warpAffine(augmented_frame, matrix, (width, height))

                augmented_frames.append(augmented_frame)

            return augmented_frames

        def augment_video(row):
            try:
                frames = load_frames(row.get(video_key))
                if not frames:
                    row[output_key] = []
                    return row

                augmented_videos = [augment_frames(frames) for _ in range(num_augmentations)]
                row[output_key] = augmented_videos[0] if num_augmentations == 1 else augmented_videos
            except Exception:
                row[output_key] = []

            return row

        return dataset.map(augment_video)
