import os
import shutil
from typing import Any, Dict

from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_cv2_backend


class VideoResizeAspectRatioOperation(AugmentationOperation):
    """Video aspect ratio adjustment

    Adjust the video aspect ratio by adding black bars (letterboxing) to keep the video content intact.
    Supports setting a target aspect ratio range; if the video is already within the range, no adjustment is made.

    Note: this operation requires the opencv-python library.
    Note: because OpenCV may have compatibility issues with Ray distributed execution, test in a local environment first.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Adjust video aspect ratio by adding letterbox padding",
        "params": {
            "video_key": {
                "type": "string",
                "required": False,
                "default": "video_path",
                "description": "Input video path field name"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "output_video_path",
                "description": "Output video path field name"
            },
            "output_folder": {
                "type": "string",
                "required": False,
                "default": "./output/videos/",
                "description": "Output video directory"
            },
            "target_aspect_ratio_range": {
                "type": "array",
                "required": False,
                "default": [1.7, 1.9],
                "description": "Target aspect-ratio range [min, max] (width/height)"
            }
        },
        "example": {
            "video_key": "video_path",
            "output_folder": "./output/videos/",
            "target_aspect_ratio_range": [1.7, 1.9]
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        video_key = self.params.get('video_key', 'video_path')
        output_key = self.params.get('output_key', 'output_video_path')
        output_folder = self.params.get('output_folder', './output/videos/')
        target_aspect_ratio_range = self.params.get('target_aspect_ratio_range', [1.7, 1.9])

        # Use the shared CV2 backend
        cv2_backend = get_cv2_backend()

        def resize_video(row):
            input_path = row.get(video_key)
            if not input_path or not os.path.exists(input_path):
                row[output_key] = None
                return row

            try:
                output_video_path = os.path.join(output_folder, "output_" + os.path.basename(input_path))

                # Open the video using the shared backend
                cap = cv2_backend.VideoCapture(input_path)
                if not cap or not cap.isOpened():
                    row[output_key] = None
                    return row

                os.makedirs(output_folder, exist_ok=True)

                fps = int(cap.get(cv2_backend.CAP_PROP_FPS)) if cap.get(cv2_backend.CAP_PROP_FPS) else 30
                width = int(cap.get(cv2_backend.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2_backend.CAP_PROP_FRAME_HEIGHT))

                original_aspect_ratio = width / height if height != 0 else 0

                target_aspect_ratio = (target_aspect_ratio_range[0] + target_aspect_ratio_range[1]) / 2

                # The current aspect ratio is within the target range; no adjustment is needed
                if target_aspect_ratio_range[0] <= original_aspect_ratio <= target_aspect_ratio_range[1]:
                    shutil.copy(input_path, output_video_path)
                    cap.release()
                    row[output_key] = output_video_path
                    return row

                if original_aspect_ratio > target_aspect_ratio:
                    # Too wide; add black bars at the top and bottom
                    new_height = int(width / target_aspect_ratio)
                    padding_top = (new_height - height) // 2
                    padding_bottom = new_height - height - padding_top
                    padding_left = 0
                    padding_right = 0
                    out_width = width
                    out_height = new_height
                else:
                    # Too tall; add black bars on the left and right
                    new_width = int(height * target_aspect_ratio)
                    padding_left = (new_width - width) // 2
                    padding_right = new_width - width - padding_left
                    padding_top = 0
                    padding_bottom = 0
                    out_width = new_width
                    out_height = height

                fourcc = cv2_backend.VideoWriter_fourcc(*'mp4v')
                out = cv2_backend.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if padding_top > 0 or padding_bottom > 0 or padding_left > 0 or padding_right > 0:
                        padded_frame = cv2_backend.copyMakeBorder(
                            frame,
                            padding_top, padding_bottom,
                            padding_left, padding_right,
                            cv2_backend.BORDER_CONSTANT,
                            value=[0, 0, 0]
                        )
                    else:
                        padded_frame = frame

                    if (out_width, out_height) != (padded_frame.shape[1], padded_frame.shape[0]):
                        padded_frame = cv2_backend.resize(padded_frame, (out_width, out_height))

                    out.write(padded_frame)

                cap.release()
                out.release()

                row[output_key] = output_video_path

            except Exception:
                row[output_key] = None

            return row

        return dataset.map(resize_video)
