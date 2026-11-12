import os
import random
from typing import Any, Dict, Optional

from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_cv2_backend


class ImageRandomFlipOperation(AugmentationOperation):
    """Random image flipping

    Apply random flipping and rotation transforms to images, including horizontal flips, vertical flips, and random rotation.
    Multiple augmented samples can be generated, suitable for data augmentation.

    Note: this operation requires the opencv-python library.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Apply random flips and rotations to images",
        "params": {
            "image_key": {
                "type": "string",
                "required": False,
                "default": "image_path",
                "description": "Input image field name"
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
                "default": 0.3,
                "description": "Vertical flip probability"
            },
            "rotation_prob": {
                "type": "number",
                "required": False,
                "default": 0.3,
                "description": "Rotation probability"
            },
            "num_augmentations": {
                "type": "integer",
                "required": False,
                "default": 3,
                "description": "Number of augmented samples generated per image"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "flip_image_data_list",
                "description": "Output augmented-image list field name"
            }
        },
        "example": {
            "image_key": "image_path",
            "flip_horizontal_prob": 0.3,
            "flip_vertical_prob": 0.3,
            "num_augmentations": 3
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        image_key = self.params.get('image_key', 'image_path')
        flip_horizontal_prob = self.params.get('flip_horizontal_prob', 0.3)
        flip_vertical_prob = self.params.get('flip_vertical_prob', 0.3)
        rotation_prob = self.params.get('rotation_prob', 0.3)
        num_augmentations = self.params.get('num_augmentations', 3)
        output_key = self.params.get('output_key', 'flip_image_data_list')

        # Use the shared CV2 backend
        cv2_backend = get_cv2_backend()

        def random_flip(row):
            image_data = row.get(image_key)
            if image_data is None:
                row[output_key] = []
                return row

            try:
                # Determine whether the input is a file path string or already loaded image data (ndarray)
                if isinstance(image_data, str):
                    # Path string: read from a file
                    image = cv2_backend.imread(image_data)
                else:
                    # Already an ndarray: use it directly
                    image = image_data
                    if len(image.shape) == 2:
                        image = cv2_backend.cvtColor(image, cv2_backend.COLOR_GRAY2BGR)
                
                if image is None or (isinstance(image, list) and len(image) == 0):
                    row[output_key] = []
                    return row

                augment_data_list = []

                for _ in range(num_augmentations):
                    augmented_image = image.copy()

                    if random.random() < flip_horizontal_prob:
                        augmented_image = cv2_backend.flip(augmented_image, 1)

                    if random.random() < flip_vertical_prob:
                        augmented_image = cv2_backend.flip(augmented_image, 0)

                    if random.random() < rotation_prob:
                        h, w = image.shape[:2]
                        import cv2
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, random.randint(1, 359), 1.0)
                        augmented_image = cv2.warpAffine(augmented_image, M, (w, h))

                    augment_data_list.append(augmented_image)

                # If there is only one augmentation result, return a single image instead of a list (keeping the Tensor format serializable)
                if num_augmentations == 1:
                    row[output_key] = augment_data_list[0]
                else:
                    row[output_key] = augment_data_list

            except Exception:
                row[output_key] = []

            return row

        return dataset.map(random_flip)
