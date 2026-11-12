import os
import random
from typing import Any, Dict, Optional

from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_cv2_backend


class ImageRandomCropOperation(AugmentationOperation):
    """Random image cropping

    Randomly crop multiple sub-images from an image. You can specify the crop count, scale range, and
    aspect ratio range. Suitable for data augmentation and training-sample expansion.

    Note: this operation requires the opencv-python library.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Randomly crop multiple sub-images from each image",
        "params": {
            "image_key": {
                "type": "string",
                "required": False,
                "default": "image_path",
                "description": "Input image field name"
            },
            "min_scale": {
                "type": "number",
                "required": False,
                "default": 0.3,
                "description": "Minimum scale ratio"
            },
            "max_scale": {
                "type": "number",
                "required": False,
                "default": 0.6,
                "description": "Maximum scale ratio"
            },
            "aspect_ratio_range": {
                "type": "array",
                "required": False,
                "default": [0.75, 1.33],
                "description": "Crop aspect-ratio range [min, max]"
            },
            "num_crops": {
                "type": "integer",
                "required": False,
                "default": 3,
                "description": "Number of crops per image"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "crop_image_data_list",
                "description": "Output cropped-image list field name"
            }
        },
        "example": {
            "image_key": "image_path",
            "min_scale": 0.3,
            "max_scale": 0.6,
            "num_crops": 3
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        image_key = self.params.get('image_key', 'image_path')
        min_scale = self.params.get('min_scale', 0.3)
        max_scale = self.params.get('max_scale', 0.6)
        aspect_ratio_range = self.params.get('aspect_ratio_range', [0.75, 1.33])
        num_crops = self.params.get('num_crops', 3)
        output_key = self.params.get('output_key', 'crop_image_data_list')

        # Use the shared CV2 backend
        cv2_backend = get_cv2_backend()

        def random_crop(row):
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
                
                if image is None or (isinstance(image, list) and len(image) == 0):
                    row[output_key] = []
                    return row

                if len(image.shape) == 2:
                    import numpy as np
                    image = np.expand_dims(image, axis=-1)

                img_height, img_width = image.shape[:2]
                crop_data_list = []

                for _ in range(num_crops):
                    scale_h = random.uniform(min_scale, max_scale)
                    scale_w = random.uniform(min_scale, max_scale)
                    aspect_ratio = scale_w / scale_h

                    min_ratio = aspect_ratio_range[0]
                    max_ratio = aspect_ratio_range[1]
                    if aspect_ratio < min_ratio:
                        scale_w = scale_h * min_ratio
                    elif aspect_ratio > max_ratio:
                        scale_w = scale_h * max_ratio

                    crop_height = int(img_height * scale_h)
                    crop_width = int(img_width * scale_w)

                    y = random.randint(0, img_height - crop_height) if img_height > crop_height else 0
                    x = random.randint(0, img_width - crop_width) if img_width > crop_width else 0

                    crop = image[y:y + crop_height, x:x + crop_width]
                    crop_data_list.append(crop)

                # If there is only one crop, return a single image instead of a list (keeping the Tensor format serializable)
                if num_crops == 1:
                    row[output_key] = crop_data_list[0]
                else:
                    row[output_key] = crop_data_list

            except Exception:
                row[output_key] = []

            return row

        return dataset.map(random_crop)
