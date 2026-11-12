import os
from typing import Any, Dict, Optional
from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class ImageAspectRatioFilterOperation(FilteringOperation):
    """Image aspect ratio filter

    Filter based on image aspect ratio. You can set minimum and maximum aspect ratio ranges,
    and filtering supports single or multiple images. Commonly used to filter out images that are too narrow or too wide.
    """

    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Filter images by aspect ratio",
        "params": {
            "image_key": {
                "type": "string",
                "required": False,
                "default": "image",
                "description": "Image field name"
            },
            "min_ratio": {
                "type": "number",
                "required": False,
                "default": 0.333,
                "description": "Minimum aspect ratio"
            },
            "max_ratio": {
                "type": "number",
                "required": False,
                "default": 3.0,
                "description": "Maximum aspect ratio"
            },
            "any_or_all": {
                "type": "string",
                "required": False,
                "default": "any",
                "description": "Multi-image mode: 'any' matches any or 'all' matches all"
            }
        },
        "example": {
            "image_key": "image",
            "min_ratio": 0.5,
            "max_ratio": 2.0,
            "any_or_all": "any"
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        any_or_all_val = self.params.get('any_or_all', 'any')
        if any_or_all_val.lower() not in ('any', 'all'):
            raise ValueError("any_or_all parameter must be 'any' or 'all'")

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        image_key = self.params.get('image_key', 'image')
        min_ratio = self.params.get('min_ratio', 0.333)
        max_ratio = self.params.get('max_ratio', 3.0)
        any_or_all = self.params.get('any_or_all', 'any').lower()

        def compute_aspect_ratio(image_data):
            if image_data is None:
                return None
            try:
                # Determine whether the input is a file path string or already loaded image data (ndarray)
                if isinstance(image_data, str):
                    # Path string: read from a file
                    if not os.path.exists(image_data):
                        return None
                    from PIL import Image

                    with Image.open(image_data) as img:
                        width, height = img.size
                        if height == 0:
                            return 0.0
                        return width / height
                else:
                    # Already an ndarray: compute directly
                    import numpy as np
                    if len(image_data.shape) == 3:
                        height, width = image_data.shape[:2]
                    elif len(image_data.shape) == 2:
                        height, width = image_data.shape
                    else:
                        return None
                    if height == 0:
                        return 0.0
                    return width / height
            except Exception:
                return None

        def aspect_ratio_filter(row):
            images = row.get(image_key)

            if images is None:
                images = []
            elif isinstance(images, str):
                images = [images]
            elif not isinstance(images, list):
                # ndarray (single image, not in a list)
                images = [images]

            if not images:
                return True

            ratios = []
            for img_data in images:
                ratio = compute_aspect_ratio(img_data)
                ratios.append(ratio)

            check_results = []
            for r in ratios:
                if r is None:
                    check_results.append(False)
                else:
                    check_results.append(min_ratio <= r <= max_ratio)

            if any_or_all == 'any':
                return any(check_results)
            else:
                return all(check_results)

        return dataset.filter(aspect_ratio_filter)
