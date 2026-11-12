import os
from typing import Any, Dict, Optional
from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class ImageSizeFilterOperation(FilteringOperation):
    """Image file size filter

    Filter based on image file size. Supports multiple units (bytes, kb, mb, gb, tb),
    and you can set minimum and maximum file sizes. Filtering supports single or multiple images.
    """

    operation_type = OperationType.FILTERING

    UNIT_MULTIPLIERS = {
        "bytes": 1,
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3,
        "tb": 1024 ** 4
    }

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Filter images by file size",
        "params": {
            "image_key": {
                "type": "string",
                "required": False,
                "default": "image",
                "description": "Image field name"
            },
            "min_size": {
                "type": "number",
                "required": False,
                "default": 0,
                "description": "Minimum file size"
            },
            "max_size": {
                "type": "number",
                "required": False,
                "default": 1e308,
                "description": "Maximum file size"
            },
            "size_unit": {
                "type": "string",
                "required": False,
                "default": "bytes",
                "description": "Size unit: bytes, kb, mb, gb, tb"
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
            "min_size": 1,
            "max_size": 10,
            "size_unit": "mb"
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        size_unit_val = self.params.get('size_unit', 'bytes')
        if size_unit_val.lower() not in self.UNIT_MULTIPLIERS:
            raise ValueError("Unsupported size unit")
        any_or_all_val = self.params.get('any_or_all', 'any')
        if any_or_all_val.lower() not in ('any', 'all'):
            raise ValueError("any_or_all parameter must be 'any' or 'all'")

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        image_key = self.params.get('image_key', 'image')
        min_size = self.params.get('min_size', 0)
        max_size = self.params.get('max_size', float('inf'))
        size_unit = self.params.get('size_unit', 'bytes').lower()
        any_or_all = self.params.get('any_or_all', 'any').lower()

        multiplier = self.UNIT_MULTIPLIERS.get(size_unit, 1)

        def get_image_size(image_data):
            if image_data is None:
                return None
            try:
                # Determine whether the input is a file path string or already loaded image data (ndarray)
                if isinstance(image_data, str):
                    # Path string: read from a file
                    if not os.path.exists(image_data):
                        return None
                    size_in_bytes = os.path.getsize(image_data)
                    return size_in_bytes / multiplier
                else:
                    # Already an ndarray: compute memory size
                    import numpy as np
                    size_in_bytes = image_data.nbytes
                    return size_in_bytes / multiplier
            except Exception:
                return None

        def image_size_filter(row):
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

            check_results = []

            for img_data in images:
                size = get_image_size(img_data)
                if size is None:
                    check_results.append(False)
                else:
                    check_results.append(min_size <= size <= max_size)

            if any_or_all == 'any':
                return any(check_results)
            else:
                return all(check_results)

        return dataset.filter(image_size_filter)
