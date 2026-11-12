from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_cv2_backend
import numpy as np

# Note: do not import cv2 at the top of the file; use get_cv2_backend() for lazy loading


class ImageCannyAugmentationOperation(AugmentationOperation):
    """Canny edge detection enhancement

    Apply the Canny edge detection algorithm to an image to produce an edge map. Useful for image preprocessing,
    feature extraction, or data augmentation.

    Note: this operation requires the opencv-python library.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Generate edge maps with Canny edge detection",
        "params": {
            "image_key": {
                "type": "string",
                "required": False,
                "default": "image_path",
                "description": "Input image field name"
            },
            "low_threshold": {
                "type": "integer",
                "required": False,
                "default": 30,
                "description": "Canny low threshold"
            },
            "high_threshold": {
                "type": "integer",
                "required": False,
                "default": 100,
                "description": "Canny high threshold"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "canny_image_data",
                "description": "Output edge-map field name"
            }
        },
        "example": {
            "image_key": "image_path",
            "low_threshold": 30,
            "high_threshold": 100
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        image_key = self.params.get('image_key', 'image_path')
        low_threshold = self.params.get('low_threshold', 30)
        high_threshold = self.params.get('high_threshold', 100)
        output_key = self.params.get('output_key', 'canny_image_data')

        cv2_backend = get_cv2_backend()

        def apply_canny(row):
            image_data = row.get(image_key)
            
            if image_data is None:
                row[output_key] = None
                return row

            try:
                # Determine whether the input is a file path string or already loaded image data (ndarray)
                if isinstance(image_data, str):
                    # Path string: read a grayscale image from a file
                    image = cv2_backend.imread(image_data, cv2_backend.IMREAD_GRAYSCALE)
                else:
                    # Already an ndarray: use it directly
                    image = image_data
                    # If it is a color image, convert it to grayscale
                    # Images read by Ray Data are in RGB format
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        # Use the standard grayscale conversion formula: 0.299*R + 0.587*G + 0.114*B
                        image = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
                
                if image is None or (isinstance(image, list) and len(image) == 0):
                    row[output_key] = None
                    return row

                # Canny edge detection
                canny_edges = cv2_backend.Canny(image, low_threshold, high_threshold)
                row[output_key] = canny_edges

            except Exception as e:
                row[output_key] = None

            return row

        return dataset.map(apply_canny)
