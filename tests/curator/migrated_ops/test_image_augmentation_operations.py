"""
Test migrated image augmentation operators
"""
import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from base.dataset import MultimodalDataset
from curator.tracking.tracking import OperationType

# Try to import ray
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    pytestmark = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")


class TestImageCannyAugmentationOperation:
    """Test ImageCannyAugmentationOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import ImageCannyAugmentationOperation
        assert ImageCannyAugmentationOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import ImageCannyAugmentationOperation
        assert "params" in ImageCannyAugmentationOperation.PARAMS_SCHEMA
        assert "image_key" in ImageCannyAugmentationOperation.PARAMS_SCHEMA["params"]
        assert "low_threshold" in ImageCannyAugmentationOperation.PARAMS_SCHEMA["params"]
        assert "high_threshold" in ImageCannyAugmentationOperation.PARAMS_SCHEMA["params"]
        assert "output_key" in ImageCannyAugmentationOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import ImageCannyAugmentationOperation
        op = ImageCannyAugmentationOperation({})
        assert op.params.get("image_key") == "image_path"
        assert op.params.get("low_threshold") == 30
        assert op.params.get("high_threshold") == 100
        assert op.params.get("output_key") == "canny_image_data"


class TestImageCaptioningOperation:
    """Test ImageCaptioningOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import ImageCaptioningOperation
        assert ImageCaptioningOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import ImageCaptioningOperation
        assert "params" in ImageCaptioningOperation.PARAMS_SCHEMA
        assert "image_key" in ImageCaptioningOperation.PARAMS_SCHEMA["params"]
        assert "output_key" in ImageCaptioningOperation.PARAMS_SCHEMA["params"]
        assert "model_path" in ImageCaptioningOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import ImageCaptioningOperation
        op = ImageCaptioningOperation({})
        assert op.params.get("image_key") == "image_path"
        assert op.params.get("output_key") == "caption"
        assert op.params.get("model_path") == "Salesforce/blip-image-captioning-base"


class TestImageRandomCropOperation:
    """Test ImageRandomCropOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import ImageRandomCropOperation
        assert ImageRandomCropOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import ImageRandomCropOperation
        assert "params" in ImageRandomCropOperation.PARAMS_SCHEMA
        assert "image_key" in ImageRandomCropOperation.PARAMS_SCHEMA["params"]
        assert "min_scale" in ImageRandomCropOperation.PARAMS_SCHEMA["params"]
        assert "max_scale" in ImageRandomCropOperation.PARAMS_SCHEMA["params"]
        assert "num_crops" in ImageRandomCropOperation.PARAMS_SCHEMA["params"]
        assert "output_key" in ImageRandomCropOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import ImageRandomCropOperation
        op = ImageRandomCropOperation({})
        assert op.params.get("image_key") == "image_path"
        assert op.params.get("min_scale") == 0.3
        assert op.params.get("max_scale") == 0.6
        assert op.params.get("num_crops") == 3
        assert op.params.get("output_key") == "crop_image_data_list"


class TestImageRandomFlipOperation:
    """Test ImageRandomFlipOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import ImageRandomFlipOperation
        assert ImageRandomFlipOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import ImageRandomFlipOperation
        assert "params" in ImageRandomFlipOperation.PARAMS_SCHEMA
        assert "image_key" in ImageRandomFlipOperation.PARAMS_SCHEMA["params"]
        assert "flip_horizontal_prob" in ImageRandomFlipOperation.PARAMS_SCHEMA["params"]
        assert "flip_vertical_prob" in ImageRandomFlipOperation.PARAMS_SCHEMA["params"]
        assert "rotation_prob" in ImageRandomFlipOperation.PARAMS_SCHEMA["params"]
        assert "num_augmentations" in ImageRandomFlipOperation.PARAMS_SCHEMA["params"]
        assert "output_key" in ImageRandomFlipOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import ImageRandomFlipOperation
        op = ImageRandomFlipOperation({})
        assert op.params.get("image_key") == "image_path"
        assert op.params.get("flip_horizontal_prob") == 0.3
        assert op.params.get("flip_vertical_prob") == 0.3
        assert op.params.get("rotation_prob") == 0.3
        assert op.params.get("num_augmentations") == 3
        assert op.params.get("output_key") == "flip_image_data_list"
