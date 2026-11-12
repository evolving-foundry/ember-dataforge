"""
Test migrated image filter operators
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


class TestImageAspectRatioFilterOperation:
    """Test ImageAspectRatioFilterOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.filtering import ImageAspectRatioFilterOperation
        assert ImageAspectRatioFilterOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.filtering import ImageAspectRatioFilterOperation
        assert "params" in ImageAspectRatioFilterOperation.PARAMS_SCHEMA
        assert "image_key" in ImageAspectRatioFilterOperation.PARAMS_SCHEMA["params"]
        assert "min_ratio" in ImageAspectRatioFilterOperation.PARAMS_SCHEMA["params"]
        assert "max_ratio" in ImageAspectRatioFilterOperation.PARAMS_SCHEMA["params"]
        assert "any_or_all" in ImageAspectRatioFilterOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.filtering import ImageAspectRatioFilterOperation
        op = ImageAspectRatioFilterOperation({})
        assert op.params.get("image_key") == "image"
        assert op.params.get("min_ratio") == 0.333
        assert op.params.get("max_ratio") == 3.0
        assert op.params.get("any_or_all") == "any"

    def test_invalid_any_or_all(self):
        """Test invalid any_or_all parameter"""
        from curator.operations.filtering import ImageAspectRatioFilterOperation
        
        with pytest.raises(ValueError, match="any_or_all parameter must be"):
            ImageAspectRatioFilterOperation({"any_or_all": "invalid"})


class TestImageNsfwFilterOperation:
    """Test ImageNsfwFilterOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.filtering import ImageNsfwFilterOperation
        assert ImageNsfwFilterOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.filtering import ImageNsfwFilterOperation
        assert "params" in ImageNsfwFilterOperation.PARAMS_SCHEMA
        assert "image_key" in ImageNsfwFilterOperation.PARAMS_SCHEMA["params"]
        assert "threshold" in ImageNsfwFilterOperation.PARAMS_SCHEMA["params"]
        assert "model_path" in ImageNsfwFilterOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.filtering import ImageNsfwFilterOperation
        op = ImageNsfwFilterOperation({})
        assert op.params.get("image_key") == "image"
        assert op.params.get("threshold") == 0.5
        assert op.params.get("model_path") == "Falconsai/nsfw_image_detection"


class TestImageSizeFilterOperation:
    """Test ImageSizeFilterOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.filtering import ImageSizeFilterOperation
        assert ImageSizeFilterOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.filtering import ImageSizeFilterOperation
        assert "params" in ImageSizeFilterOperation.PARAMS_SCHEMA
        assert "image_key" in ImageSizeFilterOperation.PARAMS_SCHEMA["params"]
        assert "min_size" in ImageSizeFilterOperation.PARAMS_SCHEMA["params"]
        assert "max_size" in ImageSizeFilterOperation.PARAMS_SCHEMA["params"]
        assert "size_unit" in ImageSizeFilterOperation.PARAMS_SCHEMA["params"]
        assert "any_or_all" in ImageSizeFilterOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.filtering import ImageSizeFilterOperation
        op = ImageSizeFilterOperation({})
        assert op.params.get("image_key") == "image"
        assert op.params.get("min_size") == 0
        assert op.params.get("size_unit") == "bytes"
        assert op.params.get("any_or_all") == "any"

    def test_unit_multipliers(self):
        """Test unit conversion"""
        from curator.operations.filtering import ImageSizeFilterOperation
        
        assert ImageSizeFilterOperation.UNIT_MULTIPLIERS["bytes"] == 1
        assert ImageSizeFilterOperation.UNIT_MULTIPLIERS["kb"] == 1024
        assert ImageSizeFilterOperation.UNIT_MULTIPLIERS["mb"] == 1024 ** 2

    def test_invalid_size_unit(self):
        """Test invalid units"""
        from curator.operations.filtering import ImageSizeFilterOperation
        
        with pytest.raises(ValueError, match="Unsupported size unit"):
            ImageSizeFilterOperation({"size_unit": "invalid"})
