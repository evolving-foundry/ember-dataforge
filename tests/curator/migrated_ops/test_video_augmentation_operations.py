"""
Test migrated video augmentation operators
"""
import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Try to import ray
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

pytestmark = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")

if HAS_RAY:
    from base.dataset import MultimodalDataset
    from curator.tracking.tracking import OperationType


class TestVideoResizeAspectRatioOperation:
    """Test VideoResizeAspectRatioOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import VideoResizeAspectRatioOperation
        assert VideoResizeAspectRatioOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import VideoResizeAspectRatioOperation
        assert "params" in VideoResizeAspectRatioOperation.PARAMS_SCHEMA
        assert "video_key" in VideoResizeAspectRatioOperation.PARAMS_SCHEMA["params"]
        assert "output_key" in VideoResizeAspectRatioOperation.PARAMS_SCHEMA["params"]
        assert "output_folder" in VideoResizeAspectRatioOperation.PARAMS_SCHEMA["params"]
        assert "target_aspect_ratio_range" in VideoResizeAspectRatioOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import VideoResizeAspectRatioOperation
        op = VideoResizeAspectRatioOperation({})
        assert op.params.get("video_key") == "video_path"
        assert op.params.get("output_key") == "output_video_path"
        assert op.params.get("output_folder") == "./output/videos/"
        assert op.params.get("target_aspect_ratio_range") == [1.7, 1.9]

    def test_custom_params(self):
        """Test custom parameters"""
        from curator.operations.augmentation import VideoResizeAspectRatioOperation
        op = VideoResizeAspectRatioOperation({
            "video_key": "my_video",
            "output_key": "resized_video",
            "output_folder": "./custom/output/",
            "target_aspect_ratio_range": [1.6, 1.8]
        })
        assert op.params.get("video_key") == "my_video"
        assert op.params.get("output_key") == "resized_video"
        assert op.params.get("output_folder") == "./custom/output/"
        assert op.params.get("target_aspect_ratio_range") == [1.6, 1.8]
