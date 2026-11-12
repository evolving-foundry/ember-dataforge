"""
Test module - DataSaverFactory class tests
"""
import pytest
import os

# Try to import ray
try:
    import ray
    HAS_RAY = True
except ImportError:
    ray = None
    HAS_RAY = False
    pytestmark = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")

from curator.utils.io.data_saver_factory import DataSaverFactory


class TestDataSaverFactoryDetectFormat:
    """Test detect_format_from_path method"""

    def test_detect_json_format(self):
        """Test detecting JSON format"""
        assert DataSaverFactory.detect_format_from_path("data.json") == "json"
        assert DataSaverFactory.detect_format_from_path("output.jsonl") == "json"

    def test_detect_parquet_format(self):
        """Test detecting Parquet format"""
        assert DataSaverFactory.detect_format_from_path("data.parquet") == "parquet"

    def test_detect_csv_format(self):
        """Test detecting CSV format"""
        assert DataSaverFactory.detect_format_from_path("data.csv") == "csv"

    def test_detect_text_format(self):
        """Test detecting text format"""
        assert DataSaverFactory.detect_format_from_path("data.txt") == "text"

    def test_detect_default_format(self):
        """Test default format"""
        assert DataSaverFactory.detect_format_from_path("data.xyz") == "json"


class TestDataSaverFactorySaveDataset:
    """Test save_dataset method"""

    @pytest.mark.skip(reason="Ray+pandas compatibility issue")
    def test_save_as_json(self, tmp_path):
        """Test saving as JSON"""
        ray_ds = ray.data.from_items([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        output_path = str(tmp_path / "output.json")
        
        DataSaverFactory.save_dataset(ray_ds, output_path)
        
        assert os.path.exists(output_path)

    def test_save_as_csv(self, tmp_path):
        """Test saving as CSV"""
        ray_ds = ray.data.from_items([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        output_path = str(tmp_path / "output.csv")
        
        DataSaverFactory.save_dataset(ray_ds, output_path)
        
        assert os.path.exists(output_path)

    @pytest.mark.skip(reason="Ray+pandas compatibility issue")
    def test_save_creates_output_directory(self, tmp_path):
        """Test creating the output directory when saving"""
        ray_ds = ray.data.from_items([{"id": 1}])
        output_path = str(tmp_path / "subdir" / "output.json")
        
        DataSaverFactory.save_dataset(ray_ds, output_path)
        
        assert os.path.exists(output_path)


class TestDataSaverFactoryGetSupportedFormats:
    """Test get_supported_formats method"""

    def test_get_supported_formats(self):
        """Test getting supported formats"""
        formats = DataSaverFactory.get_supported_formats()
        
        assert isinstance(formats, list)
        assert "json" in formats
        assert "parquet" in formats
        assert "csv" in formats
