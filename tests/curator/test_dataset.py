"""
Test module - MultimodalDataset class tests
"""
import pytest

# Try to import ray
try:
    import ray
    HAS_RAY = True
except ImportError:
    ray = None
    HAS_RAY = False
    pytestmark = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")

from base.dataset import (
    MultimodalDataset,
    DataValidationConfig,
    ValidationResult
)


class TestDataValidationConfig:
    """Test DataValidationConfig class"""

    def test_default_config(self):
        """Test default config"""
        config = DataValidationConfig()
        assert config.required_fields == []
        assert config.field_types == {}
        assert config.text_min_length == 1
        assert config.text_max_length == 10000

    def test_custom_config(self):
        """Test custom config"""
        config = DataValidationConfig(
            required_fields=["id", "text"],
            field_types={"text": "text"},
            text_min_length=5,
            text_max_length=5000
        )
        assert config.required_fields == ["id", "text"]
        assert config.field_types == {"text": "text"}


class TestValidationResult:
    """Test ValidationResult class"""

    def test_valid_result(self):
        """Test a valid validation result"""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_result_with_errors(self):
        """Test an invalid validation result"""
        result = ValidationResult(
            is_valid=False,
            errors=["Missing field: title"],
            warnings=["Text exceeds max length"]
        )
        assert result.is_valid is False
        assert "Missing field: title" in result.errors


class TestMultimodalDataset:
    """Test MultimodalDataset class"""

    def test_init_with_valid_dataset(self):
        """Test initialization with a valid dataset"""
        ray_ds = ray.data.from_items([{"id": 1}, {"id": 2}])
        dataset = MultimodalDataset(dataset=ray_ds, modalities=["text"])
        
        assert dataset.dataset is not None
        assert dataset.modalities == ["text"]

    def test_init_with_none_dataset_raises_error(self):
        """Test that initializing with a None dataset raises"""
        with pytest.raises(ValueError, match="Dataset cannot be None"):
            MultimodalDataset(dataset=None)

    def test_validate_data(self):
        """Test data validation"""
        ray_ds = ray.data.from_items([{"id": 1, "text": "hello"}])
        config = DataValidationConfig(required_fields=["id"])
        dataset = MultimodalDataset(dataset=ray_ds, validation_config=config)
        
        result = dataset.validate_data()
        assert isinstance(result, ValidationResult)

    def test_map(self):
        """Test map method"""
        ray_ds = ray.data.from_items([{"value": 1}, {"value": 2}])
        dataset = MultimodalDataset(dataset=ray_ds, modalities=["text"])
        
        def add_10(row):
            row["value"] = row["value"] + 10
            return row
        
        result = dataset.map(add_10)
        assert isinstance(result, MultimodalDataset)

    def test_filter(self):
        """Test filter method"""
        ray_ds = ray.data.from_items([{"value": 1}, {"value": 5}, {"value": 10}])
        dataset = MultimodalDataset(dataset=ray_ds, modalities=["text"])
        
        result = dataset.filter(lambda row: row["value"] > 2)
        assert isinstance(result, MultimodalDataset)
        assert result.dataset.count() == 2

    def test_get_stats(self):
        """Test get_stats method"""
        ray_ds = ray.data.from_items([{"id": 1}, {"id": 2}])
        dataset = MultimodalDataset(dataset=ray_ds, modalities=["text"])
        
        stats = dataset.get_stats()
        
        assert "num_rows" in stats
        assert stats["num_rows"] == 2
