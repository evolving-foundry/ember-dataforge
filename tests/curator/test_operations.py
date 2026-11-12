"""
Test module - operation tests
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

from curator.core.operation import Operation
from base.dataset import MultimodalDataset
from curator.tracking.tracking import OperationType


class TestOperationBase:
    """Test the Operation base class"""

    def test_operation_get_params(self):
        """Test getting parameters"""
        from curator.operations.loading import TextLoadingOperation
        
        op = TextLoadingOperation({"data_paths": ["test.json"]})
        params = op.get_params()
        
        assert params == {"data_paths": ["test.json"]}

    def test_operation_set_params(self):
        """Test setting parameters"""
        from curator.operations.loading import TextLoadingOperation

        op = TextLoadingOperation({"data_paths": []})
        op.set_params({"data_paths": ["test.json"]})

        assert op.params["data_paths"] == ["test.json"]

    def test_operation_repr(self):
        """Test operation representation"""
        from curator.operations.loading import TextLoadingOperation
        
        op = TextLoadingOperation({"data_paths": []})
        repr_str = repr(op)
        
        assert "TextLoadingOperation" in repr_str


class TestTextLoadingOperation:
    """Test TextLoadingOperation"""

    def test_execute_with_data_paths(self, sample_json_path):
        """Test execution with a data path"""
        from curator.operations.loading import TextLoadingOperation
        
        op = TextLoadingOperation({"data_paths": [sample_json_path]})
        result = op.execute(None)
        
        assert isinstance(result, MultimodalDataset)
        assert result.dataset.count() == 3

    def test_execute_without_data_paths_raises_error(self):
        """Test that execution without a data path raises"""
        from curator.operations.loading import TextLoadingOperation

        op = TextLoadingOperation({"data_paths": []})

        with pytest.raises(ValueError, match="requires 'data_paths'"):
            op.execute(None)

    def test_execute_with_existing_dataset_returns_dataset(self):
        """Test returning the original dataset when one already exists"""
        from curator.operations.loading import TextLoadingOperation
        
        existing_ds = ray.data.from_items([{"id": 1}])
        existing = MultimodalDataset(dataset=existing_ds)
        
        op = TextLoadingOperation({"data_paths": []})
        result = op.execute(existing)
        
        assert result == existing


class TestMultimodalLoadingOperation:
    """Test MultiModalLoadingOperation"""

    def test_execute_with_data_paths(self, sample_json_path):
        """Test execution with a data path"""
        from curator.operations.loading import MultiModalLoadingOperation
        
        op = MultiModalLoadingOperation({"data_paths": [sample_json_path]})
        result = op.execute(None)
        
        # Returns a Ray Dataset, not a MultimodalDataset
        assert result is not None


class TestOperationTypes:
    """Test various operation types"""

    def test_loading_operation_type(self):
        """Test the loading operation type"""
        from curator.operations.loading import TextLoadingOperation

        op = TextLoadingOperation({"data_paths": []})
        assert op.operation_type == OperationType.LOADING

    def test_cleaning_operation_type(self):
        """Test the cleaning operation type"""
        from curator.operations.cleaning import InvalidDataRemovalOperation
        
        op = InvalidDataRemovalOperation({})
        assert op.operation_type == OperationType.CLEANING

    def test_mapping_operation_type(self):
        """Test the mapping operation type"""
        from curator.operations.mapping import TextChunkMappingOperation
        
        op = TextChunkMappingOperation({})
        assert op.operation_type == OperationType.MAPPING

    def test_saving_operation_type(self):
        """Test the saving operation type"""
        from curator.operations.saving import TextSavingOperation

        op = TextSavingOperation({"output_path": "./"})
        assert op.operation_type == OperationType.SAVING
