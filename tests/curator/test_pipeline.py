"""
Test module - MultimodalPipeline class tests
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

from curator.core.pipeline import MultimodalPipeline
from curator.core.operation import Operation
from base.dataset import MultimodalDataset
from curator.tracking.tracking import OperationType


class TestMultimodalPipelineInit:
    """Test MultimodalPipeline initialization"""

    def test_init_default(self):
        """Test default initialization"""
        pipeline = MultimodalPipeline()
        
        assert pipeline.config == {}
        assert pipeline.operations == []
        assert pipeline.enable_tracking is True

    def test_init_with_config_dict(self):
        """Test initialization with a dictionary config"""
        config = {"pipeline": {"operations": []}}
        pipeline = MultimodalPipeline(config=config)
        
        assert pipeline.config == config

    def test_init_with_disable_tracking(self):
        """Test initialization with tracking disabled"""
        pipeline = MultimodalPipeline(enable_tracking=False)
        
        assert pipeline.enable_tracking is False
        assert pipeline.tracker is None


class TestOperationRegistry:
    """Test operation registry"""

    def test_register_operation(self):
        """Test manually registering an operation"""
        from curator.core.operation_registry import OperationRegistry, register_operation
        
        @register_operation("test_op")
        class TestOperation(Operation):
            operation_type = OperationType.CLEANING
            
            def execute(self, dataset):
                return dataset
        
        # Verify the operation is registered
        op_class = OperationRegistry.get_operation_class("test_op")
        assert op_class is not None
        
        # Clean up
        OperationRegistry._operations.pop("test_op", None)

    def test_create_operation(self):
        """Test creating an operation instance"""
        from curator.core.pipeline import OperationRegistry
        
        op = OperationRegistry.create_operation("text_loading", {"data_paths": []})
        assert op is not None
        assert isinstance(op, Operation)

    def test_create_unknown_operation_raises_error(self):
        """Test that creating an unknown operation raises"""
        from curator.core.pipeline import OperationRegistry
        
        with pytest.raises(ValueError, match="Unknown operation"):
            OperationRegistry.create_operation("nonexistent_operation", {})


class TestPipelineAddOperation:
    """Test adding operations to the pipeline"""

    def test_add_operation_instance(self):
        """Test adding an operation instance"""
        pipeline = MultimodalPipeline(enable_tracking=False)
        from curator.operations.loading import TextLoadingOperation
        
        op = TextLoadingOperation({"data_paths": ["./data/sample.json"]})
        pipeline.add_operation(op)
        
        assert len(pipeline.operations) == 1

    def test_add_operation_string(self):
        """Test adding an operation by string"""
        pipeline = MultimodalPipeline(enable_tracking=False)
        
        pipeline.add_operation("text_loading", {"data_paths": []})
        
        assert len(pipeline.operations) == 1

    def test_add_invalid_operation_type_raises_error(self):
        """Test that adding an invalid operation type raises"""
        pipeline = MultimodalPipeline(enable_tracking=False)
        
        with pytest.raises(ValueError, match="Invalid operation type"):
            pipeline.add_operation(123)


class TestPipelineRun:
    """Test pipeline execution"""

    def test_run_with_ray_dataset(self):
        """Test running with a Ray Dataset"""
        pipeline = MultimodalPipeline(enable_tracking=False)
        ray_ds = ray.data.from_items([{"id": 1}, {"id": 2}])
        
        result = pipeline.run(ray_ds)
        
        assert isinstance(result, MultimodalDataset)
        assert result.dataset.count() == 2

    def test_run_with_multimodal_dataset(self):
        """Test running with a MultimodalDataset"""
        pipeline = MultimodalPipeline(enable_tracking=False)
        ray_ds = ray.data.from_items([{"id": 1}])
        dataset = MultimodalDataset(dataset=ray_ds)
        
        result = pipeline.run(dataset)
        
        assert isinstance(result, MultimodalDataset)

    def test_run_with_invalid_dataset_type_raises_error(self):
        """Test that an invalid dataset type raises"""
        pipeline = MultimodalPipeline(enable_tracking=False)
        
        with pytest.raises(ValueError, match="Invalid dataset type"):
            pipeline.run("invalid_dataset")


class TestPipelineOtherMethods:
    """Test other pipeline methods"""

    def test_get_tracking_report(self):
        """Test getting the tracking report"""
        pipeline = MultimodalPipeline(enable_tracking=True)
        
        report = pipeline.get_tracking_report()
        assert report is not None

    def test_list_checkpoints(self):
        """Test listing checkpoints"""
        pipeline = MultimodalPipeline(enable_tracking=False)
        
        checkpoints = pipeline.list_checkpoints()
        assert isinstance(checkpoints, list)

    def test_cleanup(self):
        """Test cleaning up resources"""
        pipeline = MultimodalPipeline(enable_tracking=False)
        
        # Should clean up normally
        pipeline.cleanup()
