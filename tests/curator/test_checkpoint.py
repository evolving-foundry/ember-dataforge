"""
Test module - Checkpoint model tests
"""
import pytest
from datetime import datetime
from curator.checkpoint.checkpoint_models import (
    CheckpointStatus,
    CheckpointType,
    OperationState,
    CheckpointMetadata,
    CheckpointConfig
)


class TestCheckpointStatus:
    """Test CheckpointStatus enum"""

    def test_all_statuses(self):
        """Test all status values"""
        assert CheckpointStatus.ACTIVE.value == "active"
        assert CheckpointStatus.COMPLETED.value == "completed"
        assert CheckpointStatus.FAILED.value == "failed"
        assert CheckpointStatus.EXPIRED.value == "expired"


class TestCheckpointType:
    """Test CheckpointType enum"""

    def test_all_types(self):
        """Test all type values"""
        assert CheckpointType.OPERATION_LEVEL.value == "operation_level"
        assert CheckpointType.BATCH_LEVEL.value == "batch_level"
        assert CheckpointType.TIME_BASED.value == "time_based"
        assert CheckpointType.CUSTOM.value == "custom"


class TestOperationState:
    """Test OperationState class"""

    def test_create_state(self):
        """Test creating operation state"""
        state = OperationState(
            operation_id="op_001",
            operation_name="text_loading",
            operation_type="loading"
        )
        
        assert state.operation_id == "op_001"
        assert state.operation_name == "text_loading"
        assert state.operation_type == "loading"
        assert state.status == "pending"
        assert state.start_time is not None

    def test_custom_state(self):
        """Test custom state"""
        state = OperationState(
            operation_id="op_002",
            operation_name="filter",
            operation_type="filtering",
            status="completed",
            parameters={"threshold": 0.5}
        )
        
        assert state.status == "completed"
        assert state.parameters == {"threshold": 0.5}


class TestCheckpointMetadata:
    """Test CheckpointMetadata class"""

    def test_create_metadata(self):
        """Test creating Checkpoint metadata"""
        metadata = CheckpointMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001",
            checkpoint_type=CheckpointType.OPERATION_LEVEL,
            storage_path="./checkpoints/test"
        )
        
        assert metadata.pipeline_id == "pipeline_001"
        assert metadata.session_id == "session_001"
        assert metadata.checkpoint_type == CheckpointType.OPERATION_LEVEL
        assert metadata.status == CheckpointStatus.ACTIVE
        assert metadata.progress_percentage == 0.0

    def test_update_status(self):
        """Test updating status"""
        metadata = CheckpointMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001",
            checkpoint_type=CheckpointType.OPERATION_LEVEL,
            storage_path="./checkpoints/test"
        )
        
        metadata.update_status(CheckpointStatus.COMPLETED)
        
        assert metadata.status == CheckpointStatus.COMPLETED

    def test_add_completed_operation(self):
        """Test adding a completed operation"""
        metadata = CheckpointMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001",
            checkpoint_type=CheckpointType.OPERATION_LEVEL,
            storage_path="./checkpoints/test",
            total_operations=3
        )
        
        state = OperationState(
            operation_id="op_001",
            operation_name="load",
            operation_type="loading",
            status="completed"
        )
        
        metadata.add_completed_operation(state)
        
        assert len(metadata.completed_operations) == 1
        assert metadata.current_operation_index == 1
        assert metadata.progress_percentage > 0

    def test_add_multiple_operations(self):
        """Test adding multiple operations"""
        metadata = CheckpointMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001",
            checkpoint_type=CheckpointType.OPERATION_LEVEL,
            storage_path="./checkpoints/test",
            total_operations=5
        )
        
        for i in range(3):
            state = OperationState(
                operation_id=f"op_{i}",
                operation_name=f"operation_{i}",
                operation_type="processing",
                status="completed"
            )
            metadata.add_completed_operation(state)
        
        assert metadata.current_operation_index == 3
        assert metadata.progress_percentage == 60.0


class TestCheckpointConfig:
    """Test CheckpointConfig class"""

    def test_default_config(self):
        """Test default config"""
        config = CheckpointConfig()
        
        assert config.enabled is True
        assert config.auto_save_interval_ops == 1
        assert config.auto_save_interval_time is None
        assert config.max_checkpoints_to_keep == 5
        assert config.checkpoint_dir == "./checkpoints"
        assert config.checkpoint_on_failure is True
        assert config.checkpoint_on_success is False
        assert config.resume_from_latest is True

    def test_custom_config(self):
        """Test custom config"""
        config = CheckpointConfig(
            enabled=False,
            auto_save_interval_ops=5,
            max_checkpoints_to_keep=10,
            checkpoint_dir="/custom/checkpoints",
            checkpoint_on_failure=False,
            checkpoint_on_success=True
        )
        
        assert config.enabled is False
        assert config.auto_save_interval_ops == 5
        assert config.max_checkpoints_to_keep == 10
        assert config.checkpoint_dir == "/custom/checkpoints"
        assert config.checkpoint_on_failure is False
        assert config.checkpoint_on_success is True
