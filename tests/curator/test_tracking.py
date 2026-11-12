"""
Test module - tracking and report class tests
"""
import pytest
from datetime import datetime
from curator.tracking.tracking import (
    OperationType,
    ChangeType,
    DataChangeRecord,
    OperationTrackingInfo,
    TrackingMetadata
)


class TestOperationType:
    """Test OperationType enum"""

    def test_all_operation_types(self):
        """Test all operation types"""
        assert OperationType.LOADING.value == "loading"
        assert OperationType.FILTERING.value == "filtering"
        assert OperationType.CLEANING.value == "cleaning"
        assert OperationType.AUGMENTATION.value == "augmentation"
        assert OperationType.DEDUPLICATION.value == "deduplication"
        assert OperationType.FORMATTING.value == "formatting"
        assert OperationType.GENERATION.value == "generation"
        assert OperationType.AGGREGATION.value == "aggregation"
        assert OperationType.GROUPING.value == "grouping"
        assert OperationType.MAPPING.value == "mapping"
        assert OperationType.SELECTION.value == "selection"
        assert OperationType.SAVING.value == "saving"
        assert OperationType.UTILITY.value == "utility"


class TestChangeType:
    """Test ChangeType enum"""

    def test_all_change_types(self):
        """Test all change types"""
        assert ChangeType.ROW_ADDED.value == "row_added"
        assert ChangeType.ROW_REMOVED.value == "row_removed"
        assert ChangeType.ROW_MODIFIED.value == "row_modified"
        assert ChangeType.COLUMN_ADDED.value == "column_added"
        assert ChangeType.COLUMN_REMOVED.value == "column_removed"
        assert ChangeType.COLUMN_MODIFIED.value == "column_modified"
        assert ChangeType.DUPLICATE_FOUND.value == "duplicate_found"
        assert ChangeType.DUPLICATE_REMOVED.value == "duplicate_removed"


class TestDataChangeRecord:
    """Test DataChangeRecord class"""

    def test_create_record(self):
        """Test creating a change record"""
        record = DataChangeRecord(
            change_type=ChangeType.ROW_ADDED,
            record_ids=[1, 2, 3],
            count=3
        )
        
        assert record.change_type == ChangeType.ROW_ADDED
        assert record.record_ids == [1, 2, 3]
        assert record.count == 3
        assert record.timestamp is not None

    def test_with_columns_affected(self):
        """Test a record with affected columns"""
        record = DataChangeRecord(
            change_type=ChangeType.COLUMN_MODIFIED,
            columns_affected=["text", "label"],
            count=2
        )
        
        assert record.columns_affected == ["text", "label"]

    def test_with_metadata(self):
        """Test a record with metadata"""
        record = DataChangeRecord(
            change_type=ChangeType.ROW_REMOVED,
            record_ids=[1],
            metadata={"reason": "invalid_data"},
            count=1
        )
        
        assert record.metadata == {"reason": "invalid_data"}


class TestOperationTrackingInfo:
    """Test OperationTrackingInfo class"""

    def test_create_tracking_info(self):
        """Test creating tracking info"""
        info = OperationTrackingInfo(
            operation_id="op_001",
            operation_name="text_loading",
            operation_type=OperationType.LOADING
        )
        
        assert info.operation_id == "op_001"
        assert info.operation_name == "text_loading"
        assert info.operation_type == OperationType.LOADING
        assert info.start_time is not None
        assert info.success is True

    def test_finish_tracking(self):
        """Test completing tracking"""
        info = OperationTrackingInfo(
            operation_id="op_001",
            operation_name="filter",
            operation_type=OperationType.FILTERING
        )
        
        info.finish_tracking(success=True)
        
        assert info.end_time is not None
        assert info.success is True

    def test_finish_tracking_with_error(self):
        """Test completing tracking with an error"""
        info = OperationTrackingInfo(
            operation_id="op_001",
            operation_name="filter",
            operation_type=OperationType.FILTERING
        )
        
        info.finish_tracking(success=False, error_message="Test error")
        
        assert info.end_time is not None
        assert info.success is False
        assert info.error_message == "Test error"


class TestTrackingMetadata:
    """Test TrackingMetadata class"""

    def test_create_metadata(self):
        """Test creating tracking metadata"""
        metadata = TrackingMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001"
        )
        
        assert metadata.pipeline_id == "pipeline_001"
        assert metadata.session_id == "session_001"
        assert metadata.operations == []
        assert metadata.total_input_records == 0
        assert metadata.total_output_records == 0

    def test_add_operation(self):
        """Test adding an operation"""
        metadata = TrackingMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001"
        )
        
        info = OperationTrackingInfo(
            operation_id="op_001",
            operation_name="loading",
            operation_type=OperationType.LOADING,
            output_stats={"num_rows": 100}
        )
        
        metadata.add_operation(info)
        
        assert len(metadata.operations) == 1
        assert metadata.total_output_records == 100

    def test_add_multiple_operations(self):
        """Test adding multiple operations"""
        metadata = TrackingMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001"
        )
        
        metadata.add_operation(OperationTrackingInfo(
            operation_id="op_001",
            operation_name="load",
            operation_type=OperationType.LOADING,
            output_stats={"num_rows": 100}
        ))
        
        metadata.add_operation(OperationTrackingInfo(
            operation_id="op_002",
            operation_name="filter",
            operation_type=OperationType.FILTERING,
            output_stats={"num_rows": 80}
        ))
        
        assert len(metadata.operations) == 2
        assert metadata.total_output_records == 80
