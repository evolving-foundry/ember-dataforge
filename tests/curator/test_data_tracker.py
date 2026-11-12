"""
Test module - DataTracker class tests
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

from curator.tracking.data_tracker import DataTracker
from curator.tracking.tracking import (
    OperationType,
    ChangeType,
    DataChangeRecord
)


class TestDataTrackerInit:
    """Test DataTracker initialization"""

    def test_init(self):
        """Test initialization"""
        tracker = DataTracker("pipeline_001", "session_001")
        
        assert tracker.pipeline_id == "pipeline_001"
        assert tracker.session_id == "session_001"
        assert tracker.tracking_metadata is not None


class TestDataTrackerMethods:
    """Test DataTracker methods"""

    def test_start_operation_tracking(self):
        """Test starting operation tracking"""
        tracker = DataTracker("pipeline_001", "session_001")
        
        tracker.start_operation_tracking(
            operation_name="text_loading",
            operation_type=OperationType.LOADING,
            parameters={"data_paths": ["data.json"]}
        )
        
        assert tracker.current_operation_info is not None
        assert tracker.current_operation_info.operation_name == "text_loading"

    def test_track_operation(self):
        """Test tracking an operation"""
        tracker = DataTracker("pipeline_001", "session_001")
        
        input_ds = ray.data.from_items([{"id": 1}, {"id": 2}])
        output_ds = ray.data.from_items([{"id": 1}, {"id": 2}, {"id": 3}])
        
        info = tracker.track_operation(
            operation_name="synthetic_data_generation",
            operation_type=OperationType.GENERATION,
            input_dataset=input_ds,
            output_dataset=output_ds
        )
        
        assert info is not None
        assert info.operation_name == "synthetic_data_generation"

    def test_track_operation_with_changes(self):
        """Test tracking with an explicit change record"""
        tracker = DataTracker("pipeline_001", "session_001")
        
        input_ds = ray.data.from_items([{"id": 1}])
        output_ds = ray.data.from_items([{"id": 1}, {"id": 2}])
        
        changes = [
            DataChangeRecord(change_type=ChangeType.ROW_ADDED, count=1)
        ]
        
        info = tracker.track_operation(
            operation_name="filter",
            operation_type=OperationType.FILTERING,
            input_dataset=input_ds,
            output_dataset=output_ds,
            changes=changes
        )
        
        assert len(info.changes) == 1

    def test_calculate_changes_row_added(self):
        """Test computing row addition changes"""
        tracker = DataTracker("pipeline_001", "session_001")
        
        input_ds = ray.data.from_items([{"id": 1}])
        output_ds = ray.data.from_items([{"id": 1}, {"id": 2}, {"id": 3}])
        
        changes = tracker._calculate_changes(input_ds, output_ds)
        
        assert any(c.change_type == ChangeType.ROW_ADDED for c in changes)

    def test_calculate_changes_row_removed(self):
        """Test computing row deletion changes"""
        tracker = DataTracker("pipeline_001", "session_001")
        
        input_ds = ray.data.from_items([{"id": 1}, {"id": 2}, {"id": 3}])
        output_ds = ray.data.from_items([{"id": 1}])
        
        changes = tracker._calculate_changes(input_ds, output_ds)
        
        assert any(c.change_type == ChangeType.ROW_REMOVED for c in changes)

    def test_get_tracking_metadata(self):
        """Test getting tracking metadata"""
        tracker = DataTracker("pipeline_001", "session_001")
        
        metadata = tracker.get_tracking_metadata()
        
        assert metadata.pipeline_id == "pipeline_001"

    def test_reset(self):
        """Test resetting the tracker"""
        tracker = DataTracker("pipeline_001", "session_001")
        
        input_ds = ray.data.from_items([{"id": 1}])
        tracker.track_operation("test", OperationType.UTILITY, input_ds, input_ds)
        
        tracker.reset()
        
        assert len(tracker.tracking_metadata.operations) == 0
