"""
Test module - TrackingReportGenerator tests
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
from curator.tracking.report_generator import (
    TrackingReportGenerator,
    print_readable_report
)
from curator.tracking.tracking import (
    OperationType,
    ChangeType,
    DataChangeRecord,
    TrackingMetadata,
    OperationTrackingInfo
)


class TestTrackingReportGenerator:
    """Test TrackingReportGenerator class"""

    def test_init(self):
        """Test initialization"""
        metadata = TrackingMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001"
        )
        generator = TrackingReportGenerator(metadata)
        
        assert generator.tracking_metadata == metadata

    def test_generate_summary_report(self):
        """Test generating a summary report"""
        input_ds = ray.data.from_items([{"id": 1}])
        tracker = DataTracker("pipeline_001", "session_001")
        tracker.track_operation(
            operation_name="load",
            operation_type=OperationType.LOADING,
            input_dataset=input_ds,
            output_dataset=input_ds
        )
        
        generator = TrackingReportGenerator(tracker.get_tracking_metadata())
        summary = generator.generate_summary_report()
        
        assert summary["pipeline_id"] == "pipeline_001"
        assert summary["total_operations"] == 1
        assert summary["successful_operations"] == 1

    def test_generate_detailed_report(self):
        """Test generating a detailed report"""
        input_ds = ray.data.from_items([{"id": 1}])
        tracker = DataTracker("pipeline_001", "session_001")
        tracker.track_operation(
            operation_name="filter",
            operation_type=OperationType.FILTERING,
            input_dataset=input_ds,
            output_dataset=input_ds
        )
        
        generator = TrackingReportGenerator(tracker.get_tracking_metadata())
        detailed = generator.generate_detailed_report()
        
        assert "summary" in detailed
        assert "operations" in detailed

    def test_generate_changes_summary(self):
        """Test generating a change summary"""
        input_ds = ray.data.from_items([{"id": 1}, {"id": 2}])
        output_ds = ray.data.from_items([{"id": 1}])
        
        tracker = DataTracker("pipeline_001", "session_001")
        tracker.track_operation(
            operation_name="filter",
            operation_type=OperationType.FILTERING,
            input_dataset=input_ds,
            output_dataset=output_ds
        )
        
        generator = TrackingReportGenerator(tracker.get_tracking_metadata())
        changes_summary = generator.generate_changes_summary()
        
        assert "total_changes" in changes_summary
        assert "by_type" in changes_summary

    def test_generate_data_quality_report(self):
        """Test generating a data quality report"""
        metadata = TrackingMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001"
        )
        
        generator = TrackingReportGenerator(metadata)
        quality_report = generator.generate_data_quality_report()
        
        assert "data_reduction_analysis" in quality_report
        assert "duplicate_analysis" in quality_report
        assert "schema_changes" in quality_report

    def test_export_report_json(self, tmp_path):
        """Test exporting a JSON report"""
        metadata = TrackingMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001"
        )
        
        generator = TrackingReportGenerator(metadata)
        
        output_file = tmp_path / "report.json"
        result = generator.export_report(format_type="json", filepath=str(output_file))
        
        assert "Report exported to" in result
        assert output_file.exists()

    def test_export_report_json_string(self):
        """Test exporting a JSON string"""
        metadata = TrackingMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001"
        )
        
        generator = TrackingReportGenerator(metadata)
        content = generator.export_report(format_type="json")
        
        assert content is not None
        assert "pipeline_001" in content


class TestPrintReadableReport:
    """Test print_readable_report function"""

    def test_print_readable_report(self):
        """Test printing a readable report"""
        metadata = TrackingMetadata(
            pipeline_id="pipeline_001",
            session_id="session_001"
        )
        
        # Should not raise
        print_readable_report(metadata)
