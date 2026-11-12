"""
Test the video context orchestration filter operator and video utility functions
"""
import os
import sys

import pytest

try:
    import ray
    HAS_RAY = True
except ImportError:
    ray = None
    HAS_RAY = False

pytestmark = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")

if HAS_RAY:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

    from base.dataset import MultimodalDataset
    from curator.core.operation import Operation
    from curator.core.operation_registry import OperationRegistry
    from curator.tracking.tracking import OperationType
    from curator.utils.video_ops_utils import (
        attach_context_windows,
        build_filter_hints,
        merge_stitch_segments,
    )


class TestVideoContextOrchestrationOperation:
    def test_operation_type(self):
        from curator.operations.filtering import VideoContextOrchestrationOperation
        assert VideoContextOrchestrationOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        from curator.operations.filtering import VideoContextOrchestrationOperation
        assert "params" in VideoContextOrchestrationOperation.PARAMS_SCHEMA
        assert "segments_key" in VideoContextOrchestrationOperation.PARAMS_SCHEMA["params"]
        assert "orchestration_mode" in VideoContextOrchestrationOperation.PARAMS_SCHEMA["params"]
        assert "merge_gap_sec" in VideoContextOrchestrationOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        from curator.operations.filtering import VideoContextOrchestrationOperation
        op = VideoContextOrchestrationOperation({})
        assert op.params.get("segments_key") == "segments"
        assert op.params.get("output_key") == "segments"
        assert op.params.get("orchestration_mode") == "MERGE_STITCH"
        assert op.params.get("merge_gap_sec") == 0.5

    def test_custom_params(self):
        from curator.operations.filtering import VideoContextOrchestrationOperation
        op = VideoContextOrchestrationOperation({
            "segments_key": "my_segments",
            "output_key": "my_output",
            "orchestration_mode": "WINDOW_ATTACH",
            "context_window_sec": 3.0,
        })
        assert op.params.get("segments_key") == "my_segments"
        assert op.params.get("output_key") == "my_output"
        assert op.params.get("orchestration_mode") == "WINDOW_ATTACH"
        assert op.params.get("context_window_sec") == 3.0

    def test_execute_merge_stitch_mode(self):
        from curator.operations.filtering import VideoContextOrchestrationOperation

        ds = ray.data.from_items([
            {
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "text": "hello",
                        "duration_sec": 0.6,
                        "timestamp_range": {"start_sec": 0.0, "end_sec": 0.6},
                    },
                    {
                        "segment_id": "seg_0002",
                        "text": "world",
                        "duration_sec": 0.7,
                        "timestamp_range": {"start_sec": 0.8, "end_sec": 1.5},
                    },
                ]
            }
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoContextOrchestrationOperation({
            "orchestration_mode": "MERGE_STITCH",
            "merge_gap_sec": 0.3,
            "max_merge_span_sec": 5.0,
            "min_valid_duration_sec": 1.0,
        })

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert row["orchestration_mode"] == "MERGE_STITCH"
        assert row["orchestration_metrics"]["total_segments"] == 2
        assert row["segments"][0]["stitch_group_id"] == "stitch_0001"
        assert row["segments"][1]["stitch_group_id"] == "stitch_0001"

    def test_execute_window_attach_mode(self):
        from curator.operations.filtering import VideoContextOrchestrationOperation

        ds = ray.data.from_items([
            {
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "text": "first",
                        "duration_sec": 1.0,
                        "timestamp_range": {"start_sec": 0.0, "end_sec": 1.0},
                    },
                    {
                        "segment_id": "seg_0002",
                        "text": "second",
                        "duration_sec": 1.0,
                        "timestamp_range": {"start_sec": 1.5, "end_sec": 2.5},
                    },
                    {
                        "segment_id": "seg_0003",
                        "text": "third",
                        "duration_sec": 1.0,
                        "timestamp_range": {"start_sec": 5.0, "end_sec": 6.0},
                    },
                ]
            }
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoContextOrchestrationOperation({
            "orchestration_mode": "WINDOW_ATTACH",
            "context_window_sec": 1.0,
            "max_context_segments": 2,
        })

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert row["segments"][0]["context_window_refs"] == ["seg_0002"]
        assert row["segments"][1]["context_window_refs"] == ["seg_0001"]
        assert row["segments"][2]["context_window_refs"] == []

    def test_registry_create_operation(self):
        op = OperationRegistry.create_operation(
            "video_context_orchestration",
            {"orchestration_mode": "FILTER_HINT"},
        )
        assert op is not None
        assert isinstance(op, Operation)


class TestVideoContextOrchestrationUtils:
    def test_merge_stitch_segments(self):
        segments = merge_stitch_segments(
            segments=[
                {
                    "segment_id": "seg_0001",
                    "text": "a",
                    "duration_sec": 0.5,
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 0.5},
                },
                {
                    "segment_id": "seg_0002",
                    "text": "b",
                    "duration_sec": 0.5,
                    "timestamp_range": {"start_sec": 0.7, "end_sec": 1.2},
                },
            ],
            merge_gap_sec=0.3,
            max_merge_span_sec=5.0,
            min_valid_duration_sec=1.0,
        )
        assert segments[0]["stitch_group_id"] == "stitch_0001"
        assert segments[1]["stitch_group_id"] == "stitch_0001"

    def test_attach_context_windows(self):
        segments = attach_context_windows(
            segments=[
                {
                    "segment_id": "seg_0001",
                    "text": "a",
                    "duration_sec": 1.0,
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 1.0},
                },
                {
                    "segment_id": "seg_0002",
                    "text": "b",
                    "duration_sec": 1.0,
                    "timestamp_range": {"start_sec": 1.5, "end_sec": 2.5},
                },
            ],
            context_window_sec=1.0,
            max_context_segments=4,
        )
        assert segments[0]["context_window_refs"] == ["seg_0002"]
        assert segments[1]["context_window_refs"] == ["seg_0001"]

    def test_build_filter_hints(self):
        segments = build_filter_hints(
            segments=[
                {
                    "segment_id": "seg_0001",
                    "text": "",
                    "duration_sec": 0.5,
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 0.5},
                },
                {
                    "segment_id": "seg_0002",
                    "text": "Hello",
                    "duration_sec": 1.2,
                    "timestamp_range": {"start_sec": 0.6, "end_sec": 1.8},
                },
                {
                    "segment_id": "seg_0003",
                    "text": "hello",
                    "duration_sec": 1.1,
                    "timestamp_range": {"start_sec": 2.0, "end_sec": 3.1},
                },
            ],
            min_valid_duration_sec=1.0,
            merge_gap_sec=0.5,
        )
        assert "too_short" in segments[0]["cleanup_flags"]
        assert "blank_text" in segments[0]["cleanup_flags"]
        assert "near_duplicate" in segments[2]["cleanup_flags"]
