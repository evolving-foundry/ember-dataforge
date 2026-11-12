"""
Test the video temporal structure operator and temporal structure utility functions
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
        build_event_chain_structures,
        build_event_window_structures,
        build_temporal_description_map,
        build_temporal_sample_map,
        build_temporal_text_map,
    )


class TestVideoTemporalStructureOperation:
    def test_operation_type(self):
        from curator.operations.augmentation import VideoTemporalStructureOperation
        assert VideoTemporalStructureOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        from curator.operations.augmentation import VideoTemporalStructureOperation
        assert "params" in VideoTemporalStructureOperation.PARAMS_SCHEMA
        assert "segments_key" in VideoTemporalStructureOperation.PARAMS_SCHEMA["params"]
        assert "samples_key" in VideoTemporalStructureOperation.PARAMS_SCHEMA["params"]
        assert "temporal_mode" in VideoTemporalStructureOperation.PARAMS_SCHEMA["params"]
        assert "chain_window_size" in VideoTemporalStructureOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        from curator.operations.augmentation import VideoTemporalStructureOperation
        op = VideoTemporalStructureOperation({})
        assert op.params.get("segments_key") == "segments"
        assert op.params.get("samples_key") == "samples"
        assert op.params.get("temporal_mode") == "EVENT_WINDOW"
        assert op.params.get("chain_window_size") == 3

    def test_custom_params(self):
        from curator.operations.augmentation import VideoTemporalStructureOperation
        op = VideoTemporalStructureOperation({
            "segments_key": "my_segments",
            "samples_key": "my_samples",
            "temporal_mode": "EVENT_CHAIN",
            "chain_window_size": 4,
        })
        assert op.params.get("segments_key") == "my_segments"
        assert op.params.get("samples_key") == "my_samples"
        assert op.params.get("temporal_mode") == "EVENT_CHAIN"
        assert op.params.get("chain_window_size") == 4

    def test_registry_create_operation(self):
        op = OperationRegistry.create_operation(
            "video_temporal_structure",
            {"temporal_mode": "EVENT_WINDOW"},
        )
        assert op is not None
        assert isinstance(op, Operation)

    def test_extract_event_window_mode(self):
        from curator.operations.augmentation import VideoTemporalStructureOperation

        op = VideoTemporalStructureOperation({
            "temporal_mode": "EVENT_WINDOW",
            "max_temporal_units": 2,
        })
        result = op.extract(
            segments=[
                {
                    "segment_id": "seg_0001",
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 1.0},
                },
                {
                    "segment_id": "seg_0002",
                    "timestamp_range": {"start_sec": 1.0, "end_sec": 2.0},
                },
            ],
            samples=[
                {
                    "sample_id": "smp_000001",
                    "segment_id": "seg_0001",
                    "frame_path": "frame_a.jpg",
                },
                {
                    "sample_id": "smp_000002",
                    "segment_id": "seg_0002",
                    "frame_path": "frame_b.jpg",
                },
            ],
            description_units=[
                {
                    "segment_id": "seg_0001",
                    "description_text": "person opens the door",
                },
                {
                    "segment_id": "seg_0002",
                    "description_text": "person enters the room",
                },
            ],
            text_signals=[
                {
                    "segment_id": "seg_0001",
                    "text": "door opens",
                },
            ],
        )

        assert result["temporal_mode"] == "EVENT_WINDOW"
        assert result["global_metrics"]["total_temporal_structures"] == 2
        assert result["temporal_structures"][0]["temporal_unit_id"] == "tmp_000001"
        assert result["temporal_structures"][0]["segment_ids"] == ["seg_0001"]

    def test_execute_event_window_mode(self):
        from curator.operations.augmentation import VideoTemporalStructureOperation

        ds = ray.data.from_items([
            {
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "timestamp_range": {"start_sec": 0.0, "end_sec": 1.0},
                    },
                    {
                        "segment_id": "seg_0002",
                        "timestamp_range": {"start_sec": 1.0, "end_sec": 2.0},
                    },
                ],
                "samples": [
                    {"sample_id": "smp_000001", "segment_id": "seg_0001", "frame_path": "frame_a.jpg"},
                    {"sample_id": "smp_000002", "segment_id": "seg_0002", "frame_path": "frame_b.jpg"},
                ],
                "description_units": [
                    {"segment_id": "seg_0001", "description_text": "person opens the door"},
                    {"segment_id": "seg_0002", "description_text": "person enters the room"},
                ],
                "text_signals": [
                    {"segment_id": "seg_0001", "text": "door opens"},
                ],
            }
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoTemporalStructureOperation({"temporal_mode": "EVENT_WINDOW"})

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert row["temporal_mode"] == "EVENT_WINDOW"
        assert row["temporal_metrics"]["total_segments"] == 2
        assert len(row["temporal_structures"]) == 2

    def test_extract_event_chain_mode(self):
        from curator.operations.augmentation import VideoTemporalStructureOperation

        op = VideoTemporalStructureOperation({
            "temporal_mode": "EVENT_CHAIN",
            "chain_window_size": 2,
            "max_temporal_units": 2,
        })
        op._generate_event_chain = lambda window_segments, description_map, text_map, sample_map: (
            "then ".join(str(segment["segment_id"]) for segment in window_segments)
        )

        result = op.extract(
            segments=[
                {
                    "segment_id": "seg_0001",
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 1.0},
                },
                {
                    "segment_id": "seg_0002",
                    "timestamp_range": {"start_sec": 1.0, "end_sec": 2.0},
                },
                {
                    "segment_id": "seg_0003",
                    "timestamp_range": {"start_sec": 2.0, "end_sec": 3.0},
                },
            ],
            samples=[
                {"sample_id": "s1", "segment_id": "seg_0001", "frame_path": "frame_a.jpg"},
                {"sample_id": "s2", "segment_id": "seg_0002", "frame_path": "frame_b.jpg"},
                {"sample_id": "s3", "segment_id": "seg_0003", "frame_path": "frame_c.jpg"},
            ],
        )

        assert result["temporal_mode"] == "EVENT_CHAIN"
        assert result["global_metrics"]["total_temporal_structures"] == 2
        assert result["temporal_structures"][0]["structure_type"] == "event_chain"
        assert result["temporal_structures"][0]["summary"] == "seg_0001then seg_0002"


class TestVideoTemporalStructureUtils:
    def test_build_temporal_description_map(self):
        result = build_temporal_description_map([
            {"segment_id": "seg_0001", "description_text": "first"},
            {"segment_id": "seg_0001", "description_text": "second"},
        ])
        assert result == {"seg_0001": ["first", "second"]}

    def test_build_temporal_text_map(self):
        result = build_temporal_text_map([
            {"segment_id": "seg_0001", "text": "subtitle"},
            {"segment_id": "seg_0001", "content": "ocr"},
        ])
        assert result == {"seg_0001": ["subtitle", "ocr"]}

    def test_build_temporal_sample_map(self):
        result = build_temporal_sample_map([
            {"segment_id": "seg_0001", "sample_id": "s1"},
            {"segment_id": "seg_0001", "sample_id": "s2"},
        ])
        assert result["seg_0001"][0]["sample_id"] == "s1"
        assert result["seg_0001"][1]["sample_id"] == "s2"

    def test_build_event_window_structures(self):
        result = build_event_window_structures(
            segments=[
                {
                    "segment_id": "seg_0001",
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 1.0},
                },
                {
                    "segment_id": "seg_0002",
                    "timestamp_range": {"start_sec": 1.0, "end_sec": 2.0},
                },
            ],
            max_temporal_units=2,
            description_map={"seg_0001": ["first"]},
            text_map={"seg_0002": ["second"]},
            sample_map={},
        )
        assert result[0]["structure_type"] == "event_window"
        assert result[0]["segment_ids"] == ["seg_0001"]
        assert "first" in result[0]["summary"]

    def test_build_event_chain_structures(self):
        result = build_event_chain_structures(
            segments=[
                {
                    "segment_id": "seg_0001",
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 1.0},
                },
                {
                    "segment_id": "seg_0002",
                    "timestamp_range": {"start_sec": 1.0, "end_sec": 2.0},
                },
                {
                    "segment_id": "seg_0003",
                    "timestamp_range": {"start_sec": 2.0, "end_sec": 3.0},
                },
            ],
            max_temporal_units=2,
            chain_window_size=2,
            description_map={},
            text_map={},
            sample_map={},
            chain_summary_builder=lambda window_segments, description_map, text_map, sample_map: ":".join(
                str(segment["segment_id"]) for segment in window_segments
            ),
        )
        assert result[0]["structure_type"] == "event_chain"
        assert result[0]["segment_ids"] == ["seg_0001", "seg_0002"]
        assert result[0]["summary"] == "seg_0001:seg_0002"
