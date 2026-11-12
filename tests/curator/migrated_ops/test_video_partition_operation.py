"""
Test the video partition operator and video utility functions
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
        compute_video_semantic_embeddings,
        cosine_distance,
        sec_to_frame,
    )


class TestVideoPartitionOperation:
    def test_operation_type(self):
        from curator.operations.augmentation import VideoPartitionOperation
        assert VideoPartitionOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        from curator.operations.augmentation import VideoPartitionOperation
        assert "params" in VideoPartitionOperation.PARAMS_SCHEMA
        assert "video_key" in VideoPartitionOperation.PARAMS_SCHEMA["params"]
        assert "partition_mode" in VideoPartitionOperation.PARAMS_SCHEMA["params"]
        assert "video_duration_sec" in VideoPartitionOperation.PARAMS_SCHEMA["params"]
        assert "shot_fps" in VideoPartitionOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        from curator.operations.augmentation import VideoPartitionOperation
        op = VideoPartitionOperation({})
        assert op.params.get("video_key") == "video_path"
        assert op.params.get("partition_mode") == "FIXED"
        assert op.params.get("chunk_duration_sec") == 5.0
        assert op.params.get("shot_fps") == 1.0

    def test_custom_params(self):
        from curator.operations.augmentation import VideoPartitionOperation
        op = VideoPartitionOperation({
            "video_key": "my_video",
            "partition_mode": "SHOT",
            "shot_threshold": 0.8,
            "shot_fps": 2.0,
        })
        assert op.params.get("video_key") == "my_video"
        assert op.params.get("partition_mode") == "SHOT"
        assert op.params.get("shot_threshold") == 0.8
        assert op.params.get("shot_fps") == 2.0

    def test_execute_fixed_mode(self):
        from curator.operations.augmentation import VideoPartitionOperation

        ds = ray.data.from_items([
            {"video_path": "demo.mp4", "video_duration_sec": 12.0}
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoPartitionOperation({"chunk_duration_sec": 5.0, "shot_fps": 2.0})

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert len(row["segments"]) == 3
        assert row["segments"][0]["timestamp_range"] == {"start_sec": 0.0, "end_sec": 5.0}
        assert row["segments"][1]["timestamp_range"] == {"start_sec": 5.0, "end_sec": 10.0}
        assert row["segments"][2]["timestamp_range"] == {"start_sec": 10.0, "end_sec": 12.0}
        assert row["partition_metrics"]["total_segments"] == 3

    def test_execute_semantic_mode(self):
        from curator.operations.augmentation import VideoPartitionOperation

        ds = ray.data.from_items([
            {
                "video_path": "demo.mp4",
                "semantic_embeddings": [
                    {"start_sec": 0.0, "end_sec": 4.0, "embedding": [1.0, 0.0]},
                    {"start_sec": 4.0, "end_sec": 8.0, "embedding": [0.0, 1.0]},
                ],
            }
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoPartitionOperation({
            "partition_mode": "SEMANTIC",
            "semantic_distance_threshold": 0.5,
            "min_segment_duration_sec": 1.0,
        })

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert len(row["segments"]) == 2
        assert row["segments"][1]["boundary_type"] == "SEMANTIC_SHIFT"
        assert row["partition_result"]["partition_details"]["window_count"] == 2

    def test_registry_create_operation(self):
        op = OperationRegistry.create_operation("video_partition", {"video_duration_sec": 6.0})
        assert op is not None
        assert isinstance(op, Operation)

    def test_partition_shot_mode_row_level_shot_fps(self):
        from curator.operations.augmentation import VideoPartitionOperation

        op = VideoPartitionOperation(
            {
                "partition_mode": "SHOT",
                "shot_threshold": 0.5,
                "shot_fps": 1.0,
                "min_segment_duration_sec": 0.0,
            }
        )

        result = op.partition(
            video_path="demo.mp4",
            shot_scores=[0.0, 0.9, 0.0, 0.0],
            shot_fps=2.0,
        )

        assert result["partition_details"]["fps"] == pytest.approx(2.0)
        assert result["segments"][0]["timestamp_range"]["end_sec"] == pytest.approx(0.5)
        assert result["segments"][0]["frame_range"]["end_frame"] == 1


class TestVideoOpsUtils:
    def test_cosine_distance_same_vector(self):
        assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_cosine_distance_orthogonal(self):
        assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_sec_to_frame(self):
        assert sec_to_frame(2.5, 4) == 10

    def test_compute_video_semantic_embeddings_window_validation(self):
        with pytest.raises(ValueError, match="window_sec must be positive"):
            compute_video_semantic_embeddings("demo.mp4", window_sec=0)

    def test_compute_video_semantic_embeddings_batch_validation(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            compute_video_semantic_embeddings("demo.mp4", batch_size=0)
