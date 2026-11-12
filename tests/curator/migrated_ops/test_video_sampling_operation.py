"""
Test the video sampling operator and video utility functions
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
        resolve_frame_count,
        select_motion_diverse_indices,
        select_semantic_diverse_indices,
        select_state_frame_indices,
        select_uniform_frame_indices,
        unique_preserve_order,
    )


class TestVideoSamplingOperation:
    def test_operation_type(self):
        from curator.operations.augmentation import VideoSamplingOperation
        assert VideoSamplingOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        from curator.operations.augmentation import VideoSamplingOperation
        assert "params" in VideoSamplingOperation.PARAMS_SCHEMA
        assert "video_key" in VideoSamplingOperation.PARAMS_SCHEMA["params"]
        assert "segments_key" in VideoSamplingOperation.PARAMS_SCHEMA["params"]
        assert "sampling_mode" in VideoSamplingOperation.PARAMS_SCHEMA["params"]
        assert "num_frames" in VideoSamplingOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        from curator.operations.augmentation import VideoSamplingOperation
        op = VideoSamplingOperation({})
        assert op.params.get("video_key") == "video_path"
        assert op.params.get("segments_key") == "segments"
        assert op.params.get("sampling_mode") == "CENTER"
        assert op.params.get("num_frames") == 4

    def test_custom_params(self):
        from curator.operations.augmentation import VideoSamplingOperation
        op = VideoSamplingOperation({
            "video_key": "my_video",
            "segments_key": "my_segments",
            "sampling_mode": "STATE",
            "padding_frames": 3,
        })
        assert op.params.get("video_key") == "my_video"
        assert op.params.get("segments_key") == "my_segments"
        assert op.params.get("sampling_mode") == "STATE"
        assert op.params.get("padding_frames") == 3

    def test_execute_center_mode(self):
        from curator.operations.augmentation import VideoSamplingOperation

        ds = ray.data.from_items([
            {
                "video_path": "demo.mp4",
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "timestamp_range": {"start_sec": 0.0, "end_sec": 2.0},
                        "frame_range": {"start_frame": 0, "end_frame": 4},
                    }
                ],
                "video_fps": 2.0,
            }
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoSamplingOperation({"sampling_mode": "CENTER"})

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert len(row["samples"]) == 1
        assert row["samples"][0]["frame_index"] == 1
        assert row["samples"][0]["timestamp_sec"] == pytest.approx(0.5)
        assert row["sampling_metrics"]["total_samples"] == 1

    def test_execute_center_mode_uses_timestamp_range_with_video_fps(self):
        from curator.operations.augmentation import VideoSamplingOperation

        ds = ray.data.from_items([
            {
                "video_path": "demo.mp4",
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "timestamp_range": {"start_sec": 6.0, "end_sec": 11.261},
                        "frame_range": {"start_frame": 6, "end_frame": 11},
                    }
                ],
                "video_fps": 23.976,
            }
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoSamplingOperation({"sampling_mode": "CENTER"})

        result = op.execute(dataset)
        sample = result.dataset.take(1)[0]["samples"][0]

        assert sample["frame_index"] > 11
        assert 6.0 <= sample["timestamp_sec"] <= 11.261

    def test_execute_semantic_mode(self):
        from curator.operations.augmentation import VideoSamplingOperation

        ds = ray.data.from_items([
            {
                "video_path": "demo.mp4",
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "timestamp_range": {"start_sec": 0.0, "end_sec": 2.0},
                        "frame_range": {"start_frame": 0, "end_frame": 4},
                    }
                ],
                "video_fps": 2.0,
                "semantic_frame_features": {
                    0: [1.0, 0.0],
                    1: [1.0, 0.0],
                    2: [0.0, 1.0],
                    3: [-1.0, 0.0],
                },
            }
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoSamplingOperation({
            "sampling_mode": "SEMANTIC_AWARE",
            "num_frames": 3,
            "semantic_min_distance_threshold": 0.5,
        })

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert len(row["samples"]) >= 1
        assert row["sampling_result"]["sampling_mode"] == "SEMANTIC_AWARE"
        assert row["sampling_metrics"]["total_segments"] == 1

    def test_registry_create_operation(self):
        op = OperationRegistry.create_operation("video_sampling", {"sampling_mode": "CENTER"})
        assert op is not None
        assert isinstance(op, Operation)

    def test_resolve_video_fps_from_segments(self):
        from curator.operations.augmentation import VideoSamplingOperation

        fps = VideoSamplingOperation._resolve_video_fps(
            [
                {
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 2.0},
                    "frame_range": {"start_frame": 0, "end_frame": 4},
                }
            ]
        )
        assert fps == pytest.approx(2.0)

    def test_sample_frame_path_generation(self, tmp_path):
        from curator.operations.augmentation import VideoSamplingOperation

        op = VideoSamplingOperation({
            "frame_output_dir": str(tmp_path),
            "frame_image_format": ".jpg",
        })
        sample = {"sample_id": "smp_000001", "frame_index": 7}
        output_path = op._sample_frame_path("/videos/demo.mp4", sample)

        assert output_path.endswith(os.path.join("smp_000001_7.jpg"))
        assert str(tmp_path) in output_path

    def test_save_sample_frames_reuses_existing_file(self, tmp_path, monkeypatch):
        from curator.operations.augmentation import video_sampling_operation
        from curator.operations.augmentation import VideoSamplingOperation

        class FakeCapture:
            def __init__(self):
                self.read_called = False

            def isOpened(self):
                return True

            def get(self, _prop):
                return 10

            def set(self, _prop, _value):
                return True

            def read(self):
                self.read_called = True
                return True, object()

            def release(self):
                return None

        class FakeCV2:
            CAP_PROP_POS_FRAMES = 1

            def __init__(self):
                self.imwrite_called = False

            def imwrite(self, _path, _frame):
                self.imwrite_called = True
                return True

        class FakeBackend:
            CAP_PROP_FRAME_COUNT = 7

            def __init__(self):
                self.cv2 = FakeCV2()
                self.capture = FakeCapture()

            def get_cv2(self):
                return self.cv2

            def VideoCapture(self, _path):
                return self.capture

        fake_backend = FakeBackend()
        monkeypatch.setattr(video_sampling_operation, "get_cv2_backend", lambda: fake_backend)

        op = VideoSamplingOperation({
            "save_sample_frames": True,
            "frame_output_dir": str(tmp_path),
        })
        sample = {"sample_id": "smp_000001", "frame_index": 3}
        existing_path = op._sample_frame_path("demo.mp4", sample)
        os.makedirs(os.path.dirname(existing_path), exist_ok=True)
        with open(existing_path, "wb") as file_obj:
            file_obj.write(b"existing")

        op._save_sample_frames("demo.mp4", [sample])

        assert sample["frame_path"] == existing_path
        assert fake_backend.capture.read_called is False
        assert fake_backend.cv2.imwrite_called is False

    def test_save_sample_frames_missing_video(self, tmp_path, monkeypatch):
        from curator.operations.augmentation import video_sampling_operation
        from curator.operations.augmentation import VideoSamplingOperation

        class FakeCapture:
            def isOpened(self):
                return False

            def release(self):
                return None

        class FakeBackend:
            def get_cv2(self):
                return object()

            def VideoCapture(self, _path):
                return FakeCapture()

        monkeypatch.setattr(video_sampling_operation, "get_cv2_backend", lambda: FakeBackend())
        op = VideoSamplingOperation({
            "save_sample_frames": True,
            "frame_output_dir": str(tmp_path),
        })

        with pytest.raises(ValueError, match="Cannot open video"):
            op._save_sample_frames("missing.mp4", [{"sample_id": "smp_000001", "frame_index": 0}])


class TestVideoSamplingUtils:
    def test_unique_preserve_order(self):
        assert unique_preserve_order([1, 2, 2, 3, 1]) == [1, 2, 3]

    def test_resolve_frame_count(self):
        assert resolve_frame_count(0, 10, 8, max_frames_per_segment=3) == 3

    def test_select_uniform_frame_indices(self):
        assert select_uniform_frame_indices(0, 4, fps=2.0, num_frames=3) == [0, 2, 3]

    def test_select_state_frame_indices(self):
        indices, sample_types = select_state_frame_indices(0, 6, padding_frames=2)
        assert indices == [2, 3]
        assert sample_types == ["STATE_HEAD", "STATE_TAIL"]

    def test_select_semantic_diverse_indices(self):
        indices = select_semantic_diverse_indices(
            candidate_indices=[0, 1, 2],
            frame_count=2,
            semantic_frame_features={
                0: [1.0, 0.0],
                1: [1.0, 0.0],
                2: [0.0, 1.0],
            },
            min_distance_threshold=0.5,
        )
        assert indices == [0, 2]

    def test_select_motion_diverse_indices(self):
        indices = select_motion_diverse_indices(
            candidate_indices=[0, 1, 2],
            frame_count=2,
            semantic_frame_features={
                0: [1.0, 0.0],
                1: [1.0, 0.0],
                2: [0.0, 1.0],
            },
            min_distance_threshold=0.5,
        )
        assert indices == [0, 2]
