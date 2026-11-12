"""
Test the video multigranularity description operator and video description utility functions
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

    from curator.core.operation import Operation
    from curator.core.operation_registry import OperationRegistry
    from curator.tracking.tracking import OperationType


class TestVideoMultigranularityDescriptionOperation:
    def test_operation_type(self):
        from curator.operations.augmentation import VideoMultigranularityDescriptionOperation
        assert VideoMultigranularityDescriptionOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        from curator.operations.augmentation import VideoMultigranularityDescriptionOperation
        assert "params" in VideoMultigranularityDescriptionOperation.PARAMS_SCHEMA
        assert "segments_key" in VideoMultigranularityDescriptionOperation.PARAMS_SCHEMA["params"]
        assert "samples_key" in VideoMultigranularityDescriptionOperation.PARAMS_SCHEMA["params"]
        assert "description_mode" in VideoMultigranularityDescriptionOperation.PARAMS_SCHEMA["params"]
        assert "frame_path_key" in VideoMultigranularityDescriptionOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        from curator.operations.augmentation import VideoMultigranularityDescriptionOperation
        op = VideoMultigranularityDescriptionOperation({})
        assert op.params.get("segments_key") == "segments"
        assert op.params.get("samples_key") == "samples"
        assert op.params.get("description_mode") == "SEGMENT_DESCRIPTION"
        assert op.params.get("frame_path_key") == "frame_path"

    def test_custom_params(self):
        from curator.operations.augmentation import VideoMultigranularityDescriptionOperation
        op = VideoMultigranularityDescriptionOperation({
            "segments_key": "my_segments",
            "samples_key": "my_samples",
            "description_mode": "FRAME_DESCRIPTION",
            "frame_path_key": "image_path",
        })
        assert op.params.get("segments_key") == "my_segments"
        assert op.params.get("samples_key") == "my_samples"
        assert op.params.get("description_mode") == "FRAME_DESCRIPTION"
        assert op.params.get("frame_path_key") == "image_path"

    def test_registry_create_operation(self):
        op = OperationRegistry.create_operation(
            "video_multigranularity_description",
            {"description_mode": "SEGMENT_DESCRIPTION"},
        )
        assert op is not None
        assert isinstance(op, Operation)

    def test_describe_segments_mode(self):
        from curator.operations.augmentation import VideoMultigranularityDescriptionOperation

        op = VideoMultigranularityDescriptionOperation({"description_mode": "SEGMENT_DESCRIPTION"})
        op._load_caption_components = lambda: (object(), object(), object())
        op._caption_frame = lambda frame_path, prompt_text, model, processor, torch, caption_cache: {
            "frame_a.jpg": "cat on sofa",
            "frame_b.jpg": "cat on sofa",
            "frame_c.jpg": "dog running",
        }[frame_path]

        result = op.describe(
            segments=[
                {"segment_id": "seg_0001"},
                {"segment_id": "seg_0002"},
            ],
            samples=[
                {"sample_id": "smp_000001", "segment_id": "seg_0001", "frame_path": "frame_a.jpg"},
                {"sample_id": "smp_000002", "segment_id": "seg_0001", "frame_path": "frame_b.jpg"},
                {"sample_id": "smp_000003", "segment_id": "seg_0002", "frame_path": "frame_c.jpg"},
            ],
        )

        assert result["description_mode"] == "SEGMENT_DESCRIPTION"
        assert result["global_metrics"]["total_description_units"] == 2
        assert result["description_units"][0]["description_text"] == "cat on sofa"
        assert result["description_units"][0]["description_id"] == "desc_000001"
        assert result["description_units"][0]["description_unit_id"] == "desc_000001"
        assert result["description_units"][0]["id"] == "desc_000001"
        assert result["description_units"][1]["description_text"] == "dog running"

    def test_describe_multiple_modes_and_default_context_group(self):
        from curator.operations.augmentation import VideoMultigranularityDescriptionOperation

        op = VideoMultigranularityDescriptionOperation({
            "description_modes": [
                "FRAME_DESCRIPTION",
                "EVENT_DESCRIPTION",
                "SEGMENT_DESCRIPTION",
                "CONTEXT_DESCRIPTION",
            ],
        })
        op._load_caption_components = lambda: (object(), object(), object())
        op._caption_frame = (
            lambda frame_path, prompt_text, model, processor, torch, caption_cache:
            f"{prompt_text} {frame_path}"
        )

        result = op.describe(
            segments=[
                {"segment_id": "seg_0001"},
                {"segment_id": "seg_0002"},
            ],
            samples=[
                {"sample_id": "smp_000001", "segment_id": "seg_0001", "frame_path": "frame_a.jpg"},
                {"sample_id": "smp_000002", "segment_id": "seg_0002", "frame_path": "frame_b.jpg"},
            ],
            temporal_structures=[
                {"temporal_unit_id": "tmp_000001", "segment_ids": ["seg_0001"]},
                {"temporal_unit_id": "tmp_000002", "segment_ids": ["seg_0002"]},
            ],
        )

        assert result["description_mode"] == "MULTI_GRANULARITY"
        assert result["description_modes"] == [
            "FRAME_DESCRIPTION",
            "EVENT_DESCRIPTION",
            "SEGMENT_DESCRIPTION",
            "CONTEXT_DESCRIPTION",
        ]
        assert result["global_metrics"]["description_units_by_mode"] == {
            "FRAME_DESCRIPTION": 2,
            "EVENT_DESCRIPTION": 2,
            "SEGMENT_DESCRIPTION": 2,
            "CONTEXT_DESCRIPTION": 1,
        }
        assert {unit["granularity"] for unit in result["description_units"]} == {
            "frame",
            "event",
            "segment",
            "context",
        }
        context_units = [
            unit for unit in result["description_units"]
            if unit["description_mode"] == "CONTEXT_DESCRIPTION"
        ]
        assert context_units[0]["description_scope"] == "stitch_group:stitch_0001"
        assert context_units[0]["anchor_refs"]["segment_ids"] == ["seg_0001", "seg_0002"]
