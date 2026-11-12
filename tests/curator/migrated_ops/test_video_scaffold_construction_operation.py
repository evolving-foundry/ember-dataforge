"""
Test the video scaffold construction operator and scaffold utility functions
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
    from curator.operations.augmentation.video_scaffold_construction_operation import (
        build_action_graph_scaffold,
        build_event_tree_scaffold,
        build_narration_chunk_scaffolds,
    )
    from curator.utils.video_ops_utils import (
        build_scaffold_description_map,
        build_scaffold_text_map,
        build_temporal_sample_map,
    )


def fake_action_scene_classifier(samples, frame_path_key):
    label_map = {
        "frame_open.jpg": "opening",
        "frame_take.jpg": "taking",
        "frame_stir.jpg": "stirring",
        "frame_dance_a.jpg": "dancing",
        "frame_dance_b.jpg": "dancing",
        "frame_run.jpg": "running",
    }
    return [
        {
            "label": label_map[sample[frame_path_key]],
            "score": 0.9,
            "segment_id": str(sample["segment_id"]),
            "sample_id": str(sample["sample_id"]),
            "frame_path": sample[frame_path_key],
            "frame_index": sample.get("frame_index"),
            "timestamp_sec": sample.get("timestamp_sec"),
        }
        for sample in samples
    ]


class TestVideoScaffoldConstructionOperation:
    def _build_inputs(self):
        return {
            "description_units": [
                {"description_id": "desc_000001", "segment_id": "seg_0001", "description_text": "person opens drawer"},
                {"description_id": "desc_000002", "segment_id": "seg_0002", "description_text": "person takes spoon"},
                {"description_id": "desc_000003", "segment_id": "seg_0003", "description_text": "person stirs cup"},
            ],
            "temporal_structures": [
                {
                    "temporal_unit_id": "tmp_000001",
                    "segment_id": "seg_0001",
                    "segment_ids": ["seg_0001", "seg_0002", "seg_0003"],
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 30.0},
                    "summary": "open drawer, take spoon, and stir cup",
                    "precedence_refs": ["seg_0001"],
                }
            ],
            "text_signals": [
                {"text_signal_id": "txt_000001", "segment_id": "seg_0001", "text": "drawer"},
                {"text_signal_id": "txt_000002", "segment_id": "seg_0002", "text": "spoon"},
                {"text_signal_id": "txt_000003", "segment_id": "seg_0003", "text": "cup"},
            ],
            "samples": [
                {"sample_id": "smp_000001", "segment_id": "seg_0001", "frame_path": "frame_open.jpg", "frame_index": 10},
                {"sample_id": "smp_000002", "segment_id": "seg_0002", "frame_path": "frame_take.jpg", "frame_index": 20},
                {"sample_id": "smp_000003", "segment_id": "seg_0003", "frame_path": "frame_stir.jpg", "frame_index": 30},
            ],
        }

    def _fake_action_scene_classifier(self, samples, frame_path_key):
        return fake_action_scene_classifier(samples, frame_path_key)

    def test_operation_type(self):
        from curator.operations.augmentation import VideoScaffoldConstructionOperation
        assert VideoScaffoldConstructionOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        from curator.operations.augmentation import VideoScaffoldConstructionOperation
        assert "params" in VideoScaffoldConstructionOperation.PARAMS_SCHEMA
        assert "description_units_key" in VideoScaffoldConstructionOperation.PARAMS_SCHEMA["params"]
        assert "temporal_structures_key" in VideoScaffoldConstructionOperation.PARAMS_SCHEMA["params"]
        assert "text_signals_key" in VideoScaffoldConstructionOperation.PARAMS_SCHEMA["params"]
        assert "samples_key" in VideoScaffoldConstructionOperation.PARAMS_SCHEMA["params"]
        assert "frame_path_key" in VideoScaffoldConstructionOperation.PARAMS_SCHEMA["params"]
        assert "action_scene_model_name" in VideoScaffoldConstructionOperation.PARAMS_SCHEMA["params"]
        assert "scaffold_mode" in VideoScaffoldConstructionOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        from curator.operations.augmentation import VideoScaffoldConstructionOperation
        op = VideoScaffoldConstructionOperation({})
        assert op.params.get("description_units_key") == "description_units"
        assert op.params.get("temporal_structures_key") == "temporal_structures"
        assert op.params.get("text_signals_key") == "text_signals"
        assert op.params.get("samples_key") == "samples"
        assert op.params.get("frame_path_key") == "frame_path"
        assert op.params.get("scaffold_mode") == "NARRATION_CHUNK"

    def test_registry_create_operation(self):
        op = OperationRegistry.create_operation(
            "video_scaffold_construction",
            {"scaffold_mode": "EVENT_TREE"},
        )
        assert op is not None
        assert isinstance(op, Operation)

    @pytest.mark.parametrize("mode", ["NARRATION_CHUNK", "EVENT_TREE", "ACTION_GRAPH"])
    def test_construct_modes(self, mode):
        from curator.operations.augmentation import VideoScaffoldConstructionOperation

        op = VideoScaffoldConstructionOperation({"scaffold_mode": mode, "leaf_window_sec": 10.0})
        if mode == "ACTION_GRAPH":
            op._classify_action_scene_frames = self._fake_action_scene_classifier
        payload = self._build_inputs()
        result = op.construct(**payload)

        assert result["scaffold_mode"] == mode
        assert result["global_metrics"]["total_scaffold_units"] > 0
        assert all("scaffold_id" in scaffold_unit for scaffold_unit in result["scaffold_units"])
        assert all("scaffold_summary" in scaffold_unit for scaffold_unit in result["scaffold_units"])

        if mode == "EVENT_TREE":
            assert result["scaffold_units"][0]["root_ref"]
            assert len(result["scaffold_units"][0]["leaf_refs"]) == 3
            assert len(result["scaffold_units"][0]["tree_edges"]) == 3
        if mode == "ACTION_GRAPH":
            assert result["scaffold_units"][0]["action_nodes"]
            assert any(node["node_type"] == "action" for node in result["scaffold_units"][0]["action_nodes"])
            assert any(node["node_type"] == "frame" for node in result["scaffold_units"][0]["action_nodes"])

    def test_execute(self):
        from curator.operations.augmentation import VideoScaffoldConstructionOperation

        ds = ray.data.from_items([self._build_inputs()])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoScaffoldConstructionOperation({"scaffold_mode": "EVENT_TREE", "leaf_window_sec": 10.0})

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert row["scaffold_mode"] == "EVENT_TREE"
        assert row["scaffold_metrics"]["total_temporal_structures"] == 1
        assert len(row["scaffold_units"]) == 1


class TestVideoScaffoldConstructionUtils:
    def test_build_scaffold_description_map(self):
        result = build_scaffold_description_map([
            {"description_id": "desc_1", "segment_id": "seg_0001", "description_text": "first"},
            {"description_id": "desc_2", "segment_id": "seg_0001", "description_text": "second"},
        ])
        assert result["seg_0001"]["ids"] == ["desc_1", "desc_2"]
        assert len(result["seg_0001"]["units"]) == 2

    def test_build_scaffold_text_map(self):
        result = build_scaffold_text_map([
            {"text_signal_id": "txt_1", "segment_id": "seg_0001", "text": "subtitle"},
            {"text_signal_id": "txt_2", "segment_id": "seg_0001", "text": "ocr"},
        ])
        assert len(result["seg_0001"]) == 2
        assert result["seg_0001"][0]["text_signal_id"] == "txt_1"

    def test_build_narration_chunk_scaffolds(self):
        result = build_narration_chunk_scaffolds(
            temporal_structures=[
                {"temporal_unit_id": "tmp_000001", "segment_id": "seg_0001"},
            ],
            description_map={"seg_0001": {"ids": ["desc_000001"], "units": []}},
            text_map={
                "seg_0001": [
                    {"text_signal_id": "txt_000001", "segment_id": "seg_0001", "text": "drawer"},
                    {"text_signal_id": "txt_000002", "segment_id": "seg_0001", "text": "opens"},
                ]
            },
            scaffold_role="generation",
            max_children_per_unit=4,
            sample_map={
                "seg_0001": [
                    {
                        "sample_id": "smp_000001",
                        "segment_id": "seg_0001",
                        "frame_path": "/tmp/frame.jpg",
                        "frame_index": 10,
                        "timestamp_sec": 1.0,
                    }
                ]
            },
        )
        assert result[0]["unit_type"] == "narration_chunk"
        assert result[0]["scaffold_summary"] == "drawer / opens"
        assert result[0]["visual_evidence"][0]["sample_id"] == "smp_000001"
        assert result[0]["visual_evidence"][0]["frame_path"] == "/tmp/frame.jpg"

    def test_build_narration_chunk_scaffolds_prefers_event_summary(self):
        result = build_narration_chunk_scaffolds(
            temporal_structures=[
                {"temporal_unit_id": "tmp_000001", "segment_id": "seg_0001"},
            ],
            description_map={
                "seg_0001": {
                    "ids": ["desc_frame", "desc_segment", "desc_event"],
                    "units": [
                        {
                            "description_id": "desc_frame",
                            "segment_id": "seg_0001",
                            "description_text": "frame caption",
                            "granularity": "frame",
                        },
                        {
                            "description_id": "desc_segment",
                            "segment_id": "seg_0001",
                            "description_text": "segment caption",
                            "granularity": "segment",
                        },
                        {
                            "description_id": "desc_event",
                            "segment_id": "seg_0001",
                            "description_text": "event caption",
                            "granularity": "event",
                        },
                    ],
                },
            },
            text_map={},
            scaffold_role="generation",
            max_children_per_unit=4,
            sample_map={},
        )

        assert result[0]["scaffold_summary"] == "event caption"
        assert result[0]["source_description_ids"] == [
            "desc_frame",
            "desc_segment",
            "desc_event",
        ]

    def test_build_narration_chunk_scaffolds_falls_back_to_segment_then_frame(self):
        base = {
            "temporal_structures": [
                {"temporal_unit_id": "tmp_000001", "segment_id": "seg_0001"},
            ],
            "text_map": {},
            "scaffold_role": "generation",
            "max_children_per_unit": 4,
            "sample_map": {},
        }

        segment_result = build_narration_chunk_scaffolds(
            description_map={
                "seg_0001": {
                    "ids": ["desc_frame", "desc_segment"],
                    "units": [
                        {
                            "description_id": "desc_frame",
                            "description_text": "frame caption",
                            "description_mode": "FRAME_DESCRIPTION",
                        },
                        {
                            "description_id": "desc_segment",
                            "description_text": "segment caption",
                            "description_mode": "SEGMENT_DESCRIPTION",
                        },
                    ],
                },
            },
            **base,
        )
        frame_result = build_narration_chunk_scaffolds(
            description_map={
                "seg_0001": {
                    "ids": ["desc_frame"],
                    "units": [
                        {
                            "description_id": "desc_frame",
                            "description_text": "frame caption",
                            "description_mode": "FRAME_DESCRIPTION",
                        },
                    ],
                },
            },
            **base,
        )

        assert segment_result[0]["scaffold_summary"] == "segment caption"
        assert frame_result[0]["scaffold_summary"] == "frame caption"

    def test_build_event_tree_scaffold(self):
        result = build_event_tree_scaffold(
            temporal_structure={
                "temporal_unit_id": "tmp_000001",
                "segment_ids": ["seg_0001", "seg_0002", "seg_0003"],
                "timestamp_range": {"start_sec": 0.0, "end_sec": 30.0},
                "summary": "open drawer, take spoon, and stir cup",
            },
            description_map={
                "seg_0001": {"ids": ["desc_000001"], "units": [{"description_text": "open drawer"}]},
                "seg_0002": {"ids": ["desc_000002"], "units": [{"description_text": "take spoon"}]},
                "seg_0003": {"ids": ["desc_000003"], "units": [{"description_text": "stir cup"}]},
            },
            text_map={},
            scaffold_role="generation",
            max_children_per_unit=4,
            leaf_window_sec=10.0,
        )
        assert result["root_ref"] == "tmp_000001_root"
        assert len(result["leaf_refs"]) == 3
        assert len(result["tree_edges"]) == 3

    def test_build_action_graph_scaffold(self):
        result = build_action_graph_scaffold(
            temporal_structure={
                "temporal_unit_id": "tmp_000001",
                "segment_ids": ["seg_0001", "seg_0002"],
                "summary": "open drawer then take spoon",
            },
            description_map={
                "seg_0001": {"ids": ["desc_000001"], "units": [{"description_id": "desc_000001", "description_text": "opens drawer"}]},
                "seg_0002": {"ids": ["desc_000002"], "units": [{"description_id": "desc_000002", "description_text": "takes spoon"}]},
            },
            text_map={},
            sample_map=build_temporal_sample_map([
                {"sample_id": "smp_000001", "segment_id": "seg_0001", "frame_path": "frame_dance_a.jpg"},
                {"sample_id": "smp_000002", "segment_id": "seg_0001", "frame_path": "frame_dance_b.jpg"},
                {"sample_id": "smp_000003", "segment_id": "seg_0002", "frame_path": "frame_run.jpg"},
            ]),
            scaffold_role="generation",
            max_children_per_unit=4,
            action_scene_classifier=fake_action_scene_classifier,
        )
        action_nodes = [node for node in result["action_nodes"] if node["node_type"] == "action"]
        assert [node["label"] for node in action_nodes] == ["dancing", "running"]
        assert action_nodes[0]["sample_ids"] == ["smp_000001", "smp_000002"]
        assert any(node["node_type"] == "frame" for node in result["action_nodes"])
        assert any(edge["relation"] == "observed_in" for edge in result["action_edges"])
        assert any(edge["relation"] == "next" for edge in result["action_edges"])
