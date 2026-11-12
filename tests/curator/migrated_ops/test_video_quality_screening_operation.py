"""
Test the video quality screening operator and related video utility functions
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
    from curator.utils.video_ops_utils import build_quality_records


class TestVideoQualityScreeningOperation:
    def test_operation_type(self):
        from curator.operations.filtering import VideoQualityScreeningOperation
        assert VideoQualityScreeningOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        from curator.operations.filtering import VideoQualityScreeningOperation
        assert "params" in VideoQualityScreeningOperation.PARAMS_SCHEMA
        assert "qae_triplets_key" in VideoQualityScreeningOperation.PARAMS_SCHEMA["params"]
        assert "revisions_key" in VideoQualityScreeningOperation.PARAMS_SCHEMA["params"]
        assert "quality_mode" in VideoQualityScreeningOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        from curator.operations.filtering import VideoQualityScreeningOperation
        op = VideoQualityScreeningOperation({})
        assert op.params.get("qae_triplets_key") == "qae_triplets"
        assert op.params.get("revisions_key") == "revisions"
        assert op.params.get("quality_mode") == "RULE_FILTER"


    def test_custom_params(self):
        from curator.operations.filtering import VideoQualityScreeningOperation
        op = VideoQualityScreeningOperation({
            "qae_triplets_key": "my_triplets",
            "revisions_key": "my_revisions",
        })
        assert op.params.get("qae_triplets_key") == "my_triplets"
        assert op.params.get("revisions_key") == "my_revisions"

    def test_registry_create_operation(self):
        op = OperationRegistry.create_operation(
            "video_quality_screening",
            {"quality_mode": "RULE_FILTER"},
        )
        assert op is not None
        assert isinstance(op, Operation)


class TestVideoQualityScreeningUtils:
    def test_build_quality_records_rule_filter(self):
        quality_records = build_quality_records(
            quality_mode="RULE_FILTER",
            qae_triplets=[
                {
                    "triplet_id": "tri_000001",
                    "question": "What happens next?",
                    "answer": "The light turns on.",
                    "evidence_refs": {
                        "text": {
                            "description_ids": [],
                            "text_signal_ids": [],
                            "evidence_unit_ids": ["ev_0001"],
                        },
                        "image": {
                            "frame_refs": [],
                            "sample_ids": [],
                            "frame_paths": [],
                        },
                        "temporal": {
                            "temporal_unit_ids": [],
                        },
                    },
                    "triplet_type": "single_hop",
                    "grounding_level": "grounded",
                }
            ],
            revisions=[],
            required_dataset_tier=None,
        )
        assert quality_records[0]["target_triplet_id"] == "tri_000001"
        assert quality_records[0]["quality_mode"] == "RULE_FILTER"
        assert quality_records[0]["decision"] in {"keep", "drop"}

    def test_build_quality_records_with_revisions(self):
        quality_records = build_quality_records(
            quality_mode="TIER_ASSIGN",
            qae_triplets=[
                {
                    "triplet_id": "tri_000001",
                    "question": "",
                    "answer": "Original answer",
                    "evidence_refs": {
                        "text": {
                            "description_ids": [],
                            "text_signal_ids": [],
                            "evidence_unit_ids": ["ev_0001", "ev_0002"],
                        },
                        "image": {
                            "frame_refs": [],
                            "sample_ids": [],
                            "frame_paths": [],
                        },
                        "temporal": {
                            "temporal_unit_ids": [],
                        },
                    },
                    "triplet_type": "multihop",
                    "grounding_level": "grounded",
                }
            ],
            revisions=[
                {
                    "target_triplet_id": "tri_000001",
                    "after_fields": {"question": "What changed in the scene?"},
                }
            ],
            required_dataset_tier=None,
        )
        assert quality_records[0]["question"] == "What changed in the scene?"
        assert quality_records[0]["assigned_dataset_tier"] in {"qa", "grounded_qa", "strong_grounded_qa"}

    def test_build_quality_records_required_tier(self):
        quality_records = build_quality_records(
            quality_mode="TIER_ASSIGN",
            qae_triplets=[
                {
                    "triplet_id": "tri_000001",
                    "question": "Q1",
                    "answer": "A1",
                    "evidence_refs": {
                        "text": {
                            "description_ids": [],
                            "text_signal_ids": [],
                            "evidence_unit_ids": ["ev_0001", "ev_0002", "ev_0003"],
                        },
                        "image": {
                            "frame_refs": [],
                            "sample_ids": [],
                            "frame_paths": [],
                        },
                        "temporal": {
                            "temporal_unit_ids": [],
                        },
                    },
                    "triplet_type": "multihop",
                    "grounding_level": "strong_grounded",
                },
                {
                    "triplet_id": "tri_000002",
                    "question": "Q2",
                    "answer": "A2",
                    "evidence_refs": {
                        "text": {
                            "description_ids": [],
                            "text_signal_ids": [],
                            "evidence_unit_ids": [],
                        },
                        "image": {
                            "frame_refs": [],
                            "sample_ids": [],
                            "frame_paths": [],
                        },
                        "temporal": {
                            "temporal_unit_ids": [],
                        },
                    },
                    "triplet_type": "single_hop",
                    "grounding_level": "grounded",
                },
            ],
            revisions=[],
            required_dataset_tier="strong_grounded_qa",
        )
        decisions = {
            record["target_triplet_id"]: record["decision"]
            for record in quality_records
        }
        assert decisions["tri_000001"] == "keep"
        assert decisions["tri_000002"] == "drop"
