"""
Test the video consistency revision operator and related video utility functions
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
    from curator.utils.video_ops_utils import build_revision_records


class TestVideoConsistencyRevisionOperation:
    def test_operation_type(self):
        from curator.operations.augmentation import VideoConsistencyRevisionOperation
        assert VideoConsistencyRevisionOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        from curator.operations.augmentation import VideoConsistencyRevisionOperation
        assert "params" in VideoConsistencyRevisionOperation.PARAMS_SCHEMA
        assert "qae_triplets_key" in VideoConsistencyRevisionOperation.PARAMS_SCHEMA["params"]
        assert "scaffold_units_key" in VideoConsistencyRevisionOperation.PARAMS_SCHEMA["params"]
        assert "revision_mode" in VideoConsistencyRevisionOperation.PARAMS_SCHEMA["params"]
        assert "max_revision_rounds" in VideoConsistencyRevisionOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        from curator.operations.augmentation import VideoConsistencyRevisionOperation
        op = VideoConsistencyRevisionOperation({})
        assert op.params.get("qae_triplets_key") == "qae_triplets"
        assert op.params.get("scaffold_units_key") == "scaffold_units"
        assert op.params.get("revision_mode") == "FIELD_ALIGN"
        assert op.params.get("max_revision_rounds") == 1

    def test_custom_params(self):
        from curator.operations.augmentation import VideoConsistencyRevisionOperation
        op = VideoConsistencyRevisionOperation({
            "qae_triplets_key": "my_triplets",
            "scaffold_units_key": "my_scaffolds",
            "revision_mode": "EVIDENCE_PATCH",
            "max_revision_rounds": 2,
        })
        assert op.params.get("qae_triplets_key") == "my_triplets"
        assert op.params.get("scaffold_units_key") == "my_scaffolds"
        assert op.params.get("revision_mode") == "EVIDENCE_PATCH"
        assert op.params.get("max_revision_rounds") == 2

    def test_execute_field_align_mode(self):
        from curator.operations.augmentation import VideoConsistencyRevisionOperation

        ds = ray.data.from_items([
            {
                "qae_triplets": [
                    {
                        "triplet_id": "tri_000001",
                        "scaffold_id": "scf_0001",
                        "question": "What happens?",
                        "answer": "What happens?",
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
                    }
                ],
                "scaffold_units": [
                    {
                        "scaffold_id": "scf_0001",
                        "scaffold_summary": "A person waves to the camera.",
                    }
                ],
                "qa_drafts": [],
            }
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoConsistencyRevisionOperation({
            "revision_mode": "FIELD_ALIGN",
            "max_revision_rounds": 1,
        })

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert row["revision_mode"] == "FIELD_ALIGN"
        assert row["revision_metrics"]["total_revisions"] == 1
        assert row["revisions"][0]["target_triplet_id"] == "tri_000001"
        assert row["revisions"][0]["after_fields"]["answer"] == "A person waves to the camera."

    def test_registry_create_operation(self):
        op = OperationRegistry.create_operation(
            "video_consistency_revision",
            {"revision_mode": "FIELD_ALIGN"},
        )
        assert op is not None
        assert isinstance(op, Operation)


class TestVideoConsistencyRevisionUtils:
    def test_build_revision_records_field_align(self):
        revisions = build_revision_records(
            revision_mode="FIELD_ALIGN",
            qae_triplets=[
                {
                    "triplet_id": "tri_000001",
                    "scaffold_id": "scf_0001",
                    "question": "What happens?",
                    "answer": "What happens?",
                }
            ],
            scaffold_units=[
                {
                    "scaffold_id": "scf_0001",
                    "scaffold_summary": "A person enters the room.",
                }
            ],
            qa_drafts=[],
            max_revision_rounds=1,
        )
        assert revisions[0]["revision_action"] == "FIELD_ALIGN"
        assert revisions[0]["after_fields"]["answer"] == "A person enters the room."

    def test_build_revision_records_evidence_patch(self):
        revisions = build_revision_records(
            revision_mode="EVIDENCE_PATCH",
            qae_triplets=[
                {
                    "triplet_id": "tri_000001",
                    "scaffold_id": "scf_0001",
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
                }
            ],
            scaffold_units=[
                {
                    "scaffold_id": "scf_0001",
                    "source_evidence_unit_ids": ["ev_0001"],
                    "source_temporal_unit_ids": ["tmp_0001"],
                    "source_text_signal_ids": ["txt_0001"],
                    "source_description_ids": ["desc_0001"],
                }
            ],
            qa_drafts=[],
            max_revision_rounds=1,
        )
        assert revisions[0]["after_fields"]["evidence_refs"] == {
            "text": {
                "description_ids": ["desc_0001"],
                "text_signal_ids": ["txt_0001"],
                "evidence_unit_ids": ["ev_0001"],
            },
            "image": {
                "frame_refs": [],
                "sample_ids": [],
                "frame_paths": [],
            },
            "temporal": {
                "temporal_unit_ids": ["tmp_0001"],
            },
        }

    def test_build_revision_records_draft_reconcile(self):
        revisions = build_revision_records(
            revision_mode="DRAFT_RECONCILE",
            qae_triplets=[
                {
                    "triplet_id": "tri_000001",
                    "scaffold_id": "scf_0001",
                    "source_qa_draft_id": "",
                }
            ],
            scaffold_units=[
                {
                    "scaffold_id": "scf_0001",
                    "source_qa_draft_ids": ["draft_0001"],
                }
            ],
            qa_drafts=[
                {"qa_draft_id": "draft_0001", "question": "Q", "answer": "A"}
            ],
            max_revision_rounds=1,
        )
        assert revisions[0]["after_fields"]["source_qa_draft_id"] == "draft_0001"
