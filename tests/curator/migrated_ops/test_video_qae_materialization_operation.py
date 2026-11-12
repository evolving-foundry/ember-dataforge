"""
Test the video QAE materialization operator and related video utility functions
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
        build_qae_materialization_evidence_map,
        materialize_qae_from_scaffold,
        materialize_qae_with_temporal_grounding,
    )


class TestVideoQaeMaterializationOperation:
    def test_operation_type(self):
        from curator.operations.augmentation import VideoQaeMaterializationOperation
        assert VideoQaeMaterializationOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        from curator.operations.augmentation import VideoQaeMaterializationOperation
        assert "params" in VideoQaeMaterializationOperation.PARAMS_SCHEMA
        assert "scaffold_units_key" in VideoQaeMaterializationOperation.PARAMS_SCHEMA["params"]
        assert "qa_drafts_key" in VideoQaeMaterializationOperation.PARAMS_SCHEMA["params"]
        assert "materialization_mode" in VideoQaeMaterializationOperation.PARAMS_SCHEMA["params"]
        assert "triplet_type" in VideoQaeMaterializationOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        from curator.operations.augmentation import VideoQaeMaterializationOperation
        op = VideoQaeMaterializationOperation({"use_llm_client": False})
        assert op.params.get("scaffold_units_key") == "scaffold_units"
        assert op.params.get("qa_drafts_key") == "qa_drafts"
        assert op.params.get("materialization_mode") == "JOINT_SYNTHESIS"
        assert op.params.get("triplet_type") == "single_hop"

    def test_custom_params(self):
        from curator.operations.augmentation import VideoQaeMaterializationOperation
        op = VideoQaeMaterializationOperation({
            "use_llm_client": False,
            "scaffold_units_key": "my_scaffolds",
            "qa_drafts_key": "my_drafts",
            "materialization_mode": "DRAFT_BINDING",
            "triplet_type": "multihop",
        })
        assert op.params.get("scaffold_units_key") == "my_scaffolds"
        assert op.params.get("qa_drafts_key") == "my_drafts"
        assert op.params.get("materialization_mode") == "DRAFT_BINDING"
        assert op.params.get("triplet_type") == "multihop"

    def test_execute_joint_synthesis_mode(self):
        from curator.operations.augmentation import VideoQaeMaterializationOperation

        ds = ray.data.from_items([
            {
                "scaffold_units": [
                    {
                        "scaffold_id": "scf_0001",
                        "scaffold_summary": "A person opens the door.",
                        "question_focus": "the door opening",
                        "question_type": "what",
                        "source_qa_draft_ids": ["draft_0001"],
                        "source_evidence_unit_ids": ["ev_0001"],
                    }
                ],
                "qa_drafts": [
                    {
                        "qa_draft_id": "draft_0001",
                        "question": "What happens at the start?",
                        "answer": "A person opens the door.",
                    }
                ],
                "evidence_units": [
                    {
                        "evidence_unit_id": "ev_0001",
                        "evidence_summary": "The person reaches for and opens the door.",
                    }
                ],
            }
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoQaeMaterializationOperation({
            "use_llm_client": False,
            "materialization_mode": "JOINT_SYNTHESIS",
            "triplet_type": "single_hop",
        })

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert row["materialization_mode"] == "JOINT_SYNTHESIS"
        assert row["materialization_metrics"]["total_triplets"] == 1
        assert row["qae_triplets"][0]["triplet_id"] == "tri_000001"
        assert row["qae_triplets"][0]["question"] == "What happens at the start?"
        assert row["qae_triplets"][0]["evidence_refs"] == {
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
        }

    def test_registry_create_operation(self):
        op = OperationRegistry.create_operation(
            "video_qae_materialization",
            {"materialization_mode": "JOINT_SYNTHESIS", "use_llm_client": False},
        )
        assert op is not None
        assert isinstance(op, Operation)


class TestVideoQaeMaterializationUtils:
    def test_build_qae_materialization_evidence_map(self):
        evidence_map = build_qae_materialization_evidence_map([
            {"evidence_unit_id": "ev_0001", "evidence_summary": "first"},
            {"evidence_unit_id": "ev_0002", "evidence_summary": "second"},
        ])
        assert evidence_map["ev_0001"]["evidence_summary"] == "first"
        assert evidence_map["ev_0002"]["evidence_summary"] == "second"

    def test_materialize_qae_from_scaffold(self):
        qae_triplets = materialize_qae_from_scaffold(
            materialization_mode="DRAFT_BINDING",
            scaffold_units=[
                {
                    "scaffold_id": "scf_0001",
                    "scaffold_summary": "A dog runs across the yard.",
                    "question_focus": "the dog's movement",
                    "question_type": "what",
                    "source_qa_draft_ids": ["draft_0001"],
                    "source_evidence_unit_ids": ["ev_0001"],
                }
            ],
            qa_drafts=[
                {
                    "qa_draft_id": "draft_0001",
                    "question": "What is the dog doing?",
                    "answer": "It runs across the yard.",
                }
            ],
            evidence_map={"ev_0001": {"evidence_unit_id": "ev_0001"}},
            triplet_type="multihop",
        )
        assert qae_triplets[0]["scaffold_id"] == "scf_0001"
        assert qae_triplets[0]["source_qa_draft_id"] == "draft_0001"
        assert qae_triplets[0]["triplet_type"] == "multihop"
        assert qae_triplets[0]["materialization_mode"] == "DRAFT_BINDING"

    def test_materialize_qae_from_scaffold_fallback_evidence_refs(self):
        qae_triplets = materialize_qae_from_scaffold(
            materialization_mode="JOINT_SYNTHESIS",
            scaffold_units=[
                {
                    "scaffold_id": "scf_0001",
                    "scaffold_summary": "A person opens a drawer.",
                    "source_text_signal_ids": ["txt_000001"],
                    "source_description_ids": ["desc_000001"],
                    "temporal_unit_id": "tmp_000001",
                    "action_nodes": [
                        {
                            "node_ref": "tmp_000001_frame_01_01",
                            "node_type": "frame",
                            "sample_id": "smp_000001",
                            "frame_path": "/tmp/frame.jpg",
                        }
                    ],
                }
            ],
            qa_drafts=[],
            evidence_map={},
            triplet_type="single_hop",
        )
        assert qae_triplets[0]["evidence_refs"] == {
            "text": {
                "description_ids": ["desc_000001"],
                "text_signal_ids": ["txt_000001"],
                "evidence_unit_ids": [],
            },
            "image": {
                "frame_refs": ["tmp_000001_frame_01_01"],
                "sample_ids": ["smp_000001"],
                "frame_paths": ["/tmp/frame.jpg"],
            },
            "temporal": {
                "temporal_unit_ids": ["tmp_000001"],
            },
        }
        assert qae_triplets[0]["grounding_level"] == "strong_grounded"
        assert qae_triplets[0]["question_type"]
        assert qae_triplets[0]["question_plan"]["instruction"]

    def test_materialize_qae_from_scaffold_diversifies_rule_questions(self):
        scaffold_units = [
            {
                "scaffold_id": f"scf_{index:04d}",
                "scaffold_summary": summary,
                "source_description_ids": [f"desc_{index:04d}"],
            }
            for index, summary in enumerate(
                [
                    "A robot arm hovers over black beans in a tray.",
                    "Blue beads are scattered across a white tray.",
                    "A gripper is above a cylindrical object.",
                    "A pot and lemon are visible on a stove.",
                    "A clamp sits beside small translucent objects.",
                    "A tray is partly filled with black objects.",
                ],
                start=1,
            )
        ]

        qae_triplets = materialize_qae_from_scaffold(
            materialization_mode="JOINT_SYNTHESIS",
            scaffold_units=scaffold_units,
            qa_drafts=[],
            evidence_map={},
            triplet_type="single_hop",
        )

        question_types = {triplet["question_type"] for triplet in qae_triplets}
        questions = {triplet["question"] for triplet in qae_triplets}
        assert len(question_types) >= 3
        assert len(questions) >= 3

    def test_materialize_qae_with_temporal_grounding(self):
        qae_triplets = materialize_qae_with_temporal_grounding(
            materialization_mode="QA_FIRST_GROUNDING",
            scaffold_units=[
                {
                    "scaffold_id": "scf_0001",
                    "scaffold_summary": "A runner crosses the finish line.",
                    "question_focus": "the finish",
                    "question_type": "when",
                    "source_qa_draft_ids": ["draft_0001"],
                    "source_temporal_unit_ids": ["tmp_0001"],
                    "source_description_ids": ["desc_0001"],
                    "source_text_signal_ids": ["txt_0001"],
                }
            ],
            qa_drafts=[
                {
                    "qa_draft_id": "draft_0001",
                    "question": "When does the runner cross the finish line?",
                    "answer": "At the final moment.",
                    "segment_ids": ["seg_0001"],
                }
            ],
            temporal_structures=[
                {
                    "temporal_id": "tmp_0001",
                    "temporal_label": "final moment",
                    "segment_ids": ["seg_0001"],
                    "source_description_ids": ["desc_0001"],
                    "source_text_signal_ids": ["txt_0001"],
                },
                {
                    "temporal_id": "tmp_9999",
                    "temporal_label": "unrelated warmup",
                    "segment_ids": ["seg_9999"],
                    "source_description_ids": ["desc_9999"],
                    "source_text_signal_ids": ["txt_9999"],
                },
            ],
            description_units=[
                {"description_unit_id": "desc_0001", "segment_id": "seg_0001", "description_text": "The runner leans forward at the line."},
                {"description_unit_id": "desc_9999", "segment_id": "seg_9999", "description_text": "The runner warms up."},
            ],
            text_signals=[
                {"signal_id": "txt_0001", "segment_id": "seg_0001", "text": "finish"},
                {"signal_id": "txt_9999", "segment_id": "seg_9999", "text": "warmup"},
            ],
            triplet_type="single_hop",
        )
        assert qae_triplets[0]["scaffold_id"] == "scf_0001"
        assert qae_triplets[0]["materialization_mode"] == "QA_FIRST_GROUNDING"
        assert qae_triplets[0]["triplet_type"] == "single_hop"
        assert qae_triplets[0]["evidence_refs"]["text"]["description_ids"] == ["desc_0001"]
        assert qae_triplets[0]["evidence_refs"]["text"]["text_signal_ids"] == ["txt_0001"]
        assert qae_triplets[0]["evidence_refs"]["temporal"]["temporal_unit_ids"] == ["tmp_0001"]
        assert "desc_9999" not in qae_triplets[0]["evidence_refs"]["text"]["description_ids"]
        assert "tmp_9999" not in qae_triplets[0]["evidence_refs"]["temporal"]["temporal_unit_ids"]
        assert qae_triplets[0]["grounding_level"] in {"grounded", "strong_grounded"}
