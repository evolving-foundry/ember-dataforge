from __future__ import annotations

from tests.generation.config_helpers import executor_context, job_for_recipe

import json

import pytest

ray = pytest.importorskip("ray")

from base.dataset import MultimodalDataset
from curator.core.operation_registry import OperationRegistry
from curator.core.pipeline import MultimodalPipeline
from curator.core.pipeline_plugin import OpsQualityPlugin
from curator.operations.augmentation.image_captioning_operation import (
    ImageCaptioningOperation,
)
import curator.operations.augmentation.image_captioning_operation as image_captioning_module
from curator.operations.augmentation.video_partition_operation import (
    VideoPartitionOperation,
)
from curator.operations.augmentation.video_qae_materialization_operation import (
    VideoQaeMaterializationOperation,
)
from curator.operations.augmentation.video_sampling_operation import (
    VideoSamplingOperation,
)
from curator.operations.augmentation.video_scaffold_construction_operation import (
    VideoScaffoldConstructionOperation,
)
from curator.operations.augmentation.video_temporal_structure_operation import (
    VideoTemporalStructureOperation,
)
from data_designer.operators import SyntheticDataGenerationOperation
from data_designer.pipelines.recipes.video_temporal_qa import (
    VideoTemporalQaRecipeExecutor,
)
from curator.quality.ops_quality_assessor import OpsQualityAssessmentConfig
from curator.tracking.tracking import OperationType
from curator.utils.local_model_paths import resolve_local_model_path


def _dataset(row, modalities=None):
    return MultimodalDataset(
        ray.data.from_items([row]),
        modalities=modalities or ["image", "text"],
    )


@pytest.fixture
def sample_image(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    path = tmp_path / "butterfly.jpg"
    image_module.new("RGB", (16, 16), color=(40, 120, 180)).save(path)
    return path


@pytest.fixture
def sample_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path / "seg_0001.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (32, 32)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer unavailable")
    try:
        for index in range(8):
            writer.write(np.full((32, 32, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_registry_creates_synthetic_generation_with_defaults():
    op = OperationRegistry.create_operation("synthetic_data_generation", {"recipe": "dense_caption"})

    assert isinstance(op, SyntheticDataGenerationOperation)
    assert op.operation_type == OperationType.GENERATION
    assert op.config["recipe"] == "dense_caption"
    assert op.params["schema_version"] == "dataforge-job-v1"


def test_unknown_synthetic_recipe_raises_clear_error():
    # Recipe parsing happens at execution time (the generation service routes uniformly through RecipeRegistry).
    op = SyntheticDataGenerationOperation(job_for_recipe({"recipe": "doc_chart_qa"}))
    with pytest.raises(ValueError, match="unknown recipe"):
        op.execute(_dataset({"id": "img-unknown", "image": "unused.jpg"}))


def test_local_model_path_resolves_default_roots():
    assert resolve_local_model_path(
        "Salesforce/blip-image-captioning-base",
        ["/data/wenjt/llm_models", "/data/llm_models"],
    ).endswith("Salesforce/blip-image-captioning-base")


def test_dense_caption_recipe_reuses_image_captioning_and_writes_provenance(monkeypatch):
    calls = {}

    def fake_execute(self, dataset):
        calls["params"] = dict(self.params)

        def add_caption(row):
            row = dict(row)
            row[self.params["output_key"]] = json.dumps(
                {
                    "summary": "A person opens a blue door.",
                    "scene": {"setting": "indoor doorway", "viewpoint": "front view"},
                    "objects": [
                        {
                            "name": "person",
                            "count": 1,
                            "attributes": ["standing"],
                            "location": "near the door",
                        },
                        {"name": "blue door", "count": 1, "location": "center"},
                    ],
                    "relations": [
                        {"subject": "person", "predicate": "opens", "object": "blue door"}
                    ],
                    "actions": [
                        {
                            "actor": "person",
                            "action": "opens",
                            "target": "blue door",
                            "state": "in progress",
                        }
                    ],
                    "text_in_image": [],
                    "uncertainties": [],
                }
            )
            return row

        return dataset.map(add_caption)

    monkeypatch.setattr(ImageCaptioningOperation, "execute", fake_execute)

    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "dense_caption",
            "input_key": "image",
            "output_key": "dense_caption",
            "generator_model": "fake-vlm",
            "generator_prompt_version": "dense-v1",
            "filter_versions": ["image_text_alignment@v1"],
            "intended_split": "pretrain",
        })
    )
    result = op.execute(_dataset({"id": "img-1", "image": "unused.jpg"}))
    row = result.dataset.take(1)[0]

    assert calls["params"]["image_key"] == "image"
    assert calls["params"]["output_key"] == "_dense_caption_raw"
    assert row["dense_caption"]["schema_version"] == "dense-caption-v1"
    assert row["dense_caption"]["summary"] == "A person opens a blue door."
    assert row["dense_caption"]["scene"]["setting"] == "indoor doorway"
    assert row["dense_caption"]["objects"][0]["name"] == "person"
    assert row["dense_caption"]["relations"][0]["predicate"] == "opens"
    assert row["dense_caption"]["actions"][0]["target"] == "blue door"
    assert row["caption"] == "A person opens a blue door."
    assert "blue door" in row["dense_caption_text"]
    assert row["recipe"] == "dense_caption"
    assert row["schema_version"] == "synthetic-data-v1"
    assert row["generator_model"] == "fake-vlm"
    assert row["generator_prompt_version"] == "dense-v1"
    assert row["quality_scores"]["generation_success"] == 1.0
    assert row["audit_status"] == "auto_pass"
    assert row["provenance"]["sample_id"] == "img-1"
    assert row["provenance"]["recipe"] == "dense_caption"
    assert row["provenance"]["filter_versions"] == ["image_text_alignment@v1"]
    assert row["provenance"]["intended_split"] == "pretrain"


def test_dense_caption_forwards_local_model_params_and_defaults_to_image_path(monkeypatch):
    calls = {}

    def fake_execute(self, dataset):
        calls["params"] = dict(self.params)

        def add_caption(row):
            row = dict(row)
            row[self.params["output_key"]] = "A butterfly on a flower."
            return row

        return dataset.map(add_caption)

    monkeypatch.setattr(ImageCaptioningOperation, "execute", fake_execute)

    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "dense_caption",
            "local_model_roots": ["/data/wenjt/llm_models"],
            "local_files_only": True,
        })
    )
    result = op.execute(_dataset({"id": "img-path-1", "image_path": "unused.jpg"}))
    row = result.dataset.take(1)[0]

    assert calls["params"]["image_key"] == "image_path"
    assert calls["params"]["fallback_image_keys"] == ["image_path", "image"]
    assert calls["params"]["local_model_roots"] == ["/data/wenjt/llm_models"]
    assert calls["params"]["local_files_only"] is True
    assert calls["params"]["error_key"] == "_synthetic_generation_error"
    assert calls["params"]["resolved_model_path_key"] == "generator_model_path"
    assert row["dense_caption"]["summary"] == "A butterfly on a flower."
    assert row["dense_caption"]["objects"] == []
    assert row["dense_caption_text"] == "A butterfly on a flower."
    assert row["caption"] == "A butterfly on a flower."


def test_image_captioning_uses_npu_first_device_resolution(monkeypatch):
    monkeypatch.setattr(
        image_captioning_module,
        "resolve_torch_device",
        lambda preference: "npu:0",
    )

    op = ImageCaptioningOperation({"device_preference": "auto"})

    assert op._get_available_device() == "npu:0"


def test_dense_caption_operation_result_writes_caption_and_model_path(monkeypatch, sample_image):
    def fake_execute(self, dataset):
        def add_caption(row):
            row = dict(row)
            row[self.params["output_key"]] = f'{self.params["prompt_text"]} a butterfly'
            row[self.params["resolved_model_path_key"]] = (
                "models/Salesforce/blip-image-captioning-base"
            )
            return row

        return dataset.map(add_caption)

    monkeypatch.setattr(ImageCaptioningOperation, "execute", fake_execute)

    op = SyntheticDataGenerationOperation(
        job_for_recipe({"recipe": "dense_caption", "prompt_text": "a photography of"})
    )
    result = op.execute(_dataset({"id": "img-real-path", "image_path": str(sample_image)}))
    row = result.dataset.take(1)[0]

    assert row["dense_caption"]["summary"] == "a photography of a butterfly"
    assert row["dense_caption_text"] == "a photography of a butterfly"
    assert row["caption"] == "a photography of a butterfly"
    assert row["generator_model_path"].endswith("Salesforce/blip-image-captioning-base")
    assert row["quality_scores"]["generation_success"] == 1.0
    assert row["audit_status"] == "auto_pass"
    assert row["provenance"]["generator_model_path"].endswith("Salesforce/blip-image-captioning-base")


def test_dense_caption_operation_error_records_generation_error(monkeypatch, sample_image):
    def fake_execute(self, dataset):
        def add_error(row):
            row = dict(row)
            row[self.params["error_key"]] = (
                f'model unavailable: {self.params["model_path"]}'
            )
            return row

        return dataset.map(add_error)

    monkeypatch.setattr(ImageCaptioningOperation, "execute", fake_execute)
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "dense_caption",
            "model_path": "models/not-a-real-caption-model",
        })
    )
    result = op.execute(_dataset({"id": "img-fail", "image_path": str(sample_image)}))
    row = result.dataset.take(1)[0]

    assert row.get("dense_caption") is None
    assert "not-a-real-caption-model" in row["_synthetic_generation_error"]
    assert row["generator_model"] == "models/not-a-real-caption-model"
    assert row["quality_scores"]["generation_success"] == 0.0
    assert row["audit_status"] == "auto_fail"
    assert "not-a-real-caption-model" in row["provenance"]["generation_error"]
    assert row["provenance"]["generator_model"] == "models/not-a-real-caption-model"


def test_dense_caption_missing_image_path_records_generation_error(monkeypatch):
    def fail_if_model_loads(self):
        raise AssertionError("model should not load for a missing image file")

    monkeypatch.setattr(ImageCaptioningOperation, "_load_model", fail_if_model_loads)

    op = SyntheticDataGenerationOperation(job_for_recipe({"recipe": "dense_caption"}))
    result = op.execute(_dataset({"id": "img-missing", "image_path": "data/images/missing.jpg"}))
    row = result.dataset.take(1)[0]

    assert row.get("dense_caption") is None
    assert "image file not found" in row["_synthetic_generation_error"]
    assert row["audit_status"] == "auto_fail"
    assert "missing.jpg" in row["provenance"]["generation_error"]


def test_video_temporal_qa_recipe_runs_materialize_revision_and_quality():
    row = {
        "id": "video-1",
        "scaffold_units": [
            {
                "scaffold_id": "scf_0001",
                "scaffold_summary": "A person opens the door.",
                "source_qa_draft_ids": ["draft_0001"],
                "source_evidence_unit_ids": ["ev_0001"],
            }
        ],
        "qa_drafts": [
            {
                "qa_draft_id": "draft_0001",
                "question": "What happens?",
                "answer": "What happens?",
            }
        ],
        "evidence_units": [
            {
                "evidence_unit_id": "ev_0001",
                "evidence_summary": "The person reaches for and opens the door.",
            }
        ],
    }
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "video_temporal_qa",
            "schema_version": "synthetic-qae-v1",
            "recipe_config": {
                "materialization_params": {
                    "materialization_mode": "JOINT_SYNTHESIS",
                    "use_llm_client": False,
                },
                "revision_params": {"revision_mode": "FIELD_ALIGN"},
                "quality_params": {"quality_mode": "RULE_FILTER"},
            },
        })
    )

    result = op.execute(_dataset(row, modalities=["video", "text"]))
    output = result.dataset.take(1)[0]

    assert output["recipe"] == "video_temporal_qa"
    assert output["schema_version"] == "synthetic-qae-v1"
    assert output["qae_triplets"][0]["triplet_id"] == "tri_000001"
    assert output["revisions"][0]["revision_action"] == "FIELD_ALIGN"
    assert output["revisions"][0]["after_fields"]["answer"] == "A person opens the door."
    assert output["quality_records"][0]["decision"] == "keep"
    assert output["quality_scores"]["generation_success"] == 1.0
    assert output["quality_scores"]["quality_keep_rate"] == 1.0
    assert output["evidence_refs"][0]["target_triplet_id"] == "tri_000001"
    assert output["provenance"]["modality"] == ["video", "text"]
    assert output["provenance"]["audit_status"] == "auto_pass"


def test_video_temporal_qa_recipe_passes_question_plan_config_to_materialization():
    custom_plans = [
        {
            "type": "sports_tactic",
            "instruction": "询问可见运动战术站位。",
            "template": "What visible tactical position is supported by the evidence?",
        }
    ]
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "video_temporal_qa",
            "recipe_config": {
                "question_plan_profile": "generic",
                "question_plans": custom_plans,
                "generic_question_prefixes": ["what is shown in the match"],
                "materialization_params": {
                    "materialization_mode": "JOINT_SYNTHESIS",
                },
            },
        })
    )
    executor = VideoTemporalQaRecipeExecutor(executor_context(op))

    params = executor._video_step_params(
        "materialization_params",
        fallback_key="materialization",
    )

    assert params["question_plan_profile"] == "generic"
    assert params["question_plans"] == custom_plans
    assert params["generic_question_prefixes"] == ["what is shown in the match"]


def test_video_temporal_qa_prepares_raw_video_minimal_chain():
    row = {
        "id": "raw-video-1",
        "video_path": "data/videos/megamind_segments/seg_0001.avi",
        "video_duration_sec": 2.0,
        "video_fps": 4.0,
    }
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "video_temporal_qa",
            "recipe_config": {
                "raw_video_prepare": {
                    "description_enabled": False,
                    "partition_params": {"chunk_duration_sec": 1.0},
                    "sampling_params": {"sampling_mode": "CENTER", "save_sample_frames": False},
                    "temporal_params": {"temporal_mode": "EVENT_WINDOW"},
                    "scaffold_params": {"scaffold_mode": "NARRATION_CHUNK"},
                },
                "materialization_params": {
                    "materialization_mode": "JOINT_SYNTHESIS",
                    "use_llm_client": False,
                },
                "revision_params": {"revision_mode": "FIELD_ALIGN"},
                "quality_params": {"quality_mode": "RULE_FILTER"},
            },
        })
    )

    result = op.execute(_dataset(row, modalities=["video", "text"]))
    output = result.dataset.take(1)[0]

    assert len(output["segments"]) == 2
    assert len(output["samples"]) == 2
    assert len(output["temporal_structures"]) == 2
    assert output["scaffold_units"][0]["scaffold_summary"] == "seg_0001 with 1 samples"
    assert output["qae_triplets"][0]["answer"] == "seg_0001 with 1 samples"
    assert output["quality_scores"]["generation_success"] == 1.0
    assert output["audit_status"] == "auto_pass"


def test_video_temporal_qa_prepares_four_granularity_descriptions(tmp_path, sample_video):
    class FakeDescriptionOperation:
        def __init__(self, params):
            self.params = dict(params)

        def _segments_for_modes(self, segments, modes):
            segments = [dict(segment) for segment in segments]
            if "CONTEXT_DESCRIPTION" in set(modes or []):
                for segment in segments:
                    segment.setdefault("stitch_group_id", "stitch_0001")
            return segments

        def describe_modes(
            self,
            segments,
            samples,
            temporal_structures=None,
            description_modes=None,
            id_start=1,
        ):
            units = []
            for offset, mode in enumerate(description_modes or [], start=0):
                granularity = {
                    "FRAME_DESCRIPTION": "frame",
                    "EVENT_DESCRIPTION": "event",
                    "SEGMENT_DESCRIPTION": "segment",
                    "CONTEXT_DESCRIPTION": "context",
                }[mode]
                segment_ids = [str(segment["segment_id"]) for segment in segments]
                segment_id = segment_ids[0]
                sample_ids = [str(sample["sample_id"]) for sample in samples]
                description_id = f"desc_{int(id_start) + offset:06d}"
                units.append(
                    {
                        "id": description_id,
                        "description_id": description_id,
                        "description_unit_id": description_id,
                        "segment_id": segment_id,
                        "description_text": f"{granularity} caption",
                        "description_scope": granularity,
                        "granularity": granularity,
                        "description_mode": mode,
                        "description_source": "video_multigranularity_description",
                        "anchor_refs": {
                            "segment_ids": segment_ids if mode == "CONTEXT_DESCRIPTION" else [segment_id],
                            "sample_ids": sample_ids,
                        },
                        "summary_tags": [granularity],
                    }
                )
            return {
                "description_mode": (
                    description_modes[0]
                    if description_modes and len(description_modes) == 1
                    else "MULTI_GRANULARITY"
                ),
                "description_modes": list(description_modes or []),
                "global_metrics": {
                    "total_segments": len(segments),
                    "total_samples": len(samples),
                    "total_description_units": len(units),
                    "description_units_by_mode": {mode: 1 for mode in description_modes or []},
                },
                "description_results": [
                    {
                        "description_mode": mode,
                        "granularity": unit["granularity"],
                        "total_description_units": 1,
                    }
                    for mode, unit in zip(description_modes or [], units)
                ],
                "description_units": units,
            }

    row = {
        "id": "raw-video-desc-1",
        "video_path": str(sample_video),
        "video_duration_sec": 2.0,
        "video_fps": 4.0,
    }
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "video_temporal_qa",
            "recipe_config": {
                "raw_video_prepare": {
                    "partition_params": {"chunk_duration_sec": 1.0},
                    "sampling_params": {
                        "sampling_mode": "CENTER",
                        "frame_output_dir": str(tmp_path),
                        "overwrite_sample_frames": True,
                    },
                    "temporal_params": {"temporal_mode": "EVENT_WINDOW"},
                    "description_params": {
                        "description_modes": [
                            "FRAME_DESCRIPTION",
                            "EVENT_DESCRIPTION",
                            "SEGMENT_DESCRIPTION",
                            "CONTEXT_DESCRIPTION",
                        ],
                    },
                    "scaffold_params": {"scaffold_mode": "NARRATION_CHUNK"},
                },
                "materialization_params": {
                    "materialization_mode": "JOINT_SYNTHESIS",
                    "use_llm_client": False,
                },
                "revision_params": {"revision_mode": "FIELD_ALIGN"},
                "quality_params": {"quality_mode": "RULE_FILTER"},
            },
        })
    )

    executor = VideoTemporalQaRecipeExecutor(executor_context(op))
    prepared = executor._prepare_raw_video_temporal_qa_inputs(
        dataset=_dataset(row, modalities=["video", "text"]),
        partition_cls=VideoPartitionOperation,
        sampling_cls=VideoSamplingOperation,
        temporal_cls=VideoTemporalStructureOperation,
        description_cls=FakeDescriptionOperation,
        scaffold_cls=VideoScaffoldConstructionOperation,
    )
    output = prepared.dataset.take(1)[0]

    description_modes = {unit["description_mode"] for unit in output["description_units"]}
    granularities = {unit["granularity"] for unit in output["description_units"]}
    assert description_modes == {
        "FRAME_DESCRIPTION",
        "EVENT_DESCRIPTION",
        "SEGMENT_DESCRIPTION",
        "CONTEXT_DESCRIPTION",
    }
    assert granularities == {"frame", "event", "segment", "context"}
    assert output["description_metrics"]["total_description_units"] == 4
    assert output["description_metrics"]["description_units_by_mode"] == {
        "FRAME_DESCRIPTION": 1,
        "SEGMENT_DESCRIPTION": 1,
        "EVENT_DESCRIPTION": 1,
        "CONTEXT_DESCRIPTION": 1,
    }
    assert output["segments"][0]["stitch_group_id"] == "stitch_0001"
    assert output["scaffold_units"][0]["scaffold_summary"] == "event caption"
    assert output["scaffold_metrics"]["total_description_units"] == 4


def test_video_temporal_qa_local_vlm_enables_sample_frame_paths(tmp_path, monkeypatch, sample_video):
    calls = {}

    def fake_materialize_execute(self, dataset):
        calls["params"] = dict(self.params)

        def add_qae(row):
            row = dict(row)
            row["qae_triplets"] = [
                {
                    "triplet_id": "tri_000001",
                    "id": "tri_000001",
                    "scaffold_id": row["scaffold_units"][0]["scaffold_id"],
                    "question": "What is visible?",
                    "answer": "A sampled video frame is visible.",
                    "evidence_refs": row["scaffold_units"][0].get("visual_evidence") and {
                        "text": {"description_ids": [], "text_signal_ids": [], "evidence_unit_ids": []},
                        "image": {
                            "frame_refs": [],
                            "sample_ids": [row["scaffold_units"][0]["visual_evidence"][0]["sample_id"]],
                            "frame_paths": [row["scaffold_units"][0]["visual_evidence"][0]["frame_path"]],
                        },
                        "temporal": {
                            "temporal_unit_ids": [row["scaffold_units"][0]["temporal_unit_id"]],
                        },
                    },
                    "grounding_level": "strong_grounded",
                }
            ]
            row["materialization_metrics"] = {
                "total_scaffold_units": len(row.get("scaffold_units") or []),
                "total_triplets": 1,
                "llm_materialized_triplets": 1,
            }
            row["materialization_mode"] = "JOINT_SYNTHESIS"
            return row

        return dataset.map(add_qae)

    monkeypatch.setattr(VideoQaeMaterializationOperation, "execute", fake_materialize_execute)

    row = {
        "id": "raw-video-vlm-1",
        "video_path": str(sample_video),
        "video_duration_sec": 2.0,
        "video_fps": 4.0,
    }
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "video_temporal_qa",
            "recipe_config": {
                "materialization_params": {
                    "materialization_mode": "JOINT_SYNTHESIS",
                    "use_local_vlm_client": True,
                    "vlm_model_path": "/data/llm_models/Qwen2.5-VL-7B-Instruct",
                    "vlm_device_preference": "npu:0",
                    "vlm_max_new_tokens": 128,
                    "ray_num_npus": 1,
                },
                "raw_video_prepare": {
                    "description_enabled": False,
                    "partition_params": {"chunk_duration_sec": 1.0},
                    "sampling_params": {
                        "sampling_mode": "CENTER",
                        "frame_output_dir": str(tmp_path),
                        "overwrite_sample_frames": True,
                    },
                },
                "revision_params": {"revision_mode": "FIELD_ALIGN"},
                "quality_params": {"quality_mode": "RULE_FILTER"},
            },
        })
    )

    output = op.execute(_dataset(row, modalities=["video", "text"])).dataset.take(1)[0]

    assert calls["params"]["use_local_vlm_client"] is True
    assert calls["params"]["vlm_model_path"].endswith("Qwen2.5-VL-7B-Instruct")
    assert calls["params"]["vlm_device_preference"] == "npu:0"
    assert calls["params"]["vlm_max_new_tokens"] == 128
    assert calls["params"]["ray_num_npus"] == 1
    sample_frame_paths = [sample.get("frame_path") for sample in output["samples"]]
    assert sample_frame_paths
    assert all(path and str(path).startswith(str(tmp_path)) for path in sample_frame_paths)
    assert output["scaffold_units"][0]["visual_evidence"][0]["frame_path"].startswith(str(tmp_path))
    assert output["audit_status"] == "auto_pass"


def test_pipeline_generation_recipe_records_ops_quality_report(monkeypatch):
    def fake_execute(self, dataset):
        def add_caption(row):
            row = dict(row)
            row[self.params["output_key"]] = "A small object on a table."
            return row

        return dataset.map(add_caption)

    monkeypatch.setattr(ImageCaptioningOperation, "execute", fake_execute)

    quality_plugin = OpsQualityPlugin(
        OpsQualityAssessmentConfig(sample_size=1, enabled_metrics=[]),
        assess_after_operations=True,
    )
    pipeline = MultimodalPipeline(plugins=[quality_plugin])
    pipeline.add_operation(
        "synthetic_data_generation",
        job_for_recipe({"recipe": "dense_caption", "output_key": "dense_caption"}),
    )

    result = pipeline.run(_dataset({"id": "img-2", "image": "unused.jpg"}))
    output = result.dataset.take(1)[0]
    reports = quality_plugin.get_reports()

    assert output["dense_caption"]["summary"] == "A small object on a table."
    assert output["caption"] == "A small object on a table."
    assert len(reports) == 1
    assert reports[0]["stage"] == "after_operation"
    assert reports[0]["operation_name"] == "SyntheticDataGenerationOperation"


def test_pipeline_generation_recipe_records_nonempty_ops_quality_metrics():
    quality_plugin = OpsQualityPlugin(
        OpsQualityAssessmentConfig(
            sample_size=1,
            enabled_metrics=["text_noise_contamination", "qae_grounding_alignment"],
            metric_configs={"qae_grounding_alignment": {"enable_visual_filter": False}},
        ),
        assess_after_operations=True,
    )
    pipeline = MultimodalPipeline(plugins=[quality_plugin])
    pipeline.add_operation(
        "synthetic_data_generation",
        job_for_recipe({
            "recipe": "video_temporal_qa",
            "recipe_config": {
                "materialization_params": {
                    "materialization_mode": "JOINT_SYNTHESIS",
                    "use_llm_client": False,
                },
                "revision_params": {"revision_mode": "FIELD_ALIGN"},
                "quality_params": {"quality_mode": "RULE_FILTER"},
            },
        }),
    )
    row = {
        "id": "video-quality-1",
        "scaffold_units": [
            {
                "scaffold_id": "scf_0001",
                "scaffold_summary": "A person opens a door.",
                "source_evidence_unit_ids": ["ev_0001"],
            }
        ],
        "evidence_units": [
            {
                "evidence_unit_id": "ev_0001",
                "evidence_summary": "A person opens a door.",
            }
        ],
    }

    result = pipeline.run(_dataset(row, modalities=["video", "text"]))
    output = result.dataset.take(1)[0]
    reports = quality_plugin.get_reports()
    metrics = reports[0]["result"].metric_results

    assert output["audit_status"] == "auto_pass"
    assert metrics["text_noise_contamination"].status == "warning"
    assert "ngram_repetition_rate_mean" in metrics["text_noise_contamination"].scores
    assert metrics["qae_grounding_alignment"].status == "warning"
    assert metrics["qae_grounding_alignment"].scores["evidence_hit_rate"] == 1.0
