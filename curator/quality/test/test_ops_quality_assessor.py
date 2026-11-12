from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import ray

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
pythonpath_entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
if str(REPO_ROOT) not in pythonpath_entries:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *[entry for entry in pythonpath_entries if entry]]
    )

from base.dataset import MultimodalDataset
from curator.core.pipeline import MultimodalPipeline
from curator.core.pipeline_plugin import OpsQualityPlugin
from curator.quality.ops_quality_assessor import (
    OpsQualityAssessmentConfig,
    OpsQualityAssessor,
)
from curator.quality.ops_quality_assessor.metrics.chair_object_hallucination import (
    ChairObjectHallucinationAssessor,
    ChairObjectHallucinationConfig,
)


MESSI_IMAGE_PATH = REPO_ROOT / "data" / "images" / "messi5.jpg"
MEGAMIND_VIDEO_PATH = REPO_ROOT / "data" / "videos" / "Megamind.avi"
MEGAMIND_FIXTURE_PATH = REPO_ROOT / "data" / "ops_quality_megamind_fixture.json"
LOCAL_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/data/llm_models"))
LOCAL_CLIP_MODEL_PATH = str(
    LOCAL_MODEL_ROOT / "laion" / "CLIP-ViT-B-32-laion2B-s34B-b79K"
)
LOCAL_IMAGE_CAPTION_MODEL_PATH = str(LOCAL_MODEL_ROOT / "Qwen3-VL-2B-Instruct")
LOCAL_RECOGNITION_MODEL_PATH = Path(
    LOCAL_MODEL_ROOT / "Salesforce" / "blip-image-captioning-base"
)
LOCAL_RECOGNITION_PROCESSOR_PATH = LOCAL_RECOGNITION_MODEL_PATH
LOCAL_DETR_MODEL_PATH = str(
    LOCAL_MODEL_ROOT / "facebook-detr-resnet-50-panoptic"
)
LOCAL_INTERNVL_MODEL_PATH = str(LOCAL_MODEL_ROOT / "InternVL2-4B")
FOCUSED_METRIC_NAMES = {
    "chair_object_hallucination",
    "visual_robustness",
    "video_image_alignment",
}

VALID_METRIC_STATUSES = {
    "passed",
    "warning",
    "failed",
    "not_applicable",
    "failed_precondition",
}

MetricSpec = tuple[str, list[str], dict[str, Any]]


class _OpsQualityAfterAugmentationPlugin(OpsQualityPlugin):
    name = "ops_quality_after_augmentation"

    def __init__(self, metric_specs_by_operation: dict[str, list[MetricSpec]]):
        super().__init__(
            quality_config=OpsQualityAssessmentConfig(sample_size=1, enabled_metrics=[]),
            assess_after_operations=False,
        )
        self.metric_specs_by_operation = metric_specs_by_operation
        self.executed_metric_names: set[str] = set()
        self.metric_results = {}

    def after_operation(
        self,
        pipeline,
        operation,
        op_index: int,
        input_ds,
        output_ds,
        duration_sec: float,
    ) -> None:
        for metric_name, modalities, metric_config in self.metric_specs_by_operation.get(
            operation.name,
            [],
        ):
            metric = self._assess_metric(
                metric_name=metric_name,
                operation_name=operation.name,
                op_index=op_index,
                duration_sec=duration_sec,
                dataset=output_ds,
                modalities=modalities,
                metric_config=metric_config,
            )
            self.executed_metric_names.add(metric_name)
            self.metric_results[metric_name] = metric

    def _assess_metric(
        self,
        metric_name: str,
        operation_name: str,
        op_index: int,
        duration_sec: float,
        dataset: MultimodalDataset,
        modalities: list[str],
        metric_config: dict[str, Any] | None = None,
    ):
        assessment_dataset = MultimodalDataset(dataset.dataset, modalities=modalities)
        self._assessor = OpsQualityAssessor(
            OpsQualityAssessmentConfig(
                sample_size=1,
                enabled_metrics=[metric_name],
                metric_configs={metric_name: metric_config or {}},
            )
        )

        report_count_before = len(self._reports)
        self._assess_snapshot(
            dataset=assessment_dataset,
            rows=None,
            stage="after_operation",
            operation_name=operation_name,
            op_index=op_index,
            duration_sec=duration_sec,
        )
        assert len(self._reports) == report_count_before + 1

        latest_report = self._reports[-1]
        metric = latest_report["result"].metric_results[metric_name]
        _print_metric_result(metric_name, metric)
        assert metric.status in VALID_METRIC_STATUSES
        assert metric.scores or metric.skipped_scores or metric.notices or metric.metadata
        return metric


def _print_metric_result(metric_name: str, metric) -> None:
    payload = {
        "status": metric.status,
        "scores": metric.scores,
        "skipped_scores": metric.skipped_scores,
        "notices": metric.notices,
        "metadata": metric.metadata,
    }
    rendered_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    print(
        f'Calling {metric_name} operation; evaluation result: "{metric_name}": {rendered_payload}',
        flush=True,
    )


def _build_pipeline_input_dataset() -> MultimodalDataset:
    row = json.loads(MEGAMIND_FIXTURE_PATH.read_text(encoding="utf-8"))
    row["image"] = str(MESSI_IMAGE_PATH)
    return MultimodalDataset(ray.data.from_items([row]), modalities=["image", "video"])


def _metric_specs_by_operation() -> dict[str, list[MetricSpec]]:
    return {
        "ImageCaptioningOperation": [
            (
                "chair_object_hallucination",
                ["image", "text"],
                {
                    "image_field": "image",
                    "text_field": "caption",
                    "subject_mention_threshold": 0.55,
                    "subject_mismatch_threshold": 0.35,
                    "segmentation_model_path": LOCAL_DETR_MODEL_PATH,
                    "recognition_model_path": str(LOCAL_RECOGNITION_MODEL_PATH),
                    "recognition_processor_path": str(LOCAL_RECOGNITION_PROCESSOR_PATH),
                    "recognition_config_path": str(LOCAL_RECOGNITION_PROCESSOR_PATH),
                    "recognition_use_safetensors": True,
                    "embedding_model_path": LOCAL_CLIP_MODEL_PATH,
                    "device": "cpu",
                },
            ),
        ],
        "ImageRandomCropOperation": [
            (
                "visual_robustness",
                ["image"],
                {
                    "image_field": "image",
                    "transform_fields": ["crop_image_data_list"],
                    "embedding_model_path": LOCAL_CLIP_MODEL_PATH,
                    "device": "cpu",
                },
            ),
        ],
        "VideoSamplingOperation": [
            (
                "video_image_alignment",
                ["video", "image"],
                {
                    "image_embedding_model_path": LOCAL_INTERNVL_MODEL_PATH,
                    "device": "cpu",
                },
            ),
        ],
    }


def _run_focused_ops_quality_smokes() -> _OpsQualityAfterAugmentationPlugin:
    plugin = _OpsQualityAfterAugmentationPlugin(_metric_specs_by_operation())
    pipeline = MultimodalPipeline(enable_tracking=False, plugins=[plugin])
    pipeline.add_operation(
        "image_captioning",
        {
            "image_key": "image",
            "output_key": "caption",
            "model_path": LOCAL_IMAGE_CAPTION_MODEL_PATH,
            "device_preference": "cpu",
            "model_type": "qwen3_vl",
            "prompt_text": "Describe the main subject in one short English sentence.",
            "max_new_tokens": 12,
        },
    )
    pipeline.add_operation(
        "image_random_crop",
        {
            "image_key": "image",
            "num_crops": 1,
            "min_scale": 0.5,
            "max_scale": 0.5,
            "output_key": "crop_image_data_list",
        },
    )
    pipeline.add_operation(
        "video_sampling",
        {
            "video_key": "video_path",
            "segments_key": "segments",
            "sampling_mode": "CENTER",
            "num_frames": 1,
            "save_sample_frames": True,
            "frame_output_dir": str(REPO_ROOT / "data" / "ops_quality_sample_frames" / "focused_smoke"),
            "overwrite_sample_frames": True,
        },
    )
    try:
        pipeline.run(dataset=_build_pipeline_input_dataset())
    finally:
        pipeline.cleanup()

    assert plugin.executed_metric_names == FOCUSED_METRIC_NAMES
    return plugin


def test_focused_ops_quality_scores():
    plugin = _run_focused_ops_quality_smokes()

    chair_metric = plugin.metric_results["chair_object_hallucination"]
    assert "subject_mention_rate" in chair_metric.scores
    assert "subject_mismatch_rate" in chair_metric.scores

    visual_metric = plugin.metric_results["visual_robustness"]
    assert "structural_similarity_mean" in visual_metric.scores

    video_metric = plugin.metric_results["video_image_alignment"]
    assert "segment_frame_alignment_mean" in video_metric.scores
    assert "temporal_coverage_rate" in video_metric.scores


def test_chair_subject_recognition_does_not_fall_back_to_segmentation_label(monkeypatch):
    def fake_subject_crop(cls, image_value, metric_config):
        return object(), "orange"

    monkeypatch.setattr(
        ChairObjectHallucinationAssessor,
        "_subject_crop",
        classmethod(fake_subject_crop),
    )
    monkeypatch.setattr(
        ChairObjectHallucinationAssessor,
        "_recognize_crop",
        staticmethod(lambda image, metric_config: None),
    )

    subject = ChairObjectHallucinationAssessor._recognize_visual_subject(
        image_value="unused.jpg",
        metric_config=ChairObjectHallucinationConfig(),
    )

    assert subject is None


if __name__ == "__main__":
    _run_focused_ops_quality_smokes()
