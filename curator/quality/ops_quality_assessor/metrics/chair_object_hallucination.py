from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

from pydantic import BaseModel

from ..base import BaseMetricAssessor, InvalidMetricInputError
from ..context import EvaluationContext
from ..results import MetricResult
from .common import (
    as_list, cosine_similarity, get_path_values, load_image_array, mean,
    non_empty_text,
    round_score,
    text_embeddings,
    mask_box,
    score_value, mask_area
)


class ChairObjectHallucinationConfig(BaseModel):
    image_field: str = "image"
    text_field: str = "caption"
    sample_frame_path: str = "samples[].frame_path"
    segmentation_model_path: str = "facebook/detr-resnet-50-panoptic"
    recognition_model_path: str = "Salesforce/blip-image-captioning-base"
    recognition_processor_path: str | None = None
    recognition_config_path: str | None = None
    recognition_use_safetensors: bool | None = None
    embedding_model_path: str = "BAAI/bge-m3"
    device: str | None = None
    subject_mention_threshold: float = 0.55
    subject_mismatch_threshold: float = 0.35
    min_subject_score: float = 0.0


class ChairObjectHallucinationAssessor(BaseMetricAssessor):
    metric_name = "chair_object_hallucination"
    output_scores = ("subject_mention_rate", "subject_mismatch_rate")
    config_model = ChairObjectHallucinationConfig

    def _assess(self, context: EvaluationContext) -> MetricResult:
        metric_config = cast(
            ChairObjectHallucinationConfig,
            self.resolve_metric_config(context) or ChairObjectHallucinationConfig(),
        )

        if not (
            context.has_field(metric_config.image_field)
            or context.has_field("samples")
        ):
            return MetricResult(
                metric_name=self.metric_name,
                status="not_applicable",
                scores={},
                skipped_scores={
                    score: "dataset does not contain image fields"
                    for score in self.output_scores
                },
                notices=["dataset does not contain image fields"],
            )

        pair_count = 0
        mention_count = 0
        mismatch_count = 0
        similarity_scores = []
        failed_subject_count = 0

        for row in context.rows:
            image_values = self._row_images(row, metric_config)
            texts = self._row_texts(row, context, metric_config)
            if not image_values or not texts:
                continue
            for image_value in image_values:
                subject_name = self._recognize_visual_subject(image_value, metric_config)
                if not subject_name:
                    failed_subject_count += 1
                    continue
                for text in texts:
                    similarity = self._subject_caption_similarity(
                        subject_name,
                        text,
                        metric_config,
                    )
                    if similarity is None:
                        continue
                    pair_count += 1
                    similarity_scores.append(similarity)
                    if similarity >= metric_config.subject_mention_threshold:
                        mention_count += 1
                    if similarity < metric_config.subject_mismatch_threshold:
                        mismatch_count += 1

        if pair_count == 0:
            return MetricResult(
                metric_name=self.metric_name,
                status="failed_precondition",
                scores={},
                skipped_scores={
                    score: "visual subjects or caption similarities could not be computed"
                    for score in self.output_scores
                },
                notices=["visual subjects or caption similarities could not be computed"],
                metadata={"failed_subject_count": failed_subject_count},
            )

        notices = []
        status = "passed"
        if failed_subject_count:
            status = "warning"
            notices.append(f"{failed_subject_count} images had no usable visual subject")

        return MetricResult(
            metric_name=self.metric_name,
            status=status,
            scores={
                "subject_mention_rate": round_score(mention_count / pair_count),
                "subject_mismatch_rate": round_score(mismatch_count / pair_count),
            },
            notices=notices,
            metadata={
                "pair_count": pair_count,
                "failed_subject_count": failed_subject_count,
                "subject_caption_similarity_mean": mean(similarity_scores),
                "segmentation_model_path": metric_config.segmentation_model_path,
                "recognition_model_path": metric_config.recognition_model_path,
                "embedding_model_path": metric_config.embedding_model_path,
                "subject_mention_threshold": metric_config.subject_mention_threshold,
                "subject_mismatch_threshold": metric_config.subject_mismatch_threshold,
            },
        )

    @classmethod
    def _row_images(
        cls,
        row: dict[str, Any],
        metric_config: ChairObjectHallucinationConfig,
    ) -> list[Any]:
        images = as_list(row.get(metric_config.image_field))
        images.extend(get_path_values(row, metric_config.sample_frame_path))
        return [image for image in images if image is not None]

    @classmethod
    def _row_texts(
        cls,
        row: dict[str, Any],
        context: EvaluationContext,
        metric_config: ChairObjectHallucinationConfig,
    ) -> list[str]:
        if not context.has_field(metric_config.text_field):
            raise InvalidMetricInputError(
                f"configured text field is missing: {metric_config.text_field}"
            )
        values = as_list(row.get(metric_config.text_field))
        texts = [non_empty_text(value) for value in values]
        return [text for text in texts if text]

    @classmethod
    def _recognize_visual_subject(
        cls,
        image_value: Any,
        metric_config: ChairObjectHallucinationConfig,
    ) -> str | None:
        subject_crop, _segmentation_label = cls._subject_crop(image_value, metric_config)
        if subject_crop is None:
            return None
        return cls._recognize_crop(subject_crop, metric_config)

    @staticmethod
    def _subject_caption_similarity(
        subject_name: str,
        caption: str,
        metric_config: ChairObjectHallucinationConfig,
    ) -> float | None:
        try:
            vectors = text_embeddings(
                [subject_name, caption],
                model_repo_or_path=metric_config.embedding_model_path,
                device=metric_config.device,
            )
        except (ImportError, RuntimeError, ValueError):
            return None
        if vectors[0] is None or vectors[1] is None:
            return None
        return cosine_similarity(vectors[0], vectors[1])

    @classmethod
    def _subject_crop(
        cls,
        image_value: Any,
        metric_config: ChairObjectHallucinationConfig,
    ) -> tuple[Any | None, str | None]:
        array = load_image_array(image_value)
        if array is None:
            return None, None
        try:
            from PIL import Image
        except ImportError:
            return None, None

        image = Image.fromarray(array).convert("RGB")
        segment = cls._largest_segment(image, metric_config)
        if segment is None:
            return image, None
        box = mask_box(segment.get("mask"), image.size)
        label = non_empty_text(segment.get("label"))
        if not box:
            return image, label or None
        return image.crop(box), label or None

    @staticmethod
    def _largest_segment(
        image: Any,
        metric_config: ChairObjectHallucinationConfig,
    ) -> dict[str, Any] | None:
        try:
            segmenter = _segmentation_pipeline(
                metric_config.segmentation_model_path,
                metric_config.device,
            )
            segments = segmenter(image)
        except Exception:
            return None
        if isinstance(segments, dict):
            segments = as_list(segments.get("segments") or segments.get("masks"))
        candidates = [item for item in as_list(segments) if isinstance(item, dict)]
        if not candidates:
            return None

        def segment_rank(item: dict[str, Any]) -> float:
            score = score_value(item.get("score"))
            if score < metric_config.min_subject_score:
                return -1.0
            area = mask_area(item.get("mask"))
            return area * score

        ranked = sorted(candidates, key=segment_rank, reverse=True)
        return ranked[0] if ranked and segment_rank(ranked[0]) > 0 else None

    @staticmethod
    def _recognize_crop(
        image: Any,
        metric_config: ChairObjectHallucinationConfig,
    ) -> str | None:
        try:
            if _uses_blip_recognition_backend(metric_config.recognition_model_path):
                return _blip_recognition_caption(
                    image,
                    metric_config,
                )
            recognizer = _recognition_pipeline(
                metric_config.recognition_model_path,
                metric_config.device,
            )
            outputs = recognizer(image)
        except Exception:
            return None
        for item in as_list(outputs):
            if isinstance(item, dict):
                text = non_empty_text(
                    item.get("generated_text")
                    or item.get("caption")
                )
                if text:
                    return text
            else:
                text = non_empty_text(item)
                if text:
                    return text
        return None


@lru_cache(maxsize=4)
def _segmentation_pipeline(model_path: str, device: str | None):
    from transformers import pipeline
    kwargs = {}
    if device:
        kwargs["device"] = device
    return pipeline("image-segmentation", model=model_path, **kwargs)


def _uses_blip_recognition_backend(model_path: str) -> bool:
    return "blip" in str(model_path).lower()


def _blip_recognition_caption(
    image: Any,
    metric_config: ChairObjectHallucinationConfig,
) -> str | None:
    processor, model, torch, resolved_device = _blip_recognition_backend(
        metric_config.recognition_model_path,
        metric_config.recognition_processor_path,
        metric_config.recognition_config_path,
        metric_config.recognition_use_safetensors,
        metric_config.device,
    )
    inputs = processor(images=image, return_tensors="pt")
    inputs = {
        key: value.to(resolved_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=20)
    decoder = getattr(processor, "decode", None) or processor.tokenizer.decode
    return non_empty_text(decoder(generated_ids[0], skip_special_tokens=True))


@lru_cache(maxsize=2)
def _blip_recognition_backend(
    model_path: str,
    processor_path: str | None,
    config_path: str | None,
    use_safetensors: bool | None,
    device: str | None,
):
    import torch
    from transformers import (
        BlipConfig,
        BlipForConditionalGeneration,
        BlipProcessor,
        GenerationConfig,
    )

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor_source = processor_path or model_path
    config_source = config_path or processor_source
    processor = BlipProcessor.from_pretrained(
        processor_source,
        local_files_only=True,
    )
    config = BlipConfig.from_pretrained(
        config_source,
        local_files_only=True,
    )
    generation_config = GenerationConfig.from_model_config(config)
    model_kwargs: dict[str, Any] = {
        "config": config,
        "generation_config": generation_config,
        "local_files_only": True,
    }
    if use_safetensors is not None:
        model_kwargs["use_safetensors"] = use_safetensors
    model = BlipForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs,
    )
    model.to(resolved_device)
    model.eval()
    return processor, model, torch, resolved_device


@lru_cache(maxsize=4)
def _recognition_pipeline(model_path: str, device: str | None):
    from transformers import pipeline
    kwargs = {}
    if device:
        kwargs["device"] = device
    return pipeline("image-to-text", model=model_path, **kwargs)




