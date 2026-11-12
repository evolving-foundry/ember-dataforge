from typing import Any, Dict, Optional

from .video_augmentation_operator import VideoAugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.local_model_paths import local_model_roots_from_params, resolve_local_model_path
from ...utils.video_ops_utils import (
    merge_description_captions
)

SUPPORTED_VIDEO_DESCRIPTION_MODES = {
    "FRAME_DESCRIPTION",
    "EVENT_DESCRIPTION",
    "SEGMENT_DESCRIPTION",
    "CONTEXT_DESCRIPTION",
}

VIDEO_DESCRIPTION_PROMPTS = {
    "FRAME_DESCRIPTION": "a picture of",
    "EVENT_DESCRIPTION": "an event showing",
    "SEGMENT_DESCRIPTION": "a video segment showing",
    "CONTEXT_DESCRIPTION": "a broader video context showing",
}

MODEL_LIST = {
    "blip_base": "Salesforce/blip-image-captioning-base",
    "blip_large": "Salesforce/blip-image-captioning-large",
    "git_base": "microsoft/git-base-coco",
    "vit_gpt2": "nlpconnect/vit-gpt2-image-captioning",
    "local_qwen_vlm": None,
}


class VideoMultigranularityDescriptionOperation(VideoAugmentationOperation):
    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Generate multi-granularity descriptions from video sampling results",
        "params": {
            "segments_key": {
                "type": "string",
                "required": False,
                "default": "segments",
                "description": "Input segment-result field name"
            },
            "samples_key": {
                "type": "string",
                "required": False,
                "default": "samples",
                "description": "Input sampling-result field name"
            },
            "temporal_structures_key": {
                "type": "string",
                "required": False,
                "default": "temporal_structures",
                "description": "Temporal-structure field name used by event-description mode"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "description_units",
                "description": "Output description-unit field name"
            },
            "result_key": {
                "type": "string",
                "required": False,
                "default": "description_result",
                "description": "Output description-result field name"
            },
            "description_mode": {
                "type": "string",
                "required": False,
                "default": "SEGMENT_DESCRIPTION",
                "description": "Description mode: FRAME_DESCRIPTION, EVENT_DESCRIPTION, SEGMENT_DESCRIPTION, CONTEXT_DESCRIPTION"
            },
            "description_modes": {
                "type": "array",
                "required": False,
                "default": None,
                "description": "Generate multiple description modes at once; uses description_mode when empty"
            },
            "append_to_existing": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Append newly generated descriptions to the existing output_key field"
            },
            "model_name": {
                "type": "string",
                "required": False,
                "default": "blip_base",
                "description": "Caption model name; supports blip_base, blip_large, git_base, vit_gpt2, local_qwen_vlm"
            },
            "model_path": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Caption model path; resolved from model_name when empty"
            },
            "local_model_roots": {
                "type": "array",
                "required": False,
                "default": None,
                "description": "Local model search roots; defaults to MODEL_ROOTS and checks /data/llm_models"
            },
            "local_files_only": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Load models only from local files"
            },
            "max_new_tokens": {
                "type": "integer",
                "required": False,
                "default": 64,
                "description": "Maximum generated tokens"
            },
            "frame_path_key": {
                "type": "string",
                "required": False,
                "default": "frame_path",
                "description": "Frame path field name in sampling results"
            },
            "device_preference": {
                "type": "string",
                "required": False,
                "default": "auto",
                "description": "Device selection: auto (prefers NPU > CUDA > CPU), npu, npu:0, cuda, cuda:0, cpu, etc.; a device index may be specified"
            },
            "vlm_model_path": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Local Qwen-VL model path used by local_qwen_vlm"
            },
            "vlm_device_preference": {
                "type": "string",
                "required": False,
                "default": "auto",
                "description": "Device selection for local_qwen_vlm; defaults to preferring NPU"
            },
            "vlm_device_map": {
                "type": "string",
                "required": False,
                "default": "auto",
                "description": "local_qwen_vlm device_map"
            },
            "vlm_torch_dtype": {
                "type": "string",
                "required": False,
                "default": "auto",
                "description": "local_qwen_vlm torch dtype"
            },
            "vlm_min_pixels": {
                "type": "integer",
                "required": False,
                "default": 50176,
                "description": "Minimum image pixels for local_qwen_vlm"
            },
            "vlm_max_pixels": {
                "type": "integer",
                "required": False,
                "default": 409600,
                "description": "Maximum image pixels for local_qwen_vlm"
            }
        },
        "example": {
            "segments_key": "segments",
            "samples_key": "samples",
            "description_mode": "SEGMENT_DESCRIPTION",
            "model_name": "blip_base"
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        description_mode = str(self.params.get("description_mode", "SEGMENT_DESCRIPTION")).upper()
        model_name = self.params.get("model_name", "blip_base")

        if description_mode not in SUPPORTED_VIDEO_DESCRIPTION_MODES:
            raise ValueError(f"Unsupported description mode: {description_mode}")
        if model_name not in MODEL_LIST:
            raise ValueError(f"Unsupported model name: {model_name}")

        self.segments_key = self.params.get("segments_key", "segments")
        self.samples_key = self.params.get("samples_key", "samples")
        self.temporal_structures_key = self.params.get("temporal_structures_key", "temporal_structures")
        self.output_key = self.params.get("output_key", "description_units")
        self.result_key = self.params.get("result_key", "description_result")
        self.description_mode = description_mode
        self.description_modes = self._normalize_description_modes(
            self.params.get("description_modes"),
            fallback_mode=description_mode,
        )
        self.model_name = model_name
        configured_model_path = (
            self.params.get("vlm_model_path") or self.params.get("model_path")
            if model_name == "local_qwen_vlm"
            else self.params.get("model_path") or MODEL_LIST[model_name]
        )
        self.model_path = (
            configured_model_path
            if model_name == "local_qwen_vlm"
            else resolve_local_model_path(
                configured_model_path,
                local_model_roots_from_params(self.params.get("local_model_roots")),
            )
        )
        self.max_new_tokens = int(self.params.get("max_new_tokens", 64))
        self.frame_path_key = self.params.get("frame_path_key", "frame_path")
        self.append_to_existing = bool(self.params.get("append_to_existing", False))
        self._model = None
        self._processor = None
        self._torch = None
        self._caption_backend = model_name

    @staticmethod
    def _normalize_description_modes(raw_modes, fallback_mode):
        if raw_modes is None:
            values = [fallback_mode]
        elif isinstance(raw_modes, str):
            values = [part.strip() for part in raw_modes.split(",")]
        else:
            values = list(raw_modes)

        normalized = []
        for value in values:
            mode = str(value or "").strip().upper()
            if not mode:
                continue
            if mode not in SUPPORTED_VIDEO_DESCRIPTION_MODES:
                raise ValueError(f"Unsupported description mode: {mode}")
            if mode not in normalized:
                normalized.append(mode)
        if not normalized:
            raise ValueError("description_modes must contain at least one supported mode.")
        return normalized

    def _get_available_device(self) -> str:
        """Get an available compute device, with priority NPU > CUDA > CPU

        Supports formats with a device index, such as npu:0 and cuda:1
        """
        preference = self.params.get('device_preference', 'auto')
        if preference.startswith('npu'):
            try:
                import torch_npu
                if torch_npu.npu.is_available():
                    return preference
            except ImportError:
                pass
            raise RuntimeError(f"NPU device requested but not available: {preference}")
        elif preference.startswith('cuda'):
            import torch
            if torch.cuda.is_available():
                return preference
            raise RuntimeError(f"CUDA device requested but not available: {preference}")
        elif preference == 'cpu':
            return 'cpu'
        # auto
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                return 'npu'
        except ImportError:
            pass
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def _load_caption_components(self):
        if self._model is None or self._processor is None or self._torch is None:
            if self.model_name == "local_qwen_vlm":
                from ...utils.local_qwen_vlm_client import LocalQwenVlmClient

                self._model = LocalQwenVlmClient(
                    model_path=self.model_path,
                    local_model_roots=self.params.get("local_model_roots"),
                    local_files_only=bool(self.params.get("local_files_only", True)),
                    device_preference=self.params.get(
                        "vlm_device_preference",
                        self.params.get("device_preference", "auto"),
                    ),
                    device_map=self.params.get("vlm_device_map", "auto"),
                    torch_dtype=self.params.get("vlm_torch_dtype", "auto"),
                    min_pixels=int(self.params.get("vlm_min_pixels", 224 * 224)),
                    max_pixels=int(self.params.get("vlm_max_pixels", 640 * 640)),
                    max_new_tokens=self.max_new_tokens,
                )
                self.model_path = str(self._model.model_path)
                self._processor = self._model
                self._torch = object()
                return self._model, self._processor, self._torch

            try:
                import torch
                from transformers import AutoProcessor
            except ImportError as exc:
                raise ImportError(
                    "transformers and torch are required for video description generation. "
                    "Please install transformers and torch."
                ) from exc

            self._torch = torch
            device = self._get_available_device()
            local_files_only = bool(self.params.get("local_files_only", True))
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            if self.model_name in {"blip_base", "blip_large"}:
                from transformers import BlipForConditionalGeneration

                self._model = BlipForConditionalGeneration.from_pretrained(
                    self.model_path,
                    local_files_only=local_files_only,
                )
            else:
                from transformers import AutoModelForVision2Seq

                self._model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                )
            self._model = self._model.to(device)
            self._model.eval()

        return self._model, self._processor, self._torch

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        def describe_row(row):
            segments = row.get(self.segments_key)
            if not segments:
                raise ValueError("Missing segments.")

            samples = row.get(self.samples_key)
            if not samples:
                raise ValueError("Missing samples.")

            segments = self._segments_for_modes(segments, self.description_modes)
            row[self.segments_key] = segments
            existing_units = list(row.get(self.output_key) or []) if self.append_to_existing else []
            description_result = self.describe_modes(
                segments=segments,
                samples=samples,
                temporal_structures=row.get(self.temporal_structures_key),
                description_modes=self.description_modes,
                id_start=len(existing_units) + 1,
            )
            if existing_units:
                description_result = self._merge_existing_description_units(
                    existing_units,
                    description_result,
                )
            row[self.output_key] = description_result["description_units"]
            row[self.result_key] = description_result
            row["description_mode"] = description_result["description_mode"]
            row["description_modes"] = description_result["description_modes"]
            row["description_metrics"] = description_result["global_metrics"]
            return row

        return dataset.map(describe_row)

    def describe(self, segments, samples, temporal_structures=None):
        return self.describe_modes(
            segments=segments,
            samples=samples,
            temporal_structures=temporal_structures,
            description_modes=self.description_modes,
        )

    def describe_modes(self, segments, samples, temporal_structures=None, description_modes=None, id_start=1):
        model, processor, torch = self._load_caption_components()
        caption_cache = {}
        modes = self._normalize_description_modes(description_modes, self.description_mode)
        segments = self._segments_for_modes(segments, modes)
        all_description_units = []
        per_mode_results = []

        for mode in modes:
            description_units = self._describe_mode(
                mode=mode,
                segments=segments,
                samples=samples,
                temporal_structures=temporal_structures,
                model=model,
                processor=processor,
                torch=torch,
                caption_cache=caption_cache,
            )
            for description_unit in description_units:
                description_unit["description_mode"] = mode
                description_unit["description_source"] = "video_multigranularity_description"
                description_unit["description_model_name"] = self.model_name
                description_unit["description_model_path"] = str(self.model_path)
            all_description_units.extend(description_units)
            per_mode_results.append(
                {
                    "description_mode": mode,
                    "granularity": self._mode_granularity(mode),
                    "total_description_units": len(description_units),
                }
            )

        for index, description_unit in enumerate(all_description_units, start=int(id_start)):
            description_unit["description_id"] = f"desc_{index:06d}"
            description_unit["description_unit_id"] = description_unit["description_id"]
            description_unit["id"] = description_unit["description_id"]

        return {
            "description_mode": modes[0] if len(modes) == 1 else "MULTI_GRANULARITY",
            "description_modes": modes,
            "global_metrics": {
                "total_segments": len(segments),
                "total_samples": len(samples),
                "total_description_units": len(all_description_units),
                "description_units_by_mode": {
                    item["description_mode"]: item["total_description_units"]
                    for item in per_mode_results
                },
            },
            "description_results": per_mode_results,
            "description_units": all_description_units,
        }

    def _describe_mode(
        self,
        mode,
        segments,
        samples,
        temporal_structures,
        model,
        processor,
        torch,
        caption_cache,
    ):
        if mode == "FRAME_DESCRIPTION":
            return self._describe_frames(samples, model, processor, torch, caption_cache)
        if mode == "EVENT_DESCRIPTION":
            if not temporal_structures:
                raise ValueError("EVENT_DESCRIPTION requires 'temporal_structures'.")
            return self._describe_events(
                temporal_structures=temporal_structures,
                samples=samples,
                model=model,
                processor=processor,
                torch=torch,
                caption_cache=caption_cache,
            )
        if mode == "SEGMENT_DESCRIPTION":
            return self._describe_segments(
                segments=segments,
                samples=samples,
                model=model,
                processor=processor,
                torch=torch,
                caption_cache=caption_cache,
            )
        return self._describe_contexts(
            segments=self._segments_for_modes(segments, [mode]),
            samples=samples,
            model=model,
            processor=processor,
            torch=torch,
            caption_cache=caption_cache,
        )

    @staticmethod
    def _mode_granularity(mode):
        return {
            "FRAME_DESCRIPTION": "frame",
            "EVENT_DESCRIPTION": "event",
            "SEGMENT_DESCRIPTION": "segment",
            "CONTEXT_DESCRIPTION": "context",
        }[mode]

    def _segments_for_modes(self, segments, modes):
        normalized_segments = [dict(segment) for segment in segments or []]
        if "CONTEXT_DESCRIPTION" not in set(modes or []):
            return normalized_segments
        default_group_id = str(self.params.get("default_stitch_group_id") or "stitch_0001")
        for segment in normalized_segments:
            if not segment.get("stitch_group_id"):
                segment["stitch_group_id"] = default_group_id
        return normalized_segments

    @staticmethod
    def _merge_existing_description_units(existing_units, description_result):
        result = dict(description_result)
        generated_units = list(result.get("description_units") or [])
        result["description_units"] = list(existing_units or []) + generated_units
        metrics = dict(result.get("global_metrics") or {})
        metrics["total_existing_description_units"] = len(existing_units or [])
        metrics["total_generated_description_units"] = len(generated_units)
        metrics["total_description_units"] = len(result["description_units"])
        result["global_metrics"] = metrics
        return result

    def _caption_frame(self, frame_path, prompt_text, model, processor, torch, caption_cache):
        cache_key = f"{prompt_text}::{frame_path}"
        if cache_key in caption_cache:
            return caption_cache[cache_key]

        if self.model_name == "local_qwen_vlm":
            caption = self._caption_frame_with_qwen(frame_path, prompt_text, model)
            caption_cache[cache_key] = caption
            return caption

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required. Please install Pillow>=8.0.0.") from exc

        raw_image = Image.open(frame_path).convert("RGB")
        try:
            with torch.inference_mode():
                inputs = processor(images=raw_image, text=prompt_text, return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        finally:
            raw_image.close()

        caption_cache[cache_key] = caption
        return caption

    def _caption_frame_with_qwen(self, frame_path, prompt_text, client):
        prompt = (
            "Generate one concise, visually grounded English caption for this video frame. "
            f"Focus: {prompt_text}. Do not mention unseen facts."
        )
        response = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return str(response.choices[0].message.content or "").strip()

    def _describe_frames(self, samples, model, processor, torch, caption_cache):
        description_units = []
        for sample in samples:
            sample_id = str(sample["sample_id"])
            segment_id = str(sample["segment_id"])
            description_units.append(
                {
                    "segment_id": segment_id,
                    "description_text": self._caption_frame(
                        sample[self.frame_path_key],
                        VIDEO_DESCRIPTION_PROMPTS["FRAME_DESCRIPTION"],
                        model,
                        processor,
                        torch,
                        caption_cache,
                    ),
                    "description_scope": f"sample:{sample_id}",
                    "granularity": "frame",
                    "anchor_refs": {
                        "sample_ids": [sample_id],
                        "segment_ids": [segment_id],
                    },
                    "summary_tags": ["frame"],
                }
            )
        return description_units

    def _describe_events(self, temporal_structures, samples, model, processor, torch, caption_cache):
        description_units = []
        for temporal_structure in temporal_structures:
            temporal_unit_id = str(temporal_structure["temporal_unit_id"])
            segment_ids = [str(segment_id) for segment_id in temporal_structure["segment_ids"]]
            event_samples = [sample for sample in samples if str(sample["segment_id"]) in segment_ids]
            captions = [
                self._caption_frame(
                    sample[self.frame_path_key],
                    VIDEO_DESCRIPTION_PROMPTS["EVENT_DESCRIPTION"],
                    model,
                    processor,
                    torch,
                    caption_cache,
                )
                for sample in event_samples
            ]
            description_units.append(
                {
                    "segment_id": segment_ids[0],
                    "description_text": merge_description_captions(captions),
                    "description_scope": f"temporal_unit:{temporal_unit_id}",
                    "granularity": "event",
                    "anchor_refs": {
                        "temporal_unit_ids": [temporal_unit_id],
                        "segment_ids": segment_ids,
                        "sample_ids": [str(sample["sample_id"]) for sample in event_samples],
                    },
                    "summary_tags": ["event"],
                }
            )
        return description_units

    def _describe_segments(self, segments, samples, model, processor, torch, caption_cache):
        description_units = []
        for segment in segments:
            segment_id = str(segment["segment_id"])
            segment_samples = [sample for sample in samples if str(sample["segment_id"]) == segment_id]
            captions = [
                self._caption_frame(
                    sample[self.frame_path_key],
                    VIDEO_DESCRIPTION_PROMPTS["SEGMENT_DESCRIPTION"],
                    model,
                    processor,
                    torch,
                    caption_cache,
                )
                for sample in segment_samples
            ]
            description_units.append(
                {
                    "segment_id": segment_id,
                    "description_text": merge_description_captions(captions),
                    "description_scope": f"segment:{segment_id}",
                    "granularity": "segment",
                    "anchor_refs": {
                        "segment_ids": [segment_id],
                        "sample_ids": [str(sample["sample_id"]) for sample in segment_samples],
                    },
                    "summary_tags": ["segment"],
                }
            )
        return description_units

    def _describe_contexts(self, segments, samples, model, processor, torch, caption_cache):
        grouped_segments = {}
        for segment in segments:
            stitch_group_id = segment.get("stitch_group_id")
            if stitch_group_id is None:
                continue
            grouped_segments.setdefault(str(stitch_group_id), []).append(str(segment["segment_id"]))

        if not grouped_segments:
            raise ValueError("CONTEXT_DESCRIPTION requires 'segments[].stitch_group_id'.")

        description_units = []
        for stitch_group_id, segment_ids in grouped_segments.items():
            context_samples = [sample for sample in samples if str(sample["segment_id"]) in segment_ids]
            captions = [
                self._caption_frame(
                    sample[self.frame_path_key],
                    VIDEO_DESCRIPTION_PROMPTS["CONTEXT_DESCRIPTION"],
                    model,
                    processor,
                    torch,
                    caption_cache,
                )
                for sample in context_samples
            ]
            description_units.append(
                {
                    "segment_id": segment_ids[0],
                    "description_text": merge_description_captions(captions),
                    "description_scope": f"stitch_group:{stitch_group_id}",
                    "granularity": "context",
                    "anchor_refs": {
                        "segment_ids": segment_ids,
                        "sample_ids": [str(sample["sample_id"]) for sample in context_samples],
                    },
                    "summary_tags": ["context"],
                }
            )
        return description_units
