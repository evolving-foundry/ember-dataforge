from typing import Any, Dict, Optional

from .video_augmentation_operator import VideoAugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.device import resolve_torch_device
from ...utils.video_ops_utils import (
    build_event_chain_prompt,
    build_temporal_structure_item,
    build_segment_text,
    build_temporal_description_map,
    build_temporal_sample_map,
    build_temporal_text_map
)

SUPPORTED_TEMPORAL_MODES = {"EVENT_WINDOW", "EVENT_CHAIN"}
DEFAULT_CHAIN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class VideoTemporalStructureOperation(VideoAugmentationOperation):
    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Build temporal structure from existing segments, samples, and descriptions",
        "params": {
            "segments_key": {
                "type": "string",
                "required": False,
                "default": "segments",
                "description": "Input segment field name"
            },
            "samples_key": {
                "type": "string",
                "required": False,
                "default": "samples",
                "description": "Input sampling field name"
            },
            "description_units_key": {
                "type": "string",
                "required": False,
                "default": "description_units",
                "description": "Input description-unit field name"
            },
            "text_signals_key": {
                "type": "string",
                "required": False,
                "default": "text_signals",
                "description": "Input text-evidence field name"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "temporal_structures",
                "description": "Output temporal-structure field name"
            },
            "result_key": {
                "type": "string",
                "required": False,
                "default": "temporal_result",
                "description": "Output result field name"
            },
            "temporal_mode": {
                "type": "string",
                "required": False,
                "default": "EVENT_WINDOW",
                "description": "Temporal structure mode: EVENT_WINDOW, EVENT_CHAIN"
            },
            "max_temporal_units": {
                "type": "integer",
                "required": False,
                "default": 12,
                "description": "Maximum number of temporal-structure units to process"
            },
            "chain_window_size": {
                "type": "integer",
                "required": False,
                "default": 3,
                "description": "EVENT_CHAIN mode window size"
            },
            "model_path": {
                "type": "string",
                "required": False,
                "default": DEFAULT_CHAIN_MODEL,
                "description": "Model path used by EVENT_CHAIN mode"
            },
            "max_new_tokens": {
                "type": "integer",
                "required": False,
                "default": 96,
                "description": "Maximum tokens for event-chain summaries"
            },
            "device_preference": {
                "type": "string",
                "required": False,
                "default": "auto",
                "description": "Device selection: auto (prefers NPU > CUDA > CPU), npu, npu:0, cuda, cuda:0, cpu, etc.; a device index may be specified"
            }
        },
        "example": {
            "segments_key": "segments",
            "samples_key": "samples",
            "temporal_mode": "EVENT_WINDOW"
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        temporal_mode = str(self.params.get("temporal_mode", "EVENT_WINDOW")).upper()
        if temporal_mode not in SUPPORTED_TEMPORAL_MODES:
            raise ValueError(f"Unsupported temporal mode: {temporal_mode}")

        self.segments_key = self.params.get("segments_key", "segments")
        self.samples_key = self.params.get("samples_key", "samples")
        self.description_units_key = self.params.get("description_units_key", "description_units")
        self.text_signals_key = self.params.get("text_signals_key", "text_signals")
        self.output_key = self.params.get("output_key", "temporal_structures")
        self.result_key = self.params.get("result_key", "temporal_result")
        self.temporal_mode = temporal_mode
        self.max_temporal_units = int(self.params.get("max_temporal_units", 12))
        self.chain_window_size = int(self.params.get("chain_window_size", 3))
        self.model_path = self.params.get("model_path", DEFAULT_CHAIN_MODEL)
        self.max_new_tokens = int(self.params.get("max_new_tokens", 96))
        self._chain_components = None

    def _get_available_device(self) -> str:
        """Get an available compute device, with priority NPU > CUDA > CPU

        Supports formats with a device index, such as npu:0 and cuda:1
        """
        return resolve_torch_device(self.params.get('device_preference', 'auto'))

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        def extract_row(row):
            segments = row.get(self.segments_key)
            if not segments:
                raise ValueError("Missing segments.")

            samples = row.get(self.samples_key)
            if not samples:
                raise ValueError("Missing samples.")

            temporal_result = self.extract(
                segments=segments,
                samples=samples,
                description_units=row.get(self.description_units_key) or [],
                text_signals=row.get(self.text_signals_key) or [],
            )
            row[self.output_key] = temporal_result["temporal_structures"]
            row[self.result_key] = temporal_result
            row["temporal_mode"] = temporal_result["temporal_mode"]
            row["temporal_metrics"] = temporal_result["global_metrics"]
            return row

        return dataset.map(extract_row)

    def extract(self, segments, samples, description_units=None, text_signals=None):
        description_map = build_temporal_description_map(description_units)
        text_map = build_temporal_text_map(text_signals)
        sample_map = build_temporal_sample_map(samples)

        if self.temporal_mode == "EVENT_WINDOW":
            temporal_structures = build_event_window_structures(
                segments=segments,
                max_temporal_units=self.max_temporal_units,
                description_map=description_map,
                text_map=text_map,
                sample_map=sample_map,
            )
        else:
            temporal_structures = build_event_chain_structures(
                segments=segments,
                max_temporal_units=self.max_temporal_units,
                chain_window_size=max(2, self.chain_window_size),
                description_map=description_map,
                text_map=text_map,
                sample_map=sample_map,
                chain_summary_builder=self._generate_event_chain,
            )

        for index, temporal_structure in enumerate(temporal_structures, start=1):
            temporal_structure["id"] = f"tmp_{index:06d}"
            temporal_structure["temporal_unit_id"] = temporal_structure["id"]

        return {
            "temporal_mode": self.temporal_mode,
            "global_metrics": {
                "total_segments": len(segments),
                "total_samples": len(samples),
                "total_temporal_structures": len(temporal_structures),
            },
            "temporal_structures": temporal_structures,
        }

    def _load_chain_components(self):
        if self._chain_components is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for event chain generation. Please install transformers and torch."
                ) from exc

            device = self._get_available_device()
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
            model = model.to(device)
            model.eval()
            self._chain_components = (model, tokenizer)

        return self._chain_components

    def _generate_event_chain(self, window_segments, description_map, text_map, sample_map):
        import torch

        model, tokenizer = self._load_chain_components()
        prompt = build_event_chain_prompt(window_segments, description_map, text_map, sample_map)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0].strip()


def build_event_window_structures(segments, max_temporal_units, description_map, text_map, sample_map):
    scoped_segments = segments[:max_temporal_units]
    segment_lookup = {str(segment["segment_id"]): segment for segment in segments}
    temporal_structures = []
    for segment in scoped_segments:
        segment_id = str(segment["segment_id"])
        temporal_structures.append(
            build_temporal_structure_item(
                [segment_id],
                "event_window",
                build_segment_text(segment_id, description_map, text_map, sample_map),
                segment_lookup,
            )
        )
    return temporal_structures


def build_event_chain_structures(
    segments,
    max_temporal_units,
    chain_window_size,
    description_map,
    text_map,
    sample_map,
    chain_summary_builder,
):
    segment_lookup = {str(segment["segment_id"]): segment for segment in segments}
    temporal_structures = []
    for end_index in range(1, len(segments)):
        if len(temporal_structures) >= int(max_temporal_units):
            break
        start_index = max(0, end_index - chain_window_size + 1)
        window_segments = segments[start_index : end_index + 1]
        summary = chain_summary_builder(window_segments, description_map, text_map, sample_map)
        segment_ids = [str(segment["segment_id"]) for segment in window_segments]
        temporal_structures.append(
            build_temporal_structure_item(segment_ids, "event_chain", summary, segment_lookup)
        )
    return temporal_structures
