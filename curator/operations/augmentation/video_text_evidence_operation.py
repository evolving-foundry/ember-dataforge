from typing import Any, Dict, Optional

from .video_augmentation_operator import VideoAugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.video_ops_utils import (
    load_text_entries,
    merge_text_signals_by_segment,
)

MODEL_LIST = {
    "trocr_small_printed": "microsoft/trocr-small-printed",
    "trocr_base_printed": "microsoft/trocr-base-printed",
}

SUPPORTED_TEXT_EVIDENCE_MODES = {
    "OCR",
    "SUBTITLE_PARSE",
    "ASR_ALIGN",
}

SUPPORTED_MERGE_STRATEGIES = {"KEEP_ALL", "BY_SEGMENT"}


class VideoTextEvidenceOperation(VideoAugmentationOperation):
    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Build text evidence from video segments, samples, and text sources",
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
            "video_key": {
                "type": "string",
                "required": False,
                "default": "video",
                "description": "Input video metadata field name"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "text_signals",
                "description": "Output text-evidence-signal field name"
            },
            "result_key": {
                "type": "string",
                "required": False,
                "default": "text_evidence_result",
                "description": "Output text-evidence-result field name"
            },
            "text_evidence_mode": {
                "type": "string",
                "required": False,
                "default": "OCR",
                "description": "Text evidence mode: OCR, SUBTITLE_PARSE, ASR_ALIGN"
            },
            "merge_strategy": {
                "type": "string",
                "required": False,
                "default": "BY_SEGMENT",
                "description": "Merge strategy: KEEP_ALL, BY_SEGMENT"
            },
            "default_language": {
                "type": "string",
                "required": False,
                "default": "zh",
                "description": "Default language"
            },
            "use_huggingface_ocr": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Use HuggingFace OCR"
            },
            "model_name": {
                "type": "string",
                "required": False,
                "default": "trocr_small_printed",
                "description": "OCR model name"
            },
            "frame_path_key": {
                "type": "string",
                "required": False,
                "default": "frame_path",
                "description": "Frame path field name in sampling results"
            },
            "ocr_text_key": {
                "type": "string",
                "required": False,
                "default": "ocr_text",
                "description": "Pre-extracted OCR text field name in sampling results"
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
            "text_evidence_mode": "OCR"
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        text_evidence_mode = str(self.params.get("text_evidence_mode", "OCR")).upper()
        merge_strategy = str(self.params.get("merge_strategy", "BY_SEGMENT")).upper()
        model_name = self.params.get("model_name", "trocr_small_printed")

        if text_evidence_mode not in SUPPORTED_TEXT_EVIDENCE_MODES:
            raise ValueError(f"Unsupported text evidence mode: {text_evidence_mode}")
        if merge_strategy not in SUPPORTED_MERGE_STRATEGIES:
            raise ValueError(f"Unsupported merge strategy: {merge_strategy}")
        if model_name not in MODEL_LIST:
            raise ValueError(f"Unsupported model name: {model_name}")

        self.segments_key = self.params.get("segments_key", "segments")
        self.samples_key = self.params.get("samples_key", "samples")
        self.video_key = self.params.get("video_key", "video")
        self.output_key = self.params.get("output_key", "text_signals")
        self.result_key = self.params.get("result_key", "text_evidence_result")
        self.text_evidence_mode = text_evidence_mode
        self.merge_strategy = merge_strategy
        self.default_language = self.params.get("default_language", "zh")
        self.use_huggingface_ocr = bool(self.params.get("use_huggingface_ocr", True))
        self.model_name = model_name
        self.model_path = MODEL_LIST[model_name]
        self.frame_path_key = self.params.get("frame_path_key", "frame_path")
        self.ocr_text_key = self.params.get("ocr_text_key", "ocr_text")
        self._ocr_components = None

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

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        def extract_row(row):
            segments = row.get(self.segments_key)
            if not segments:
                raise ValueError("Missing segments.")

            samples = row.get(self.samples_key)
            if not samples:
                raise ValueError("Missing samples.")

            text_evidence_result = self.extract(
                segments=segments,
                samples=samples,
                video=row.get(self.video_key) or {},
            )
            row[self.output_key] = text_evidence_result["text_signals"]
            row[self.result_key] = text_evidence_result
            row["text_evidence_mode"] = text_evidence_result["text_evidence_mode"]
            row["text_evidence_metrics"] = text_evidence_result["global_metrics"]
            return row

        return dataset.map(extract_row)

    def extract(self, segments, samples, video=None):
        if self.text_evidence_mode == "OCR":
            text_signals = self._extract_ocr_signals(segments=segments, samples=samples)
        else:
            text_signals = self._extract_timed_text_signals(segments=segments, video=video or {})

        if self.merge_strategy == "BY_SEGMENT":
            text_signals = merge_text_signals_by_segment(text_signals)

        for index, text_signal in enumerate(text_signals, start=1):
            text_signal["id"] = f"txt_{index:06d}"
            text_signal["text_signal_id"] = text_signal["id"]
            text_signal["signal_id"] = text_signal["id"]

        return {
            "text_evidence_mode": self.text_evidence_mode,
            "global_metrics": {
                "total_segments": len(segments),
                "total_samples": len(samples),
                "total_text_signals": len(text_signals),
            },
            "text_signals": text_signals,
        }

    def _extract_ocr_signals(self, segments, samples):
        segment_map = {str(segment["segment_id"]): segment for segment in segments}
        model = processor = None
        if self.use_huggingface_ocr:
            model, processor = self._load_ocr_components()

        text_signals = []
        for sample in samples:
            segment_id = str(sample["segment_id"])
            if self.use_huggingface_ocr:
                text_value = self._run_huggingface_ocr(sample[self.frame_path_key], model, processor)
            else:
                text_value = str(sample.get(self.ocr_text_key) or "").strip()

            if not text_value:
                continue

            timestamp_sec = sample.get("timestamp_sec")
            if timestamp_sec is None:
                timestamp_range = dict(segment_map[segment_id]["timestamp_range"])
            else:
                timestamp_range = {
                    "start_sec": float(timestamp_sec),
                    "end_sec": float(timestamp_sec),
                }

            text_signals.append(
                {
                    "segment_id": segment_id,
                    "timestamp_range": timestamp_range,
                    "text_type": "OCR",
                    "text": text_value,
                    "language": str(sample.get("language") or self.default_language),
                }
            )
        return text_signals

    def _extract_timed_text_signals(self, segments, video):
        if self.text_evidence_mode == "SUBTITLE_PARSE":
            entries = load_text_entries(video.get("subtitle_uri"))
            text_type = "SUBTITLE_PARSE"
        else:
            entries = load_text_entries(video.get("asr_uri"))
            text_type = "ASR_ALIGN"

        text_signals = []
        for entry in entries:
            entry_text = str(entry.get("text") or "").strip()
            if not entry_text:
                continue

            entry_start_sec = float(entry["start_sec"])
            entry_end_sec = float(entry.get("end_sec", entry_start_sec))
            entry_language = str(entry.get("language") or self.default_language)

            for segment in segments:
                segment_id = str(segment["segment_id"])
                segment_start_sec = float(segment["timestamp_range"]["start_sec"])
                segment_end_sec = float(segment["timestamp_range"]["end_sec"])
                if entry_end_sec < segment_start_sec or entry_start_sec > segment_end_sec:
                    continue

                text_signals.append(
                    {
                        "segment_id": segment_id,
                        "timestamp_range": {
                            "start_sec": max(segment_start_sec, entry_start_sec),
                            "end_sec": min(segment_end_sec, entry_end_sec),
                        },
                        "text_type": text_type,
                        "text": entry_text,
                        "language": entry_language,
                    }
                )
        return text_signals

    def _load_ocr_components(self):
        if self._ocr_components is None:
            try:
                import torch
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for OCR generation. Please install transformers and torch."
                ) from exc

            device = self._get_available_device()
            processor = TrOCRProcessor.from_pretrained(self.model_path)
            model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            model = model.to(device)
            model.eval()
            self._ocr_components = (model, processor)

        return self._ocr_components

    @staticmethod
    def _run_huggingface_ocr(frame_path, model, processor):
        import torch
        from PIL import Image

        raw_image = Image.open(frame_path).convert("RGB")
        try:
            with torch.no_grad():
                pixel_values = processor(images=raw_image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(model.device)
                generated_ids = model.generate(pixel_values)
                return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        finally:
            raw_image.close()
