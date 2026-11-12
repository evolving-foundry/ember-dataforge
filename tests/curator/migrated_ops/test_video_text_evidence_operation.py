"""
Test the video text evidence operator and text evidence utility functions
"""
import json
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
        load_text_entries,
        merge_text_signals_by_segment,
    )


class TestVideoTextEvidenceOperation:
    def test_operation_type(self):
        from curator.operations.augmentation import VideoTextEvidenceOperation
        assert VideoTextEvidenceOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        from curator.operations.augmentation import VideoTextEvidenceOperation
        assert "params" in VideoTextEvidenceOperation.PARAMS_SCHEMA
        assert "segments_key" in VideoTextEvidenceOperation.PARAMS_SCHEMA["params"]
        assert "samples_key" in VideoTextEvidenceOperation.PARAMS_SCHEMA["params"]
        assert "text_evidence_mode" in VideoTextEvidenceOperation.PARAMS_SCHEMA["params"]
        assert "merge_strategy" in VideoTextEvidenceOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        from curator.operations.augmentation import VideoTextEvidenceOperation
        op = VideoTextEvidenceOperation({})
        assert op.params.get("segments_key") == "segments"
        assert op.params.get("samples_key") == "samples"
        assert op.params.get("text_evidence_mode") == "OCR"
        assert op.params.get("merge_strategy") == "BY_SEGMENT"

    def test_custom_params(self):
        from curator.operations.augmentation import VideoTextEvidenceOperation
        op = VideoTextEvidenceOperation({
            "segments_key": "my_segments",
            "samples_key": "my_samples",
            "text_evidence_mode": "SUBTITLE_PARSE",
            "merge_strategy": "KEEP_ALL",
        })
        assert op.params.get("segments_key") == "my_segments"
        assert op.params.get("samples_key") == "my_samples"
        assert op.params.get("text_evidence_mode") == "SUBTITLE_PARSE"
        assert op.params.get("merge_strategy") == "KEEP_ALL"

    def test_registry_create_operation(self):
        op = OperationRegistry.create_operation(
            "video_text_evidence",
            {"text_evidence_mode": "OCR", "use_huggingface_ocr": False},
        )
        assert op is not None
        assert isinstance(op, Operation)

    def test_extract_ocr_mode(self):
        from curator.operations.augmentation import VideoTextEvidenceOperation

        op = VideoTextEvidenceOperation({
            "text_evidence_mode": "OCR",
            "use_huggingface_ocr": False,
        })
        result = op.extract(
            segments=[
                {
                    "segment_id": "seg_0001",
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 1.0},
                }
            ],
            samples=[
                {
                    "sample_id": "smp_000001",
                    "segment_id": "seg_0001",
                    "timestamp_sec": 0.4,
                    "ocr_text": "欢迎光临",
                    "language": "zh",
                },
                {
                    "sample_id": "smp_000002",
                    "segment_id": "seg_0001",
                    "timestamp_sec": 0.7,
                    "ocr_text": "欢迎光临",
                    "language": "zh",
                },
            ],
        )

        assert result["text_evidence_mode"] == "OCR"
        assert result["global_metrics"]["total_text_signals"] == 1
        assert result["text_signals"][0]["text_signal_id"] == "txt_000001"
        assert result["text_signals"][0]["signal_id"] == "txt_000001"
        assert result["text_signals"][0]["text"] == "欢迎光临"

    def test_extract_subtitle_parse_mode(self, tmp_path):
        from curator.operations.augmentation import VideoTextEvidenceOperation

        subtitle_path = tmp_path / "subtitle.json"
        subtitle_path.write_text(json.dumps([
            {"start_sec": 0.0, "end_sec": 0.8, "text": "第一句", "language": "zh"},
            {"start_sec": 0.7, "end_sec": 1.4, "text": "第二句", "language": "zh"},
        ]), encoding="utf-8")

        op = VideoTextEvidenceOperation({
            "text_evidence_mode": "SUBTITLE_PARSE",
            "merge_strategy": "BY_SEGMENT",
            "use_huggingface_ocr": False,
        })
        result = op.extract(
            segments=[
                {
                    "segment_id": "seg_0001",
                    "timestamp_range": {"start_sec": 0.0, "end_sec": 1.0},
                },
                {
                    "segment_id": "seg_0002",
                    "timestamp_range": {"start_sec": 1.0, "end_sec": 2.0},
                },
            ],
            samples=[],
            video={"subtitle_uri": str(subtitle_path)},
        )

        assert result["text_evidence_mode"] == "SUBTITLE_PARSE"
        assert result["global_metrics"]["total_text_signals"] == 2
        assert result["text_signals"][0]["text"] == "第一句 第二句"
        assert result["text_signals"][1]["text"] == "第二句"

    def test_execute_ocr_mode(self):
        from curator.operations.augmentation import VideoTextEvidenceOperation

        ds = ray.data.from_items([
            {
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "timestamp_range": {"start_sec": 0.0, "end_sec": 1.0},
                    }
                ],
                "samples": [
                    {
                        "sample_id": "smp_000001",
                        "segment_id": "seg_0001",
                        "timestamp_sec": 0.2,
                        "ocr_text": "EXIT",
                        "language": "en",
                    }
                ],
                "video": {},
            }
        ])
        dataset = MultimodalDataset(dataset=ds)
        op = VideoTextEvidenceOperation({
            "text_evidence_mode": "OCR",
            "use_huggingface_ocr": False,
        })

        result = op.execute(dataset)
        row = result.dataset.take(1)[0]

        assert row["text_evidence_mode"] == "OCR"
        assert row["text_evidence_metrics"]["total_segments"] == 1
        assert len(row["text_signals"]) == 1


class TestVideoTextEvidenceUtils:
    def test_load_text_entries_json_dict(self, tmp_path):
        text_path = tmp_path / "entries.json"
        text_path.write_text(
            json.dumps({"entries": [{"start_sec": 0.0, "text": "hello"}]}),
            encoding="utf-8",
        )
        assert load_text_entries(str(text_path)) == [{"start_sec": 0.0, "text": "hello"}]

    def test_merge_text_signals_by_segment(self):
        result = merge_text_signals_by_segment([
            {
                "segment_id": "seg_0001",
                "timestamp_range": {"start_sec": 0.2, "end_sec": 0.4},
                "text_type": "OCR",
                "text": "HELLO",
                "language": "en",
            },
            {
                "segment_id": "seg_0001",
                "timestamp_range": {"start_sec": 0.5, "end_sec": 0.6},
                "text_type": "OCR",
                "text": "WORLD",
                "language": "en",
            },
        ])
        assert len(result) == 1
        assert result[0]["timestamp_range"] == {"start_sec": 0.2, "end_sec": 0.6}
        assert result[0]["text"] == "HELLO WORLD"
