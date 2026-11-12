from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curator.quality.ops_quality_assessor.config import (
    OpsQualityAssessmentConfig,
)
from curator.quality.ops_quality_assessor.context import EvaluationContext
from curator.quality.ops_quality_assessor.metrics import (
    coherence_score as coherence_score_module,
)
from curator.quality.ops_quality_assessor.metrics.coherence_score import (
    CoherenceScoreAssessor,
)


def _context(row: dict[str, Any]) -> EvaluationContext:
    return EvaluationContext(
        dataset=object(),  # type: ignore[arg-type]
        rows=[row],
        schema_fields=set(row),
        data={key: [value] for key, value in row.items()},
        row_count=1,
        config=OpsQualityAssessmentConfig(
            sample_size=1,
            enabled_metrics=["coherence_score"],
        ),
    )


def test_coherence_score_prefers_subtitle_entries_over_segment_text_signals(monkeypatch):
    row = {
        "segments": [
            {"segment_id": "seg_0001", "timestamp_range": {"start_sec": 0, "end_sec": 10}},
            {"segment_id": "seg_0002", "timestamp_range": {"start_sec": 10, "end_sec": 20}},
        ],
        "video": {"subtitle_uri": "/tmp/subtitle.entries.json"},
        "text_signals": [
            {
                "text_signal_id": "txt_0001",
                "segment_id": "seg_0001",
                "timestamp_range": {"start_sec": 0, "end_sec": 10},
                "text": "merged segment subtitle window",
            },
            {
                "text_signal_id": "txt_0002",
                "segment_id": "seg_0002",
                "timestamp_range": {"start_sec": 10, "end_sec": 20},
                "text": "merged segment subtitle window",
            },
        ],
        "description_units": [
            {"segment_id": "seg_0001", "description_text": "same generated caption"},
            {"segment_id": "seg_0002", "description_text": "same generated caption"},
        ],
    }
    seen_texts = []

    monkeypatch.setattr(
        coherence_score_module,
        "load_text_entries",
        lambda subtitle_uri: [
            {
                "start_sec": 18.7,
                "end_sec": 21.5,
                "text": "This blade has a dark past.",
            },
            {
                "start_sec": 22.8,
                "end_sec": 26.8,
                "text": "It has shed much innocent blood.",
            },
        ],
    )

    def fake_text_embeddings(texts, model_repo_or_path, device):
        seen_texts.extend(texts)
        return [[1.0, 0.0], [1.0, 0.0]]

    monkeypatch.setattr(
        coherence_score_module,
        "text_embeddings",
        fake_text_embeddings,
    )

    result = CoherenceScoreAssessor().assess(_context(row))

    assert seen_texts == [
        "This blade has a dark past.",
        "It has shed much innocent blood.",
    ]
    assert result.status == "passed"
    assert result.scores["coherence_score"] == 1.0
    assert result.scores["duplicate_description_rate"] == 0.0
    assert result.metadata["semantic_candidate_count"] == 1
    assert result.metadata["text_source_counts"] == {"subtitle_entries": 1}


def test_coherence_score_prefers_text_signals_over_description_units(monkeypatch):
    row = {
        "segments": [
            {"segment_id": "seg_0001", "timestamp_range": {"start_sec": 0, "end_sec": 10}},
            {"segment_id": "seg_0002", "timestamp_range": {"start_sec": 10, "end_sec": 20}},
        ],
        "description_units": [
            {"segment_id": "seg_0001", "description_text": "same generated caption"},
            {"segment_id": "seg_0002", "description_text": "same generated caption"},
        ],
        "text_signals": [
            {
                "text_signal_id": "txt_0002",
                "segment_id": "seg_0002",
                "timestamp_range": {"start_sec": 10, "end_sec": 20},
                "text": "It has shed much innocent blood.",
            },
            {
                "text_signal_id": "txt_0001",
                "segment_id": "seg_0001",
                "timestamp_range": {"start_sec": 0, "end_sec": 10},
                "text": "This blade has a dark past.",
            },
        ],
    }
    seen_texts = []

    def fake_text_embeddings(texts, model_repo_or_path, device):
        seen_texts.extend(texts)
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(
        coherence_score_module,
        "text_embeddings",
        fake_text_embeddings,
    )

    result = CoherenceScoreAssessor().assess(_context(row))

    assert seen_texts == [
        "This blade has a dark past.",
        "It has shed much innocent blood.",
    ]
    assert result.status == "passed"
    assert result.scores["coherence_score"] == 1.0
    assert result.scores["duplicate_description_rate"] == 0.0
    assert result.metadata["text_source_counts"] == {"text_signals": 1}


def test_coherence_score_falls_back_to_description_units(monkeypatch):
    row = {
        "segments": [
            {"segment_id": "seg_0001", "timestamp_range": {"start_sec": 0, "end_sec": 10}},
            {"segment_id": "seg_0002", "timestamp_range": {"start_sec": 10, "end_sec": 20}},
        ],
        "description_units": [
            {"segment_id": "seg_0001", "description_text": "same generated caption"},
            {"segment_id": "seg_0002", "description_text": "same generated caption"},
        ],
    }
    seen_texts = []

    def fake_text_embeddings(texts, model_repo_or_path, device):
        seen_texts.extend(texts)
        return [[1.0, 0.0], [1.0, 0.0]]

    monkeypatch.setattr(
        coherence_score_module,
        "text_embeddings",
        fake_text_embeddings,
    )

    result = CoherenceScoreAssessor().assess(_context(row))

    assert seen_texts == ["same generated caption", "same generated caption"]
    assert result.status == "passed"
    assert result.scores["coherence_score"] == 0.0
    assert result.scores["duplicate_description_rate"] == 1.0
    assert result.metadata["text_source_counts"] == {"description_units": 1}
