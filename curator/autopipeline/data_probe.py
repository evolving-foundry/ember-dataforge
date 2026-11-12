from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..quality.quality_assessor import DataQualityAssessor
from ..quality.quality_metrics import DatasetQualityReport, QualityAssessmentConfig


@dataclass
class DataProbeResult:
    """Data probe collection result."""
    total_rows: int
    num_columns: int
    schema_fields: Dict[str, str]                  # field_name → type
    detected_modalities: List[str]                 # text / image / audio / video
    text_fields: List[str]
    text_stats: Dict[str, Dict[str, Any]]          # field → {avg_length, min, max, empty_rate}
    null_rates: Dict[str, float]                   # field → null_rate
    duplicate_rate: float
    quality_report: Optional[DatasetQualityReport] = None

    def to_description(self) -> str:
        """Convert to a human-readable natural-language description for LLM understanding."""
        lines = [
            f"Dataset contains {self.total_rows} rows and {self.num_columns} columns.",
            f"Fields: {', '.join(self.schema_fields.keys())}",
        ]
        if self.detected_modalities:
            lines.append(f"Detected modalities: {', '.join(self.detected_modalities)}")
        if self.text_fields:
            lines.append(f"Text fields: {', '.join(self.text_fields)}")
            for tf, stats in self.text_stats.items():
                lines.append(
                    f"  - {tf}: average length={stats.get('avg_length', 0):.0f}, "
                    f"null rate={stats.get('empty_rate', 0):.1%}"
                )
        high_null = [(f, r) for f, r in self.null_rates.items() if r > 0.05]
        if high_null:
            lines.append("Fields with null rate >5%: " + ", ".join(f"{f}({r:.1%})" for f, r in high_null))
        lines.append(f"Estimated duplicate rate: {self.duplicate_rate:.1%}")
        if self.quality_report:
            lines.append(f"Overall quality score: {self.quality_report.overall_score:.2f}")
        return "\n".join(lines)


class DataProbe:
    """Data probe: collect features from a MultimodalDataset to provide decision support for AutoPipeline."""

    def __init__(
        self,
        sample_size: int = 500,
        run_quality_assessment: bool = True,
        quality_config: Optional[QualityAssessmentConfig] = None,
    ):
        self.sample_size = sample_size
        self.run_quality_assessment = run_quality_assessment
        self.quality_config = quality_config

    def probe(self, dataset) -> DataProbeResult:
        """Run probe collection and return a DataProbeResult."""
        sample = self._get_sample(dataset)
        if not sample:
            return DataProbeResult(
                total_rows=0,
                num_columns=0,
                schema_fields={},
                detected_modalities=[],
                text_fields=[],
                text_stats={},
                null_rates={},
                duplicate_rate=0.0,
            )

        fields = list(sample[0].keys())
        field_types = self._infer_field_types(sample, fields)
        text_fields = self._detect_text_fields(fields, field_types, sample)
        modalities = self._detect_modalities(fields, field_types, sample)
        text_stats = self._analyze_text_fields(sample, text_fields)
        null_rates = self._compute_null_rates(sample, fields)
        duplicate_rate = self._estimate_duplicate_rate(sample)

        quality_report = None
        if self.run_quality_assessment:
            try:
                cfg = self.quality_config or QualityAssessmentConfig(
                    sample_size=self.sample_size,
                    text_fields=text_fields,
                )
                assessor = DataQualityAssessor(cfg)
                quality_report = assessor.assess(dataset, operation_name="data_probe")
            except Exception:
                pass

        try:
            total_rows = dataset.dataset.count()
        except Exception:
            total_rows = len(sample)

        return DataProbeResult(
            total_rows=total_rows,
            num_columns=len(fields),
            schema_fields=field_types,
            detected_modalities=modalities,
            text_fields=text_fields,
            text_stats=text_stats,
            null_rates=null_rates,
            duplicate_rate=duplicate_rate,
            quality_report=quality_report,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sample(self, dataset) -> List[Dict[str, Any]]:
        try:
            return dataset.dataset.limit(self.sample_size).take_all()
        except Exception:
            return []

    def _infer_field_types(
        self, sample: List[Dict[str, Any]], fields: List[str]
    ) -> Dict[str, str]:
        type_map: Dict[str, str] = {}
        for field in fields:
            values = [r.get(field) for r in sample if r.get(field) is not None]
            if not values:
                type_map[field] = "unknown"
                continue
            # Count the most common type
            type_counts: Dict[str, int] = {}
            for v in values:
                t = type(v).__name__
                type_counts[t] = type_counts.get(t, 0) + 1
            type_map[field] = max(type_counts, key=type_counts.__getitem__)
        return type_map

    def _detect_text_fields(
        self,
        fields: List[str],
        field_types: Dict[str, str],
        sample: List[Dict[str, Any]],
    ) -> List[str]:
        text_fields = []
        for field in fields:
            if field_types.get(field) == "str":
                text_fields.append(field)
        return text_fields

    def _detect_modalities(
        self,
        fields: List[str],
        field_types: Dict[str, str],
        sample: List[Dict[str, Any]],
    ) -> List[str]:
        """Infer modality from keywords in field names."""
        modalities = set()
        text_keywords = {"text", "content", "sentence", "document", "desc", "caption"}
        image_keywords = {"image", "img", "photo", "picture", "visual"}
        audio_keywords = {"audio", "sound", "speech", "wav", "mp3"}
        video_keywords = {"video", "clip", "movie", "mp4"}

        for field in fields:
            fl = field.lower()
            if any(kw in fl for kw in text_keywords) and field_types.get(field) == "str":
                modalities.add("text")
            if any(kw in fl for kw in image_keywords):
                modalities.add("image")
            if any(kw in fl for kw in audio_keywords):
                modalities.add("audio")
            if any(kw in fl for kw in video_keywords):
                modalities.add("video")

        # If str fields exist without matching keywords, still treat them as containing text
        if not modalities and any(t == "str" for t in field_types.values()):
            modalities.add("text")

        return sorted(modalities)

    def _analyze_text_fields(
        self, sample: List[Dict[str, Any]], text_fields: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for field in text_fields:
            texts = [r.get(field, "") for r in sample if isinstance(r.get(field), str)]
            if not texts:
                continue
            lengths = [len(t) for t in texts]
            empty_count = sum(1 for t in texts if t.strip() == "")
            stats[field] = {
                "avg_length": sum(lengths) / len(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "empty_rate": empty_count / len(texts),
            }
        return stats

    def _compute_null_rates(
        self, sample: List[Dict[str, Any]], fields: List[str]
    ) -> Dict[str, float]:
        total = len(sample)
        if total == 0:
            return {}
        rates: Dict[str, float] = {}
        for field in fields:
            null_count = sum(
                1 for r in sample if r.get(field) is None or r.get(field) == ""
            )
            rates[field] = round(null_count / total, 4)
        return rates

    def _estimate_duplicate_rate(self, sample: List[Dict[str, Any]]) -> float:
        """Estimate the duplicate rate via hash-based deduplication."""
        if not sample:
            return 0.0
        hashes = set()
        dup_count = 0
        for row in sample:
            try:
                h = hashlib.md5(str(sorted(row.items())).encode()).hexdigest()
            except Exception:
                h = str(id(row))
            if h in hashes:
                dup_count += 1
            hashes.add(h)
        return round(dup_count / len(sample), 4)
