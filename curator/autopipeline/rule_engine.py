from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .data_probe import DataProbeResult


@dataclass
class OperationRecommendation:
    operation_name: str
    priority: int               # 1=required  2=strongly recommended  3=optional
    params: Dict[str, Any]
    reason: str
    estimated_impact: str


@dataclass
class PipelineRecommendation:
    operations: List[OperationRecommendation]
    target_task: str
    estimated_quality_improvement: float
    warnings: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _Rule:
    condition: Callable[[DataProbeResult, Optional[str]], bool]
    recommendations: List[OperationRecommendation]
    description: str


class RuleBasedRecommender:
    """Rule-based operator recommendation engine.

    Default rule set:
      - Any field with null_rate > 5% -> invalid_data_removal (required)
      - duplicate_rate > 5% -> exact_duplicate_removal (required)
      - Text field present -> token_count_mapping (recommended)
      - Very long text plus a quality field -> quality_based_filter (recommended)

    Saving operators are excluded from rule recommendations. AutoPipeline.build()
    appends them only when the caller explicitly provides output_path, avoiding
    empty save steps that cannot be executed.
    """

    def __init__(self) -> None:
        self._rules: List[_Rule] = self._build_default_rules()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def recommend(
        self,
        probe_result: DataProbeResult,
        target_task: str = "general_preprocessing",
        text_field: Optional[str] = None,
    ) -> PipelineRecommendation:
        """Recommend an operator list based on probe results.

        Only operators registered in Curator OperationRegistry are recommended.
        """
        implemented = self._get_implemented_ops()

        collected: List[OperationRecommendation] = []
        warnings: List[str] = []

        for rule in self._rules:
            try:
                if rule.condition(probe_result, text_field):
                    for rec in rule.recommendations:
                        if rec.operation_name in implemented:
                            collected.append(rec)
                        else:
                            warnings.append(
                                f"Operation '{rec.operation_name}' is not implemented; skipped"
                            )
            except Exception:
                pass

        # Deduplicate by name, keeping the first occurrence and preserving priority order
        seen = set()
        deduped: List[OperationRecommendation] = []
        for rec in sorted(collected, key=lambda r: r.priority):
            if rec.operation_name not in seen:
                seen.add(rec.operation_name)
                deduped.append(rec)

        # Sort by priority
        deduped.sort(key=lambda r: r.priority)

        # Estimate quality improvement
        estimated_improvement = self._estimate_improvement(probe_result, deduped)

        return PipelineRecommendation(
            operations=deduped,
            target_task=target_task,
            estimated_quality_improvement=estimated_improvement,
            warnings=warnings,
        )

    def register_rule(
        self,
        condition: Callable[[DataProbeResult, Optional[str]], bool],
        recommendations: List[OperationRecommendation],
        description: str = "",
    ) -> None:
        """Register a custom rule to extend the rule set."""
        self._rules.append(
            _Rule(condition=condition, recommendations=recommendations, description=description)
        )

    def to_pipeline_config(self, recommendation: PipelineRecommendation) -> Dict[str, Any]:
        """Convert recommendation results into a format accepted by MultimodalPipeline(config=...)."""
        ops = [
            {"name": rec.operation_name, "params": rec.params}
            for rec in recommendation.operations
        ]
        return {
            "pipeline": {
                "operations": ops
            }
        }

    # ------------------------------------------------------------------
    # Default rules
    # ------------------------------------------------------------------

    def _build_default_rules(self) -> List[_Rule]:
        return [
            _Rule(
                condition=lambda p, _: any(r > 0.05 for r in p.null_rates.values()),
                recommendations=[
                    OperationRecommendation(
                        operation_name="invalid_data_removal",
                        priority=1,
                        params={},
                        reason="Detected a field null rate > 5%",
                        estimated_impact="Remove invalid rows to improve the completeness score",
                    )
                ],
                description="High null rate triggered cleaning",
            ),
            _Rule(
                condition=lambda p, _: p.duplicate_rate > 0.05,
                recommendations=[
                    OperationRecommendation(
                        operation_name="exact_duplicate_removal",
                        priority=1,
                        params={},
                        reason=f"Detected an excessive duplicate rate",
                        estimated_impact="Remove duplicate records to improve the uniqueness score",
                    )
                ],
                description="High duplicate rate triggered deduplication",
            ),
            _Rule(
                condition=lambda p, tf: bool(p.text_fields),
                recommendations=[
                    OperationRecommendation(
                        operation_name="token_count_mapping",
                        priority=2,
                        params={"text_field": None},  # Replaced during recommendation
                        reason="Text fields are present; token counting supports downstream filtering",
                        estimated_impact="Add a token_count field for downstream filtering",
                    )
                ],
                description="Text fields triggered token counting",
            ),
            _Rule(
                condition=lambda p, _: any(
                    s.get("max_length", 0) > 10000 for s in p.text_stats.values()
                ),
                recommendations=[
                    OperationRecommendation(
                        operation_name="quality_based_filter",
                        priority=2,
                        params={},
                        reason="Very long text (> 10000 characters) detected; quality filtering is recommended",
                        estimated_impact="Filter abnormally long text to improve the text quality score",
                    )
                ],
                description="Very long text triggered quality filtering",
            ),
            _Rule(
                condition=lambda p, _: len(p.detected_modalities) > 1,
                recommendations=[
                    OperationRecommendation(
                        operation_name="schema_alignment",
                        priority=2,
                        params={},
                        reason="Multimodal data detected; schema alignment is recommended",
                        estimated_impact="Unify multimodal field structure to improve the consistency score",
                    )
                ],
                description="Multimodal data triggered schema alignment",
            ),
            _Rule(
                condition=lambda p, _: True,
                recommendations=[
                    OperationRecommendation(
                        operation_name="text_saving",
                        priority=3,
                        params={},
                        reason="Save processing results",
                        estimated_impact="Persist processing results to disk",
                    )
                ],
                description="Default: final save",
            ),
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_implemented_ops() -> set:
        try:
            from ...skills.schema_extractor import get_implemented_operations  # type: ignore
            return set(get_implemented_operations())
        except Exception:
            pass
        try:
            from skills.schema_extractor import get_implemented_operations  # type: ignore
            return set(get_implemented_operations())
        except Exception:
            pass
        # Fall back to the known implemented operations
        return {
            "text_loading", "image_loading", "video_loading", "audio_loading",
            "multimodal_loading", "text_saving", "image_saving", "video_saving",
            "audio_saving", "multimodal_saving", "exact_duplicate_removal",
            "quality_based_filter", "token_count_mapping", "invalid_data_removal",
        }

    @staticmethod
    def _estimate_improvement(
        probe: DataProbeResult, recs: List[OperationRecommendation]
    ) -> float:
        """Roughly estimate the quality improvement (0-1)."""
        base = probe.quality_report.overall_score if probe.quality_report else 0.5
        # Each priority=1 operation is expected to improve quality by 0.05; priority=2 by 0.02
        improvement = sum(0.05 if r.priority == 1 else 0.02 for r in recs)
        return round(min(base + improvement, 1.0) - base, 4)
