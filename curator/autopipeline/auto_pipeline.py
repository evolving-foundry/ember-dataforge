from __future__ import annotations

from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)

from .data_probe import DataProbe, DataProbeResult
from .rule_engine import PipelineRecommendation, RuleBasedRecommender


class AutoPipeline:
    """AutoPipeline intelligent build entry point.

    Three-step workflow:
      1. analyze(dataset) → DataProbeResult
      2. recommend(...) → PipelineRecommendation
      3. build(recommendation) → MultimodalPipeline

    Also provides a one-click interface: auto_build_and_run(dataset, ...)
    """

    def __init__(
        self,
        llm_advisor_enabled: bool = False,
        llm_client: Optional[Any] = None,  # anthropic.Anthropic or a compatible client
        sample_size: int = 500,
    ):
        self.llm_advisor_enabled = llm_advisor_enabled
        self.llm_client = llm_client
        self._probe = DataProbe(sample_size=sample_size)
        self._recommender = RuleBasedRecommender()
        self._last_probe: Optional[DataProbeResult] = None
        self._last_recommendation: Optional[PipelineRecommendation] = None

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def analyze(self, dataset) -> DataProbeResult:
        """Step 1: Collect dataset features."""
        self._last_probe = self._probe.probe(dataset)
        return self._last_probe

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    def recommend(
        self,
        target_task: str = "general_preprocessing",
        user_requirements: Optional[str] = None,
        text_field: Optional[str] = None,
        probe_result: Optional[DataProbeResult] = None,
    ) -> PipelineRecommendation:
        """Step 2: Recommend an operator list.

        Args:
            target_task: Target task description (for example, 'general_preprocessing' or 'text_quality_cleaning')
            user_requirements: Natural-language user requirements (used for LLM enhancement)
            text_field: Primary text field (overrides automatic detection)
            probe_result: Probe result; defaults to the result from the previous analyze() call
        """
        probe = probe_result or self._last_probe
        if probe is None:
            raise ValueError("Call analyze() or pass probe_result first")

        recommendation = self._recommender.recommend(probe, target_task, text_field)

        # Populate the text_field parameter of token_count_mapping with the text field
        actual_text_field = text_field or (probe.text_fields[0] if probe.text_fields else "text")
        for rec in recommendation.operations:
            if rec.operation_name == "token_count_mapping":
                rec.params["text_field"] = actual_text_field

        if self.llm_advisor_enabled and self.llm_client:
            try:
                recommendation = self._enhance_with_llm(
                    probe, recommendation, user_requirements
                )
            except Exception as e:
                logger.warning("[AutoPipeline] LLM enhancement failed; falling back to rule-based recommendations: %s", e)

        self._last_recommendation = recommendation
        return recommendation

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------

    def build(
        self,
        recommendation: Optional[PipelineRecommendation] = None,
        auto_run: bool = False,
        dataset=None,
    ):
        """Step 3: Build a MultimodalPipeline instance.

        Args:
            recommendation: Recommendation result; defaults to the result from the previous recommend() call
            auto_run: Whether to execute immediately
            dataset: Input dataset passed when auto_run=True
            output_path: Optional persistence path; text_saving is appended only when provided
            output_format: Format used by text_saving

        Returns:
            A MultimodalPipeline instance (auto_run=False), or
            (MultimodalPipeline, MultimodalDataset) (auto_run=True)
        """
        rec = recommendation or self._last_recommendation
        if rec is None:
            raise ValueError("Call recommend() or pass recommendation first")

        from ..core.pipeline import MultimodalPipeline

        config = self._recommender.to_pipeline_config(rec)
        pipeline = MultimodalPipeline(config=config)

        if auto_run:
            if dataset is None:
                raise ValueError("dataset is required when auto_run=True")
            result = pipeline.run(dataset)
            return pipeline, result

        return pipeline

    # ------------------------------------------------------------------
    # One-shot interface
    # ------------------------------------------------------------------

    def auto_build_and_run(
        self,
        dataset,
        target_task: str = "general_preprocessing",
        user_requirements: Optional[str] = None,
        text_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One-click interface: probe → recommend → build → execute.

        Returns:
            {
                'probe_result': DataProbeResult,
                'recommendation': PipelineRecommendation,
                'pipeline_config': dict,
                'output_dataset': MultimodalDataset,
                'tracking_report': TrackingMetadata,
            }
        """
        probe_result = self.analyze(dataset)
        recommendation = self.recommend(
            target_task=target_task,
            user_requirements=user_requirements,
            text_field=text_field,
            probe_result=probe_result,
        )
        pipeline_config = self._recommender.to_pipeline_config(recommendation)
        pipeline, output_dataset = self.build(
            recommendation=recommendation, auto_run=True, dataset=dataset
        )
        tracking_report = pipeline.get_tracking_report()

        return {
            "probe_result": probe_result,
            "recommendation": recommendation,
            "pipeline_config": pipeline_config,
            "output_dataset": output_dataset,
            "tracking_report": tracking_report,
        }

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain_recommendation(
        self, recommendation: Optional[PipelineRecommendation] = None
    ) -> str:
        """Generate a human-readable recommendation description."""
        rec = recommendation or self._last_recommendation
        if rec is None:
            return "No recommendation is available; call recommend() first."

        priority_label = {1: "[Required]", 2: "[Strongly recommended]", 3: "[Optional]"}
        lines = [
            f"Target task: {rec.target_task}",
            f"Expected quality improvement: {rec.estimated_quality_improvement:+.1%}",
            "",
            "Recommended operations:",
        ]
        for op in rec.operations:
            prio = priority_label.get(op.priority, "")
            lines.append(f"  {prio} {op.operation_name}")
            lines.append(f"    Reason: {op.reason}")
            lines.append(f"    Expected impact: {op.estimated_impact}")
            if op.params:
                lines.append(f"    Parameters: {op.params}")

        if rec.warnings:
            lines.append("")
            lines.append("Warning: ")
            for w in rec.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM enhancement
    # ------------------------------------------------------------------

    def _enhance_with_llm(
        self,
        probe: DataProbeResult,
        recommendation: PipelineRecommendation,
        user_requirements: Optional[str],
    ) -> PipelineRecommendation:
        """Enhance recommendation results through an LLM.

        Send the probe description, current recommendations, and user requirements to an Anthropic-compatible client.
        Fall back to rule-based recommendations if parsing fails.
        """
        try:
            from skills.function_tools import FUNCTION_TOOLS  # type: ignore
        except ImportError:
            FUNCTION_TOOLS = []

        probe_desc = probe.to_description()
        current_ops = [r.operation_name for r in recommendation.operations]

        system_prompt = (
            "You are a multimodal data processing expert. Based on the data characteristics and user requirements,"
            "optimize or confirm the recommended data processing operation list."
            "Recommend only available operations and explain each choice."
        )
        user_msg = f"""Data characteristics:
{probe_desc}

Operations currently recommended by rules:{', '.join(current_ops)}

User requirements:{user_requirements or 'No special requirements; follow the standard data quality improvement workflow'}

Confirm or adjust the operation recommendations and return JSON in this format:
{{"operations": [{{"name": "operation name", "reason": "reason"}}]}}"""

        response = self.llm_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            tools=FUNCTION_TOOLS or [],
        )

        # Parse the LLM response
        import json
        for block in response.content:
            if hasattr(block, "text"):
                text = block.text
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    try:
                        data = json.loads(text[start:end])
                        ops_data = data.get("operations", [])
                        new_ops = []
                        for op in ops_data:
                            name = op.get("name", "")
                            reason = op.get("reason", "LLM recommendation")
                            new_ops.append(
                                __import__(
                                    "curator.autopipeline.rule_engine",
                                    fromlist=["OperationRecommendation"],
                                ).OperationRecommendation(
                                    operation_name=name,
                                    priority=2,
                                    params={},
                                    reason=reason,
                                    estimated_impact="LLM evaluation",
                                )
                            )
                        if new_ops:
                            recommendation.operations = new_ops
                        return recommendation
                    except Exception:
                        pass

        return recommendation
