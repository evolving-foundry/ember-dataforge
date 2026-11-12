from __future__ import annotations

import json
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from ..base import BaseMetricAssessor
from ..context import EvaluationContext
from ..results import MetricResult
from .common import (
    as_list,
    non_empty_text,
    round_score,
)
from ....utils.local_qwen_vlm_client import LocalQwenVlmClient, extract_json_object
from ....utils.llm_api_config import (
    OpenAICompatibleConfig,
    create_openai_compatible_client,
)
from ....utils.video_ops_utils import (
    evidence_image_frame_paths,
    evidence_image_lookup_ids,
    evidence_temporal_ref_ids,
    evidence_text_ref_ids,
)
from .image_text_alignment import ImageTextAlignmentAssessor, ImageTextAlignmentConfig

JUDGE_PROMPT_TEMPLATE = """
You are a video question-answering data auditing expert. Determine whether the provided "answer" is fully and accurately supported by the provided "evidence text".

### Data to evaluate:
- **Question**: {question}
- **Answer**: {answer}
- **Evidence**: {evidence}

### Evaluation criteria:
1. **Factuality**: Can the core facts in the answer be found in the evidence?
2. **Relevance**: Does the evidence directly address the question?
3. **Entailment**: Can the evidence logically entail the answer?

### Output requirements:
Provide a brief logical analysis, then a score (1-5):
- 5: Fully supported; no logical flaws.
- 4: Mostly supported; only minor non-core details are missing.
- 3: Relevant but insufficient for a definitive conclusion.
- 2: Weakly supported; obvious factual deviations.
- 1: Completely irrelevant or the evidence directly contradicts the answer.

Return JSON in this format:
{{"analysis": "analysis text", "score": score_number}}
"""

VLM_JUDGE_PROMPT_TEMPLATE = """
You are a video question-answering grounding auditing expert. Use the provided sampled video frames and evidence text to determine whether the answer is fully supported.

### Data to evaluate:
- **Question**: {question}
- **Answer**: {answer}
- **Evidence**: {evidence}

### Evaluation criteria:
1. Whether the answer is consistent with visible content in the image frames.
2. Whether the evidence text or temporal structure directly supports the answer.
3. Whether the question, answer, and evidence disagree on subject, action, quantity, location, or temporal order.

Return JSON: {{"analysis": "brief analysis", "score": integer from 1 to 5}}. A score of 4 or above is considered a pass.
"""

class QaeGroundingAlignmentConfig(BaseModel):
    enable_visual_filter: bool = True
    visual_support_threshold: float = 0.25  # Discard directly when CLIP cosine similarity is below this value
    visual_config: ImageTextAlignmentConfig = Field(
        default_factory=ImageTextAlignmentConfig
    )

    judge_model: str | None = None
    temperature: float = 0.0
    llm_support_threshold_score: int = 4
    llm_client: Any | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_config_path: str | None = None
    require_llm_config: bool = False
    use_local_vlm_client: bool = False
    vlm_model_path: str | None = None
    local_model_roots: list[str] | str | None = None
    vlm_local_files_only: bool = True
    vlm_device_preference: str | None = "auto"
    vlm_device_map: str | dict[str, Any] | None = "auto"
    vlm_torch_dtype: str | None = "auto"
    vlm_min_pixels: int = 224 * 224
    vlm_max_pixels: int = 640 * 640
    vlm_max_new_tokens: int = 128
    max_evidence_images: int = 3


class QaeGroundingAlignmentAssessor(BaseMetricAssessor):
    metric_name = "qae_grounding_alignment"
    output_scores = ("answer_support_rate", "evidence_hit_rate", "visual_pass_rate", "llm_pass_rate")
    config_model = QaeGroundingAlignmentConfig

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client

    def _assess(self, context: EvaluationContext) -> MetricResult:
        metric_config = cast(
            QaeGroundingAlignmentConfig,
            self.resolve_metric_config(context) or QaeGroundingAlignmentConfig(),
        )

        if not context.has_field("qae_triplets"):
            return MetricResult(
                metric_name=self.metric_name,
                status="not_applicable",
                scores={},
                skipped_scores={score: "missing qae_triplets" for score in self.output_scores},
                notices=["missing qae_triplets"],
            )

        triplets_by_row = [
            (row, triplet)
            for row in context.rows
            for triplet in as_list(row.get("qae_triplets"))
            if isinstance(triplet, dict)
        ]
        if not triplets_by_row:
            return MetricResult(
                metric_name=self.metric_name,
                status="not_applicable",
                scores={},
                skipped_scores={score: "qae_triplets are empty" for score in self.output_scores},
            )

        llm_client = self._resolve_llm_client(metric_config)
        if llm_client is None and metric_config.require_llm_config:
            notice = "llm_client missing for qae_grounding_alignment"
            return MetricResult(
                metric_name=self.metric_name,
                status="failed_precondition",
                scores={},
                skipped_scores={score: notice for score in self.output_scores},
                notices=[notice],
                metadata={"triplet_count": len(triplets_by_row)},
            )

        triplet_count = 0
        evidence_hit_count = 0
        visual_passed_count = 0
        llm_judged_count = 0
        final_supported_count = 0
        llm_error_count = 0
        llm_error_messages = []

        for row, triplet in triplets_by_row:
            evidence_index = self._build_evidence_index(row)

            triplet_count += 1
            answer = non_empty_text(triplet.get("answer"))
            if not answer:
                triplet["status"] = "missing_answer"
                continue

            evidence_refs = triplet.get("evidence_refs")
            evidence_text_refs = (
                evidence_text_ref_ids(evidence_refs)
                + evidence_temporal_ref_ids(evidence_refs)
            )
            evidence_texts = [
                evidence_index[ref]["text"]
                for ref in evidence_text_refs
                if ref in evidence_index and evidence_index[ref].get("text")
            ]
            image_values = self._resolve_image_evidence_values(evidence_refs, evidence_index)
            if evidence_texts or image_values:
                evidence_hit_count += 1
            else:
                triplet["status"] = "missing_evidence"
                continue

            visual_passed = True
            if metric_config.enable_visual_filter:
                if not image_values:
                    visual_passed = False
                    triplet["status"] = "missing_visual_evidence"
                else:
                    visual_score = self._compute_max_visual_score(
                        image_values=image_values,
                        text_value=answer,
                        config=metric_config,
                    )
                    triplet["clip_score"] = visual_score

                    if visual_score is None or visual_score < metric_config.visual_support_threshold:
                        visual_passed = False
                        triplet["status"] = "rejected_by_vision"

            if visual_passed:
                visual_passed_count += 1
            else:
                continue

            can_judge_image_only = (
                llm_client is not None
                and getattr(llm_client, "supports_images", False)
                and bool(image_values)
            )
            if not evidence_texts and not can_judge_image_only:
                triplet["status"] = "missing_text_evidence"
                continue
            if llm_client is None:
                triplet["status"] = "evidence_prechecked"
                continue

            judgment, llm_error = self._call_llm_judge(
                llm_client=llm_client,
                question=triplet.get("question", "N/A"),
                answer=answer,
                evidence_texts=evidence_texts,
                image_values=image_values[: metric_config.max_evidence_images],
                config=metric_config,
            )
            if llm_error:
                llm_error_count += 1
                if len(llm_error_messages) < 3:
                    llm_error_messages.append(llm_error)
                triplet["status"] = "llm_judge_failed"
                continue

            if judgment:
                llm_judged_count += 1
                score = int(judgment.get("score", 0) or 0)
                triplet["llm_judge_score"] = score
                triplet["llm_judge_analysis"] = judgment.get("analysis")

                if score >= metric_config.llm_support_threshold_score:
                    final_supported_count += 1
                    triplet["status"] = "fully_supported"
                else:
                    triplet["status"] = "rejected_by_llm"

        scores = {
            "evidence_hit_rate": round_score(evidence_hit_count / triplet_count),
            "visual_pass_rate": round_score(visual_passed_count / triplet_count),
        }
        if visual_passed_count > 0:
            if llm_client is not None:
                scores["llm_pass_rate"] = round_score(final_supported_count / visual_passed_count)
                scores["answer_support_rate"] = round_score(final_supported_count / triplet_count)

        notices = []
        status = "passed"
        if llm_client is None:
            status = "warning"
            notices.append(
                "llm_client missing for qae_grounding_alignment; computed evidence/visual prechecks only."
            )
        if llm_error_count:
            status = "warning"
            notices.append(f"{llm_error_count} LLM judge calls failed.")
            notices.extend(llm_error_messages)
        if evidence_hit_count < triplet_count:
            status = "warning"
            notices.append(f"{triplet_count - evidence_hit_count} triplets have unresolved evidence refs.")
        if llm_client is not None and final_supported_count == 0 and triplet_count > 0:
            status = "warning"
            notices.append("No triplets passed the combined visual and LLM validation.")

        return MetricResult(
            metric_name=self.metric_name,
            status=status,
            scores=scores,
            notices=notices,
            metadata={
                "triplet_count": triplet_count,
                "evidence_hit_count": evidence_hit_count,
                "visual_passed_count": visual_passed_count,
                "llm_judged_count": llm_judged_count,
                "llm_error_count": llm_error_count,
                "final_supported_count": final_supported_count,
                "visual_threshold": metric_config.visual_support_threshold,
                "llm_threshold": metric_config.llm_support_threshold_score,
            },
        )

    def _resolve_llm_client(self, config: QaeGroundingAlignmentConfig) -> Any | None:
        if self.llm_client is not None:
            return self.llm_client
        if config.llm_client is not None:
            return config.llm_client
        if config.llm_base_url and config.llm_api_key and (config.llm_model or config.judge_model):
            return create_openai_compatible_client(
                config=OpenAICompatibleConfig(
                    base_url=config.llm_base_url,
                    api_key=config.llm_api_key,
                    model=config.llm_model or str(config.judge_model),
                )
            )
        if config.llm_config_path:
            try:
                return create_openai_compatible_client(
                    config_path=config.llm_config_path,
                    require_complete=config.require_llm_config,
                )
            except ValueError:
                return None
        if config.use_local_vlm_client:
            return LocalQwenVlmClient(
                model_path=config.vlm_model_path,
                local_model_roots=config.local_model_roots,
                local_files_only=config.vlm_local_files_only,
                device_preference=config.vlm_device_preference,
                device_map=config.vlm_device_map,
                torch_dtype=config.vlm_torch_dtype,
                min_pixels=config.vlm_min_pixels,
                max_pixels=config.vlm_max_pixels,
                max_new_tokens=config.vlm_max_new_tokens,
            )
        return None

    def _compute_max_visual_score(
        self,
        image_values: list[Any],
        text_value: str,
        config: QaeGroundingAlignmentConfig,
    ) -> float | None:
        max_score = -1.0

        for img in image_values:
            score = ImageTextAlignmentAssessor._score_from_embeddings(
                image_value=img,
                text_value=text_value,
                metric_config=config.visual_config,
            )

            if score is not None and score > max_score:
                max_score = score

        return max_score if max_score >= 0 else None

    def _call_llm_judge(
        self,
        llm_client: Any,
        question: str,
        answer: str,
        evidence_texts: list[str],
        image_values: list[Any],
        config: QaeGroundingAlignmentConfig,
    ) -> tuple[dict[str, Any] | None, str | None]:
        evidence_blob = " ".join(evidence_texts)
        image_values = [image for image in image_values or [] if image]
        if not answer or (not evidence_blob and not image_values):
            return None, None

        prompt_template = (
            VLM_JUDGE_PROMPT_TEMPLATE
            if image_values and getattr(llm_client, "supports_images", False)
            else JUDGE_PROMPT_TEMPLATE
        )
        prompt = prompt_template.format(
            question=question,
            answer=answer,
            evidence=evidence_blob or "N/A",
        )
        if image_values and getattr(llm_client, "supports_images", False):
            content = [{"type": "image", "image": image} for image in image_values]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        try:
            response = llm_client.chat(
                messages=messages,
                model=config.judge_model,
                temperature=config.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return extract_json_object(content) or json.loads(content), None
        except Exception as exc:
            return None, f"LLM judge failed: {exc}"

    @classmethod
    def _resolve_image_evidence_values(
        cls,
        evidence_refs: Any,
        evidence_index: dict[str, dict[str, Any]],
    ) -> list[Any]:
        values = list(evidence_image_frame_paths(evidence_refs))
        for ref in evidence_image_lookup_ids(evidence_refs):
            item = evidence_index.get(ref) or {}
            value = item.get("image")
            if value is not None:
                values.append(value)
        return _unique_non_empty(values)

    @classmethod
    def _build_evidence_index(cls, row: dict[str, Any]) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for field_name in (
            "evidence_units",
            "scaffold_units",
            "description_units",
            "text_signals",
            "temporal_structures",
            "samples",
        ):
            for item in [entry for entry in as_list(row.get(field_name)) if isinstance(entry, dict)]:
                for item_id in cls._item_ids(item):
                    text = cls._item_text(item)
                    image = cls._item_image(item)
                    existing = index.setdefault(item_id, {})
                    if text and not existing.get("text"):
                        existing["text"] = text
                    if image is not None and existing.get("image") is None:
                        existing["image"] = image
        return index

    @staticmethod
    def _item_ids(item: dict[str, Any]) -> list[str]:
        ids = []
        for key in (
            "evidence_unit_id", "scaffold_id", "description_unit_id",
            "description_id", "text_signal_id", "signal_id",
            "temporal_unit_id", "temporal_id", "sample_id", "id",
        ):
            if item.get(key):
                ids.append(str(item[key]))
        return _unique_non_empty(ids)

    @staticmethod
    def _item_id(item: dict[str, Any]) -> Optional[str]:
        ids = QaeGroundingAlignmentAssessor._item_ids(item)
        return ids[0] if ids else None

    @staticmethod
    def _item_text(item: dict[str, Any]) -> str:
        parts = []
        for key in (
            "evidence_summary", "scaffold_summary", "description_text",
            "text", "content", "summary", "temporal_label",
        ):
            value = non_empty_text(item.get(key))
            if value:
                parts.append(value)
        return " ".join(parts)

    @staticmethod
    def _item_image(item: dict[str, Any]) -> Any:
        for key in ("frame_path", "image_path", "image"):
            value = item.get(key)
            if value:
                return value
        return None


def _unique_non_empty(values: list[Any]) -> list[Any]:
    unique = []
    seen = set()
    for value in values:
        if value is None:
            continue
        marker = str(value)
        if not marker or marker in seen:
            continue
        unique.append(value)
        seen.add(marker)
    return unique
