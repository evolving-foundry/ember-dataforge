from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from . import question_planning
from .video_augmentation_operator import VideoAugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.device import is_cuda_device, is_npu_device, resolve_torch_device
from ...utils.local_qwen_vlm_client import (
    LocalQwenVlmClient,
    extract_json_object,
)
from ...utils.llm_api_config import OpenAICompatibleConfig, create_openai_compatible_client
from ...utils.video_ops_utils import (
    evidence_image_frame_paths,
    evidence_image_lookup_ids,
    unique_preserve_order,
)

SUPPORTED_MATERIALIZATION_MODES = {"JOINT_SYNTHESIS", "DRAFT_BINDING", "ANSWER_VERIFY_FILL", "QA_FIRST_GROUNDING"}

QAE_LLM_PROMPT_TEMPLATE = """
You are a video question-answering evidence materialization assistant. Generate or revise exactly one QAE triple using only the provided evidence.

### Question plan for this item
Question type: {question_type}
Question objective: {question_instruction}
Avoid: do not use generic questions such as "What is happening?" or "What is shown?", and do not repeatedly ask what the subject is doing for non-action questions.

### Scaffold summary
{scaffold_summary}

### Evidence text
{evidence_text}

### Visual evidence candidates
{visual_evidence_text}

### Existing question-answer draft
Question: {draft_question}
Answer: {draft_answer}

### Output requirements
Return a JSON object with the fields question, answer, analysis, evidence_frame_indices, and evidence_frame_rationale. question must be answerable from the evidence; answer must be directly supported by the evidence and must not introduce outside facts. question must not exceed 18 English words and must not copy the scaffold summary verbatim; answer must not exceed 25 English words. evidence_frame_indices is the list of candidate frame indices that support answer; return an empty list when there are no visual candidates.
"""

QAE_VLM_PROMPT_TEMPLATE = """
You are a high-quality video question-answering data generation assistant. Use the provided sampled video frames, temporal structure, and evidence text to generate or revise a strongly grounded QAE triple.

### Question plan for this item
Question type: {question_type}
Question objective: {question_instruction}
Avoid: do not use generic questions such as "What is happening?" or "What is shown?", and do not repeatedly ask what the subject is doing for non-action questions.

### Scaffold summary
{scaffold_summary}

### Evidence text
{evidence_text}

### Visual evidence candidates
{visual_evidence_text}

### Existing question-answer draft
Question: {draft_question}
Answer: {draft_answer}

### Generation requirements
1. question must be directly answerable from the current frame and evidence; do not ask for background knowledge outside the evidence.
2. Follow the question type in the "Question plan for this item" exactly, covering one of visible actions, subject state, object relationships, temporal order, or scene changes; do not ask the same type of question for every item.
3. Keep answer concise and verifiable; do not add details absent from the frames or evidence. When visual candidates exist, answer must be directly supported by the image frames selected in evidence_frame_indices, not only by the text description.
4. evidence_frame_indices must include only candidate frame indices that directly support answer. Do not list frames from the same segment that do not support answer; multiple frames may be selected when answer describes a change or temporal order.
5. Except for temporal_change questions that explicitly compare changes or temporal order, select the single strongest evidence frame; do not use a frame already showing the next state as evidence for the previous state.
6. Avoid speculative words such as ready, likely, suggesting, and intends unless visible physical evidence in the selected frames directly supports them.
7. question must not exceed 18 English words and answer must not exceed 25 English words; do not copy the scaffold summary or long descriptions into question.
8. analysis briefly explains which visual or textual evidence supports the answer.

Return only a JSON object with the fields question, answer, analysis, evidence_frame_indices, and evidence_frame_rationale.
"""


class VideoQaeMaterializationOperation(VideoAugmentationOperation):
    operation_type = OperationType.AUGMENTATION
    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Generate QAE triples from QA pairs and their bound clip/segment/time",
        "params": {
            "scaffold_units_key": {"type": "string", "required": False, "default": "scaffold_units"},
            "qa_drafts_key": {"type": "string", "required": False, "default": "qa_drafts"},
            "evidence_units_key": {"type": "string", "required": False, "default": "evidence_units"},
            "temporal_structures_key": {"type": "string", "required": False, "default": "temporal_structures"},
            "description_units_key": {"type": "string", "required": False, "default": "description_units"},
            "text_signals_key": {"type": "string", "required": False, "default": "text_signals"},
            "output_key": {"type": "string", "required": False, "default": "qae_triplets"},
            "result_key": {"type": "string", "required": False, "default": "materialization_result"},
            "materialization_mode": {"type": "string", "required": False, "default": "JOINT_SYNTHESIS"},
            "triplet_type": {"type": "string", "required": False, "default": "single_hop"},
            "llm_config_path": {"type": "string", "required": False, "default": None},
            "use_llm_client": {"type": "boolean", "required": False, "default": True},
            "require_llm_client": {"type": "boolean", "required": False, "default": True},
            "llm_base_url": {"type": "string", "required": False, "default": None},
            "llm_api_key": {"type": "string", "required": False, "default": None},
            "llm_model": {"type": "string", "required": False, "default": None},
            "temperature": {"type": "number", "required": False, "default": 0.0},
            "use_local_vlm_client": {"type": "boolean", "required": False, "default": False},
            "vlm_model_path": {"type": "string", "required": False, "default": None},
            "local_model_roots": {"type": "array", "required": False, "default": None},
            "vlm_local_files_only": {"type": "boolean", "required": False, "default": True},
            "vlm_device_preference": {"type": "string", "required": False, "default": "auto"},
            "vlm_device_map": {"type": "string", "required": False, "default": "auto"},
            "vlm_torch_dtype": {"type": "string", "required": False, "default": "auto"},
            "vlm_min_pixels": {"type": "integer", "required": False, "default": 50176},
            "vlm_max_pixels": {"type": "integer", "required": False, "default": 409600},
            "vlm_max_new_tokens": {"type": "integer", "required": False, "default": 256},
            "max_evidence_images": {"type": "integer", "required": False, "default": 3},
            "llm_json_retry_count": {"type": "integer", "required": False, "default": 1},
            "question_plan_profile": {"type": "string", "required": False, "default": "generic"},
            "question_plans": {"type": "array", "required": False, "default": None},
            "question_plan_config_path": {"type": "string", "required": False, "default": None},
            "question_plan_config": {"type": "object", "required": False, "default": None},
            "generic_question_prefixes": {"type": "array", "required": False, "default": None},
            "ray_num_gpus": {"type": "number", "required": False, "default": None},
            "ray_num_npus": {"type": "number", "required": False, "default": None},
        },
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None, llm_client: Any | None = None):
        super().__init__(params)
        self.scaffold_units_key = self.params.get("scaffold_units_key", "scaffold_units")
        self.qa_drafts_key = self.params.get("qa_drafts_key", "qa_drafts")
        self.evidence_units_key = self.params.get("evidence_units_key", "evidence_units")
        self.temporal_structures_key = self.params.get("temporal_structures_key", "temporal_structures")
        self.description_units_key = self.params.get("description_units_key", "description_units")
        self.text_signals_key = self.params.get("text_signals_key", "text_signals")
        self.output_key = self.params.get("output_key", "qae_triplets")
        self.result_key = self.params.get("result_key", "materialization_result")
        self.materialization_mode = str(self.params.get("materialization_mode", "JOINT_SYNTHESIS")).upper()
        self.triplet_type = self.params.get("triplet_type", "single_hop")
        self.llm_config_path = self.params.get("llm_config_path")
        self.use_llm_client = bool(self.params.get("use_llm_client", True))
        self.require_llm_client = bool(self.params.get("require_llm_client", True))
        self.llm_base_url = self.params.get("llm_base_url")
        self.llm_api_key = self.params.get("llm_api_key")
        self.llm_model = self.params.get("llm_model")
        self.temperature = float(self.params.get("temperature", 0.0))
        self.use_local_vlm_client = bool(self.params.get("use_local_vlm_client", False))
        self.vlm_model_path = self.params.get("vlm_model_path")
        self.vlm_device_preference = self.params.get(
            "vlm_device_preference",
            self.params.get("device_preference", "auto"),
        )
        self.max_evidence_images = int(self.params.get("max_evidence_images", 3))
        self.llm_json_retry_count = int(self.params.get("llm_json_retry_count", 1))
        self.question_plan_config = question_planning.resolve_question_plan_config(self.params)
        self.question_plans = self.question_plan_config["question_plans"]
        self.generic_question_prefixes = self.question_plan_config["generic_question_prefixes"]
        self.llm_client = llm_client or self.params.get("llm_client")
        if self.materialization_mode not in SUPPORTED_MATERIALIZATION_MODES:
            raise ValueError(f"Unsupported materialization mode: {self.materialization_mode}")

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        def materialize_row(row):
            result = self.materialize(
                scaffold_units=row.get(self.scaffold_units_key) or [],
                qa_drafts=row.get(self.qa_drafts_key) or [],
                evidence_units=row.get(self.evidence_units_key) or [],
                temporal_structures=row.get(self.temporal_structures_key) or [],
                description_units=row.get(self.description_units_key) or [],
                text_signals=row.get(self.text_signals_key) or [],
                samples=row.get("samples") or [],
            )
            row[self.output_key] = result["qae_triplets"]
            row[self.result_key] = result
            row["materialization_mode"] = result["materialization_mode"]
            row["materialization_metrics"] = result["global_metrics"]
            return row

        if self.use_local_vlm_client:
            ray_map_kwargs = self._local_vlm_ray_map_kwargs()
            return MultimodalDataset(
                dataset.dataset.map(materialize_row, **ray_map_kwargs),
                dataset.modalities,
                dataset.validation_config,
                **dataset.kwargs,
            )
        return dataset.map(materialize_row)

    def _local_vlm_ray_map_kwargs(self) -> dict[str, Any]:
        runtime_device = resolve_torch_device(self.vlm_device_preference)
        if is_npu_device(runtime_device):
            num_npus = self.params.get("ray_num_npus")
            num_npus = 1 if num_npus is None else float(num_npus)
            return {"resources": {"NPU": num_npus}}
        if is_cuda_device(runtime_device):
            num_gpus = self.params.get("ray_num_gpus")
            num_gpus = 1 if num_gpus is None else float(num_gpus)
            return {"num_gpus": num_gpus}
        return {}

    def materialize(self, scaffold_units, qa_drafts, evidence_units=None, temporal_structures=None, description_units=None, text_signals=None, samples=None):
        evidence_index = build_materialization_evidence_index(evidence_units or [], temporal_structures or [], description_units or [], text_signals or [], samples or [])
        if self.materialization_mode == "QA_FIRST_GROUNDING":
            triplets = materialize_qae_with_temporal_grounding(
                self.materialization_mode, scaffold_units, qa_drafts, temporal_structures or [],
                description_units or [], text_signals or [], self.triplet_type, samples or [],
                question_plans=self.question_plans,
                evidence_index=evidence_index,
            )
        else:
            triplets = materialize_qae_from_scaffold(
                self.materialization_mode,
                scaffold_units,
                qa_drafts,
                evidence_units or [],
                self.triplet_type,
                question_plans=self.question_plans,
                evidence_index=evidence_index,
                temporal_structures=temporal_structures or [],
                description_units=description_units or [],
                text_signals=text_signals or [],
                samples=samples or [],
            )

        llm_count = 0
        if self.use_llm_client:
            llm_client = self._resolve_llm_client()
            if llm_client is None and self.require_llm_client:
                raise ValueError("Missing llm_client for video_qae_materialization.")
            for triplet_index, triplet in enumerate(triplets):
                question_plan = triplet.get("question_plan") or question_planning.choose_question_plan(
                    scaffold_summary=triplet.get("scaffold_summary") or triplet.get("answer") or "",
                    evidence_refs=triplet.get("evidence_refs") or {},
                    index=triplet_index,
                    question_plans=self.question_plans,
                )
                triplet["question_type"] = question_plan["type"]
                evidence_image_items = resolve_triplet_evidence_image_items(triplet, evidence_index)[: self.max_evidence_images]
                payload = self._call_llm_materializer(
                    llm_client, triplet.get("scaffold_summary") or triplet.get("answer") or "",
                    triplet.get("question") or "", triplet.get("answer") or "",
                    resolve_triplet_evidence_texts(triplet, evidence_index),
                    evidence_image_items,
                    question_plan=question_plan,
                ) if llm_client is not None else None
                if payload:
                    triplet["question"] = str(payload.get("question") or triplet.get("question") or "").strip()
                    triplet["answer"] = str(payload.get("answer") or triplet.get("answer") or "").strip()
                    grounded_answer = remove_inference_language(triplet["answer"])
                    if grounded_answer != triplet["answer"]:
                        triplet["llm_original_answer"] = triplet["answer"]
                        triplet["answer"] = grounded_answer
                        triplet["answer_rewrite_reason"] = "remove_unsupported_inference_language"
                    if question_planning.is_generic_repeated_question(
                        triplet["question"],
                        generic_prefixes=self.generic_question_prefixes,
                    ):
                        triplet["llm_original_question"] = triplet["question"]
                        triplet["question"] = question_plan["template"]
                        triplet["question_rewrite_reason"] = "avoid_repeated_generic_question"
                    if payload.get("analysis"):
                        triplet["llm_materialization_analysis"] = str(payload["analysis"])
                    apply_selected_evidence_frames(
                        triplet=triplet,
                        payload=payload,
                        evidence_image_items=evidence_image_items,
                        evidence_index=evidence_index,
                    )
                    triplet["materialized_by"] = (
                        "local_qwen_vlm" if getattr(llm_client, "supports_images", False)
                        else "llm_client"
                    )
                    if getattr(llm_client, "model_path", None):
                        triplet["materializer_model_path"] = str(llm_client.model_path)
                    llm_count += 1

        for index, triplet in enumerate(triplets, start=1):
            triplet_id = f"tri_{index:06d}"
            triplet["id"] = triplet_id
            triplet["triplet_id"] = triplet_id
            triplet.pop("scaffold_summary", None)
        return {
            "materialization_mode": self.materialization_mode,
            "global_metrics": {"total_scaffold_units": len(scaffold_units or []), "total_triplets": len(triplets), "llm_materialized_triplets": llm_count},
            "qae_triplets": triplets,
        }

    def _resolve_llm_client(self) -> Any | None:
        if self.llm_client is not None:
            return self.llm_client
        if self.llm_base_url and self.llm_api_key and self.llm_model:
            self.llm_client = create_openai_compatible_client(config=OpenAICompatibleConfig(base_url=self.llm_base_url, api_key=self.llm_api_key, model=self.llm_model))
            return self.llm_client
        if self.llm_config_path:
            try:
                self.llm_client = create_openai_compatible_client(config_path=self.llm_config_path, require_complete=self.require_llm_client)
            except ValueError:
                return None
            return self.llm_client
        if self.use_local_vlm_client:
            model_path = self.vlm_model_path or self.params.get("model_path")
            self.llm_client = LocalQwenVlmClient(
                model_path=model_path,
                local_model_roots=self.params.get("local_model_roots"),
                local_files_only=bool(self.params.get("vlm_local_files_only", True)),
                device_preference=self.vlm_device_preference,
                device_map=self.params.get("vlm_device_map", "auto"),
                torch_dtype=self.params.get("vlm_torch_dtype", "auto"),
                min_pixels=int(self.params.get("vlm_min_pixels", 224 * 224)),
                max_pixels=int(self.params.get("vlm_max_pixels", 640 * 640)),
                max_new_tokens=int(self.params.get("vlm_max_new_tokens", 256)),
            )
            self.llm_model = self.llm_model or str(self.llm_client.model_path)
            return self.llm_client
        return None

    def _call_llm_materializer(self, llm_client, scaffold_summary, draft_question, draft_answer, evidence_texts, evidence_images=None, question_plan=None):
        evidence_text = "\n".join(evidence_texts).strip()
        evidence_image_items = normalize_evidence_image_items(evidence_images or [])
        evidence_image_paths = [item["frame_path"] for item in evidence_image_items if item.get("frame_path")]
        if not evidence_text and not evidence_image_paths:
            return None
        question_plan = question_plan or self.question_plans[0]
        template = QAE_VLM_PROMPT_TEMPLATE if evidence_image_paths and getattr(llm_client, "supports_images", False) else QAE_LLM_PROMPT_TEMPLATE
        prompt = template.format(
            question_type=question_plan["type"],
            question_instruction=question_plan["instruction"],
            scaffold_summary=scaffold_summary,
            evidence_text=evidence_text or "N/A",
            visual_evidence_text=format_visual_evidence_candidates(evidence_image_items),
            draft_question=draft_question or "N/A",
            draft_answer=draft_answer or "N/A",
        )
        last_content = ""
        for retry_index in range(max(self.llm_json_retry_count, 0) + 1):
            retry_prompt = prompt
            if retry_index:
                candidate_indices = ", ".join(
                    str(item["candidate_index"])
                    for item in normalize_evidence_image_items(evidence_image_items)
                ) or "none"
                retry_prompt = (
                    f"{prompt}\n\n"
                    "Important: the previous output failed validation. Regenerate it and return only one JSON object;"
                    "do not include Markdown, explanatory prose, or empty output.\n"
                    f"The only valid visual candidate frame indices are: {candidate_indices}.\n"
                    "When valid visual candidate frame indices are not none, evidence_frame_indices must be a non-empty array,"
                    "and every value must come from those indices. For non-temporal_change questions, select only the single strongest evidence frame.\n"
                    "Use this JSON schema:"
                    "{\"question\":\"...\",\"answer\":\"...\",\"analysis\":\"...\","
                    "\"evidence_frame_indices\":[1],\"evidence_frame_rationale\":\"...\"}"
                )
            messages = build_materializer_messages(
                retry_prompt,
                evidence_image_paths,
                supports_images=getattr(llm_client, "supports_images", False),
            )
            response = llm_client.chat(
                messages=messages,
                model=self.llm_model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            last_content = response.choices[0].message.content or ""
            payload = parse_llm_json_payload(last_content)
            if payload is not None and evidence_frame_indices_are_valid(payload, evidence_image_items):
                return payload
        raise ValueError(f"VLM materializer returned invalid JSON or evidence frame indices: {last_content[:200]!r}")


def materialize_qae_with_temporal_grounding(
    materialization_mode,
    scaffold_units,
    qa_drafts,
    temporal_structures,
    description_units,
    text_signals,
    triplet_type,
    samples=None,
    question_plans=None,
    evidence_index=None,
):
    qa_by_id = {str(item["qa_draft_id"]): item for item in qa_drafts or []}
    evidence_index = evidence_index or build_materialization_evidence_index([], temporal_structures or [], description_units or [], text_signals or [], samples or [])
    triplets = []
    for scaffold_index, scaffold in enumerate(scaffold_units or []):
        draft_ids = value_list(scaffold, "source_qa_draft_ids")
        qa = qa_by_id.get(draft_ids[0]) if draft_ids else None
        if not qa:
            raise ValueError(f"Missing QA draft for scaffold {scaffold.get('scaffold_id')}.")
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        if not question or not answer:
            raise ValueError(f"QA draft {qa.get('qa_draft_id')} must include question and answer.")
        refs = build_qa_clip_evidence_refs(qa, temporal_structures, description_units, text_signals, samples or [])
        if not evidence_ref_count(refs):
            raise ValueError(f"No evidence found for QA draft {qa.get('qa_draft_id')}.")
        planning = question_planning.choose_evidence_aware_question_plan(
            scaffold_summary=scaffold.get("scaffold_summary") or answer,
            evidence_refs=refs,
            index=scaffold_index,
            question_plans=question_plans,
            evidence_index=evidence_index,
        )
        question_plan = planning["question_plan"]
        triplets.append({
            "scaffold_id": str(scaffold["scaffold_id"]),
            "source_qa_draft_id": str(qa.get("qa_draft_id") or ""),
            "question": question,
            "answer": answer,
            "evidence_refs": refs,
            "triplet_type": normalized_triplet_type(triplet_type),
            "question_type": question_plan["type"],
            "question_plan": question_plan,
            "planning_trace": planning["planning_trace"],
            "grounding_level": "strong_grounded",
            "materialization_mode": materialization_mode,
            "scaffold_summary": str(scaffold.get("scaffold_summary") or ""),
        })
    return triplets


def build_qa_clip_evidence_refs(qa, temporal_structures, description_units, text_signals, samples):
    clip_ids = value_list(qa, "clip_id", "clip_ids")
    segment_ids = value_list(qa, "segment_id", "segment_ids")
    time_range = get_time_range(qa)
    if not clip_ids and not segment_ids and not time_range:
        raise ValueError(f"QA draft {qa.get('qa_draft_id')} is missing clip, segment, or time grounding.")
    temporal = [item for item in temporal_structures or [] if item_in_scope(item, clip_ids, segment_ids, time_range)]
    clip_ids = unique_preserve_order(clip_ids + collect_values(temporal, "clip_id", "clip_ids"))
    segment_ids = unique_preserve_order(segment_ids + collect_values(temporal, "segment_id", "segment_ids"))
    desc_ids = collect_values(temporal, "source_description_ids")
    text_ids = collect_values(temporal, "source_text_signal_ids")
    desc = [item for item in description_units or [] if item_in_scope(item, clip_ids, segment_ids, time_range) or has_common(value_list(item, "description_unit_id", "description_id", "id"), desc_ids)]
    text = [item for item in text_signals or [] if item_in_scope(item, clip_ids, segment_ids, time_range) or has_common(value_list(item, "signal_id", "text_signal_id", "id"), text_ids)]
    sample = [item for item in samples or [] if item_in_scope(item, clip_ids, segment_ids, time_range)]
    return {
        "text": {
            "description_ids": unique_preserve_order(desc_ids + collect_values(desc, "description_unit_id", "description_id", "id")),
            "text_signal_ids": unique_preserve_order(text_ids + collect_values(text, "signal_id", "text_signal_id", "id")),
            "evidence_unit_ids": collect_values(temporal, "source_evidence_unit_ids"),
        },
        "image": {"frame_refs": collect_values(sample, "frame_ref", "frame_refs", "node_ref"), "sample_ids": collect_values(sample, "sample_id", "id"), "frame_paths": collect_values(sample, "frame_path")},
        "temporal": {"temporal_unit_ids": collect_values(temporal, "temporal_unit_id", "temporal_id", "id")},
    }


def materialize_qae_from_scaffold(
    materialization_mode,
    scaffold_units,
    qa_drafts,
    evidence_map=None,
    triplet_type="single_hop",
    question_plans=None,
    evidence_index=None,
    temporal_structures=None,
    description_units=None,
    text_signals=None,
    samples=None,
    **_,
):
    qa_by_id = {str(item["qa_draft_id"]): item for item in qa_drafts or []}
    evidence_index = evidence_index or build_materialization_evidence_index(evidence_map or [], temporal_structures or [], description_units or [], text_signals or [], samples or [])
    triplets = []
    for scaffold_index, scaffold in enumerate(scaffold_units or []):
        draft_ids = value_list(scaffold, "source_qa_draft_ids")
        qa = qa_by_id.get(draft_ids[0]) if draft_ids else {}
        refs = {
            "text": {"description_ids": value_list(scaffold, "source_description_ids"), "text_signal_ids": value_list(scaffold, "source_text_signal_ids"), "evidence_unit_ids": value_list(scaffold, "source_evidence_unit_ids")},
            "image": question_planning.scaffold_image_refs(scaffold),
            "temporal": {"temporal_unit_ids": value_list(scaffold, "source_temporal_unit_ids", "temporal_unit_id")},
        }
        planning = question_planning.choose_evidence_aware_question_plan(
            scaffold_summary=scaffold.get("scaffold_summary") or scaffold.get("answer") or "",
            evidence_refs=refs,
            index=scaffold_index,
            question_plans=question_plans,
            evidence_index=evidence_index,
        )
        question_plan = planning["question_plan"]
        question = str(
            qa.get("question")
            or scaffold.get("question")
            or question_planning.build_rule_question_from_scaffold(scaffold, question_plan, question_plans)
        ).strip()
        answer = str(qa.get("answer") or scaffold.get("answer") or scaffold.get("scaffold_summary") or "").strip()
        if not answer:
            answer = best_evidence_answer(refs, evidence_map or {})
        if not question or not answer:
            raise ValueError(f"Missing question or answer for scaffold {scaffold.get('scaffold_id')}.")
        triplets.append({
            "scaffold_id": str(scaffold["scaffold_id"]),
            "source_qa_draft_id": str(qa.get("qa_draft_id") or ""),
            "question": question,
            "answer": answer,
            "evidence_refs": refs,
            "triplet_type": normalized_triplet_type(triplet_type),
            "question_type": question_plan["type"],
            "question_plan": question_plan,
            "planning_trace": planning["planning_trace"],
            "grounding_level": "strong_grounded" if evidence_ref_count(refs) else "ungrounded",
            "materialization_mode": materialization_mode,
            "scaffold_summary": str(scaffold.get("scaffold_summary") or ""),
        })
    return triplets


def item_in_scope(item, clip_ids, segment_ids, time_range):
    item_clip_ids = value_list(item, "clip_id", "clip_ids")
    item_segment_ids = value_list(item, "segment_id", "segment_ids")
    anchors = item.get("anchor_refs") or {}
    item_clip_ids += value_list(anchors, "clip_id", "clip_ids")
    item_segment_ids += value_list(anchors, "segment_id", "segment_ids")
    if has_common(clip_ids, item_clip_ids + item_segment_ids):
        return True
    if has_common(segment_ids, item_segment_ids + item_clip_ids):
        return True
    item_time_range = get_time_range(item)
    return bool(time_range and item_time_range and ranges_overlap(time_range, item_time_range))


def build_materialization_evidence_index(evidence_units, temporal_structures, description_units, text_signals, samples=None):
    index = {}
    for item in list(evidence_units or []) + list(temporal_structures or []) + list(description_units or []) + list(text_signals or []) + list(samples or []):
        if isinstance(item, dict):
            for item_id in item_ids(item):
                index[item_id] = {"text": item_text(item), "image": item_image(item), "item": item}
    return index


def resolve_triplet_evidence_texts(qae_triplet, evidence_index):
    ids = evidence_text_ref_ids(qae_triplet.get("evidence_refs")) + evidence_temporal_ref_ids(qae_triplet.get("evidence_refs"))
    return unique_preserve_order([evidence_index[item_id]["text"] for item_id in ids if evidence_index.get(item_id, {}).get("text")])


def resolve_triplet_evidence_images(qae_triplet, evidence_index):
    refs = qae_triplet.get("evidence_refs")
    values = list(evidence_image_frame_paths(refs))
    for item_id in evidence_image_lookup_ids(refs):
        image = evidence_index.get(item_id, {}).get("image")
        if image:
            values.append(image)
    return unique_preserve_order(values)


def resolve_triplet_evidence_image_items(qae_triplet, evidence_index):
    refs = qae_triplet.get("evidence_refs") or {}
    image_refs = (refs.get("image") or {})
    frame_paths = value_list(image_refs, "frame_paths")
    sample_ids = value_list(image_refs, "sample_ids")
    frame_refs = value_list(image_refs, "frame_refs")
    items = []
    max_len = max(len(frame_paths), len(sample_ids), len(frame_refs), 0)
    for index in range(max_len):
        sample_id = sample_ids[index] if index < len(sample_ids) else ""
        frame_ref = frame_refs[index] if index < len(frame_refs) else ""
        frame_path = frame_paths[index] if index < len(frame_paths) else ""
        if not frame_path:
            lookup_id = sample_id or frame_ref
            frame_path = str(evidence_index.get(lookup_id, {}).get("image") or "")
        items.append(
            {
                "candidate_index": index + 1,
                "sample_id": sample_id,
                "frame_ref": frame_ref,
                "frame_path": frame_path,
            }
        )
    if not items:
        for item_id in evidence_image_lookup_ids(refs):
            image = evidence_index.get(item_id, {}).get("image")
            if image:
                items.append(
                    {
                        "candidate_index": len(items) + 1,
                        "sample_id": item_id,
                        "frame_ref": "",
                        "frame_path": str(image),
                    }
                )
    return [
        item
        for item in items
        if item.get("frame_path") or item.get("sample_id") or item.get("frame_ref")
    ]


def normalize_evidence_image_items(evidence_images):
    items = []
    for index, image in enumerate(evidence_images or [], start=1):
        if isinstance(image, dict):
            item = dict(image)
            item["candidate_index"] = int(item.get("candidate_index") or index)
            item["frame_path"] = str(item.get("frame_path") or item.get("image") or "")
        else:
            item = {
                "candidate_index": index,
                "sample_id": "",
                "frame_ref": "",
                "frame_path": str(image),
            }
        items.append(item)
    return [item for item in items if item.get("frame_path")]


def build_materializer_messages(prompt, evidence_image_paths, supports_images=False):
    if evidence_image_paths and supports_images:
        content = [{"type": "image", "image": image} for image in evidence_image_paths]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": prompt}]


def parse_llm_json_payload(content):
    if not str(content or "").strip():
        return None
    extracted = extract_json_object(content)
    if extracted is not None:
        return extracted
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None


def evidence_frame_indices_are_valid(payload, evidence_image_items):
    image_items = normalize_evidence_image_items(evidence_image_items)
    if not image_items:
        return True
    selected_indices = parse_evidence_frame_indices(payload.get("evidence_frame_indices"))
    if not selected_indices:
        return False
    candidate_indices = {
        int(item.get("candidate_index") or 0)
        for item in image_items
        if int(item.get("candidate_index") or 0) > 0
    }
    return bool(candidate_indices) and set(selected_indices) <= candidate_indices


def format_visual_evidence_candidates(evidence_image_items):
    items = normalize_evidence_image_items(evidence_image_items)
    if not items:
        return "N/A"
    lines = []
    for item in items:
        parts = [f"Frame {item['candidate_index']}"]
        if item.get("sample_id"):
            parts.append(f"sample_id={item['sample_id']}")
        if item.get("frame_ref"):
            parts.append(f"frame_ref={item['frame_ref']}")
        if item.get("frame_path"):
            parts.append(f"path={item['frame_path']}")
        lines.append("; ".join(parts))
    return "\n".join(lines)


def apply_selected_evidence_frames(triplet, payload, evidence_image_items, evidence_index):
    selected_indices = parse_evidence_frame_indices(payload.get("evidence_frame_indices"))
    if not selected_indices:
        return
    if should_use_single_frame_evidence(triplet) and len(selected_indices) > 1:
        selected_indices = selected_indices[:1]
    image_items = normalize_evidence_image_items(evidence_image_items)
    selected = [
        item
        for item in image_items
        if int(item.get("candidate_index") or 0) in selected_indices
    ]
    if not selected:
        return
    image_refs = (triplet.setdefault("evidence_refs", {}).setdefault("image", {}))
    image_refs["frame_paths"] = unique_preserve_order(
        item["frame_path"] for item in selected if item.get("frame_path")
    )
    image_refs["sample_ids"] = unique_preserve_order(
        item["sample_id"] for item in selected if item.get("sample_id")
    )
    image_refs["frame_refs"] = unique_preserve_order(
        item["frame_ref"] for item in selected if item.get("frame_ref")
    )
    image_refs["selected_frame_indices"] = selected_indices
    if should_use_single_frame_evidence(triplet):
        image_refs["selection_policy"] = "single_strongest_frame_for_non_temporal_question"
    if payload.get("evidence_frame_rationale"):
        image_refs["selection_rationale"] = remove_inference_language(
            str(payload["evidence_frame_rationale"])
        )
    triplet["evidence_frame_indices"] = selected_indices
    if payload.get("evidence_frame_rationale"):
        triplet["evidence_frame_rationale"] = image_refs.get("selection_rationale")
    prune_unselected_description_refs(triplet, evidence_index)


def remove_inference_language(text):
    value = str(text or "").strip()
    if not value:
        return value
    patterns = (
        r",?\s+ready to\b[^.]*",
        r",?\s+prepared to\b[^.]*",
        r",\s*likely\b[^.]*",
        r",\s*suggesting\b[^.]*",
        r",?\s+indicating its readiness\b[^.]*",
        r",?\s+intending to\b[^.]*",
    )
    for pattern in patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ,;")
    if value and value[-1] not in ".!?":
        value = f"{value}."
    return value


def should_use_single_frame_evidence(triplet):
    question_type = str((triplet or {}).get("question_type") or "").strip()
    return question_type != "temporal_change"


def parse_evidence_frame_indices(value):
    if value is None:
        return []
    if isinstance(value, int):
        raw_values = [value]
    elif isinstance(value, str):
        raw_values = [part.strip() for part in value.replace("，", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        return []
    parsed = []
    for raw_value in raw_values:
        try:
            parsed.append(int(raw_value))
        except (TypeError, ValueError):
            continue
    return unique_preserve_order(index for index in parsed if index > 0)


def prune_unselected_description_refs(triplet, evidence_index):
    refs = triplet.get("evidence_refs") or {}
    text_refs = refs.get("text") or {}
    selected_sample_ids = set(value_list((refs.get("image") or {}), "sample_ids"))
    if not selected_sample_ids:
        return
    require_subset = should_use_single_frame_evidence(triplet)
    kept_description_ids = []
    for description_id in value_list(text_refs, "description_ids"):
        item = evidence_index.get(description_id, {}).get("item") or {}
        anchors = item.get("anchor_refs") or {}
        sample_ids = set(value_list(anchors, "sample_ids"))
        if not sample_ids:
            kept_description_ids.append(description_id)
            continue
        if require_subset and sample_ids <= selected_sample_ids:
            kept_description_ids.append(description_id)
        elif not require_subset and sample_ids & selected_sample_ids:
            kept_description_ids.append(description_id)
    text_refs["description_ids"] = unique_preserve_order(kept_description_ids)


def best_evidence_answer(evidence_refs, evidence_index):
    texts = resolve_triplet_evidence_texts({"evidence_refs": evidence_refs}, evidence_index)
    return texts[0] if texts else ""


def value_list(item, *keys):
    values = []
    for key in keys:
        value = (item or {}).get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(str(part).strip() for part in value if str(part).strip())
        elif str(value).strip():
            values.append(str(value).strip())
    return unique_preserve_order(values)


def collect_values(items, *keys):
    values = []
    for item in items or []:
        values.extend(value_list(item, *keys))
    return unique_preserve_order(values)


def item_ids(item):
    return value_list(item, "evidence_unit_id", "scaffold_id", "description_unit_id", "description_id", "text_signal_id", "signal_id", "temporal_unit_id", "temporal_id", "sample_id", "id")


def item_text(item):
    return " ".join(value_list(item, "evidence_summary", "scaffold_summary", "description_text", "text", "content", "summary", "temporal_label"))


def item_image(item):
    for key in ("frame_path", "image_path", "image"):
        value = (item or {}).get(key)
        if value:
            return value
    return None


def has_common(left, right):
    return bool(set(left or []) & set(right or []))


def get_time_range(item):
    raw_range = (item or {}).get("timestamp_range") or {}
    start = raw_range.get("start_sec", raw_range.get("start"))
    end = raw_range.get("end_sec", raw_range.get("end"))
    if start is None:
        start = (item or {}).get("start_sec", (item or {}).get("start"))
    if end is None:
        end = (item or {}).get("end_sec", (item or {}).get("end"))
    if start is None or end is None:
        return None
    start, end = float(start), float(end)
    if end < start:
        start, end = end, start
    return {"start_sec": start, "end_sec": end}


def ranges_overlap(left, right):
    return left["start_sec"] <= right["end_sec"] and right["start_sec"] <= left["end_sec"]


def evidence_ref_count(evidence_refs):
    return len(evidence_text_ref_ids(evidence_refs)) + len(evidence_image_frame_paths(evidence_refs)) + len(evidence_image_lookup_ids(evidence_refs)) + len(evidence_temporal_ref_ids(evidence_refs))


def evidence_refs_have_values(evidence_refs):
    return evidence_ref_count(evidence_refs) > 0


def evidence_text_ref_ids(evidence_refs):
    return value_list((evidence_refs or {}).get("text") or {}, "description_ids", "text_signal_ids", "evidence_unit_ids")


def evidence_image_lookup_ids(evidence_refs):
    return value_list((evidence_refs or {}).get("image") or {}, "frame_refs", "sample_ids")


def evidence_image_frame_paths(evidence_refs):
    return value_list((evidence_refs or {}).get("image") or {}, "frame_paths")


def evidence_temporal_ref_ids(evidence_refs):
    return value_list((evidence_refs or {}).get("temporal") or {}, "temporal_unit_ids")


def normalized_triplet_type(triplet_type):
    return "multihop" if str(triplet_type) == "multihop" else "single_hop"
