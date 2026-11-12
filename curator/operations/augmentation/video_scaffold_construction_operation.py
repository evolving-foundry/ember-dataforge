import math
from functools import lru_cache
from typing import Any, Dict, Optional

from .video_augmentation_operator import VideoAugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.video_ops_utils import (
    unique_preserve_order,
    build_scaffold_description_map,
    build_scaffold_text_map,
    build_temporal_sample_map,
    load_action_scene_pipeline,
    merge_visual_actions
)

SUPPORTED_SCAFFOLD_MODES = {
    "NARRATION_CHUNK",
    "EVENT_TREE",
    "ACTION_GRAPH",
}

ACTION_SCENE_MODEL_LIST = {
    "human_action_vit": "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224",
    "human_action_siglip": "prithivMLmods/Human-Action-Recognition",
    "human_pose_vit": "Harsha901/vit-human-pose-classification-model",
}
DEFAULT_ACTION_SCENE_MODEL_NAME = "human_action_vit"

class VideoScaffoldConstructionOperation(VideoAugmentationOperation):
    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Build video scaffold units from descriptions, temporal structure, and text signals",
        "params": {
            "description_units_key": {
                "type": "string",
                "required": False,
                "default": "description_units",
                "description": "Input description-unit field name"
            },
            "temporal_structures_key": {
                "type": "string",
                "required": False,
                "default": "temporal_structures",
                "description": "Input temporal-structure field name"
            },
            "text_signals_key": {
                "type": "string",
                "required": False,
                "default": "text_signals",
                "description": "Input text-signal field name"
            },
            "samples_key": {
                "type": "string",
                "required": False,
                "default": "samples",
                "description": "Input sampled-frame field name for ACTION_GRAPH mode"
            },
            "frame_path_key": {
                "type": "string",
                "required": False,
                "default": "frame_path",
                "description": "Sampled-frame path field name for ACTION_GRAPH mode"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "scaffold_units",
                "description": "Output scaffold field name"
            },
            "result_key": {
                "type": "string",
                "required": False,
                "default": "scaffold_result",
                "description": "Output result field name"
            },
            "scaffold_mode": {
                "type": "string",
                "required": False,
                "default": "NARRATION_CHUNK",
                "description": "Scaffold mode: NARRATION_CHUNK, EVENT_TREE, ACTION_GRAPH"
            },
            "scaffold_role": {
                "type": "string",
                "required": False,
                "default": "generation",
                "description": "Scaffold role"
            },
            "max_children_per_unit": {
                "type": "integer",
                "required": False,
                "default": 4,
                "description": "Maximum child references retained per scaffold unit"
            },
            "leaf_window_sec": {
                "type": "number",
                "required": False,
                "default": 10.0,
                "description": "Leaf window length for EVENT_TREE mode (seconds)"
            },
            "action_scene_model_name": {
                "type": "string",
                "required": False,
                "default": DEFAULT_ACTION_SCENE_MODEL_NAME,
                "description": "Frame action-recognition model name used by ACTION_GRAPH mode"
            },
            "action_scene_model_path": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Hugging Face model path used by ACTION_GRAPH mode; resolved from action_scene_model_name when empty"
            },
            "action_scene_top_k": {
                "type": "integer",
                "required": False,
                "default": 1,
                "description": "Number of action classification candidates retained per frame in ACTION_GRAPH mode"
            },
            "action_scene_device": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Inference device for ACTION_GRAPH models. When None, auto-detects NPU > CUDA > CPU. Explicit values such as npu, cuda, cuda:0, and cpu are supported"
            }
        },
        "example": {
            "description_units_key": "description_units",
            "temporal_structures_key": "temporal_structures",
            "text_signals_key": "text_signals",
            "samples_key": "samples",
            "scaffold_mode": "NARRATION_CHUNK"
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        scaffold_mode = str(self.params.get("scaffold_mode", "NARRATION_CHUNK")).upper()
        if scaffold_mode not in SUPPORTED_SCAFFOLD_MODES:
            raise ValueError(f"Unsupported scaffold mode: {scaffold_mode}")

        self.description_units_key = self.params.get("description_units_key", "description_units")
        self.temporal_structures_key = self.params.get("temporal_structures_key", "temporal_structures")
        self.text_signals_key = self.params.get("text_signals_key", "text_signals")
        self.samples_key = self.params.get("samples_key", "samples")
        self.frame_path_key = self.params.get("frame_path_key", "frame_path")
        self.output_key = self.params.get("output_key", "scaffold_units")
        self.result_key = self.params.get("result_key", "scaffold_result")
        self.scaffold_mode = scaffold_mode
        self.scaffold_role = self.params.get("scaffold_role", "generation")
        self.max_children_per_unit = int(self.params.get("max_children_per_unit", 4))
        self.leaf_window_sec = float(self.params.get("leaf_window_sec", 10.0))
        self.action_scene_model_name = self.params.get("action_scene_model_name", DEFAULT_ACTION_SCENE_MODEL_NAME)
        if self.action_scene_model_name not in ACTION_SCENE_MODEL_LIST:
            raise ValueError(f"Unsupported action scene model name: {self.action_scene_model_name}")
        self.action_scene_model_path = (
            self.params.get("action_scene_model_path")
            or ACTION_SCENE_MODEL_LIST[self.action_scene_model_name]
        )
        self.action_scene_top_k = int(self.params.get("action_scene_top_k", 1))
        self.action_scene_device = self.params.get("action_scene_device")

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        def construct_row(row):
            scaffold_result = self.construct(
                description_units=row.get(self.description_units_key) or [],
                temporal_structures=row.get(self.temporal_structures_key) or [],
                text_signals=row.get(self.text_signals_key) or [],
                samples=row.get(self.samples_key) or [],
            )
            row[self.output_key] = scaffold_result["scaffold_units"]
            row[self.result_key] = scaffold_result
            row["scaffold_mode"] = scaffold_result["scaffold_mode"]
            row["scaffold_metrics"] = scaffold_result["global_metrics"]
            return row

        return dataset.map(construct_row)

    def construct(self, description_units, temporal_structures, text_signals, samples=None):
        description_map = build_scaffold_description_map(description_units)
        text_map = build_scaffold_text_map(text_signals)
        sample_map = build_temporal_sample_map(samples or [])

        if self.scaffold_mode == "NARRATION_CHUNK":
            scaffold_units = build_narration_chunk_scaffolds(
                temporal_structures=temporal_structures,
                description_map=description_map,
                text_map=text_map,
                scaffold_role=self.scaffold_role,
                max_children_per_unit=self.max_children_per_unit,
                sample_map=sample_map,
            )
        elif self.scaffold_mode == "EVENT_TREE":
            if not temporal_structures:
                raise ValueError("EVENT_TREE requires 'temporal_structures'.")
            scaffold_units = [
                build_event_tree_scaffold(
                    temporal_structure=temporal_structure,
                    description_map=description_map,
                    text_map=text_map,
                    scaffold_role=self.scaffold_role,
                    max_children_per_unit=self.max_children_per_unit,
                    leaf_window_sec=self.leaf_window_sec,
                )
                for temporal_structure in temporal_structures
            ]
        else:
            if not temporal_structures:
                raise ValueError("ACTION_GRAPH requires 'temporal_structures'.")
            if not samples:
                raise ValueError("ACTION_GRAPH requires 'samples' with frame paths.")
            scaffold_units = [
                build_action_graph_scaffold(
                    temporal_structure=temporal_structure,
                    description_map=description_map,
                    text_map=text_map,
                    sample_map=sample_map,
                    scaffold_role=self.scaffold_role,
                    max_children_per_unit=self.max_children_per_unit,
                    frame_path_key=self.frame_path_key,
                    action_scene_model_path=self.action_scene_model_path,
                    action_scene_top_k=self.action_scene_top_k,
                    action_scene_device=self.action_scene_device,
                    action_scene_classifier=self._classify_action_scene_frames,
                )
                for temporal_structure in temporal_structures
            ]

        for index, scaffold_unit in enumerate(scaffold_units, start=1):
            scaffold_unit["id"] = f"scf_{index:06d}"
            scaffold_unit["scaffold_id"] = scaffold_unit["id"]

        return {
            "scaffold_mode": self.scaffold_mode,
            "global_metrics": {
                "total_description_units": len(description_units),
                "total_temporal_structures": len(temporal_structures),
                "total_text_signals": len(text_signals),
                "total_samples": len(samples or []),
                "total_scaffold_units": len(scaffold_units),
            },
            "scaffold_units": scaffold_units,
        }

    def _classify_action_scene_frames(self, samples, frame_path_key):
        return classify_action_scene_frames(
            samples=samples,
            frame_path_key=frame_path_key,
            model_path=self.action_scene_model_path,
            top_k=self.action_scene_top_k,
            device=self.action_scene_device,
        )

def build_action_graph_scaffold(
    temporal_structure,
    description_map,
    text_map,
    sample_map,
    scaffold_role,
    max_children_per_unit,
    frame_path_key="frame_path",
    action_scene_model_path=ACTION_SCENE_MODEL_LIST[DEFAULT_ACTION_SCENE_MODEL_NAME],
    action_scene_top_k=1,
    action_scene_device=None,
    action_scene_classifier=None,
):
    temporal_unit_id = str(temporal_structure["temporal_unit_id"])
    segment_ids = [str(segment_id) for segment_id in temporal_structure.get("segment_ids") or []]
    if not segment_ids and temporal_structure.get("segment_id"):
        segment_ids = [str(temporal_structure["segment_id"])]

    description_units = [
        unit
        for segment_id in segment_ids
        for unit in (description_map.get(segment_id) or {}).get("units") or []
    ]
    summary = str(temporal_structure.get("summary") or "").strip()
    if not summary:
        summary = " / ".join(
            unique_preserve_order(
                [str(unit.get("description_text") or "").strip() for unit in description_units if str(unit.get("description_text") or "").strip()]
            )[: int(max_children_per_unit)]
        )

    visual_samples = [
        sample
        for segment_id in segment_ids
        for sample in sample_map.get(segment_id) or []
    ]
    visual_samples = [sample for sample in visual_samples if sample.get(frame_path_key)]
    if not visual_samples:
        raise ValueError("ACTION_GRAPH requires sampled frames with frame paths.")

    classifier = action_scene_classifier or (
        lambda samples, frame_path_key: classify_action_scene_frames(
            samples=samples,
            frame_path_key=frame_path_key,
            model_path=action_scene_model_path,
            top_k=action_scene_top_k,
            device=action_scene_device,
        )
    )
    visual_actions = classifier(visual_samples, frame_path_key)
    action_groups = merge_visual_actions(visual_actions)[: int(max_children_per_unit)]
    if not action_groups:
        raise ValueError("ACTION_GRAPH action scene model returned no usable actions.")
    if not summary:
        summary = " / ".join(group["label"] for group in action_groups)

    action_nodes = []
    action_edges = []
    previous_action_ref = None
    for index, action_group in enumerate(action_groups, start=1):
        action_ref = f"{temporal_unit_id}_action_{index:02d}"
        action_nodes.append(
            {
                "node_ref": action_ref,
                "node_type": "action",
                "label": action_group["label"],
                "score": action_group["score"],
                "segment_ids": action_group["segment_ids"],
                "sample_ids": action_group["sample_ids"],
                "frame_paths": action_group["frame_paths"],
                "visual_evidence": action_group["visual_evidence"],
            }
        )
        for evidence_index, evidence in enumerate(action_group["visual_evidence"], start=1):
            frame_ref = f"{temporal_unit_id}_frame_{index:02d}_{evidence_index:02d}"
            action_nodes.append(
                {
                    "node_ref": frame_ref,
                    "node_type": "frame",
                    "label": str(evidence.get("sample_id") or evidence.get("frame_path") or frame_ref),
                    "segment_id": str(evidence.get("segment_id") or ""),
                    "sample_id": str(evidence.get("sample_id") or ""),
                    "frame_path": str(evidence.get("frame_path") or ""),
                    "frame_index": evidence.get("frame_index"),
                    "timestamp_sec": evidence.get("timestamp_sec"),
                }
            )
            action_edges.append(
                {
                    "source_ref": action_ref,
                    "target_ref": frame_ref,
                    "relation": "observed_in",
                }
            )
        if previous_action_ref:
            action_edges.append(
                {
                    "source_ref": previous_action_ref,
                    "target_ref": action_ref,
                    "relation": "next",
                }
            )
        previous_action_ref = action_ref

    return {
        "unit_type": "action_graph",
        "temporal_unit_id": temporal_unit_id,
        "scaffold_role": scaffold_role,
        "segment_ids": segment_ids,
        "scaffold_summary": summary,
        "action_nodes": action_nodes,
        "action_edges": action_edges,
        "action_scene_model": action_scene_model_path,
        "text_signals": {
            segment_id: [
                str(text_entry.get("text") or "").strip()
                for text_entry in (text_map.get(segment_id) or [])
                if str(text_entry.get("text") or "").strip()
            ][: int(max_children_per_unit)]
            for segment_id in segment_ids
        },
    }


def classify_action_scene_frames(samples, frame_path_key, model_path, top_k=1, device=None):
    if int(top_k) <= 0:
        raise ValueError("action_scene_top_k must be positive.")

    classifier = load_action_scene_pipeline(model_path, device)
    visual_actions = []
    for sample in samples:
        frame_path = str(sample.get(frame_path_key) or "").strip()
        if not frame_path:
            raise ValueError(f"Missing frame path key '{frame_path_key}' in ACTION_GRAPH sample.")
        predictions = classifier(frame_path, top_k=int(top_k))
        if not predictions:
            raise ValueError(f"Action scene model returned no predictions for frame: {frame_path}")
        top_prediction = predictions[0] if isinstance(predictions, list) else predictions
        label = str(top_prediction.get("label") or "").strip()
        if not label:
            raise ValueError(f"Action scene model returned an empty label for frame: {frame_path}")
        visual_actions.append(
            {
                "label": label,
                "score": float(top_prediction.get("score") or 0.0),
                "segment_id": str(sample.get("segment_id") or ""),
                "sample_id": str(sample.get("sample_id") or sample.get("id") or ""),
                "frame_path": frame_path,
                "frame_index": sample.get("frame_index"),
                "timestamp_sec": sample.get("timestamp_sec"),
            }
        )
    return visual_actions

def build_event_tree_scaffold(
    temporal_structure,
    description_map,
    text_map,
    scaffold_role,
    max_children_per_unit,
    leaf_window_sec,
):
    temporal_unit_id = str(temporal_structure["temporal_unit_id"])
    segment_ids = [str(segment_id) for segment_id in temporal_structure.get("segment_ids") or []]
    if not segment_ids and temporal_structure.get("segment_id"):
        segment_ids = [str(temporal_structure["segment_id"])]

    timestamp_range = temporal_structure.get("timestamp_range") or {}
    start_sec = float(timestamp_range.get("start_sec") or 0.0)
    end_sec = float(timestamp_range.get("end_sec") or start_sec)
    duration_sec = max(0.0, end_sec - start_sec)
    leaf_count = max(len(segment_ids), int(math.ceil(duration_sec / float(leaf_window_sec))) if duration_sec > 0 else 1)
    leaf_refs = [f"{temporal_unit_id}_leaf_{index:02d}" for index in range(1, leaf_count + 1)]
    root_ref = f"{temporal_unit_id}_root"

    summary = str(temporal_structure.get("summary") or "").strip()
    if not summary:
        summary = " / ".join(
            unique_preserve_order(
                [
                    text
                    for segment_id in segment_ids
                    for text in (description_map.get(segment_id) or [])
                    if text
                ]
            )[: int(max_children_per_unit)]
        )

    tree_nodes = [
        {
            "node_ref": root_ref,
            "node_type": "root",
            "summary": summary,
        }
    ]
    tree_edges = []
    for index, leaf_ref in enumerate(leaf_refs):
        segment_id = segment_ids[min(index, len(segment_ids) - 1)] if segment_ids else None
        tree_nodes.append(
            {
                "node_ref": leaf_ref,
                "node_type": "leaf",
                "segment_id": segment_id,
                "text_signals": [
                    str(text_entry.get("text") or "").strip()
                    for text_entry in (text_map.get(segment_id) or [])
                    if str(text_entry.get("text") or "").strip()
                ][: int(max_children_per_unit)] if segment_id else [],
            }
        )
        tree_edges.append(
            {
                "source_ref": root_ref,
                "target_ref": leaf_ref,
                "relation": "contains",
            }
        )

    return {
        "unit_type": "event_tree",
        "temporal_unit_id": temporal_unit_id,
        "scaffold_role": scaffold_role,
        "segment_ids": segment_ids,
        "scaffold_summary": summary,
        "root_ref": root_ref,
        "leaf_refs": leaf_refs,
        "tree_nodes": tree_nodes,
        "tree_edges": tree_edges,
        "composition_rule": "fixed_window_event_tree",
    }

def build_narration_chunk_scaffolds(
    temporal_structures,
    description_map,
    text_map,
    scaffold_role,
    max_children_per_unit,
    sample_map=None,
):
    scaffold_units = []
    for temporal_structure in temporal_structures or []:
        temporal_unit_id = str(temporal_structure["temporal_unit_id"])
        segment_ids = [str(segment_id) for segment_id in temporal_structure.get("segment_ids") or []]
        if not segment_ids and temporal_structure.get("segment_id"):
            segment_ids = [str(temporal_structure["segment_id"])]

        description_refs = []
        text_refs = []
        description_units = []
        text_summary_parts = []
        for segment_id in segment_ids:
            description_bucket = description_map.get(segment_id) or {}
            description_refs.extend(str(description_id) for description_id in description_bucket.get("ids") or [])
            description_units.extend(description_bucket.get("units") or [])
            text_entries = text_map.get(segment_id) or []
            text_refs.extend(
                str(text_entry["text_signal_id"])
                for text_entry in text_entries
                if text_entry.get("text_signal_id")
            )
            text_summary_parts.extend(
                str(text_entry.get("text") or "").strip()
                for text_entry in text_entries
                if str(text_entry.get("text") or "").strip()
            )
        summary_parts = select_narration_scaffold_summary_parts(
            description_units=description_units,
            text_summary_parts=text_summary_parts,
            temporal_summary=str(temporal_structure.get("summary") or "").strip(),
            max_parts=int(max_children_per_unit),
        )
        visual_evidence = [
            {
                "sample_id": str(sample.get("sample_id") or sample.get("id") or ""),
                "segment_id": str(sample.get("segment_id") or ""),
                "frame_path": str(sample.get("frame_path") or ""),
                "frame_index": sample.get("frame_index"),
                "timestamp_sec": sample.get("timestamp_sec"),
            }
            for segment_id in segment_ids
            for sample in (sample_map or {}).get(segment_id, [])
            if sample.get("frame_path")
        ][: int(max_children_per_unit)]

        scaffold_units.append(
            {
                "unit_type": "narration_chunk",
                "temporal_unit_id": temporal_unit_id,
                "scaffold_role": scaffold_role,
                "segment_ids": segment_ids,
                "source_description_ids": unique_preserve_order(description_refs)[: int(max_children_per_unit)],
                "source_text_signal_ids": unique_preserve_order(text_refs)[: int(max_children_per_unit)],
                "visual_evidence": visual_evidence,
                "scaffold_summary": " / ".join(summary_parts),
            }
        )
    return scaffold_units


def select_narration_scaffold_summary_parts(
    description_units,
    text_summary_parts,
    temporal_summary,
    max_parts,
):
    """Choose one summary level for QAE scaffolds: event > segment > frame."""
    max_parts = max(1, int(max_parts))
    priority_granularities = ("event", "segment", "frame")
    for granularity in priority_granularities:
        selected = [
            str(unit.get("description_text") or "").strip()
            for unit in description_units or []
            if description_unit_granularity(unit) == granularity
            and str(unit.get("description_text") or "").strip()
        ]
        selected = unique_preserve_order(selected)
        if selected:
            return selected[:max_parts]

    fallback_parts = unique_preserve_order(
        [
            str(part).strip()
            for part in (text_summary_parts or [])
            if str(part).strip()
        ]
    )
    if fallback_parts:
        return fallback_parts[:max_parts]
    if temporal_summary:
        return [temporal_summary]
    return []


def description_unit_granularity(unit):
    granularity = str(unit.get("granularity") or "").strip().lower()
    if granularity:
        return granularity
    mode = str(unit.get("description_mode") or "").strip().upper()
    if mode == "EVENT_DESCRIPTION":
        return "event"
    if mode == "SEGMENT_DESCRIPTION":
        return "segment"
    if mode == "FRAME_DESCRIPTION":
        return "frame"
    if mode == "CONTEXT_DESCRIPTION":
        return "context"
    return ""

def build_qae_materialization_evidence_map(evidence_units):
    return {
        str(evidence_unit["evidence_unit_id"]): evidence_unit
        for evidence_unit in evidence_units or []
    }
