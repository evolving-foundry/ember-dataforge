from typing import Any, Dict, Optional

from .question_planning import (
    build_evidence_refs_by_modality,
    build_rule_question_from_scaffold,
)
from .video_augmentation_operator import VideoAugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.video_ops_utils import (
    evidence_refs_have_values,
)

SUPPORTED_REVISION_MODES = {
    "FIELD_ALIGN",
    "EVIDENCE_PATCH",
    "QUESTION_REWRITE",
    "DRAFT_RECONCILE",
}

class VideoConsistencyRevisionOperation(VideoAugmentationOperation):
    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Revise QAE triples for consistency using the data structure and question-answer drafts",
        "params": {
            "qae_triplets_key": {
                "type": "string",
                "required": False,
                "default": "qae_triplets",
                "description": "Input QAE triple field name"
            },
            "scaffold_units_key": {
                "type": "string",
                "required": False,
                "default": "scaffold_units",
                "description": "Input scaffold field name"
            },
            "qa_drafts_key": {
                "type": "string",
                "required": False,
                "default": "qa_drafts",
                "description": "Input question-answer draft field name"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "revisions",
                "description": "Output revision field name"
            },
            "result_key": {
                "type": "string",
                "required": False,
                "default": "revision_result",
                "description": "Output result field name"
            },
            "revision_mode": {
                "type": "string",
                "required": False,
                "default": "FIELD_ALIGN",
                "description": "Revision mode: FIELD_ALIGN, EVIDENCE_PATCH, QUESTION_REWRITE, DRAFT_RECONCILE"
            },
            "max_revision_rounds": {
                "type": "integer",
                "required": False,
                "default": 1,
                "description": "Maximum triplet scan multiplier per round"
            }
        },
        "example": {
            "qae_triplets_key": "qae_triplets",
            "scaffold_units_key": "scaffold_units",
            "revision_mode": "FIELD_ALIGN",
            "max_revision_rounds": 1
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        revision_mode = str(self.params.get("revision_mode", "FIELD_ALIGN")).upper()
        if revision_mode not in SUPPORTED_REVISION_MODES:
            raise ValueError(f"Unsupported revision mode: {revision_mode}")

        self.qae_triplets_key = self.params.get("qae_triplets_key", "qae_triplets")
        self.scaffold_units_key = self.params.get("scaffold_units_key", "scaffold_units")
        self.qa_drafts_key = self.params.get("qa_drafts_key", "qa_drafts")
        self.output_key = self.params.get("output_key", "revisions")
        self.result_key = self.params.get("result_key", "revision_result")
        self.revision_mode = revision_mode
        self.max_revision_rounds = int(self.params.get("max_revision_rounds", 1))

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        def revise_row(row):
            qae_triplets = row.get(self.qae_triplets_key)
            if not qae_triplets:
                raise ValueError("Missing qae_triplets.")

            revision_result = self.revise(
                qae_triplets=qae_triplets,
                scaffold_units=row.get(self.scaffold_units_key) or [],
                qa_drafts=row.get(self.qa_drafts_key) or [],
            )
            row[self.output_key] = revision_result["revisions"]
            row[self.result_key] = revision_result
            row["revision_mode"] = revision_result["revision_mode"]
            row["revision_metrics"] = revision_result["global_metrics"]
            return row

        return dataset.map(revise_row)

    def revise(self, qae_triplets, scaffold_units=None, qa_drafts=None):
        revisions = build_revision_records(
            revision_mode=self.revision_mode,
            qae_triplets=qae_triplets,
            scaffold_units=scaffold_units,
            qa_drafts=qa_drafts,
            max_revision_rounds=self.max_revision_rounds,
        )
        return {
            "revision_mode": self.revision_mode,
            "global_metrics": {
                "total_triplets": len(qae_triplets),
                "total_revisions": len(revisions),
            },
            "revisions": revisions,
        }

def build_revision_records(revision_mode, qae_triplets, scaffold_units=None, qa_drafts=None, max_revision_rounds=1):
    scaffold_map = {str(unit["scaffold_id"]): unit for unit in scaffold_units or []}
    qa_draft_map = {str(draft["qa_draft_id"]): draft for draft in qa_drafts or []}
    revisions = []
    limit = int(max_revision_rounds) * max(1, len(qae_triplets or []))

    for triplet in (qae_triplets or [])[:limit]:
        scaffold = scaffold_map.get(str(triplet["scaffold_id"]))
        after_fields = {}
        reason = ""

        if revision_mode == "FIELD_ALIGN":
            question = str(triplet.get("question") or "").strip()
            answer = str(triplet.get("answer") or "").strip()
            scaffold_summary = str((scaffold or {}).get("scaffold_summary") or "").strip()
            if scaffold_summary and answer == question and answer != scaffold_summary:
                after_fields["answer"] = scaffold_summary
                reason = "fix misplaced answer field"
        elif revision_mode == "EVIDENCE_PATCH":
            if scaffold is None:
                continue
            scaffold_evidence_refs = build_evidence_refs_by_modality(scaffold)
            if evidence_refs_have_values(scaffold_evidence_refs) and triplet.get("evidence_refs") != scaffold_evidence_refs:
                after_fields["evidence_refs"] = scaffold_evidence_refs
                reason = "fill or replace evidence refs from scaffold sources"
        elif revision_mode == "DRAFT_RECONCILE":
            source_qa_draft_id = str(triplet.get("source_qa_draft_id") or "")
            if source_qa_draft_id:
                continue
            if scaffold is None or not scaffold.get("source_qa_draft_ids"):
                continue
            fallback_draft_id = str(scaffold["source_qa_draft_ids"][0])
            fallback_draft = qa_draft_map.get(fallback_draft_id)
            if fallback_draft is None:
                continue
            after_fields["source_qa_draft_id"] = fallback_draft_id
            reason = "restore qa draft lineage from scaffold"
        else:
            question = str(triplet.get("question") or "").strip()
            if len(question) < 3 and scaffold is not None:
                after_fields["question"] = build_rule_question_from_scaffold(scaffold)
                reason = "rewrite underspecified question"

        if not after_fields:
            continue

        revision_id = f"rev_{len(revisions) + 1:06d}"
        revisions.append(
            {
                "id": revision_id,
                "revision_id": revision_id,
                "target_triplet_id": str(triplet["triplet_id"]),
                "revision_action": revision_mode,
                "before_fields": {key: triplet.get(key, "") for key in after_fields},
                "after_fields": after_fields,
                "reason": reason,
            }
        )

    return revisions
