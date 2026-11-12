from typing import Any, Dict, Optional

from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.video_ops_utils import evidence_ref_count, evidence_refs_have_values

SUPPORTED_QUALITY_MODES = {
    "RULE_FILTER",
    "TIER_ASSIGN",
}


class VideoQualityScreeningOperation(FilteringOperation):
    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Screen video QAE samples using revision results and scoring rules",
        "params": {
            "qae_triplets_key": {
                "type": "string",
                "required": False,
                "default": "qae_triplets",
                "description": "Input QAE triple field name"
            },
            "revisions_key": {
                "type": "string",
                "required": False,
                "default": "revisions",
                "description": "Input revision field name"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "quality_records",
                "description": "Output quality-record field name"
            },
            "result_key": {
                "type": "string",
                "required": False,
                "default": "quality_result",
                "description": "Output result field name"
            },
            "quality_mode": {
                "type": "string",
                "required": False,
                "default": "RULE_FILTER",
                "description": "Quality mode: RULE_FILTER, TIER_ASSIGN"
            },
            "required_dataset_tier": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Required dataset tier"
            }
        },
        "example": {
            "qae_triplets_key": "qae_triplets",
            "revisions_key": "revisions",
            "quality_mode": "RULE_FILTER"
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        quality_mode = str(self.params.get("quality_mode", "RULE_FILTER")).upper()
        if quality_mode not in SUPPORTED_QUALITY_MODES:
            raise ValueError(f"Unsupported quality mode: {quality_mode}")

        self.qae_triplets_key = self.params.get("qae_triplets_key", "qae_triplets")
        self.revisions_key = self.params.get("revisions_key", "revisions")
        self.output_key = self.params.get("output_key", "quality_records")
        self.result_key = self.params.get("result_key", "quality_result")
        self.quality_mode = quality_mode
        self.required_dataset_tier = self.params.get("required_dataset_tier")

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        def screen_row(row):
            qae_triplets = row.get(self.qae_triplets_key)
            if not qae_triplets:
                raise ValueError("Missing qae_triplets.")

            quality_result = self.screen(
                qae_triplets=qae_triplets,
                revisions=row.get(self.revisions_key) or [],
            )
            row[self.output_key] = quality_result["quality_records"]
            row[self.result_key] = quality_result
            row["quality_mode"] = quality_result["quality_mode"]
            row["quality_metrics"] = quality_result["global_metrics"]
            return row

        return dataset.map(screen_row)

    def screen(self, qae_triplets, revisions=None):
        quality_records = build_quality_records(
            quality_mode=self.quality_mode,
            qae_triplets=qae_triplets,
            revisions=revisions or [],
            required_dataset_tier=self.required_dataset_tier,
        )
        return {
            "quality_mode": self.quality_mode,
            "global_metrics": {
                "total_triplets": len(qae_triplets),
                "total_quality_records": len(quality_records),
            },
            "quality_records": quality_records,
        }

def build_quality_records(
    quality_mode,
    qae_triplets,
    revisions=None,
    required_dataset_tier=None,
):
    revision_map = {}
    for revision in revisions or []:
        target_triplet_id = str(revision["target_triplet_id"])
        revision_map.setdefault(target_triplet_id, []).append(revision)

    quality_records = []
    for qae_triplet in qae_triplets or []:
        final_triplet = dict(qae_triplet)
        for revision in revision_map.get(str(qae_triplet["triplet_id"])) or []:
            final_triplet.update(revision.get("after_fields") or {})

        evidence_refs = final_triplet.get("evidence_refs")
        evidence_score = evidence_ref_count(evidence_refs)
        completeness_score = 1.0 if final_triplet.get("question") and final_triplet.get("answer") else 0.0
        grounding_level = str(final_triplet.get("grounding_level") or "")

        assigned_dataset_tier = "qa"
        if evidence_score > 1.0 or grounding_level == "strong_grounded":
            assigned_dataset_tier = "strong_grounded_qa"
        elif evidence_score == 1.0:
            assigned_dataset_tier = "grounded_qa"

        decision = "keep"
        reason = "passed quality screening"
        if quality_mode == "RULE_FILTER":
            if completeness_score == 0.0:
                decision = "reject"
                reason = "missing question or answer"
            elif not evidence_refs_have_values(evidence_refs):
                decision = "reject"
                reason = "missing evidence refs"
        elif quality_mode == "TIER_ASSIGN":
            reason = "assigned dataset tier by grounding and evidence"

        if required_dataset_tier and assigned_dataset_tier != required_dataset_tier:
            decision = "reject"
            reason = "dataset tier does not match requirement"

        quality_records.append(
            {
                "target_triplet_id": str(qae_triplet["triplet_id"]),
                "question": str(final_triplet.get("question") or ""),
                "quality_mode": quality_mode,
                "scores": {
                    "evidence_score": evidence_score,
                    "completeness_score": completeness_score
                },
                "decision": "drop" if decision == "reject" else decision,
                "reason": reason,
                "rank_in_video": None,
                "assigned_dataset_tier": assigned_dataset_tier,
            }
        )

    for index, quality_record in enumerate(quality_records, start=1):
        quality_record["rank_in_video"] = index

    for index, quality_record in enumerate(quality_records, start=1):
        quality_id = f"qlt_{index:06d}"
        quality_record["id"] = quality_id
        quality_record["quality_id"] = quality_id

    return quality_records
