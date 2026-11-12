from data_designer.contracts.algorithms import AlgorithmResult
from data_designer.contracts.datasets import RecordEnvelope
from data_designer.contracts.provenance import content_digest
from .descriptors import AGENT_TASK


class AgentTaskAlgorithm:
    def describe(self):
        return AGENT_TASK

    def run(self, request, context):
        record = request.records[0]
        row = record.payload
        task_text = str(row.get("task_text") or "").strip()
        if not task_text:
            return AlgorithmResult((), {"empty_task_text": 1})
        item = {
            "seed_id": str(row.get("seed_id") or record.id),
            "task_text": task_text,
            "tools_available": list(row.get("tools_available") or []),
            "success_criteria": str(row.get("success_criteria") or "").strip(),
            "ground_truth_action_sequence": list(row.get("ground_truth_action_sequence") or []),
            "scoring_rubric": str(row.get("scoring_rubric") or "").strip(),
            "difficulty": str(row.get("difficulty") or "unspecified").strip(),
        }
        item["item_digest"] = content_digest(item)
        item["task_id"] = "task:" + record.id
        return AlgorithmResult((RecordEnvelope("task:" + record.id, item,
                                              AGENT_TASK.output_schema, (record.id,)),))
