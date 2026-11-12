from __future__ import annotations

from data_designer.contracts.algorithms import AlgorithmResult
from data_designer.contracts.datasets import RecordEnvelope
from .descriptors import TRAJECTORY_CLEAN, TRAJECTORY_LESSON


class TrajectoryCleanAlgorithm:
    def describe(self):
        return TRAJECTORY_CLEAN

    def run(self, request, context):
        from .trajectory_data import TrajectoryCorpus
        from .trajectory_trajectory_loading import TrajectoryNormalizationOperation
        from .trajectory_cleaning import ReasoningStripOperation, ToolReceiptTruncationOperation, PIIScrubbingOperation
        from .trajectory_degenerate import DegenerateTrajectoryDetectionOperation
        record = request.records[0]
        corpus = TrajectoryCorpus(records=[record.payload])
        TrajectoryNormalizationOperation({"require_core_meta": request.config["require_core_meta"]}).execute(corpus)
        ReasoningStripOperation().execute(corpus)
        ToolReceiptTruncationOperation({"max_chars": request.config["max_tool_chars"]}).execute(corpus)
        PIIScrubbingOperation().execute(corpus)
        DegenerateTrajectoryDetectionOperation().execute(corpus)
        row = corpus.records[0].model_dump(mode="json")
        return AlgorithmResult((RecordEnvelope("clean:" + record.id, row, TRAJECTORY_CLEAN.output_schema,
                                              (record.id,), {"omissions": corpus.omissions.to_report(),
                                                             "degenerate_flags": corpus.degenerate_flags}),))


class TrajectoryLessonAlgorithm:
    def describe(self):
        return TRAJECTORY_LESSON

    def run(self, request, context):
        record = request.records[0]
        row = record.payload
        if row.get("outcome", {}).get("status") != "failure":
            return AlgorithmResult((), {"not_failed_trajectory": 1})
        errors = [step for step in row.get("steps", []) if step.get("error") and str(step.get("observation", "")).strip()]
        if not errors:
            return AlgorithmResult((), {"no_error_evidence": 1})
        step = errors[0]
        quote = str(step["observation"]).strip()[:request.config["max_quote_chars"]]
        output = {"statement": f"Before repeating {step.get('action', 'the action')}, verify the preconditions implicated by: {quote}",
                  "task": row.get("task", ""),
                  "evidence": {"trajectory_id": row.get("trajectory_id", record.id), "step_index": step["index"], "quote": quote}}
        return AlgorithmResult((RecordEnvelope("lesson:" + record.id, output, TRAJECTORY_LESSON.output_schema, (record.id,)),))
