from typing import Any, Dict, Optional

from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType

SUPPORTED_VIDEO_ORCHESTRATION_MODES = {
    "MERGE_STITCH",
    "WINDOW_ATTACH",
    "FILTER_HINT",
}

class VideoContextOrchestrationOperation(FilteringOperation):
    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Apply context orchestration to existing video segment results",
        "params": {
            "segments_key": {
                "type": "string",
                "required": False,
                "default": "segments",
                "description": "Input segment-result field name"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "segments",
                "description": "Output segment-result field name"
            },
            "result_key": {
                "type": "string",
                "required": False,
                "default": "orchestration_result",
                "description": "Orchestration-result field name"
            },
            "orchestration_mode": {
                "type": "string",
                "required": False,
                "default": "MERGE_STITCH",
                "description": "Orchestration mode: MERGE_STITCH, WINDOW_ATTACH, FILTER_HINT"
            },
            "merge_gap_sec": {
                "type": "number",
                "required": False,
                "default": 0.5,
                "description": "Maximum gap in seconds for MERGE_STITCH mode"
            },
            "max_merge_span_sec": {
                "type": "number",
                "required": False,
                "default": 8.0,
                "description": "Maximum merge span in seconds for MERGE_STITCH mode"
            },
            "context_window_sec": {
                "type": "number",
                "required": False,
                "default": 2.0,
                "description": "Context window in seconds for WINDOW_ATTACH mode"
            },
            "max_context_segments": {
                "type": "integer",
                "required": False,
                "default": 4,
                "description": "Maximum context segments for WINDOW_ATTACH mode"
            },
            "min_valid_duration_sec": {
                "type": "number",
                "required": False,
                "default": 1.0,
                "description": "Minimum valid duration in seconds for FILTER_HINT mode"
            }
        },
        "example": {
            "segments_key": "segments",
            "orchestration_mode": "MERGE_STITCH",
            "merge_gap_sec": 0.5,
            "max_merge_span_sec": 8.0
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        orchestration_mode = str(self.params.get("orchestration_mode", "MERGE_STITCH")).upper()
        if orchestration_mode not in SUPPORTED_VIDEO_ORCHESTRATION_MODES:
            raise ValueError(f"Unsupported orchestration mode: {orchestration_mode}")

        self.segments_key = self.params.get("segments_key", "segments")
        self.output_key = self.params.get("output_key", "segments")
        self.result_key = self.params.get("result_key", "orchestration_result")
        self.orchestration_mode = orchestration_mode
        self.merge_gap_sec = float(self.params.get("merge_gap_sec", 0.5))
        self.max_merge_span_sec = float(self.params.get("max_merge_span_sec", 8.0))
        self.context_window_sec = float(self.params.get("context_window_sec", 2.0))
        self.max_context_segments = int(self.params.get("max_context_segments", 4))
        self.min_valid_duration_sec = float(self.params.get("min_valid_duration_sec", 1.0))

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        def orchestrate_row(row):
            segments = row.get(self.segments_key)
            if not segments:
                raise ValueError("Missing segments.")

            orchestration_result = self.orchestrate(segments)
            row[self.output_key] = orchestration_result["segments"]
            row[self.result_key] = orchestration_result
            row["orchestration_mode"] = orchestration_result["orchestration_mode"]
            row["orchestration_metrics"] = orchestration_result["global_metrics"]
            return row

        return dataset.map(orchestrate_row)

    def orchestrate(self, segments):
        if self.orchestration_mode == "MERGE_STITCH":
            orchestrated_segments = merge_stitch_segments(
                segments=segments,
                merge_gap_sec=self.merge_gap_sec,
                max_merge_span_sec=self.max_merge_span_sec,
                min_valid_duration_sec=self.min_valid_duration_sec,
            )
        elif self.orchestration_mode == "WINDOW_ATTACH":
            orchestrated_segments = attach_context_windows(
                segments=segments,
                context_window_sec=self.context_window_sec,
                max_context_segments=self.max_context_segments,
            )
        else:
            orchestrated_segments = build_filter_hints(
                segments=segments,
                min_valid_duration_sec=self.min_valid_duration_sec,
                merge_gap_sec=self.merge_gap_sec,
            )

        return {
            "orchestration_mode": self.orchestration_mode,
            "global_metrics": {
                "total_segments": len(orchestrated_segments),
            },
            "segments": orchestrated_segments,
        }


def attach_context_windows(segments, context_window_sec, max_context_segments):
    output_segments = []

    for anchor_index, segment in enumerate(segments):
        anchor_start_sec = float(segment["timestamp_range"]["start_sec"])
        anchor_end_sec = float(segment["timestamp_range"]["end_sec"])
        lower_bound = anchor_start_sec - float(context_window_sec)
        upper_bound = anchor_end_sec + float(context_window_sec)

        context_window_refs = []
        for candidate_index, candidate in enumerate(segments):
            if candidate_index == anchor_index:
                continue
            if (
                float(candidate["timestamp_range"]["end_sec"]) >= lower_bound
                and float(candidate["timestamp_range"]["start_sec"]) <= upper_bound
            ):
                context_window_refs.append(str(candidate["segment_id"]))

        output_segments.append(
            {
                **segment,
                "stitch_group_id": None,
                "context_window_refs": context_window_refs[: int(max_context_segments)],
                "cleanup_flags": [],
            }
        )

    return output_segments

# Cleaning operator
def build_filter_hints(segments, min_valid_duration_sec, merge_gap_sec):
    output_segments = []

    for index, segment in enumerate(segments):
        cleanup_flags = []
        if float(segment["duration_sec"]) < float(min_valid_duration_sec):
            cleanup_flags.append("too_short")

        text = str(segment.get("text") or "")
        if not text.strip():
            cleanup_flags.append("blank_text")

        if index > 0:
            previous_segment = segments[index - 1]
            previous_text = str(previous_segment.get("text") or "")
            gap_sec = (
                float(segment["timestamp_range"]["start_sec"])
                - float(previous_segment["timestamp_range"]["end_sec"])
            )
            if (
                text.strip()
                and text.strip().lower() == previous_text.strip().lower()
                and gap_sec <= float(merge_gap_sec)
            ):
                cleanup_flags.append("near_duplicate")

        output_segments.append(
            {
                **segment,
                "stitch_group_id": None,
                "context_window_refs": [],
                "cleanup_flags": cleanup_flags,
            }
        )

    return output_segments

def merge_stitch_segments(
    segments,
    merge_gap_sec,
    max_merge_span_sec,
    min_valid_duration_sec,
):
    groups = []
    current_group = []

    for segment in segments:
        if not current_group:
            current_group = [segment]
            continue

        previous_segment = current_group[-1]
        gap_sec = (
            float(segment["timestamp_range"]["start_sec"])
            - float(previous_segment["timestamp_range"]["end_sec"])
        )
        group_start_sec = float(current_group[0]["timestamp_range"]["start_sec"])
        group_end_sec = float(segment["timestamp_range"]["end_sec"])
        merged_span_sec = group_end_sec - group_start_sec

        if (
            gap_sec <= float(merge_gap_sec)
            and merged_span_sec <= float(max_merge_span_sec)
            and (
                float(previous_segment["duration_sec"]) < float(min_valid_duration_sec)
                or float(segment["duration_sec"]) < float(min_valid_duration_sec)
            )
        ):
            current_group.append(segment)
        else:
            groups.append(current_group)
            current_group = [segment]

    if current_group:
        groups.append(current_group)

    output_segments = []
    for group_index, group in enumerate(groups, start=1):
        stitch_group_id = f"stitch_{group_index:04d}"
        for segment in group:
            output_segments.append(
                {
                    **segment,
                    "stitch_group_id": stitch_group_id,
                    "context_window_refs": [],
                    "cleanup_flags": [],
                }
            )

    return output_segments
