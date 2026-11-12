"""
Pipeline plugin interface definitions and reference implementations.

All extension capabilities (Tracking, Checkpoint, Quality, Experiment, AdaptiveScheduling)
implement the PipelinePlugin interface and are coordinated by MultimodalPipeline.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from base.dataset import MultimodalDataset
    from .operation import Operation
    from .pipeline import MultimodalPipeline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Plugin interface
# ─────────────────────────────────────────────────────────────────────────────

class PipelinePlugin(ABC):
    """Pipeline lifecycle plugin interface."""

    # Must be set by subclasses
    name: str = "base_plugin"

    def on_pipeline_init(self, pipeline: "MultimodalPipeline") -> None:
        """Called after Pipeline.__init__ completes."""
        pass

    def before_operation(
        self,
        pipeline: "MultimodalPipeline",
        operation: "Operation",
        op_index: int,
        input_ds: "MultimodalDataset",
    ) -> None:
        """Called before a single operation executes."""
        pass

    def after_operation(
        self,
        pipeline: "MultimodalPipeline",
        operation: "Operation",
        op_index: int,
        input_ds: "MultimodalDataset",
        output_ds: "MultimodalDataset",
        duration_sec: float,
    ) -> None:
        """Called after a single operation executes."""
        pass

    def on_pipeline_complete(
        self, pipeline: "MultimodalPipeline", final_dataset: "MultimodalDataset"
    ) -> None:
        """Called when the pipeline completes successfully."""
        pass

    def on_pipeline_failure(
        self,
        pipeline: "MultimodalPipeline",
        operation: "Operation",
        op_index: int,
        exception: Exception,
    ) -> None:
        """Called when pipeline execution fails."""
        pass

    def before_pipeline(self, pipeline: "MultimodalPipeline") -> None:
        """Called before Pipeline.run() starts."""
        pass

    def after_dag_node(
        self,
        pipeline: "MultimodalPipeline",
        node_id: str,
        output: "MultimodalDataset",
    ) -> None:
        """Called after a DAG node finishes executing."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. TrackingPlugin — wraps DataTracker + TrackingManager
# ─────────────────────────────────────────────────────────────────────────────

class TrackingPlugin(PipelinePlugin):
    """Data tracking plugin."""

    name = "data_tracking"

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._tracker = None
        self._manager = None

    # ── PipelinePlugin interface ──────────────────────────────────────────────

    def on_pipeline_init(self, pipeline: "MultimodalPipeline") -> None:
        if not self._enabled:
            self._manager = _NullTrackingManager()
            return

        from ..tracking.data_tracker import DataTracker
        from ..tracking.tracking_manager import TrackingManager

        # Read the force_full_snapshot setting from GlobalConfig
        force_full_snapshot = False
        if pipeline.config_model and pipeline.config_model.global_config:
            force_full_snapshot = pipeline.config_model.global_config.force_full_snapshot

        self._tracker = DataTracker(pipeline.pipeline_id, pipeline.session_id, force_full_snapshot)
        self._manager = TrackingManager(self._tracker)
        logger.info("TrackingPlugin initialized (enabled=True)")

    def after_operation(
        self,
        pipeline: "MultimodalPipeline",
        operation: "Operation",
        op_index: int,
        input_ds: "MultimodalDataset",
        output_ds: "MultimodalDataset",
        duration_sec: float,
    ) -> None:
        self._manager.track_operation(
            operation=operation,
            input_dataset=input_ds,
            output_dataset=output_ds,
        )

    def get_tracker(self):
        """Expose the tracker reference so Pipeline can retrieve tracking reports."""
        return self._tracker

    def get_manager(self):
        return self._manager


class _NullTrackingManager:
    """No-op implementation used when tracking is disabled."""

    def track_operation(self, *args, **kwargs) -> None:
        pass

    def get_tracking_report(self):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. CheckpointPlugin — wraps CheckpointManagerWrapper
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointPlugin(PipelinePlugin):
    """Checkpoint plugin."""

    name = "checkpoint"

    def __init__(self, checkpoint_config=None, pipeline_id: str = None, session_id: str = None):
        self._checkpoint_config = checkpoint_config
        self._pipeline_id = pipeline_id
        self._session_id = session_id
        self._wrapper = None

    def on_pipeline_init(self, pipeline: "MultimodalPipeline") -> None:
        from ..checkpoint.checkpoint_manager import CheckpointConfig
        from ..checkpoint.checkpoint_manager_wrapper import CheckpointManagerWrapper

        cfg = self._checkpoint_config or CheckpointConfig()
        self._wrapper = CheckpointManagerWrapper(
            cfg, pipeline.pipeline_id, pipeline.session_id
        )
        logger.info("CheckpointPlugin initialized")

    def after_operation(
        self,
        pipeline: "MultimodalPipeline",
        operation: "Operation",
        op_index: int,
        input_ds: "MultimodalDataset",
        output_ds: "MultimodalDataset",
        duration_sec: float,
    ) -> None:
        if self._wrapper is None:
            return
        # Skip Checkpointing for IO operators (side effects are not idempotent)
        if getattr(operation, "IS_IO_OPERATION", False):
            return
        completed = list(range(op_index + 1))
        remaining = list(range(op_index + 1, len(pipeline.operations)))
        self._wrapper.save_checkpoint_if_needed(
            operations=pipeline.operations,
            current_operation_idx=op_index,
            processed_dataset=output_ds,
            completed_ops_indices=completed,
            remaining_ops_indices=remaining,
        )

    def get_wrapper(self):
        return self._wrapper


# ─────────────────────────────────────────────────────────────────────────────
# 3. QualityPlugin — wraps DataQualityAssessor
# ─────────────────────────────────────────────────────────────────────────────

class QualityPlugin(PipelinePlugin):
    """Data quality assessment plugin.

    I4 (FR-I): assessment results must be persisted and enter the event stream —
    this plugin retains every assessment report (get_reports()) and, when an EventChannel
    is provided, emits quality.assessed / quality.degraded events (RSI event protocol).
    """

    name = "quality_assessment"

    def __init__(self, quality_config=None, event_channel=None, degraded_delta: float = 0.05):
        self._quality_config = quality_config
        self._assessor = None
        self._reports = []  # type: ignore[var-annotated]
        self.events = event_channel  # rsi.protocol.events.EventChannel; may be None
        self._degraded_delta = float(degraded_delta)
        self._last_score = None

    def on_pipeline_init(self, pipeline: "MultimodalPipeline") -> None:
        if self._quality_config is None:
            return
        try:
            from ..quality.quality_assessor import DataQualityAssessor
            self._assessor = DataQualityAssessor(self._quality_config)
            logger.info("QualityPlugin initialized")
        except Exception as e:
            logger.warning("QualityPlugin init failed: %s", e)

    def after_operation(
        self,
        pipeline: "MultimodalPipeline",
        operation: "Operation",
        op_index: int,
        input_ds: "MultimodalDataset",
        output_ds: "MultimodalDataset",
        duration_sec: float,
    ) -> None:
        if self._assessor is None:
            return
        try:
            report = self._assessor.assess(output_ds, operation.name)
            score = float(report.overall_score)
            self._reports.append(
                {
                    "operation": operation.name,
                    "op_index": op_index,
                    "overall_score": score,
                    "issues": list(report.issues),
                    "assessed_at": report.assessed_at.isoformat(),
                }
            )
            if self.events is not None:
                self.events.emit_simple(
                    "quality.assessed",
                    operation=operation.name,
                    overall_score=score,
                )
                if self._last_score is not None and score < self._last_score - self._degraded_delta:
                    self.events.emit_simple(
                        "quality.degraded",
                        operation=operation.name,
                        previous_score=self._last_score,
                        current_score=score,
                    )
                self._last_score = score
        except Exception as e:
            logger.warning("Quality assessment failed for %s: %s", operation.name, e)

    def get_reports(self):
        """I4: assessment results are readable (previously they were discarded directly)."""
        return list(self._reports)

    def get_assessor(self):
        return self._assessor


# ─────────────────────────────────────────────────────────────────────────────
# 3b. OpsQualityPlugin — wraps OpsQualityAssessor
# ─────────────────────────────────────────────────────────────────────────────

class OpsQualityPlugin(PipelinePlugin):
    """Quality assessment plugin for multimodal operator artifacts.

    The plugin only records assessment results and does not modify the downstream pipeline data flow.
    """

    name = "ops_quality_assessment"

    def __init__(
        self,
        quality_config=None,
        assess_after_operations: bool = True,
    ):
        self._quality_config = quality_config
        self._assess_after_operations = assess_after_operations
        self._assessor = None
        self._reports: List[Dict[str, Any]] = []

    def on_pipeline_init(self, pipeline: "MultimodalPipeline") -> None:
        try:
            from ..quality.ops_quality_assessor import (
                OpsQualityAssessmentConfig,
                OpsQualityAssessor,
            )

            config = self._quality_config or OpsQualityAssessmentConfig()
            self._assessor = OpsQualityAssessor(config)
            logger.info("OpsQualityPlugin initialized")
        except Exception as e:
            logger.warning("OpsQualityPlugin init failed: %s", e)

    def after_operation(
        self,
        pipeline: "MultimodalPipeline",
        operation: "Operation",
        op_index: int,
        input_ds: "MultimodalDataset",
        output_ds: "MultimodalDataset",
        duration_sec: float,
    ) -> None:
        if self._assessor is None or not self._assess_after_operations:
            return
        assessment_rows = self._build_after_operation_assessment_rows(
            input_ds=input_ds,
            output_ds=output_ds,
        )
        self._assess_snapshot(
            dataset=output_ds,
            rows=assessment_rows,
            stage="after_operation",
            operation_name=operation.name,
            op_index=op_index,
            duration_sec=duration_sec,
        )

    def _build_after_operation_assessment_rows(
        self,
        input_ds: "MultimodalDataset",
        output_ds: "MultimodalDataset",
    ) -> Optional[List[Dict[str, Any]]]:
        """Build temporary assessment rows without changing pipeline data."""
        if input_ds is None or output_ds is None or self._assessor is None:
            return None

        try:
            sample_size = max(0, int(self._assessor.config.sample_size))
            if sample_size == 0:
                return []

            before_rows = input_ds.dataset.limit(sample_size).take_all()
            after_rows = output_ds.dataset.limit(sample_size).take_all()
            pair_count = min(len(before_rows), len(after_rows))
            if pair_count == 0:
                return None

            rows = []
            for before_row, after_row in zip(
                before_rows[:pair_count],
                after_rows[:pair_count],
            ):
                row = dict(after_row)
                generated_text = self._generated_text_value(after_row)
                row.setdefault("generated_text", generated_text)
                row.setdefault("before_text", self._preferred_text_value(before_row))
                row.setdefault("after_text", self._preferred_text_value(after_row) or generated_text)
                rows.append(row)
            return rows
        except Exception as e:
            logger.warning("Failed to build ops quality assessment view: %s", e)
            return None

    @staticmethod
    def _preferred_text_value(row: Dict[str, Any]) -> Any:
        if row.get("output_text") is not None:
            return row.get("output_text")
        if row.get("after_text") is not None:
            return row.get("after_text")
        return row.get("text")

    @staticmethod
    def _generated_text_value(row: Dict[str, Any]) -> Any:
        for key in (
            "generated_text",
            "dense_caption_text",
            "caption",
            "dense_caption",
            "output_text",
            "text",
        ):
            value = row.get(key)
            if value not in (None, ""):
                return OpsQualityPlugin._textual_generated_value(value)
        triplets = row.get("qae_triplets")
        if isinstance(triplets, list):
            parts = []
            for triplet in triplets:
                if not isinstance(triplet, dict):
                    continue
                question = str(triplet.get("question") or "").strip()
                answer = str(triplet.get("answer") or "").strip()
                if question or answer:
                    parts.append(f"{question} {answer}".strip())
            if parts:
                return "\n".join(parts)
        return None

    @staticmethod
    def _textual_generated_value(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        for key in ("summary", "caption", "text", "dense_caption_text"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text
        parts = []
        scene = value.get("scene")
        if isinstance(scene, dict):
            parts.extend(str(item).strip() for item in scene.values() if item)
        for list_key in ("objects", "relations", "actions", "text_in_image"):
            items = value.get(list_key)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    parts.extend(str(part).strip() for part in item.values() if part)
                elif item:
                    parts.append(str(item).strip())
        text = " ".join(part for part in parts if part).strip()
        return text or value

    def _assess_snapshot(
        self,
        dataset: "MultimodalDataset",
        rows: Optional[List[Dict[str, Any]]],
        stage: str,
        operation_name: Optional[str],
        op_index: Optional[int],
        duration_sec: Optional[float],
    ) -> None:
        try:
            if rows is None:
                result = self._assessor.evaluate(dataset)
            else:
                result = self._assessor.evaluate_rows(dataset, rows)
            report_items = result.to_report_items(
                related_operations=[operation_name] if operation_name else None,
            )
            self._reports.append(
                {
                    "stage": stage,
                    "operation_name": operation_name,
                    "op_index": op_index,
                    "duration_sec": duration_sec,
                    "result": result,
                    "report_items": report_items,
                }
            )
        except Exception as e:
            label = operation_name or stage
            logger.warning("Ops quality assessment failed for %s: %s", label, e)

    def get_assessor(self):
        return self._assessor

    def get_reports(self) -> List[Dict[str, Any]]:
        return list(self._reports)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ExperimentPlugin — wraps ExperimentTracker
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentPlugin(PipelinePlugin):
    """Experiment tracking plugin."""

    name = "experiment_tracking"

    def __init__(
        self,
        experiment_tracker=None,
        experiment_id: str = None,
        experiment_tags: Dict[str, str] = None,
        experiment_notes: str = "",
    ):
        self._tracker = experiment_tracker
        self._experiment_id = experiment_id
        self._experiment_tags = experiment_tags or {}
        self._experiment_notes = experiment_notes
        self._run_id: Optional[str] = None

    def on_pipeline_init(self, pipeline: "MultimodalPipeline") -> None:
        if self._tracker is None or self._experiment_id is None:
            return
        logger.info("ExperimentPlugin initialized (experiment_id=%s)", self._experiment_id)

    def before_operation(
        self,
        pipeline: "MultimodalPipeline",
        operation: "Operation",
        op_index: int,
        input_ds: "MultimodalDataset",
    ) -> None:
        if self._run_id is None and self._tracker is not None and self._experiment_id is not None:
            try:
                run = self._tracker.start_run(
                    experiment_id=self._experiment_id,
                    pipeline_config=pipeline.config,
                    tags=self._experiment_tags,
                    notes=self._experiment_notes,
                )
                self._run_id = run.run_id
            except Exception as e:
                logger.warning("Failed to start experiment run: %s", e)

    def on_pipeline_complete(
        self, pipeline: "MultimodalPipeline", final_dataset: "MultimodalDataset"
    ) -> None:
        self._complete_run(pipeline, failed=False)

    def on_pipeline_failure(
        self,
        pipeline: "MultimodalPipeline",
        operation: "Operation",
        op_index: int,
        exception: Exception,
    ) -> None:
        self._complete_run(pipeline, failed=True)

    def _complete_run(self, pipeline: "MultimodalPipeline", failed: bool) -> None:
        if self._tracker is None or self._run_id is None:
            return
        try:
            from ..experiments.experiment_models import RunStatus
            status = RunStatus.FAILED if failed else RunStatus.COMPLETED
            self._tracker.complete_run(
                run_id=self._run_id,
                experiment_id=self._experiment_id,
                status=status,
                tracking_report=None,
            )
        except Exception as e:
            logger.warning("Failed to complete experiment run: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# 5. AdaptiveSchedulingPlugin — wraps AdaptiveResourceScheduler
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveSchedulingPlugin(PipelinePlugin):
    """Adaptive resource scheduling plugin."""

    name = "adaptive_scheduling"

    def __init__(self, history_db_path: str = None):
        self._history_db_path = history_db_path
        self._scheduler = None

    def on_pipeline_init(self, pipeline: "MultimodalPipeline") -> None:
        try:
            from .scheduler.adaptive_scheduler import AdaptiveResourceScheduler
            from .scheduler.resource_history import ResourceHistoryDB

            db = ResourceHistoryDB(self._history_db_path) if self._history_db_path else ResourceHistoryDB()
            self._scheduler = AdaptiveResourceScheduler(db)
            logger.info("AdaptiveSchedulingPlugin initialized")
        except Exception as e:
            logger.warning("AdaptiveSchedulingPlugin init failed: %s", e)

    def before_operation(
        self,
        pipeline: "MultimodalPipeline",
        operation: "Operation",
        op_index: int,
        input_ds: "MultimodalDataset",
    ) -> None:
        if self._scheduler is None:
            return
        try:
            resources = self._scheduler.recommend_resources(operation, input_ds)
            logger.info(
                "[AdaptiveScheduler] Recommended resources: CPU=%s, GPU=%s, batch_size=%s",
                resources.get("num_cpus"), resources.get("num_gpus"), resources.get("batch_size"),
            )
        except Exception as e:
            logger.warning("Resource recommendation failed: %s", e)

    def get_scheduler(self):
        return self._scheduler
