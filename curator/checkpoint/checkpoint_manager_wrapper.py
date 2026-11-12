"""
Checkpoint manager wrapper class - encapsulates checkpoint management logic
"""
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

from .checkpoint_manager import CheckpointManager
from .checkpoint_models import CheckpointConfig, CheckpointStatus, OperationState


class CheckpointManagerWrapper:
    """Checkpoint manager wrapper class that encapsulates checkpoint strategies during pipeline execution."""

    def __init__(self, checkpoint_config: CheckpointConfig, pipeline_id: str, session_id: str):
        self.checkpoint_config = checkpoint_config
        self.pipeline_id = pipeline_id
        self.session_id = session_id
        self.checkpoint_manager: Optional[CheckpointManager] = (
            CheckpointManager(checkpoint_config) if checkpoint_config.enabled else None
        )

    # ── Shared helpers for building operation state ─────────────────────────────

    def _build_completed_op_states(
        self, operations: List, indices: List[int], status: str = "completed"
    ) -> List[OperationState]:
        """Build a list of completed operation states from an index list."""
        return [
            self.checkpoint_manager.create_operation_state(
                operation_id=str(j),
                operation_name=operations[j].name,
                operation_type=operations[j].operation_type.value,
                parameters=operations[j].get_params(),
                status=status,
            )
            for j in indices
            if j < len(operations)
        ]

    @staticmethod
    def _build_remaining_ops(operations: List, indices: List[int]) -> List[Dict[str, Any]]:
        """Build a list of configuration dictionaries for remaining operations from an index list."""
        return [
            {"name": operations[j].name, "params": operations[j].get_params()}
            for j in indices
            if j < len(operations)
        ]

    # ── Resume ────────────────────────────────────────────────────────────────

    def should_resume_from_checkpoint(self, resume_from_checkpoint: Optional[bool]) -> tuple:
        """Check whether execution should resume from a checkpoint, returning (should_resume, dataset, start_index, ops)."""
        if not self.checkpoint_manager:
            return False, None, 0, []

        should_check = resume_from_checkpoint or (
            resume_from_checkpoint is None and self.checkpoint_config.resume_from_latest
        )
        if not should_check:
            return False, None, 0, []

        latest = self.checkpoint_manager.get_latest_checkpoint(self.pipeline_id)
        if latest and latest.status == CheckpointStatus.ACTIVE:
            logger.info("Resuming from checkpoint: %s", latest.checkpoint_id)
            dataset, start_index, remaining_ops = (
                self.checkpoint_manager.restore_pipeline_from_checkpoint(latest)
            )
            return True, dataset, start_index, remaining_ops

        return False, None, 0, []

    # ── Save (all three scenarios share the same build logic) ──────────────────

    def save_checkpoint_if_needed(
        self,
        operations: List,
        current_operation_idx: int,
        processed_dataset,
        completed_ops_indices: List[int],
        remaining_ops_indices: List[int],
    ) -> None:
        """Decide whether to save a periodic checkpoint based on the configured interval."""
        if not self.checkpoint_manager:
            return
        if (current_operation_idx + 1) % self.checkpoint_config.auto_save_interval_ops != 0:
            return

        self.checkpoint_manager.save_checkpoint(
            pipeline_id=self.pipeline_id,
            session_id=self.session_id,
            current_operation_index=current_operation_idx,
            completed_operations=self._build_completed_op_states(operations, completed_ops_indices),
            remaining_operations=self._build_remaining_ops(operations, remaining_ops_indices),
            dataset=processed_dataset,
        )
        logger.info("Checkpoint saved after operation %s", current_operation_idx + 1)

    def save_checkpoint_on_failure(
        self,
        operations: List,
        operation_idx: int,
        processed_dataset,
        error: Exception,
    ) -> None:
        """Save a checkpoint containing failure information when an operation fails."""
        if not self.checkpoint_manager or not self.checkpoint_config.checkpoint_on_failure:
            return

        completed_ops = self._build_completed_op_states(
            operations, list(range(operation_idx))
        )
        failed_op = self.checkpoint_manager.create_operation_state(
            operation_id=str(operation_idx),
            operation_name=operations[operation_idx].name,
            operation_type=operations[operation_idx].operation_type.value,
            parameters=operations[operation_idx].get_params(),
            status="failed",
            metadata={"error": str(error)},
        )
        completed_ops.append(failed_op)

        self.checkpoint_manager.save_checkpoint(
            pipeline_id=self.pipeline_id,
            session_id=self.session_id,
            current_operation_index=operation_idx,
            completed_operations=completed_ops,
            remaining_operations=self._build_remaining_ops(
                operations, list(range(operation_idx + 1, len(operations)))
            ),
            dataset=processed_dataset,
        )
        logger.info("Checkpoint saved after failure at operation %s", operation_idx + 1)

    def save_final_checkpoint(self, operations: List, processed_dataset) -> None:
        """Save the final checkpoint when the pipeline completes successfully."""
        if not self.checkpoint_manager or not self.checkpoint_config.checkpoint_on_success:
            return

        self.checkpoint_manager.save_checkpoint(
            pipeline_id=self.pipeline_id,
            session_id=self.session_id,
            current_operation_index=len(operations),
            completed_operations=self._build_completed_op_states(
                operations, list(range(len(operations)))
            ),
            remaining_operations=[],
            dataset=processed_dataset,
        )

    def save_manual_checkpoint(self, checkpoint_name: Optional[str] = None) -> None:
        """Manually trigger checkpoint saving.

        Note: this method can only save an empty state (no execution progress). To save
        manually during pipeline execution, call save_checkpoint_if_needed() from a
        callback inside pipeline.run().
        """
        if not self.checkpoint_manager:
            logger.debug("Checkpoint manager is not enabled")
            return

        checkpoint_metadata = self.checkpoint_manager.save_checkpoint(
            pipeline_id=self.pipeline_id,
            session_id=self.session_id,
            current_operation_index=0,
            completed_operations=[],
            remaining_operations=[],
            dataset=None,
        )
        logger.info("Manual checkpoint saved: %s", checkpoint_metadata.checkpoint_id)

    # ── Query ─────────────────────────────────────────────────────────────────

    def list_checkpoints(self) -> List:
        """List all available checkpoints for the current pipeline."""
        if not self.checkpoint_manager:
            logger.debug("Checkpoint manager is not enabled")
            return []
        return self.checkpoint_manager.list_checkpoints(self.pipeline_id)
