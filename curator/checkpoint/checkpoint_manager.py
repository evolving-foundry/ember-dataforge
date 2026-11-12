"""
Checkpoint manager responsible for saving and restoring data processing pipeline state
"""
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import logging

logger = logging.getLogger(__name__)

from .checkpoint_models import (
    CheckpointMetadata, OperationState, CheckpointStatus,
    CheckpointType, CheckpointConfig
)


class CheckpointManager:
    """Checkpoint manager responsible for saving and restoring pipeline state.

    Responsibilities:
      - Disk I/O: _load_metadata_file / _save_metadata_file / delete_checkpoint
      - Metadata normalization: _normalize_checkpoint_dict (backward compatible with old formats)
      - Lifecycle management: list_checkpoints / get_latest_checkpoint / _cleanup_old_checkpoints
      - High-level orchestration: save_checkpoint / load_checkpoint / restore_pipeline_from_checkpoint
    """

    def __init__(self, config: CheckpointConfig = None):
        self.config = config or CheckpointConfig()
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Disk I/O ──────────────────────────────────────────────────────────────

    def _load_metadata_file(self, checkpoint_dir: Path) -> Optional[CheckpointMetadata]:
        """Load and deserialize metadata.json from a directory; return None on failure."""
        metadata_file = checkpoint_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            raw = self._normalize_checkpoint_dict(raw)
            return CheckpointMetadata(**raw)
        except Exception as e:
            logger.error("Error loading checkpoint from %s: %s", checkpoint_dir, e)
            return None

    @staticmethod
    def _normalize_checkpoint_dict(d: dict) -> dict:
        """Normalize legacy 'ClassName.MEMBER' enum fields to plain string values.

        This only handles historical serialization formats; newly written files do not need it.
        """
        enum_fields = {
            "status": CheckpointStatus,
            "checkpoint_type": CheckpointType,
        }
        for field, enum_cls in enum_fields.items():
            raw = d.get(field)
            if isinstance(raw, str) and "." in raw:
                name = raw.rsplit(".", 1)[-1]  # 'ACTIVE' from 'CheckpointStatus.ACTIVE'
                try:
                    d[field] = enum_cls[name].value
                except KeyError:
                    pass  # Preserve the original value so Pydantic can report a clear error later
        return d

    def delete_checkpoint(self, checkpoint_path: str) -> None:
        """Delete the Checkpoint directory."""
        path = Path(checkpoint_path)
        if path.exists():
            shutil.rmtree(path)

    # ── Lifecycle management ──────────────────────────────────────────────────

    def list_checkpoints(self, pipeline_id: str = None) -> List[CheckpointMetadata]:
        """List available Checkpoints in descending creation-time order."""
        checkpoints = []
        for subdir in self.checkpoint_dir.iterdir():
            if not subdir.is_dir():
                continue
            if pipeline_id and not subdir.name.startswith(pipeline_id):
                continue
            metadata = self._load_metadata_file(subdir)
            if metadata is not None:
                checkpoints.append(metadata)
        checkpoints.sort(key=lambda x: x.created_at, reverse=True)
        return checkpoints

    def get_latest_checkpoint(self, pipeline_id: str) -> Optional[CheckpointMetadata]:
        """Return the latest Checkpoint, or None if none exists."""
        checkpoints = self.list_checkpoints(pipeline_id)
        return checkpoints[0] if checkpoints else None

    def _cleanup_old_checkpoints(self, pipeline_id: str) -> None:
        """Keep the latest N checkpoints and delete older ones."""
        checkpoints = self.list_checkpoints(pipeline_id)
        for old in checkpoints[self.config.max_checkpoints_to_keep:]:
            try:
                self.delete_checkpoint(old.storage_path)
                logger.info("Deleted old checkpoint: %s", old.checkpoint_id)
            except Exception as e:
                logger.error("Failed to delete checkpoint %s: %s", old.checkpoint_id, e)

    # ── High-level orchestration ──────────────────────────────────────────────

    def save_checkpoint(
        self,
        pipeline_id: str,
        session_id: str,
        current_operation_index: int,
        completed_operations: List[OperationState],
        remaining_operations: List[Dict[str, Any]],
        dataset=None,
        data_location: Optional[str] = None,
    ) -> CheckpointMetadata:
        """Save a Checkpoint and return the written metadata object."""
        checkpoint_subdir = self.checkpoint_dir / (
            f"{pipeline_id}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        checkpoint_subdir.mkdir(exist_ok=True)

        if dataset is not None:
            data_location = self._save_dataset_metadata(checkpoint_subdir, dataset, data_location)

        checkpoint_metadata = CheckpointMetadata(
            pipeline_id=pipeline_id,
            session_id=session_id,
            checkpoint_type=CheckpointType.OPERATION_LEVEL,
            current_operation_index=current_operation_index,  # ignored (computed_field)
            completed_operations=completed_operations,
            remaining_operations=remaining_operations,
            data_location=data_location,
            storage_path=str(checkpoint_subdir),
            total_operations=len(completed_operations) + len(remaining_operations),
        )

        metadata_file = checkpoint_subdir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                checkpoint_metadata.model_dump(mode="json"),
                f, indent=2, ensure_ascii=False, default=str,
            )

        self._cleanup_old_checkpoints(pipeline_id)
        return checkpoint_metadata

    @staticmethod
    def _save_dataset_metadata(
        checkpoint_subdir: Path, dataset, data_location: Optional[str]
    ) -> str:
        """Persist dataset metadata to dataset.json and return the file path string."""
        raw_ds = getattr(dataset, "dataset", dataset)
        dataset_metadata = {
            "count": raw_ds.count(),
            "schema": str(raw_ds.schema()),
            "path": data_location,
        }
        data_file = checkpoint_subdir / "dataset.pkl"
        with open(data_file, "wb") as f:
            pickle.dump(dataset_metadata, f)
        return str(data_file)

    def load_checkpoint(self, checkpoint_path: str) -> CheckpointMetadata:
        """Load Checkpoint metadata from the specified path."""
        metadata = self._load_metadata_file(Path(checkpoint_path))
        if metadata is None:
            raise FileNotFoundError(
                f"Checkpoint metadata not found: {checkpoint_path}/metadata.json"
            )
        return metadata

    def load_dataset_from_checkpoint(self, checkpoint_path: str):
        """Load a dataset from a Checkpoint.

        The current implementation only logs metadata and returns None.
        To restore the actual dataset, override this method in a subclass and reload
        data from external storage using dataset_metadata['path'].
        """
        data_file = Path(checkpoint_path) / "dataset.pkl"
        if not data_file.exists():
            return None
        with open(data_file, "rb") as f:
            dataset_metadata = pickle.load(f)
        logger.info("Dataset metadata loaded from checkpoint: %s", dataset_metadata)
        return None

    def restore_pipeline_from_checkpoint(self, checkpoint: CheckpointMetadata) -> tuple:
        """Restore a pipeline from a Checkpoint and return (dataset, start_index, remaining_ops)."""
        dataset = self.load_dataset_from_checkpoint(checkpoint.storage_path)
        return dataset, checkpoint.current_operation_index, checkpoint.remaining_operations

    # ── Factory methods ──────────────────────────────────────────────────────

    @staticmethod
    def create_operation_state(
        operation_id: str,
        operation_name: str,
        operation_type: str,
        parameters: Dict[str, Any] = None,
        status: str = "completed",
        metadata: Dict[str, Any] = None,
    ) -> OperationState:
        """Create an operation state object (used by CheckpointManagerWrapper)."""
        return OperationState(
            operation_id=operation_id,
            operation_name=operation_name,
            operation_type=operation_type,
            end_time=datetime.now(),
            status=status,
            parameters=parameters,
            metadata=metadata,
        )


