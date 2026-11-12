"""
Data tracker - an independent tracking utility for recording data changes before and after operator processing
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from uuid import uuid4

from .tracking import (
    OperationType, ChangeType, DataChangeRecord,
    OperationTrackingInfo, TrackingMetadata
)


@dataclass
class _DatasetSnapshot:
    """Collect key dataset statistics in one pass to avoid repeated access.

    Supports lazy sampling mode (is_estimated=True): prefer estimating the row count from Ray metadata / stats,
    avoiding a forced materialization of the Ray execution plan via count() at every step.
    """
    row_count: int
    schema: Any
    column_names: List[str]
    sample_keys: List[str] = field(default_factory=list)
    # New: mark whether this is a full computation (True=measured via count(), False=estimated from metadata)
    is_estimated: bool = False

    @classmethod
    def take(cls, dataset, *, force: bool = False) -> Optional["_DatasetSnapshot"]:
        """Collect a snapshot from dataset.

        Args:
            dataset: MultimodalDataset or Ray Dataset
            force: True = force count()+take(1) (breaks lazy evaluation; debugging only);
                   False = prefer sampled estimation (recommended, default)
        """
        if dataset is None:
            return None
        try:
            # ── Option A: force full materialization (force=True) ──────────────────────────
            if force:
                count = dataset.count()
                schema = dataset.schema()
                column_names = list(schema.names) if hasattr(schema, "names") else []
                sample_keys = []
                if count > 0:
                    row = dataset.take(1)[0]
                    sample_keys = list(row.keys()) if hasattr(row, "keys") else []
                return cls(
                    row_count=count,
                    schema=schema,
                    column_names=column_names,
                    sample_keys=sample_keys,
                    is_estimated=False,
                )

            # ── Option B: estimate from Ray Dataset metadata (recommended) ────────────────
            try:
                metadata = dataset._get_metadata()
                if metadata:
                    est_rows = getattr(metadata, "num_rows", None)
                    if est_rows and est_rows > 0:
                        schema = dataset.schema()
                        column_names = list(schema.names) if hasattr(schema, "names") else []
                        return cls(
                            row_count=est_rows,
                            schema=schema,
                            column_names=column_names,
                            is_estimated=True,
                        )
            except Exception:
                pass

            # ── Option C: extract estimates from Ray stats ───────────────────────────
            try:
                stats = dataset.stats()
                if stats and stats.get("dataset_metrics"):
                    est_rows = stats["dataset_metrics"].get("num_rows_estimated", 0)
                    if est_rows > 0:
                        schema = dataset.schema()
                        column_names = list(schema.names) if hasattr(schema, "names") else []
                        return cls(
                            row_count=est_rows,
                            schema=schema,
                            column_names=column_names,
                            is_estimated=True,
                        )
            except Exception:
                pass

            # ── Fallback: force full computation (only when estimation is impossible) ────────────────────────────
            return cls.take(dataset, force=True)

        except Exception as e:
            return cls(row_count=-1, schema=None, column_names=[], sample_keys=[])


class DataTracker:
    """Data tracker - an independent tracking utility."""

    def __init__(self, pipeline_id: str, session_id: str, force_full_snapshot: bool = False):
        self.pipeline_id = pipeline_id
        self.session_id = session_id
        self._force_full_snapshot = force_full_snapshot
        self.tracking_metadata = TrackingMetadata(
            pipeline_id=pipeline_id,
            session_id=session_id,
        )
        self.current_operation_info: Optional[OperationTrackingInfo] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def start_operation_tracking(
        self,
        operation_name: str,
        operation_type: OperationType,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start tracking an operation."""
        self.current_operation_info = OperationTrackingInfo(
            operation_id=str(uuid4()),
            operation_name=operation_name,
            operation_type=operation_type,
            parameters=parameters or {},
        )

    def track_operation(
        self,
        operation_name: str,
        operation_type: OperationType,
        input_dataset,
        output_dataset,
        parameters: Optional[Dict[str, Any]] = None,
        changes: Optional[List[DataChangeRecord]] = None,
    ) -> OperationTrackingInfo:
        """Track an operation's impact on the dataset and return operation tracking information."""
        input_snap = _DatasetSnapshot.take(input_dataset, force=self._force_full_snapshot)
        output_snap = _DatasetSnapshot.take(output_dataset, force=self._force_full_snapshot)

        operation_info = OperationTrackingInfo(
            operation_id=str(uuid4()),
            operation_name=operation_name,
            operation_type=operation_type,
            input_stats=self._snapshot_to_stats(input_snap),
            output_stats=self._snapshot_to_stats(output_snap),
            changes=changes or self._calculate_changes(input_snap, output_snap),
            parameters=parameters or {},
            success=True,
        )
        operation_info.finish_tracking()
        self.tracking_metadata.add_operation(operation_info)
        return operation_info

    def get_tracking_metadata(self) -> TrackingMetadata:
        """Get tracking metadata."""
        return self.tracking_metadata

    def reset(self) -> None:
        """Reset the tracker."""
        self.tracking_metadata = TrackingMetadata(
            pipeline_id=self.pipeline_id,
            session_id=self.session_id,
        )
        self.current_operation_info = None

    # ── Internal change detection ─────────────────────────────────────────────

    @staticmethod
    def _snapshot_to_stats(snap: Optional["_DatasetSnapshot"]) -> Dict[str, Any]:
        """Convert the snapshot into a statistics dictionary."""
        if snap is None:
            return {}
        return {
            "num_rows": snap.row_count,
            "schema": str(snap.schema),
            "sample_keys": snap.sample_keys,
        }

    def _calculate_changes(
        self,
        input_snap: Optional["_DatasetSnapshot"],
        output_snap: Optional["_DatasetSnapshot"],
    ) -> List[DataChangeRecord]:
        """Summarize row-count changes and schema changes, catching exceptions independently for each change type."""
        # If a Ray Dataset is passed instead of a _DatasetSnapshot, convert it to a snapshot first
        if not isinstance(input_snap, _DatasetSnapshot):
            input_snap = _DatasetSnapshot.take(input_snap)
        if not isinstance(output_snap, _DatasetSnapshot):
            output_snap = _DatasetSnapshot.take(output_snap)

        if input_snap is None or output_snap is None:
            return []

        changes: List[DataChangeRecord] = []

        try:
            changes.extend(self._detect_row_changes(input_snap.row_count, output_snap.row_count))
        except Exception as e:
            changes.append(DataChangeRecord(
                change_type=ChangeType.ROW_MODIFIED,
                count=0,
                metadata={"error_calculating_row_changes": str(e)},
            ))

        try:
            changes.extend(self._detect_schema_changes(input_snap.schema, output_snap.schema))
        except Exception as e:
            changes.append(DataChangeRecord(
                change_type=ChangeType.ROW_MODIFIED,
                count=0,
                metadata={"error_calculating_schema_changes": str(e)},
            ))

        return changes

    @staticmethod
    def _detect_row_changes(input_count: int, output_count: int) -> List[DataChangeRecord]:
        """Generate ROW_REMOVED / ROW_ADDED records based on the row-count difference."""
        # If snapshot collection failed (row_count=-1), skip row-count change detection
        if input_count < 0 or output_count < 0:
            return []
        diff = output_count - input_count
        if diff == 0:
            return []
        change_type = ChangeType.ROW_REMOVED if diff < 0 else ChangeType.ROW_ADDED
        return [DataChangeRecord(
            change_type=change_type,
            count=abs(diff),
            metadata={"reason": "operation_removed_rows" if diff < 0 else "operation_added_rows"},
        )]

    @staticmethod
    def _detect_schema_changes(input_schema, output_schema) -> List[DataChangeRecord]:
        """Compare column-name differences between two schemas and generate COLUMN_ADDED / COLUMN_REMOVED records."""
        if not (hasattr(input_schema, "names") and hasattr(output_schema, "names")):
            return []

        input_cols = set(input_schema.names)
        output_cols = set(output_schema.names)
        changes: List[DataChangeRecord] = []

        added = output_cols - input_cols
        if added:
            changes.append(DataChangeRecord(
                change_type=ChangeType.COLUMN_ADDED,
                count=len(added),
                columns_affected=list(added),
                metadata={"reason": "operation_added_columns"},
            ))

        removed = input_cols - output_cols
        if removed:
            changes.append(DataChangeRecord(
                change_type=ChangeType.COLUMN_REMOVED,
                count=len(removed),
                columns_affected=list(removed),
                metadata={"reason": "operation_removed_columns"},
            ))

        return changes
