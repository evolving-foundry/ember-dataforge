from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
from data_designer.contracts.trajectories import TrajectoryRecord

@dataclass
class OmissionEntry:
    """A single omission record (B4: omission is declaration)."""

    op: str
    trajectory_id: str
    step_index: int
    field: str
    reason: str
    chars_removed: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op, "trajectory_id": self.trajectory_id,
            "step_index": self.step_index, "field": self.field,
            "reason": self.reason, "chars_removed": self.chars_removed,
        }


@dataclass
class OmissionLog:
    """Omission log: exported with the tracking report so lossy transformations remain auditable."""

    entries: List[OmissionEntry] = field(default_factory=list)

    def add(self, entry: OmissionEntry) -> None:
        self.entries.append(entry)

    def extend(self, other: "OmissionLog") -> None:
        self.entries.extend(other.entries)

    def stats(self) -> Dict[str, Any]:
        by_op: Dict[str, int] = {}
        chars = 0
        for e in self.entries:
            by_op[e.op] = by_op.get(e.op, 0) + 1
            chars += e.chars_removed
        return {"total_omissions": len(self.entries), "chars_removed": chars, "by_op": by_op}

    def to_report(self) -> Dict[str, Any]:
        """Shape written into the tracking report (B4 acceptance)."""
        return {
            "omission_policy": "Omission is explicit: every lossy transformation must be recorded in this log; "
                               "omitted content must not be treated as known evidence in later analysis",
            **self.stats(),
            "entries": [e.as_dict() for e in self.entries],
        }


@dataclass
class TrajectoryCorpus:
    """Trajectory corpus: input/output container for operator pipelines."""

    records: List[TrajectoryRecord] = field(default_factory=list)
    omissions: OmissionLog = field(default_factory=OmissionLog)
    # Degenerate trajectory flags (B3): trajectory_id -> detection result
    degenerate_flags: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Corpus-level fingerprint (for D3 snapshot freezing)
    meta: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        from data_designer.contracts.provenance import content_digest as fingerprint_of
        return fingerprint_of([r.fingerprint() for r in self.records])

    def failed(self) -> List[TrajectoryRecord]:
        return [r for r in self.records if r.outcome.status == "failure"]

    def succeeded(self) -> List[TrajectoryRecord]:
        return [r for r in self.records if r.outcome.status == "success"]


