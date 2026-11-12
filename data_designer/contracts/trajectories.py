"""Normalized trajectory data; no host resource or RSI protocol types."""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from .provenance import content_digest

class StepRecord(BaseModel):
    """Normalized step: preserves the original message index and error marker (B1 requirement)."""

    index: int
    action: str = ""
    observation: str = ""
    tool: Optional[str] = None
    error: bool = False
    # Cleaning marker: this step's content was lossy-transformed during preprocessing (B4 evidence integrity).
    redacted: bool = False


class TaskOutcome(BaseModel):
    """Task success/failure outcome and signals."""

    status: Literal["success", "failure", "aborted"] = "failure"
    error_category: Optional[str] = None
    corrections: int = 0
    summary: str = ""


class TrajectoryMeta(BaseModel):
    """Trajectory metadata (A4): ExperimentTracker run comparison applies directly.

    Host-specific fields always belong in extensions and must not pollute the core schema (FR-A acceptance point).
    """

    agent_version: str
    skill_versions: Dict[str, str] = Field(default_factory=dict)
    model_version: str = ""
    harness: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    extensions: Dict[str, Any] = Field(default_factory=dict)


class TrajectoryRecord(BaseModel):
    """Normalized trajectory record (A3): task / steps[] / outcome / cost / meta."""

    trajectory_id: str
    task: str
    steps: List[StepRecord] = Field(default_factory=list)
    outcome: TaskOutcome = Field(default_factory=TaskOutcome)
    cost: Dict[str, float] = Field(default_factory=dict)
    meta: TrajectoryMeta

    def fingerprint(self) -> str:
        return content_digest(self.model_dump(mode="json"))

