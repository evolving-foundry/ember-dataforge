"""Trajectory loading operator (FR-A A3): normalize host export formats into TrajectoryRecord.

Acceptance point (FR-A): trajectories from different hosts under the same schema can be mixed in the same analysis DAG;
host-specific fields go into meta.extensions and must not pollute the core schema.
A4: meta must contain agent_version / skill_versions / model_version (validated programmatically).
"""

from __future__ import annotations

from typing import Any, Dict, List

from data_designer.contracts.trajectories import (
    StepRecord,
    TaskOutcome,
    TrajectoryMeta,
    TrajectoryRecord,
)
from .trajectory_data import TrajectoryCorpus
from .trajectory_processing import _Processor


class TrajectoryNormalizationOperation(_Processor):
    """Normalize an Adapter-exported dict into a TrajectoryRecord (A3/A4)."""

    name = "trajectory_normalization"
    PARAMS_SCHEMA = {
        "params": {
            "default_harness": {
                "type": "str", "required": False, "default": "unknown",
                "description": "Host identifier used when the record does not declare a harness",
            },
            "require_core_meta": {
                "type": "bool", "required": False, "default": True,
                "description": "A4: require agent_version/skill_versions/model_version to be present",
            },
        }
    }

#: Core schema fields--host-specific fields always go into meta.extensions (FR-A acceptance point)
    _META_CORE_FIELDS = {"agent_version", "skill_versions", "model_version", "harness",
                         "started_at", "finished_at"}

    def execute(self, corpus: TrajectoryCorpus) -> TrajectoryCorpus:
        require_meta = bool(self.params["require_core_meta"])
        default_harness = str(self.params["default_harness"])

        normalized: List[TrajectoryRecord] = []
        for raw in corpus.records:
            if isinstance(raw, TrajectoryRecord):
                normalized.append(raw)
                continue
            normalized.append(self._normalize_dict(raw, default_harness, require_meta))

    # A4 programmatic validation: ensure core meta is complete even when the input is already a TrajectoryRecord
        if require_meta:
            for rec in normalized:
                if rec.meta.agent_version == "" or rec.meta.model_version == "":
                    raise ValueError(
                        f"trajectory '{rec.trajectory_id}': A4 requires meta to contain "
                        f"agent_version/model_version (agent_version={rec.meta.agent_version!r}, "
                        f"model_version={rec.meta.model_version!r})"
                    )

        corpus.records = normalized
        corpus.meta["normalized"] = True
        return corpus

    def _normalize_dict(
        self, raw: Dict[str, Any], default_harness: str, require_meta: bool
    ) -> TrajectoryRecord:
        raw = dict(raw)
        meta_raw = dict(raw.pop("meta", {}) or {})
        extensions = {
            k: meta_raw.pop(k)
            for k in list(meta_raw)
            if k not in self._META_CORE_FIELDS
        }
        steps_raw = raw.pop("steps", []) or []
        steps = [
            s if isinstance(s, StepRecord)
            else StepRecord(
                index=int(s.get("index", i)),
                action=str(s.get("action", "")),
                observation=str(s.get("observation", "")),
                tool=s.get("tool"),
                error=bool(s.get("error", False)),
            )
            for i, s in enumerate(steps_raw)
        ]
        outcome_raw = raw.pop("outcome", {}) or {}
        outcome = outcome_raw if isinstance(outcome_raw, TaskOutcome) else TaskOutcome(**{
            k: v for k, v in outcome_raw.items() if k in TaskOutcome.model_fields
        })
        meta = TrajectoryMeta(
            agent_version=str(meta_raw.get("agent_version", "")),
            skill_versions=dict(meta_raw.get("skill_versions", {}) or {}),
            model_version=str(meta_raw.get("model_version", "")),
            harness=str(meta_raw.get("harness", default_harness)),
            started_at=meta_raw.get("started_at"),
            finished_at=meta_raw.get("finished_at"),
            extensions=extensions,
        )
        return TrajectoryRecord(
            trajectory_id=str(raw.get("trajectory_id", "")),
            task=str(raw.get("task", "")),
            steps=steps,
            outcome=outcome,
            cost=dict(raw.get("cost", {}) or {}),
            meta=meta,
        )
