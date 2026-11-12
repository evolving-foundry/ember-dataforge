from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from data_designer.contracts.trajectories import StepRecord, TaskOutcome, TrajectoryMeta, TrajectoryRecord
from data_designer.contracts.provenance import content_digest as fingerprint_of
logger = logging.getLogger(__name__)

class OpenCodeReader:
    """opencode trajectory reader; never installs host resources."""


    def __init__(self, workspace: str):
        self.workspace = Path(workspace)
        self.storage_dir = self.workspace / "storage"


    def export_trajectories(
        self,
        scope: Optional[Dict] = None,
        since: Optional[object] = None,
    ) -> List[TrajectoryRecord]:
        records: List[TrajectoryRecord] = []
        if not self.storage_dir.exists():
            return records
        for path in sorted(self.storage_dir.glob("*.jsonl")):
            records.extend(self._load(path, since))
        return records


    def _load(self, path: Path, since) -> List[TrajectoryRecord]:
        events: List[Dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skip unreadable opencode session %s: %s", path, e)
            return []

        sessions: Dict[str, List[Dict]] = {}
        for ev in events:
            sid = ev.get("sessionID")
            if sid:
                sessions.setdefault(sid, []).append(ev)

        records: List[TrajectoryRecord] = []
        for sid, evs in sessions.items():
            evs.sort(key=lambda e: e.get("time", ""))
            start_ev = next((e for e in evs if e.get("type") == "session.start"), None)
            user_ev = next((e for e in evs if e.get("type") == "message.user"), None)
            if user_ev is None:
                continue
            started = user_ev.get("time")
            if since is not None and started is not None and started < str(since):
                continue

            steps: List[StepRecord] = []
            outcome = TaskOutcome()
            model_version = str((start_ev or {}).get("model", "") or "unknown")
            for ev in evs:
                etype = ev.get("type")
                if etype == "tool.invoke":
                    steps.append(StepRecord(
                        index=len(steps),
                        action=str(ev.get("tool", "")),
                        observation=str(ev.get("output", "")),
                        tool=str(ev.get("tool", "")) or None,
                        error=bool(ev.get("error", False)),
                    ))
                elif etype == "session.finish":
                    outcome = TaskOutcome(
                        status=str(ev.get("status", "failure")),
                        corrections=int(ev.get("corrections", 0)),
                        summary=str(ev.get("summary", "")),
                    )

            records.append(TrajectoryRecord(
                trajectory_id=sid,
                task=str(user_ev.get("text", "")),
                steps=steps,
                outcome=outcome,
                cost={"events": float(len(evs))},
                meta=TrajectoryMeta(
                    agent_version=str((start_ev or {}).get("version", "opencode")),
                    skill_versions=dict((start_ev or {}).get("skill_versions", {}) or {}),
                    model_version=model_version,
                    harness=self.harness,
                    started_at=started,
                    finished_at=evs[-1].get("time") if evs else None,
                    extensions={"transcript_source": path.name},
                ),
            ))
        return records

