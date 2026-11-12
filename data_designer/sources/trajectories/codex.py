from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from data_designer.contracts.trajectories import StepRecord, TaskOutcome, TrajectoryMeta, TrajectoryRecord
from data_designer.contracts.provenance import content_digest as fingerprint_of
logger = logging.getLogger(__name__)

_STATUS_MAP = {"success": "success", "failed": "failure",
               "interrupted": "aborted", "error": "failure"}


class CodexReader:
    """codex trajectory reader; never installs host resources."""


    def __init__(self, workspace: str):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "sessions"


    def export_trajectories(
        self,
        scope: Optional[Dict] = None,
        since: Optional[object] = None,
    ) -> List[TrajectoryRecord]:
        records: List[TrajectoryRecord] = []
        if not self.sessions_dir.exists():
            return records
        for path in sorted(self.sessions_dir.glob("*.jsonl")):
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
            logger.warning("skip unreadable codex session %s: %s", path, e)
            return []

        sessions: Dict[str, List[Dict]] = {}
        for ev in events:
            sid = ev.get("session_id")
            if sid:
                sessions.setdefault(sid, []).append(ev)

        records: List[TrajectoryRecord] = []
        for sid, evs in sessions.items():
            evs.sort(key=lambda e: e.get("timestamp", ""))
            meta_ev = next((e for e in evs if e.get("type") == "session_meta"), None)
            user_ev = next((e for e in evs if e.get("type") == "user_message"), None)
            if user_ev is None:
                continue
            started = user_ev.get("timestamp")
            if since is not None and started is not None and started < str(since):
                continue

            steps: List[StepRecord] = []
            outcome = TaskOutcome()
            model_version = str((meta_ev or {}).get("model", "") or "unknown")
            for ev in evs:
                etype = ev.get("type")
                if etype == "function_call":
                    steps.append(StepRecord(
                        index=len(steps),
                        action=str(ev.get("name", "")),
                        observation=str(ev.get("output", "")),
                        tool=str(ev.get("name", "")) or None,
                        error=str(ev.get("status", "")) == "error",
                    ))
                elif etype == "task_complete":
                    status = _STATUS_MAP.get(str(ev.get("status", "failed")), "failure")
                    outcome = TaskOutcome(
                        status=status,
                        corrections=int(ev.get("corrections", 0)),
                        summary=str(ev.get("summary", "")),
                    )

            records.append(TrajectoryRecord(
                trajectory_id=sid,
                task=str(user_ev.get("content", "")),
                steps=steps,
                outcome=outcome,
                cost={"events": float(len(evs))},
                meta=TrajectoryMeta(
                    agent_version=str((meta_ev or {}).get("cli_version", "codex")),
                    skill_versions=dict((meta_ev or {}).get("skill_versions", {}) or {}),
                    model_version=model_version,
                    harness=self.harness,
                    started_at=started,
                    finished_at=evs[-1].get("timestamp") if evs else None,
                    extensions={"transcript_source": path.name},
                ),
            ))
        return records

