from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from data_designer.contracts.trajectories import StepRecord, TaskOutcome, TrajectoryMeta, TrajectoryRecord
from data_designer.contracts.provenance import content_digest as fingerprint_of
logger = logging.getLogger(__name__)

class ClaudeTranscriptReader:
    """claude trajectory reader; never installs host resources."""


    def __init__(self, workspace: str):
        self.workspace = Path(workspace)
        self.transcripts_dir = self.workspace / "transcripts"


    def export_trajectories(
        self,
        scope: Optional[Dict] = None,
        since: Optional[object] = None,
    ) -> List[TrajectoryRecord]:
        records: List[TrajectoryRecord] = []
        if not self.transcripts_dir.exists():
            return records
        for path in sorted(self.transcripts_dir.glob("*.jsonl")):
            records.extend(self._load_transcript(path, since))
        return records


    def _load_transcript(self, path: Path, since) -> List[TrajectoryRecord]:
        events: List[Dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skip unreadable transcript %s: %s", path, e)
            return []

        sessions: Dict[str, List[Dict]] = {}
        for ev in events:
            sid = ev.get("sessionId")
            if sid:
                sessions.setdefault(sid, []).append(ev)

        records: List[TrajectoryRecord] = []
        for sid, evs in sessions.items():
            evs.sort(key=lambda e: e.get("timestamp", ""))
            user_evs = [e for e in evs if e.get("type") == "user"]
            if not user_evs:
                continue
            first = user_evs[0]
            started = first.get("timestamp")
            if since is not None and started is not None and started < str(since):
                continue

            steps: List[StepRecord] = []
            outcome = TaskOutcome()
            model_version = str(first.get("model", "") or "unknown")
            for ev in evs:
                if ev.get("type") == "assistant":
                    model_version = str(ev.get("model", model_version)) or model_version
                    message = ev.get("message", {}) or {}
                    for block in message.get("content", []) or []:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "tool_use":
                            steps.append(StepRecord(
                                index=len(steps),
                                action=str(block.get("name", "")),
                                observation=str(block.get("output", "")),
                                tool=str(block.get("name", "")) or None,
                                error=False,
                            ))
                        elif block.get("type") == "text":
                            steps.append(StepRecord(
                                index=len(steps),
                                action="say",
                                observation=str(block.get("text", "")),
                                error=False,
                            ))
                elif ev.get("type") == "result":
                    subtype = str(ev.get("subtype", "error"))
                    status = "success" if subtype == "success" else (
                        "aborted" if subtype == "aborted" else "failure")
                    outcome = TaskOutcome(
                        status=status,
                        corrections=int(ev.get("num_corrections", 0)),
                        summary=str(ev.get("summary", "")),
                    )

            meta = TrajectoryMeta(
                agent_version=str(first.get("agentVersion", "claude-code")),
                skill_versions=dict(first.get("skill_versions", {}) or {}),
                model_version=model_version,
                harness=self.harness,
                started_at=started,
                finished_at=evs[-1].get("timestamp") if evs else None,
                extensions={"transcript_source": path.name},
            )
            records.append(TrajectoryRecord(
                trajectory_id=sid,
                task=str((first.get("message", {}) or {}).get("content", "")),
                steps=steps,
                outcome=outcome,
                cost={"events": float(len(evs))},
                meta=meta,
            ))
        return records

