from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from data_designer.contracts.trajectories import StepRecord, TaskOutcome, TrajectoryMeta, TrajectoryRecord
from data_designer.contracts.provenance import content_digest as fingerprint_of
logger = logging.getLogger(__name__)

def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _epoch_ms_to_iso(ms: Optional[int]) -> Optional[str]:
    """Convert real DSH epoch milliseconds -> ISO-8601 UTC (the unified trajectory meta time format)."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _read_dsh_log(path: Path) -> List[Dict]:
    """Read a real DSH session log (.zst/.zstd compressed or plain-text jsonl), one event dict per line."""
    magic = b""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic == b"\x28\xb5\x2f\xfd":  # zstd frame magic (independent of suffix)
        try:
            import zstandard
        except ImportError as e:
            raise RuntimeError(
                f"Real DSH sessions are zstd-compressed ({path}); the zstandard package is required: "
                "pip install zstandard") from e
        with open(path, "rb") as fh:
            try:
                text = zstandard.ZstdDecompressor().stream_reader(fh).read().decode(
                    "utf-8")
            except zstandard.ZstdError as e:
                # Truncated/corrupt frames (such as half-written files left by a DSH session crash)
                raise RuntimeError(f"zstd decompression failed {path}: {e}") from e
    else:
        text = path.read_text(encoding="utf-8")
    events: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("skip malformed line in %s: %s", path, e)
    return events


def _content_text(data: Optional[Dict]) -> str:
    """Concatenate all text items in user/message data.content[] (real DSH task input)."""
    parts = []
    for c in (data or {}).get("content") or []:
        if c.get("type") == "text":
            parts.append(str(c.get("text", "")))
    return "\n".join(parts)


_REAL_STATUS_MAP = {
    "completed": "success",
    "error": "failure",
    "aborted": "aborted",
    "interrupted": "aborted",
}


class DshReader:
    """dsh trajectory reader; never installs host resources."""


    def __init__(self, workspace: str):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "sessions"


    def export_trajectories(
        self,
        scope: Optional[Dict] = None,
        since: Optional[datetime] = None,
    ) -> List[TrajectoryRecord]:
        real_dir = self._real_sessions_dir()
        if real_dir is not None:
            return self._export_real_sessions(real_dir, scope=scope, since=since)
        records: List[TrajectoryRecord] = []
        if not self.sessions_dir.exists():
            return records
        for path in sorted(self.sessions_dir.glob("*.jsonl")):
            records.extend(self._load_session_file(path, since))
        return records


    def _real_sessions_dir(self) -> Optional[Path]:
        """Automatically detect the real DSH home layout: sessions/*/session-<id>/session.jsonl*.

        The session directory name may be session-<uuid> or a bare uuid; if detected, return the sessions
        directory; otherwise None (use the synthetic fixture path).
        """
        if not self.sessions_dir.is_dir():
            return None
        if any(self.sessions_dir.glob("*/*/session.jsonl*")):
            return self.sessions_dir
        return None


    def _export_real_sessions(
        self,
        sessions_dir: Path,
        scope: Optional[Dict],
        since: Optional[datetime],
    ) -> List[TrajectoryRecord]:
        """Real DSH: normalize at turn granularity (one turn = one trajectory).

        scope={"cwd": ...} can filter by session cwd (select sessions for a workspace);
        since filters by turn start time.
        """
        cwd_filter = (scope or {}).get("cwd")
        records: List[TrajectoryRecord] = []
        for path in sorted(sessions_dir.glob("*/*/session.jsonl*")):
            try:
                events = _read_dsh_log(path)
            except (OSError, RuntimeError, ValueError) as e:
                # ValueError covers UnicodeDecodeError (binary garbage files that are not zstd)
                logger.warning("skip unreadable DSH session %s: %s", path, e)
                continue
            records.extend(
                self._real_session_to_trajectories(events, cwd_filter, since))
        return records


    def _real_session_to_trajectories(
        self,
        events: List[Dict],
        cwd_filter: Optional[str],
        since: Optional[datetime],
    ) -> List[TrajectoryRecord]:
        header = next(
            (e for e in events if e.get("type") == "session"), None)
        session_id = str((header or {}).get("id")
                         or (header or {}).get("data", {}).get("id", ""))
        session_cwd = str((header or {}).get("cwd")
                          or (header or {}).get("data", {}).get("cwd", ""))
        if cwd_filter and session_cwd != cwd_filter:
            return []

        # model_version: the model from the most recent request/header before the turn starts
        model_headers = [
            (e.get("time"),
             ((e.get("data") or {}).get("header") or {}).get("config") or {})
            for e in events if e.get("type") == "request/header"]
        model_headers.sort(key=lambda pair: pair[0] or 0)

        # Split by the turn/start..turn/end window
        turns: List[Dict] = []
        current: Optional[Dict] = None
        for e in events:
            t = e.get("type")
            if t == "turn/start":
                current = {"start": e, "events": []}
                turns.append(current)
            elif t == "turn/end" and current is not None:
                current["end"] = e
                current = None
            elif current is not None:
                current["events"].append(e)

        records: List[TrajectoryRecord] = []
        for n, turn in enumerate(turns, start=1):
            end_ev = turn.get("end")
            start_ms = turn["start"].get("time")
            end_ms = (end_ev or {}).get("time")
            started = (_epoch_ms_to_iso(start_ms) if start_ms else None)
            # since filtering uses exact epoch milliseconds (ISO strings truncate to seconds, avoiding same-second false matches)
            if since is not None and start_ms is not None:
                if datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc) < since:
                    continue

            reason = ((end_ev or {}).get("data") or {}).get("reason") or {}
            if end_ev is None:
                status = "aborted"  # No turn/end: in flight or the session was interrupted
            else:
                status = _REAL_STATUS_MAP.get(reason.get("kind"), "failure")
            error_code = ((reason.get("error") or {}).get("code")) \
                if reason.get("kind") == "error" else None

            task = next((
                _content_text(e.get("data"))
                for e in turn["events"] if e.get("type") == "user/message"
            ), "")
            if not task:
                continue

            # Pair tool/call <-> tool/result by callId (only within the turn window)
            result_by_call: Dict[str, Dict] = {}
            for e in turn["events"]:
                if e.get("type") != "tool/result":
                    continue
                msg = (e.get("data") or {}).get("message") or {}
                items = msg.get("content") or []
                call_id = next(
                    (c.get("toolCallId") for c in items if c.get("toolCallId")),
                    (msg.get("source") or {}).get("callId"))
                if call_id:
                    result_by_call[call_id] = e

            steps: List[StepRecord] = []
            for e in turn["events"]:
                if e.get("type") != "tool/call":
                    continue
                d = e.get("data") or {}
                call_id = str(d.get("callId", ""))
                res = result_by_call.get(call_id)
                observation, is_error = "", False
                if res is not None:
                    msg = (res.get("data") or {}).get("message") or {}
                    texts = []
                    for c in msg.get("content") or []:
                        for sub in c.get("content") or []:
                            if sub.get("type") == "text":
                                texts.append(str(sub.get("text", "")))
                        if c.get("type") == "text":
                            texts.append(str(c.get("text", "")))
                    observation = "\n".join(texts)
                    is_error = any(
                        bool(c.get("isError")) for c in msg.get("content") or [])
                steps.append(StepRecord(
                    index=len(steps),
                    action=str(d.get("name", "")),
                    observation=observation,
                    tool=str(d.get("name", "")) or None,
                    error=is_error,
                ))

            model = ""
            for t_ms, cfg in reversed(model_headers):
                if t_ms is None or (start_ms is not None and t_ms <= start_ms):
                    model = str(cfg.get("model", ""))
                    break
            if not model and model_headers:
                # Fallback: the turn precedes the first request/header (for example, session resume) -> use the first header
                model = str(model_headers[0][1].get("model", ""))
            # A4 requires model_version to be non-empty; explicitly mark unknown when the session has no LLM requests at all
            model = model or "unknown"

            meta = TrajectoryMeta(
                agent_version="dsh",
                model_version=model,
                harness=self.harness,
                started_at=started,
                finished_at=_epoch_ms_to_iso(end_ms),
                extensions={"session_id": session_id, "cwd": session_cwd,
                            "turn": n},
            )
            records.append(TrajectoryRecord(
                trajectory_id=f"{session_id}#turn{n}",
                task=task,
                steps=steps,
                outcome=TaskOutcome(status=status, error_category=error_code),
                cost={"events": float(len(turn["events"])), "steps": float(len(steps))},
                meta=meta,
            ))
        return records


    def _load_session_file(
        self, path: Path, since: Optional[datetime]
    ) -> List[TrajectoryRecord]:
        events: List[Dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skip unreadable session file %s: %s", path, e)
            return []

        sessions: Dict[str, List[Dict]] = {}
        for ev in events:
            sid = ev.get("session_id")
            if sid:
                sessions.setdefault(sid, []).append(ev)

        records: List[TrajectoryRecord] = []
        for sid, evs in sessions.items():
            evs.sort(key=lambda e: e.get("ts", ""))
            task_evs = [e for e in evs if e.get("event") == "task_start"]
            if not task_evs:
                continue
            task_ev = task_evs[0]
            started = _parse_ts(task_ev.get("ts"))
            if since is not None and started is not None and started < since:
                continue

            steps: List[StepRecord] = []
            outcome = TaskOutcome()
            for ev in evs:
                kind = ev.get("event")
                if kind == "action":
                    steps.append(
                        StepRecord(
                            index=int(ev.get("index", len(steps))),
                            action=str(ev.get("action", "")),
                            observation=str(ev.get("observation", "")),
                            tool=ev.get("tool"),
                            error=bool(ev.get("error", False)),
                        )
                    )
                elif kind == "task_end":
                    status = ev.get("status", "failure")
                    if status not in ("success", "failure", "aborted"):
                        status = "failure"
                    outcome = TaskOutcome(
                        status=status,
                        error_category=ev.get("error_category"),
                        corrections=int(ev.get("corrections", 0)),
                        summary=str(ev.get("summary", "")),
                    )

            meta = TrajectoryMeta(
                agent_version=str(task_ev.get("agent_version", "unknown")),
                skill_versions=dict(task_ev.get("skill_versions", {}) or {}),
                model_version=str(task_ev.get("model_version", "")),
                harness=self.harness,
                started_at=task_ev.get("ts"),
                finished_at=evs[-1].get("ts"),
            )
            records.append(
                TrajectoryRecord(
                    trajectory_id=sid,
                    task=str(task_ev.get("task", "")),
                    steps=steps,
                    outcome=outcome,
                    cost={"events": float(len(evs))},
                    meta=meta,
                )
            )
        return records

