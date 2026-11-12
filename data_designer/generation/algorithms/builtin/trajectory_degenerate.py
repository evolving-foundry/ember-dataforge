"""Degenerate trajectory detection (FR-B B3): identify infinite loops, repeated actions, and reward-hacking patterns.

Detection results are written to corpus.degenerate_flags (never dropped directly--the honest-ledger principle:
degenerate trajectories are valuable failure evidence), so the policy layer can decide whether to use them for experience synthesis.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .trajectory_data import TrajectoryCorpus
from .trajectory_processing import _Processor


class DegenerateTrajectoryDetectionOperation(_Processor):
    """B3: Detect infinite loops / repeated actions / reward-hacking patterns."""

    name = "degenerate_detection"
    PARAMS_SCHEMA = {
        "params": {
            "repeat_action_threshold": {
                "type": "int", "required": False, "default": 5,
                "description": "Repeated identical actions beyond this count are treated as duplicate actions",
            },
            "loop_action_threshold": {
                "type": "int", "required": False, "default": 3,
                "description": "Repeated identical (action, observation) pairs beyond this count are treated as an infinite loop",
            },
            "reward_hack_signals": {
                "type": "list", "required": False,
                "default": ["skip test", "modify test", "delete test", "mock result", "hardcode"],
                "description": "Reward-hacking signal keywords (case-insensitive)",
            },
        }
    }

    def execute(self, corpus: TrajectoryCorpus) -> TrajectoryCorpus:
        repeat_th = int(self.params["repeat_action_threshold"])
        loop_th = int(self.params["loop_action_threshold"])
        hack_signals = [str(s).lower() for s in self.params["reward_hack_signals"]]

        for rec in corpus.records:
            flags: Dict[str, Any] = {}
            actions: List[str] = [s.action for s in rec.steps if s.action]

        # Consecutive repeated actions
            streak, max_streak = 1, 1
            for i in range(1, len(actions)):
                streak = streak + 1 if actions[i] == actions[i - 1] else 1
                max_streak = max(max_streak, streak)
            if max_streak > repeat_th:
                flags["repeated_action"] = {"max_streak": max_streak, "threshold": repeat_th}

        # Infinite loop: the same (action, observation) appears repeatedly
            pairs = Counter((s.action, s.observation) for s in rec.steps if s.action)
            loops = {f"{a}|{o}": c for (a, o), c in pairs.items() if c > loop_th}
            if loops:
                flags["infinite_loop"] = {"pairs": loops, "threshold": loop_th}

        # reward-hacking pattern: a cheating signal appears in a successful outcome
            if rec.outcome.status == "success":
                text = " ".join([s.action + " " + s.observation for s in rec.steps]).lower()
                hits = [sig for sig in hack_signals if sig in text]
                if hits:
                    flags["reward_hacking"] = {"signals": hits}

            if flags:
                corpus.degenerate_flags[rec.trajectory_id] = flags
        return corpus
