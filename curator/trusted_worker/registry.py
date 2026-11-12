"""evaluation mode op registry (RSI frozen surface, single source of truth).

op names are the public API (RSI §3.9.4 item 1: additions and removals require updating the contract list and following version evolution).
New operators go first into the op mode registry (``trusted_worker/op/registry.py``, which is a superset);
entering this table requires the §3.9 version-evolution process.
"""
from __future__ import annotations

OP_REGISTRY = {
    "CandidateBatchQualityAssess": {"kind": "quality"},
    "CandidateLogDeduplicate": {"kind": "dedup"},
}
