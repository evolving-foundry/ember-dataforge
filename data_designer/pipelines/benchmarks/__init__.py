"""Trusted benchmark case and suite contracts for RSI.

Scoring route principle (2026-10-08): all evaluation sets are environment-agnostic--in-pipeline scoring is always offline
(verifier dispatch / file assertions). ``runners/swe.py`` (official harness subprocess) and
``runners/osworld.py`` (DesktopEnv) are heavy-environment legacy paths and are retired: file bytes are frozen
(historical taskset runtime pins depend on their SHA-256), for historical artifact validation only; new tasksets
must not reference them. See ``runners/DEPRECATED.md`` and the two scoring re-expression addenda under docs/ for alternatives.
"""

from .contracts import FAMILIES, BenchmarkError, canonical_sha256

__all__ = ["BenchmarkError", "FAMILIES", "canonical_sha256"]
