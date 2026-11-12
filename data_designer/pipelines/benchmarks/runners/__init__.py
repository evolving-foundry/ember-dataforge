"""Trusted native benchmark runner entry points.

Retirement and scoring route principle (2026-10-08 user decision): this project is positioned as "generating
additional evaluation data from evaluation sets", and does not care about the runtime environment (for example, docker)
for any supported evaluation set--in-pipeline scoring is always offline (``data_designer.verifiers`` dispatch:
unit_test / artifact_compare, etc.).

Retired runners (in this directory):

- ``swe.py``: subprocess calls the pinned official SWE-bench harness (WSL bridge + internal
  docker container), heavy-environment scoring. Replacement: ``swe_offline.py`` (offline patch application +
  file-state comparison; see the RSI_swe scoring re-expression addendum under ``docs/``).
- ``osworld.py``: subprocess calls pinned OSWorld ``DesktopEnv(provider_name="docker")``
  to replay trajectories, heavy-environment scoring. Replacement: ``osworld_offline.py`` (state-artifact assertions:
  unit_test / artifact_compare dispatch; see the RSI_osworld scoring re-expression addendum under ``docs/``).

Out-of-repo retired runner: ``BENCHMARK_LAB_ROOT/native_derived_runner.py`` (tau);
the external new process calls the pinned official tau2-bench retail tools and ``EnvironmentEvaluator``
to replay candidate action data. Replacement: ``tau_offline.py`` (``{db_state, messages}`` final-state
artifact assertions; see the RSI_tau scoring re-expression addendum under ``docs/``); the bytes of that out-of-repo file
remain frozen and are not pinned in this repo, and historical tau taskset evaluation semantics are unchanged.

Byte-freeze constraint (important): do not modify any byte of a retired runner (including comments).
``benchmarks.runtime.verify_runtime_files`` recomputes the runner file SHA-256 before each candidate evaluation
(``_locked_file(runtime["verifier"])``); historical taskset artifact runtime pins depend on the hashes of the existing bytes.
Retired runners are retained only for historical artifact validation; verifier pins for new tasksets always point to offline runners.
"""
