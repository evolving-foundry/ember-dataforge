#!/usr/bin/env python3
"""Formal DataForge -> Gym terminal-profile environment example.

This example exercises the same four-layer DataForge pipeline as production
without requiring Curator input:

1. a terminal DomainAdapter extracts workspace/replay/task semantics;
2. the profile-neutral EnvironmentModel validates the environment;
3. the registered ``terminal`` ProfileCompiler emits Gym's terminal files;
4. PackageBuilder/gates freeze one immutable package and Publisher writes it.

When a Gym checkout with ``resources_servers/synth_env`` is discoverable
(``--gym-root`` / ``DATA_FORGE_GYM`` / sibling checkout), the example also
loads the zip through Gym's ``load_package_zip`` and executes the golden replay
command against Gym's own ``TerminalProfile``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from data_designer.pipelines.envsynth import gates, package
from data_designer.pipelines.envsynth.serialization import json_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "envsynth-terminal-e2e-example-v1"
SUMMARY_NAME = "example_summary.json"
REF_ID = "terminal-report-e2e"
COMMAND = "cat README.md > report.md"
EXPECTED_FILES = {"/report.md": "release checklist\n"}


class ExampleError(RuntimeError):
    """The example cannot continue or an expected invariant failed."""


def _terminal_domain() -> Dict[str, Any]:
    return {
        "family": "terminal",
        "schemaVersion": "domain-terminal-v1",
        "glossary": ["report", "checklist"],
        "chunks": [
            {
                "chunk_id": "chunk-terminal-1",
                "text": "Create a report file from the README checklist with cat.",
                "role": "seed",
            }
        ],
        "terminal": {
            "workspace": {"/README.md": "release checklist\n"},
            "replay": {
                "entries": [
                    {
                        "cwd": "/",
                        "cmd": COMMAND,
                        "stdout": "",
                        "fs_writes": dict(EXPECTED_FILES),
                        "done": True,
                    }
                ]
            },
            "danger_patterns": ["rm -rf /", "mkfs"],
            "command_max_length": 2000,
            "dynamics": "replay",
        },
    }


def _terminal_draft() -> Dict[str, Any]:
    gold = {"report": EXPECTED_FILES["/report.md"]}
    verifier = {
        "kind": "artifact_compare",
        "payload": {
            "artifact": {"path": "/report.md", "format": "text"},
            "gold": gold,
        },
        "generator_ref": "envsynth@example#seed=42",
    }
    return {
        "tasks": [
            {
                "task_id": "terminal-task-0001",
                "anchor_id": "terminal-anchor-0001",
                "instruction": "Create a report file from the README checklist with cat.",
                "initial_state": {"env": {"cwd": "/"}},
                "gold_solution": gold,
                "negative_solution": {"report": "wrong content\n"},
                "verifier_spec": verifier,
                "expected_files": dict(EXPECTED_FILES),
            }
        ],
        "anchors": [
            {
                "anchor_id": "terminal-anchor-0001",
                "text": "Create a report file from the README checklist with cat.",
                "gold_solution": gold,
                "negative_solution": {"report": "wrong content\n"},
            }
        ],
        "rules": [],
    }


def _discover_gym_root(explicit: Optional[str]) -> Optional[Path]:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    env_root = os.environ.get("DATA_FORGE_GYM")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.append((REPO_ROOT.parent / "AscendAgent-Gym").resolve())
    for candidate in candidates:
        if (candidate / "overlay/resources_servers/synth_env").is_dir():
            return candidate
    if explicit:
        raise ExampleError(
            "--gym-root is not a Gym checkout containing overlay/resources_servers/synth_env: {}".format(
                explicit
            )
        )
    return None


def _load_gym_runtime(gym_root: Path):
    overlay = str(gym_root / "overlay")
    if overlay not in sys.path:
        sys.path.insert(0, overlay)
    from resources_servers.synth_env.dynamics.replay import ReplayEngine
    from resources_servers.synth_env.package_loader import load_package_zip
    from resources_servers.synth_env.profiles.terminal import TerminalProfile

    return {
        "ReplayEngine": ReplayEngine,
        "TerminalProfile": TerminalProfile,
        "load_package_zip": load_package_zip,
    }


def _verify_with_gym(zip_bytes: bytes, gym_root: Path) -> Dict[str, Any]:
    runtime = _load_gym_runtime(gym_root)
    loaded = runtime["load_package_zip"](zip_bytes)
    profile = runtime["TerminalProfile"](loaded.env_spec, loaded.fs_snapshot)
    replay = runtime["ReplayEngine"](loaded.replay)
    fs = profile.snapshot()
    replay.execute(fs, "/", COMMAND)
    results = []
    for task in loaded.tasks.values():
        results.append(
            {
                "taskId": task.task_id,
                "verified": bool(
                    profile.check_success(fs, task.success_check.verifier_spec)
                ),
            }
        )
    failed = [item for item in results if not item["verified"]]
    if failed:
        raise ExampleError("Gym terminal runtime verification failed: {}".format(failed))
    return {
        "runtimeRoot": str(gym_root),
        "refId": loaded.manifest.ref_id,
        "version": loaded.manifest.version,
        "fingerprint": loaded.manifest.fingerprint,
        "taskCount": len(loaded.tasks),
        "tasks": results,
    }


def _prepare_workspace(work_dir: Optional[str]) -> Path:
    if work_dir is None:
        root = Path.cwd() / "terminal-e2e-example"
        suffix = 1
        while root.exists():
            root = Path.cwd() / "terminal-e2e-example-{}".format(suffix)
            suffix += 1
        root.mkdir(parents=True)
        return root.resolve()
    root = Path(work_dir).expanduser().resolve()
    if root.exists():
        raise ExampleError("--work-dir must be a directory that does not exist yet: {}".format(root))
    root.mkdir(parents=True)
    return root


def run_example(work_dir: Optional[str], gym_root: Optional[str]) -> Dict[str, Any]:
    root = _prepare_workspace(work_dir)
    domain = _terminal_domain()
    draft = _terminal_draft()
    (root / "domain.json").write_bytes(json_bytes(domain))
    (root / "draft.json").write_bytes(json_bytes(draft))

    compiled = package.build_env_package(
        domain,
        draft,
        ref_id=REF_ID,
        version=1,
        profile="terminal",
        seed=42,
        mode="template",
        source_files=[],
    )
    report = gates.run_gates(compiled, quality_threshold=0.5)
    if not report["passed"]:
        raise ExampleError("Gate failed: {}".format(report["failedChecks"]))
    frozen = package.finalize_bundle(compiled, report)
    release = package.write_release(str(root / "release"), frozen)
    manifest, files = package.load_env_package(str(root / "release" / "package"))

    target_gym = _discover_gym_root(gym_root)
    gym_result = _verify_with_gym(frozen.zip_bytes, target_gym) if target_gym else None
    summary = {
        "schemaVersion": SUMMARY_SCHEMA,
        "status": "ok",
        "workDir": str(root),
        "dataDesigner": {
            "family": compiled.model.family if compiled.model else None,
            "profile": manifest["profile"],
            "files": sorted(files),
            "taskCount": len(compiled.gym_tasks),
            "fingerprint": manifest["fingerprint"],
            "assetSha256": release["assetSha256"],
            "gatePassed": bool(report["passed"]),
            "releaseDir": release["assetDir"],
            "zip": str(root / "release" / package.ZIP_NAME),
        },
        "gym": {
            "runtimeVerified": gym_result is not None,
            "runtime": gym_result,
        },
    }
    (root / SUMMARY_NAME).write_bytes(json_bytes(summary))
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--gym-root", default=None)
    args = parser.parse_args(argv)
    try:
        summary = run_example(args.work_dir, args.gym_root)
    except ExampleError as error:
        print("example failed: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())