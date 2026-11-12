"""Read-only OSWorld V2 task-class importer with release and asset locks."""

from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from ..contracts import (
    BenchmarkError,
    canonical_sha256,
    file_sha256,
    require_nonempty,
    require_sha,
    require_split,
)
from ..runners.osworld_offline import VERSION as OFFLINE_RUNNER_VERSION
from ..runtime import lock_runtime
from .gaia import in_generation_checkout

OFFLINE_STATE_REGISTRY_SCHEMA = "osworld-offline-state-registry-v1"
OFFLINE_SPEC_KINDS = ("unit_test", "artifact_compare")


def _private_path(value: Any, name: str, *, directory: bool = False) -> Path:
    path = value if isinstance(value, Path) else Path(require_nonempty(value, name))
    if not path.is_absolute() or path.is_symlink():
        raise BenchmarkError(f"OSWorld {name} must be a real absolute path")
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise BenchmarkError(f"OSWorld {name} unavailable")
    if in_generation_checkout(path):
        raise BenchmarkError(f"OSWorld {name} cannot live in this repository")
    return path


def _pin(value: Any, sha_value: Any, name: str) -> Dict[str, str]:
    path = _private_path(value, name)
    sha = require_sha(sha_value, f"{name}.sha256")
    if file_sha256(path) != sha:
        raise BenchmarkError(f"OSWorld {name} SHA-256 drift")
    return {"path": str(path), "sha256": sha}


def _json(path: str, name: str) -> Dict[str, Any]:
    try:
        data = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read OSWorld {name}: {error}") from error
    if not isinstance(data, dict):
        raise BenchmarkError(f"OSWorld {name} must be an object")
    return data


def _sha_from_upstream(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise BenchmarkError(f"OSWorld {name} must be a sha256: digest")
    return require_sha(value.removeprefix("sha256:"), name)


def verify_osworld_asset_snapshot(snapshot: Dict[str, Any]) -> None:
    """Rehash the complete gated asset mirror, once per evaluation preflight."""
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("files"), dict):
        raise BenchmarkError("OSWorld source snapshot files missing")
    files = snapshot["files"]
    inventory_pin = files.get("asset_inventory")
    if not isinstance(inventory_pin, dict):
        raise BenchmarkError("OSWorld asset inventory pin missing")
    inventory_pin = _pin(
        inventory_pin.get("path"), inventory_pin.get("sha256"),
        "asset inventory",
    )
    root = _private_path(snapshot.get("asset_root"), "asset root", directory=True)
    inventory = _json(inventory_pin["path"], "asset inventory")
    entries = inventory.get("files")
    if (
        inventory.get("schema_version") != "osworld-asset-inventory-v1"
        or not isinstance(entries, dict)
        or not entries
    ):
        raise BenchmarkError("OSWorld asset inventory schema invalid")
    for relative, record in entries.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(record, dict)
            or type(record.get("size")) is not int
            or record["size"] < 0
        ):
            raise BenchmarkError("OSWorld asset inventory entry invalid")
        target = root / relative
        if not target.resolve().is_relative_to(root.resolve()):
            raise BenchmarkError("OSWorld asset path escapes root")
        if target.is_symlink() or not target.is_file():
            raise BenchmarkError(f"OSWorld asset unavailable: {relative}")
        sha = require_sha(record.get("sha256"), "asset.sha256")
        if target.stat().st_size != record["size"] or file_sha256(target) != sha:
            raise BenchmarkError(f"OSWorld asset SHA-256 drift: {relative}")
    nodes = list(root.rglob("*"))
    if any(path.is_symlink() for path in nodes):
        raise BenchmarkError("OSWorld asset root cannot contain links")
    actual = {
        path.relative_to(root).as_posix() for path in nodes if path.is_file()
    }
    if actual != set(entries):
        raise BenchmarkError("OSWorld asset inventory file set drift")


def _class_metadata(path: Path, task_id: str) -> Dict[str, Any]:
    try:
        syntax = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise BenchmarkError(f"OSWorld task class invalid: {task_id}") from error
    matches = []
    for node in syntax.body:
        if not isinstance(node, ast.ClassDef):
            continue
        values: Dict[str, Any] = {}
        methods = set()
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.add(child.name)
            if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(
                child.targets[0], ast.Name
            ):
                try:
                    values[child.targets[0].id] = ast.literal_eval(child.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
        if values.get("id") == task_id:
            matches.append((values, methods))
    if len(matches) != 1:
        raise BenchmarkError(f"OSWorld task {task_id} class ID ambiguous or missing")
    values, methods = matches[0]
    instruction = require_nonempty(values.get("instruction"), "OSWorld instruction")
    snapshot = require_nonempty(values.get("snapshot"), "OSWorld snapshot")
    apps = values.get("related_apps")
    if not isinstance(apps, list) or not all(
        isinstance(app, str) and app for app in apps
    ):
        raise BenchmarkError(f"OSWorld task {task_id} related_apps invalid")
    if "evaluate" not in methods and "evaluator" not in values:
        raise BenchmarkError(f"OSWorld task {task_id} has no evaluator")
    return {
        "instruction": instruction,
        "snapshot": snapshot,
        "related_apps": apps,
        "has_setup": "setup" in methods,
        "has_evaluate": "evaluate" in methods,
    }


def _offline_admission(task_id: str, entry: Any) -> Dict[str, Any]:
    """O4 subset admission from the offline state registry (double criterion)."""
    if not isinstance(entry, dict) or "admission" not in entry:
        raise BenchmarkError(f"OSWorld offline state entry invalid: {task_id}")
    if entry["admission"] == "excluded":
        return {
            "admission": "excluded",
            "reason": require_nonempty(entry.get("reason"), f"offline reason {task_id}"),
        }
    if entry["admission"] != "expressible":
        raise BenchmarkError(f"OSWorld offline admission invalid: {task_id}")
    files = entry.get("initial_files")
    if not isinstance(files, dict) or not files:
        raise BenchmarkError(
            f"OSWorld offline expressible entry needs initial_files: {task_id}"
        )
    pinned: Dict[str, Dict[str, str]] = {}
    for relative, pin in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise BenchmarkError(f"OSWorld offline initial file path invalid: {task_id}")
        if not isinstance(pin, dict):
            raise BenchmarkError(f"OSWorld offline initial file pin invalid: {task_id}")
        pinned[relative] = _pin(
            pin.get("path"), pin.get("sha256"),
            f"offline initial file {relative} ({task_id})",
        )
    return {"admission": "expressible", "initial_files": pinned}


def assert_osworld_runtime_assets(
    runtime: Dict[str, Any], source_snapshot: Dict[str, Any], case: Dict[str, Any]
) -> None:
    assets = runtime.get("assets") if isinstance(runtime, dict) else None
    files = source_snapshot.get("files") if isinstance(source_snapshot, dict) else None
    task_pin = (
        case.get("osworld", {}).get("task_class")
        if isinstance(case, dict) else None
    )
    if (
        not isinstance(assets, list)
        or not isinstance(files, dict)
        or not isinstance(task_pin, dict)
    ):
        raise BenchmarkError("OSWorld runtime/source assets missing")
    for pin in [*files.values(), task_pin]:
        if pin not in assets:
            raise BenchmarkError("OSWorld runtime missing pinned source asset")


def import_osworld_snapshot(
    release_path: str,
    release_sha256: str,
    *,
    task_hashes_path: str,
    task_hashes_sha256: str,
    task_class_dir: str,
    asset_inventory_path: str,
    asset_inventory_sha256: str,
    asset_root: str,
    split: str,
    license_record: Dict[str, Any],
    selection_path: Optional[str] = None,
    selection_sha256: Optional[str] = None,
    selection_use: Optional[str] = None,
    runtime_registry_path: Optional[str] = None,
    runtime_registry_sha256: Optional[str] = None,
    offline_state_registry_path: Optional[str] = None,
    offline_state_registry_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Import a matched V2 release without executing gated task Python."""
    files = {
        "release": _pin(release_path, release_sha256, "release"),
        "task_hashes": _pin(
            task_hashes_path, task_hashes_sha256, "task hashes"
        ),
        "asset_inventory": _pin(
            asset_inventory_path, asset_inventory_sha256, "asset inventory"
        ),
    }
    if (selection_path is None) != (selection_sha256 is None):
        raise BenchmarkError("OSWorld selection path and SHA-256 must be paired")
    if selection_path is not None:
        files["selection"] = _pin(selection_path, selection_sha256, "selection")
    split = require_split(split)
    if split != "holdout" and (
        "selection" not in files or selection_use != "derivation"
    ):
        raise BenchmarkError("OSWorld derivation needs explicit pinned selection")
    if split == "holdout" and selection_use == "derivation":
        raise BenchmarkError("OSWorld derivation selection cannot be holdout")
    if selection_use not in (None, "derivation", "benchmark"):
        raise BenchmarkError("OSWorld selection_use invalid")
    if (
        not isinstance(license_record, dict)
        or not isinstance(license_record.get("id"), str)
        or not license_record["id"].strip()
        or type(license_record.get("derive_allowed")) is not bool
        or license_record.get("redistribute_allowed") is not False
        or not isinstance(license_record.get("access_conditions"), str)
        or not license_record["access_conditions"].strip()
    ):
        raise BenchmarkError("OSWorld gated-data license decision required")
    release = _json(files["release"]["path"], "release")
    if release.get("schema_version") != 1 or release.get("status") != "active":
        raise BenchmarkError("OSWorld release must be active schema v1")
    release_name = require_nonempty(release.get("release"), "OSWorld release name")
    for section in (
        "osworld_code", "website_code", "tasks", "assets", "provider_images"
    ):
        if not isinstance(release.get(section), dict):
            raise BenchmarkError(f"OSWorld release missing {section}")
    hash_record = release.get("task_hash_manifest")
    if not isinstance(hash_record, dict) or _sha_from_upstream(
        hash_record.get("sha256"), "task hash manifest"
    ) != files["task_hashes"]["sha256"]:
        raise BenchmarkError("OSWorld task hash manifest release mismatch")
    hashes = _json(files["task_hashes"]["path"], "task hashes")
    task_files = hashes.get("files")
    if not isinstance(task_files, dict) or not task_files or hash_record.get(
        "task_count"
    ) != len(task_files):
        raise BenchmarkError("OSWorld task hash manifest count mismatch")
    task_dir = _private_path(task_class_dir, "task class dir", directory=True)
    if {path.name for path in task_dir.glob("task_*.py")} != set(task_files):
        raise BenchmarkError("OSWorld local task class set drift")
    task_pins: Dict[str, Dict[str, str]] = {}
    for filename, record in task_files.items():
        if (
            not isinstance(filename, str)
            or not filename.startswith("task_")
            or not filename.endswith(".py")
            or Path(filename).name != filename
            or not isinstance(record, dict)
            or type(record.get("size")) is not int
        ):
            raise BenchmarkError("OSWorld task hash entry invalid")
        path = task_dir / filename
        pin = _pin(path, record.get("sha256"), filename)
        if path.stat().st_size != record["size"]:
            raise BenchmarkError(f"OSWorld task class size drift: {filename}")
        task_pins[filename.removeprefix("task_").removesuffix(".py")] = pin
    if len(task_pins) != len(task_files):
        raise BenchmarkError("OSWorld task IDs duplicated")
    source_snapshot = {
        **files["release"],
        "format": "osworld-v2-task-classes-v1",
        "release": release_name,
        "task_class_dir": str(task_dir),
        "asset_root": str(_private_path(asset_root, "asset root", directory=True)),
        "selection_use": selection_use,
        "files": files,
    }
    verify_osworld_asset_snapshot(source_snapshot)
    if "selection" in files:
        selection = _json(files["selection"]["path"], "selection")
        ids = selection.get("tasks")
        if not isinstance(ids, list) or not ids or not all(
            isinstance(task_id, str) and task_id for task_id in ids
        ) or len(ids) != len(set(ids)):
            raise BenchmarkError("OSWorld selection.tasks invalid")
    else:
        ids = sorted(task_pins)
    if not set(ids) <= set(task_pins):
        raise BenchmarkError("OSWorld selection has unknown task IDs")
    runtimes: Dict[str, Any] | None = None
    registry_sha: str | None = None
    if runtime_registry_path is not None or runtime_registry_sha256 is not None:
        registry_pin = _pin(
            runtime_registry_path, runtime_registry_sha256, "runtime registry"
        )
        registry_sha = registry_pin["sha256"]
        registry = _json(registry_pin["path"], "runtime registry")
        if (
            registry.get("schema_version") != "benchmark-runtime-registry-v1"
            or not isinstance(registry.get("cases"), dict)
        ):
            raise BenchmarkError("invalid OSWorld runtime registry schema")
        runtimes = registry["cases"]
        if set(runtimes) != set(ids):
            raise BenchmarkError("OSWorld runtimes must match selected task IDs")
    offline_states: Dict[str, Any] | None = None
    if (
        offline_state_registry_path is not None
        or offline_state_registry_sha256 is not None
    ):
        offline_pin = _pin(
            offline_state_registry_path, offline_state_registry_sha256,
            "offline state registry",
        )
        registry = _json(offline_pin["path"], "offline state registry")
        if (
            registry.get("schema_version") != OFFLINE_STATE_REGISTRY_SCHEMA
            or not isinstance(registry.get("cases"), dict)
        ):
            raise BenchmarkError("invalid OSWorld offline state registry schema")
        offline_states = registry["cases"]
        if set(offline_states) != set(ids):
            raise BenchmarkError(
                "OSWorld offline state registry must cover selected task IDs"
            )
        source_snapshot["offline_state_registry"] = offline_pin
    cases: List[Dict[str, Any]] = []
    for task_id in ids:
        task_pin = task_pins[task_id]
        metadata = _class_metadata(Path(task_pin["path"]), task_id)
        case = {
            "source_task_id": task_id,
            "task_text": metadata["instruction"],
            "initial_state_sha256": canonical_sha256(
                {
                    "release": release_name,
                    "snapshot": metadata["snapshot"],
                    "asset_inventory_sha256": files["asset_inventory"]["sha256"],
                    "provider_images": release["provider_images"],
                }
            ),
            "expected_state_sha256": canonical_sha256(
                {"task_class_sha256": task_pin["sha256"]}
            ),
            "osworld": {
                "task_class": task_pin,
                "snapshot": metadata["snapshot"],
                "related_apps": metadata["related_apps"],
                "has_setup": metadata["has_setup"],
                "has_evaluate": metadata["has_evaluate"],
            },
        }
        if offline_states is not None:
            case["osworld"]["offline"] = _offline_admission(
                task_id, offline_states[task_id]
            )
        if runtimes is not None:
            runtime = lock_runtime(runtimes[task_id])
            by_path = {asset["path"]: asset for asset in runtime["assets"]}
            for pin in [*files.values(), task_pin]:
                existing = by_path.get(pin["path"])
                if existing is not None and existing != pin:
                    raise BenchmarkError("OSWorld runtime asset SHA-256 conflict")
                by_path[pin["path"]] = pin
            runtime["assets"] = list(by_path.values())
            case["runtime"] = runtime
        cases.append(case)
    return {
        "schema_version": "benchmark-source-v1",
        "family": "osworld",
        "release": release_name,
        "split": split,
        "license": dict(license_record),
        "source_snapshot": source_snapshot,
        "runtime_registry_sha256": registry_sha,
        "cases": cases,
    }


def verify_osworld_source_manifest(source: Dict[str, Any]) -> None:
    snapshot = source.get("source_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("format") != "osworld-v2-task-classes-v1"
        or not isinstance(snapshot.get("files"), dict)
    ):
        raise BenchmarkError("OSWorld source needs a pinned V2 release")
    files = snapshot["files"]
    if set(files) not in (
        {"release", "task_hashes", "asset_inventory"},
        {"release", "task_hashes", "asset_inventory", "selection"},
    ):
        raise BenchmarkError("OSWorld source file pins incomplete")
    offline_registry = snapshot.get("offline_state_registry")
    if offline_registry is not None and (
        not isinstance(offline_registry, dict)
        or not offline_registry.get("path")
        or not offline_registry.get("sha256")
    ):
        raise BenchmarkError("OSWorld offline state registry pin invalid")
    expected = import_osworld_snapshot(
        files["release"]["path"], files["release"]["sha256"],
        task_hashes_path=files["task_hashes"]["path"],
        task_hashes_sha256=files["task_hashes"]["sha256"],
        task_class_dir=snapshot.get("task_class_dir"),
        asset_inventory_path=files["asset_inventory"]["path"],
        asset_inventory_sha256=files["asset_inventory"]["sha256"],
        asset_root=snapshot.get("asset_root"),
        split=source.get("split"),
        license_record=source.get("license"),
        selection_path=files.get("selection", {}).get("path"),
        selection_sha256=files.get("selection", {}).get("sha256"),
        selection_use=snapshot.get("selection_use"),
        offline_state_registry_path=(
            offline_registry.get("path") if offline_registry is not None else None
        ),
        offline_state_registry_sha256=(
            offline_registry.get("sha256") if offline_registry is not None else None
        ),
    )
    if (
        source.get("release") != expected["release"]
        or snapshot != expected["source_snapshot"]
    ):
        raise BenchmarkError("OSWorld source release metadata drift")
    cases = source.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected["cases"]):
        raise BenchmarkError("OSWorld source cases incomplete")
    for actual, fresh in zip(cases, expected["cases"]):
        if not isinstance(actual, dict):
            raise BenchmarkError("OSWorld source case invalid")
        for key, value in fresh.items():
            if actual.get(key) != value:
                raise BenchmarkError(f"OSWorld source metadata drift: {key}")
        if "runtime" in actual:
            assert_osworld_runtime_assets(actual["runtime"], snapshot, actual)


def _clean_relative(relative: str) -> str:
    parts = PurePosixPath(relative).parts
    if (
        not relative.strip()
        or relative.startswith("/")
        or "\\" in relative
        or ".." in parts
        or any(part in ("", ".") for part in parts)
    ):
        raise BenchmarkError(f"must be a clean relative path: {relative!r}")
    return relative


def _file_dict(value: Any, name: str) -> Dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise BenchmarkError(f"osworld_offline {name} must be a non-empty mapping")
    normalized: Dict[str, str] = {}
    for relative, content in value.items():
        _clean_relative(str(relative))
        if not isinstance(content, str):
            raise BenchmarkError(
                f"osworld_offline {name}.{relative} must be string content"
            )
        normalized[str(relative)] = content
    return normalized


def derive_osworld_offline_spec(
    source: Dict[str, Any], source_case: Dict[str, Any], directive: Any
) -> Dict[str, Any]:
    """O5 pregeneration: build the offline verifier_spec from the state directive.

    The trusted directive supplies the offline-rebuildable initial state
    (``setup_files``) and the expected final state; this helper turns them into
    a ``unit_test`` spec whose assertion script is anchored on the expected
    state artifacts, or — for regular-shaped artifacts — an ``artifact_compare``
    spec. Only hashes and the trusted script travel into the manifest; the
    source-side oracle never does.
    """
    if not isinstance(directive, dict):
        raise BenchmarkError("osworld_offline directive must be an object")
    allowed = {
        "artifact", "expected_files", "setup_files", "gold", "fields", "timeout_s",
    }
    if set(directive) - allowed:
        raise BenchmarkError(f"osworld_offline directive accepts only {sorted(allowed)}")
    if ("expected_files" in directive) == ("gold" in directive):
        raise BenchmarkError(
            "osworld_offline directive needs exactly one of expected_files or gold"
        )
    artifact = directive.get("artifact")
    if artifact is None:
        artifact = {"path": "state_artifact.json"}
    if (
        not isinstance(artifact, dict)
        or set(artifact) - {"path"}
        or not isinstance(artifact.get("path"), str)
    ):
        raise BenchmarkError("osworld_offline artifact must be {path: relative}")
    relative = _clean_relative(artifact["path"])
    timeout = directive.get("timeout_s", 30.0)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise BenchmarkError("osworld_offline timeout_s must be a positive number")
    metadata = source_case.get("osworld")
    offline = metadata.get("offline") if isinstance(metadata, dict) else None
    task_id = source_case.get("source_task_id")
    if not isinstance(metadata, dict) or not isinstance(offline, dict) or (
        offline.get("admission") != "expressible"
    ):
        reason = offline.get("reason") if isinstance(offline, dict) else "unknown"
        raise BenchmarkError(
            f"osworld task {task_id!r} is not admitted for offline derivation "
            f"({reason})"
        )
    release = require_nonempty(source.get("release"), "release")
    if "gold" in directive:
        gold = directive["gold"]
        if not isinstance(gold, dict) or not gold:
            raise BenchmarkError("osworld_offline gold must be a non-empty object")
        fields = directive.get("fields") or {}
        if not isinstance(fields, dict):
            raise BenchmarkError("osworld_offline fields must be an object")
        spec = {
            "kind": "artifact_compare",
            "generator_ref": f"osworld-offline-state@{release}",
            "payload": {
                "artifact": {"path": relative},
                "gold": gold,
                "fields": fields,
            },
        }
        expected_input: Dict[str, Any] = {"gold": gold, "fields": fields}
        setup: Dict[str, str] = {}
    else:
        expected = _file_dict(directive["expected_files"], "expected_files")
        setup = (
            _file_dict(directive["setup_files"], "setup_files")
            if "setup_files" in directive
            else {}
        )
        literal = json.dumps(json.dumps(expected, sort_keys=True), ensure_ascii=False)
        test_script = (
            "import json\n"
            f"EXPECTED = json.loads({literal})\n"
            "missing, wrong = [], []\n"
            "for relative, content in sorted(EXPECTED.items()):\n"
            "    try:\n"
            "        with open(relative, encoding='utf-8') as handle:\n"
            "            actual = handle.read()\n"
            "    except OSError:\n"
            "        missing.append(relative)\n"
            "        continue\n"
            "    if actual != content:\n"
            "        wrong.append(relative)\n"
            "assert not missing and not wrong, (\n"
            "    'missing=%r wrong=%r' % (missing, wrong))\n"
        )
        spec = {
            "kind": "unit_test",
            "generator_ref": f"osworld-offline-state@{release}",
            "payload": {
                "artifact": {"path": relative},
                "setup_files": setup,
                "test_script": test_script,
                "timeout_s": float(timeout),
            },
        }
        expected_input = {"expected_files": expected}
    return {
        "verifier_spec": spec,
        "initial_state_sha256": canonical_sha256(
            {"snapshot": metadata.get("snapshot"), "setup_files": setup}
        ),
        "expected_state_sha256": canonical_sha256(expected_input),
    }


def assert_osworld_offline_build(
    runtime: Dict[str, Any], private: Dict[str, Any], source_case: Dict[str, Any]
) -> None:
    """M2 gate: every new osworld taskset pins the offline runner and a real spec."""
    metadata = source_case.get("osworld")
    offline = metadata.get("offline") if isinstance(metadata, dict) else None
    task_id = source_case.get("source_task_id")
    if not isinstance(offline, dict) or offline.get("admission") != "expressible":
        reason = offline.get("reason") if isinstance(offline, dict) else "unknown"
        raise BenchmarkError(
            f"osworld task {task_id!r} is excluded from offline derivation "
            f"({reason}); import with an offline state registry"
        )
    verifier = runtime.get("verifier")
    if (
        not isinstance(verifier, dict)
        or verifier.get("version") != OFFLINE_RUNNER_VERSION
    ):
        raise BenchmarkError(
            "osworld derivation must pin the offline runner "
            f"({OFFLINE_RUNNER_VERSION}); the DesktopEnv runner is retired"
        )
    spec = private.get("verifier_spec") if isinstance(private, dict) else None
    if not isinstance(spec, dict) or spec.get("kind") not in OFFLINE_SPEC_KINDS:
        raise BenchmarkError(
            "osworld trusted case lacks a unit_test/artifact_compare "
            f"verifier_spec: {task_id}"
        )
