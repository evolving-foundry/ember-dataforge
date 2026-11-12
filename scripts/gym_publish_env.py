"""Publish one Gym-native DataForge package into a Gym overlay checkout.

stdin (v0.1 store contract)::

    {"storePath": "/abs/path/to/artifact-store",
     "refId": "office-power",
     "version": 1,                     # optional, defaults to latest
     "name": "synth-office-power"}

stdin (file mirror used by the compile-time CLI)::

    {"compileResultPath": "/abs/path/to/compile-result.json",
     "name": "synth-office-power"}

The release package itself is already in Gym's schema
(``manifest.json`` / ``env_spec.json`` / ``tasks.jsonl`` / office assets); this
script never rewrites package content or fingerprints.  It only verifies the
frozen release, copies the loadable zip into
``overlay/benchmarks/<name>/assets/`` under a versioned immutable filename, and
writes the Gym benchmark shell (``prepare.py`` / ``config.yaml`` /
``manifest.yaml`` / prompt).  ``--register`` additionally updates the overlay
manifest and ``package_paths`` through Gym's own manifest tool.

The primary input is the ``ArtifactCatalog`` pair written by the v0.1 worker; a
file-mirror compile result is also accepted for CI.  No runner daemon is
started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_designer.pipelines.envsynth.package import (  # noqa: E402
    EnvPackageError,
    ZIP_NAME,
    load_env_package,
    zip_sha256,
)
from data_designer.pipelines.envsynth import artifacts, gym_contract  # noqa: E402
from data_designer.pipelines.envsynth.profile_compilers import (  # noqa: E402
    get_profile_compiler,
)
from data_designer.pipelines.envsynth.serialization import (  # noqa: E402
    json_bytes as _json_bytes,
    yaml_bytes as _yaml_bytes,
)


PLACEHOLDER_MARKERS = ("placeholder", "todo", "tbd", "fixme", "<...>", "<placeholder>")


def _safe_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    if not sanitized:
        raise ValueError(f"Name {name!r} is empty after sanitization and cannot be used as a directory name")
    return sanitized


def _write(path: Path, data: bytes, written: List[str], gym_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != data:
            raise ValueError(
                "Published asset already exists with different content; refusing to overwrite: {}".format(path)
            )
    else:
        temp_path = path.with_name(path.name + ".tmp")
        temp_path.write_bytes(data)
        os.replace(str(temp_path), str(path))
    written.append(str(path.relative_to(gym_root)).replace("\\", "/"))


def _write_replace(
    path: Path, data: bytes, written: List[str], gym_root: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == data:
        written.append(str(path.relative_to(gym_root)).replace("\\", "/"))
        return
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_bytes(data)
    os.replace(str(temp_path), str(path))
    written.append(str(path.relative_to(gym_root)).replace("\\", "/"))


def _load_compile_result_from_store(payload: Dict[str, Any]) -> Dict[str, Any]:
    store_path = payload.get("storePath")
    if not isinstance(store_path, str) or not store_path:
        raise ValueError("payload.storePath must be a non-empty string")
    store_file = Path(store_path)
    if not store_file.is_absolute():
        raise ValueError("payload.storePath must be an absolute path")
    ref_id = payload.get("refId")
    if not isinstance(ref_id, str) or not ref_id:
        raise ValueError("payload.refId must be a non-empty string")
    version = payload.get("version")
    if version is not None:
        try:
            version = int(version)
        except (TypeError, ValueError) as error:
            raise ValueError("payload.version must be an integer") from error
        if version < 1:
            raise ValueError("payload.version must be >= 1")

    from data_designer.storage.catalog import ArtifactCatalog

    store = ArtifactCatalog(str(store_file))
    try:
        release_artifact = store.get_artifact("env_release", ref_id, version)
        if release_artifact is None:
            raise ValueError(
                "Artifact directory does not contain env_release/{}/v{}".format(
                    ref_id, version if version is not None else "latest"
                )
            )
        release_payload = release_artifact["payload"]
        if release_payload.get("fingerprint") != release_artifact["fingerprint"]:
            raise ValueError("env_release artifact fingerprint does not match payload")
        taskset_ref = release_payload.get("taskset")
        if not isinstance(taskset_ref, dict):
            raise ValueError("env_release.payload.taskset is missing")
        taskset_artifact = store.get_artifact(
            "taskset", str(taskset_ref.get("refId")), int(taskset_ref.get("version"))
        )
        if taskset_artifact is None:
            raise ValueError("Artifact directory does not contain the corresponding taskset artifact")
        if taskset_artifact["fingerprint"] != taskset_ref.get("fingerprint"):
            raise ValueError("taskset artifact fingerprint does not match env_release.taskset")
        release_info = {
            "refId": release_payload["refId"],
            "version": int(release_payload["version"]),
            "fingerprint": release_payload["fingerprint"],
            "assetDir": release_payload["assetDir"],
            "assetSha256": release_payload["assetSha256"],
        }
        return {
            "schemaVersion": artifacts.COMPILE_RESULT_SCHEMA,
            "envRelease": release_info,
            "taskset": {
                "refId": taskset_ref["refId"],
                "version": int(taskset_ref["version"]),
                "fingerprint": taskset_ref["fingerprint"],
                "itemCount": int(taskset_ref["itemCount"]),
            },
            "buildReport": release_payload.get("buildReport") or {},
        }
    finally:
        store.close()


def _load_compile_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "storePath",
        "compileResultPath",
        "refId",
        "version",
        "name",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("payload contains unknown fields: {}".format(unknown))
    has_store = payload.get("storePath") is not None
    has_file = payload.get("compileResultPath") is not None
    if has_store and has_file:
        raise ValueError("storePath and compileResultPath are mutually exclusive")
    if has_store:
        return _load_compile_result_from_store(payload)
    path = payload.get("compileResultPath")
    if not isinstance(path, str) or not path:
        raise ValueError("payload requires storePath or compileResultPath")
    if not Path(path).is_absolute():
        raise ValueError("payload.compileResultPath must be an absolute path")
    return artifacts.load_compile_result(path)


def _load_verified_release(
    release: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Path]:
    """Load and verify the immutable release tree without touching RSI storage."""
    if not isinstance(release, dict):
        raise ValueError("compile result envRelease must be an object")
    for key in ("refId", "fingerprint", "assetDir", "assetSha256"):
        value = release.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError("compile result envRelease.{} must be a non-empty string".format(key))
    try:
        version = int(release.get("version"))
    except (TypeError, ValueError) as error:
        raise ValueError("compile result envRelease.version must be an integer") from error
    if version < 1:
        raise ValueError("compile result envRelease.version must be >= 1")

    raw_asset_dir = Path(release["assetDir"])
    if not raw_asset_dir.is_absolute():
        raise ValueError("compile result envRelease.assetDir must be an absolute path")
    asset_dir = raw_asset_dir.resolve()
    if not asset_dir.is_dir():
        raise ValueError(f"env release assetDir does not exist: {asset_dir}")
    release_root = asset_dir.parent
    if asset_dir != (release_root / "package").resolve():
        raise ValueError("env release assetDir must be a release/package directory")

    try:
        release_record, taskset_payload = artifacts.load_release_metadata(release_root)
    except artifacts.ArtifactConflictError as error:
        raise ValueError("env release file artifact is incomplete: {}".format(error)) from error
    if release_record.get("schemaVersion") != artifacts.ENV_RELEASE_SCHEMA:
        raise ValueError(
            "env_release.json schemaVersion must be {}".format(
                artifacts.ENV_RELEASE_SCHEMA
            )
        )
    for key in ("refId", "version", "fingerprint", "assetSha256"):
        if release_record.get(key) != release.get(key):
            raise ValueError("env_release.json.{} does not match the compile result".format(key))
    if not isinstance(release_record.get("buildFingerprint"), str) or not release_record[
        "buildFingerprint"
    ]:
        raise ValueError("env_release.json.buildFingerprint must be a non-empty string")

    try:
        manifest, package_files = load_env_package(str(asset_dir))
    except EnvPackageError as error:
        raise ValueError("Package validation failed: {}".format(error)) from error
    if manifest.get("fingerprint") != release["fingerprint"]:
        raise ValueError("package manifest fingerprint does not match release")
    if manifest.get("ref_id") != release["refId"]:
        raise ValueError("package manifest ref_id does not match release")
    try:
        manifest_version = int(manifest.get("version"))
    except (TypeError, ValueError) as error:
        raise ValueError("package manifest version must be an integer") from error
    if manifest_version != version:
        raise ValueError("package manifest version does not match release")
    if release_record.get("manifest") != manifest:
        raise ValueError("env_release.json.manifest does not match the package manifest")

    try:
        actual_zip_sha = zip_sha256(str(asset_dir))
    except EnvPackageError as error:
        raise ValueError("Failed to rebuild the package zip: {}".format(error)) from error
    if actual_zip_sha != release["assetSha256"]:
        raise ValueError(
            "Rebuilt package zip SHA-256 does not match assetSha256: {}".format(actual_zip_sha)
        )
    zip_path = release_root / ZIP_NAME
    if not zip_path.is_file():
        raise ValueError(f"release is missing {ZIP_NAME}: {zip_path}")
    if hashlib.sha256(zip_path.read_bytes()).hexdigest() != release["assetSha256"]:
        raise ValueError("release zip SHA-256 does not match assetSha256")
    try:
        package_tasks = [
            json.loads(line)
            for line in package_files["tasks.jsonl"].decode("utf-8").splitlines()
            if line.strip()
        ]
    except (KeyError, ValueError) as error:
        raise ValueError("package tasks.jsonl is not readable: {}".format(error)) from error
    return release_record, taskset_payload, manifest, zip_path, package_tasks


def _validate_taskset(
    taskset_payload: Dict[str, Any],
    taskset_ref: Dict[str, Any],
    release: Dict[str, Any],
    release_record: Dict[str, Any],
    package_tasks: List[Dict[str, Any]],
) -> None:
    if not isinstance(taskset_ref, dict):
        raise ValueError("compile result taskset must be an object")
    for key in ("refId", "version", "fingerprint", "itemCount"):
        if taskset_ref.get(key) in (None, ""):
            raise ValueError("compile result taskset.{} is missing".format(key))
    if release_record.get("taskset") != taskset_ref:
        raise ValueError("env_release.json.taskset does not match the compile result")
    if taskset_payload.get("schemaVersion") != artifacts.ENV_TASKSET_SCHEMA:
        raise ValueError(
            "taskset schemaVersion must be {}".format(artifacts.ENV_TASKSET_SCHEMA)
        )
    if taskset_payload.get("refId") != taskset_ref["refId"]:
        raise ValueError("taskset.refId does not match the compile result")
    try:
        taskset_version = int(taskset_ref["version"])
        taskset_item_count = int(taskset_ref["itemCount"])
        payload_item_count = int(taskset_payload.get("itemCount"))
    except (TypeError, ValueError) as error:
        raise ValueError("taskset version/itemCount must be integers") from error
    if taskset_version != int(release["version"]):
        raise ValueError("taskset.version does not match the env release version")
    actual_fingerprint = artifacts.fingerprint_of(taskset_payload)
    if actual_fingerprint != taskset_ref["fingerprint"]:
        raise ValueError(
            "Recomputed taskset fingerprint does not match the compile result: {}".format(actual_fingerprint)
        )
    tasks = taskset_payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("taskset.tasks is empty; refusing to publish an empty environment task set")
    if payload_item_count != len(tasks):
        raise ValueError("taskset.itemCount does not match the number of tasks")
    if taskset_item_count != len(tasks):
        raise ValueError("compile result taskset.itemCount does not match the number of tasks")
    if tasks != package_tasks:
        raise ValueError("taskset.tasks does not match tasks.jsonl in the frozen package")
    if taskset_payload.get("envRelease") != {
        key: release[key] for key in ("refId", "version", "fingerprint", "assetSha256")
    }:
        raise ValueError("taskset does not reference the same env release")
    contract = taskset_payload.get("gymContract") or {}
    if contract.get("taskSchema") != gym_contract.GYM_TASK_SCHEMA:
        raise ValueError(
            "taskset.gymContract.taskSchema must be {}".format(
                gym_contract.GYM_TASK_SCHEMA
            )
        )
    try:
        gym_contract.validate_tasks(tasks)
    except gym_contract.GymContractError as error:
        raise ValueError("taskset does not satisfy the Gym task contract: {}".format(error)) from error
    for task in tasks:
        task_release = task["env_release"]
        if task_release["ref_id"] != release["refId"]:
            raise ValueError("task {} env_release.ref_id does not match".format(task["task_id"]))
        if int(task_release["version"]) != int(release["version"]):
            raise ValueError("task {} env_release.version does not match".format(task["task_id"]))
        if task_release["fingerprint"] != release["fingerprint"]:
            raise ValueError(
                "task {} env_release.fingerprint does not match".format(task["task_id"])
            )


def _register_overlay(
    gym_root: Path, overlay_root: Path, name: str, zip_name: str
) -> Dict[str, Any]:
    """Register the benchmark component and package path, then recompute hashes."""
    manifest_path = overlay_root / "overlay-manifest.yaml"
    config_path = overlay_root / "resources_servers/synth_env/configs/synth_env.yaml"
    if not manifest_path.is_file() or not config_path.is_file():
        raise ValueError(
            "Gym overlay is missing overlay-manifest.yaml or synth_env config and cannot be registered"
        )
    import yaml

    overlay_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(overlay_manifest, dict):
        raise ValueError("overlay-manifest.yaml must be an object")
    components = overlay_manifest.setdefault("components", {})
    components["benchmarks/{}".format(name)] = {
        "kind": "component",
        "complete": True,
        "wheel_backed": False,
        "tree_sha256": "",
        "depends_on": [
            {"component": "benchmarks/prompts", "backend": "overlay"},
            {"component": "resources_servers/synth_env", "backend": "overlay"},
            {"component": "responses_api_agents/synth_env_agent", "backend": "overlay"},
            {"component": "responses_api_models/openai_model", "backend": "overlay"},
        ],
        "reviewed_by": "dataforge-publish",
    }
    manifest_path.write_text(
        yaml.safe_dump(overlay_manifest, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    node = (
        config.get("synth_env", {})
        .get("resources_servers", {})
        .get("synth_env", {})
    )
    if not isinstance(node, dict):
        raise ValueError("synth_env.yaml is missing synth_env.resources_servers.synth_env")
    paths = node.setdefault("package_paths", [])
    package_path = "benchmarks/{}/assets/{}".format(name, zip_name)
    if package_path not in paths:
        paths.append(package_path)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
    )

    tool = gym_root / "scripts/refactor/update_overlay_manifest.py"
    if not tool.is_file():
        raise ValueError("Gym is missing scripts/refactor/update_overlay_manifest.py and cannot recompute tree_sha256")
    subprocess.run(
        [sys.executable, str(tool), "--overlay", str(overlay_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "status": "registered",
        "component": "benchmarks/{}".format(name),
        "packagePath": package_path,
    }


def publish(
    payload: Dict[str, Any], gym_root: Path, *, register: bool = False
) -> Dict[str, Any]:
    compile_result = _load_compile_result(payload)
    release = compile_result["envRelease"]
    taskset_ref = compile_result["taskset"]
    release_meta, taskset_payload, manifest, zip_path, package_tasks = (
        _load_verified_release(release)
    )
    _validate_taskset(
        taskset_payload, taskset_ref, release, release_meta, package_tasks
    )
    raw_name = payload.get("name") or release["refId"]
    name = _safe_name(str(raw_name))
    if not name.startswith("synth-"):
        name = "synth-" + name
    overlay_root = gym_root / "overlay"
    if not overlay_root.is_dir():
        raise ValueError("Gym root is missing overlay/; point to the Gym checkout root")
    bench_dir = overlay_root / "benchmarks" / name
    prompt_dir = overlay_root / "benchmarks" / "prompts" / name
    prompt_compiler = get_profile_compiler(str(manifest["profile"]))
    written: List[str] = []

    asset_zip_name = "gym-env-package-v{}-{}.zip".format(
        int(release["version"]), release["fingerprint"]
    )
    _write(
        bench_dir / "assets" / asset_zip_name,
        zip_path.read_bytes(),
        written,
        gym_root,
    )
    data_name = "{}.jsonl".format(name)
    _write_replace(
        bench_dir / "prepare.py",
        gym_contract.prepare_script(
            name, zip_name=asset_zip_name, data_name=data_name
        ),
        written,
        gym_root,
    )
    _write_replace(
        bench_dir / "config.yaml",
        _yaml_bytes(
            gym_contract.benchmark_config(
                name, data_name=data_name, prompt_name=name
            )
        ),
        written,
        gym_root,
    )
    _write_replace(
        bench_dir / "manifest.yaml",
        _yaml_bytes(
            gym_contract.benchmark_manifest(
                name,
                data_name=data_name,
                description="DataForge Gym-native environment package {}".format(
                    release["refId"]
                ),
            )
        ),
        written,
        gym_root,
    )
    _write_replace(
        prompt_dir / "agent.yaml",
        _yaml_bytes(prompt_compiler.agent_prompt(name)),
        written,
        gym_root,
    )

    tasks = list(taskset_payload.get("tasks") or [])
    publish_record = {
        "schemaVersion": "env-gym-publish-v1",
        "benchmark": name,
        "status": "published",
        "runnable": True,
        "release": {
            "refId": release["refId"],
            "version": int(release["version"]),
            "fingerprint": release["fingerprint"],
            "assetSha256": release["assetSha256"],
            "buildFingerprint": release_meta.get("buildFingerprint"),
        },
        "taskset": {
            "refId": taskset_ref["refId"],
            "version": int(taskset_ref["version"]),
            "fingerprint": taskset_ref["fingerprint"],
            "itemCount": len(tasks),
        },
        "assets": {
            "packageZip": "benchmarks/{}/assets/{}".format(name, asset_zip_name),
            "config": "benchmarks/{}/config.yaml".format(name),
            "manifest": "benchmarks/{}/manifest.yaml".format(name),
            "prepareScript": "benchmarks/{}/prepare.py".format(name),
            "prompt": "benchmarks/prompts/{}/agent.yaml".format(name),
            "generatedData": "benchmarks/{}/data/{}".format(name, data_name),
        },
        "gymContract": {
            "design": gym_contract.GYM_CONTRACT_DESIGN,
            "sections": list(gym_contract.GYM_CONTRACT_SECTIONS),
            "taskSchema": gym_contract.GYM_TASK_SCHEMA,
            "taskFields": list(gym_contract.TASK_FIELDS),
        },
        "registration": {"status": "not-requested"},
    }
    if register:
        publish_record["registration"] = _register_overlay(
            gym_root, overlay_root, name, asset_zip_name
        )
    _write_replace(
        bench_dir / "publish.json",
        _json_bytes(publish_record),
        written,
        gym_root,
    )

    result = dict(publish_record)
    result["written"] = written
    result["item_count"] = len(tasks)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a Gym-native environment package to the Gym overlay")
    parser.add_argument("--gym-root", required=True, help="Gym project root")
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register benchmark components and package_paths, then recompute overlay tree_sha256",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        payload = json.load(sys.stdin)
    except ValueError as error:
        print(f"stdin is not valid JSON: {error}", file=sys.stderr)
        return 2
    gym_root = Path(args.gym_root).expanduser().resolve()
    try:
        result = publish(payload, gym_root, register=args.register)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"[gym-publish-env] Publish rejected: {error}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
