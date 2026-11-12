from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict


def _parse_suite_ref(suite_ref):
    ref_id, separator, version = suite_ref.rpartition("@")
    if not separator or not ref_id.strip():
        raise ValueError(f"suite ref must be ref_id@version: {suite_ref!r}")
    try:
        return ref_id, int(version)
    except ValueError as error:
        raise ValueError(f"suite version must be an integer: {suite_ref!r}") from error


def _verify_benchmark(args):
    from data_designer.storage.catalog import ArtifactCatalog

    if args.package:
        if args.catalog or args.suite:
            raise ValueError("verify accepts --catalog/--suite or --package, not both")
        from pathlib import Path
        from data_designer.pipelines.benchmarks.reader import read_package
        path = Path(args.package)
        if path.is_dir():
            path /= "manifest.json"
        package = read_package(path, verify_resources=True)
        return {"verified": True, "manifest_uri": package["manifest_uri"],
                "digest": package["digest"]}
    if not args.catalog or not args.suite:
        raise ValueError("verify requires --catalog with --suite, or --package")
    ref_id, version = _parse_suite_ref(args.suite)
    from data_designer.pipelines.benchmarks.suite import verify_suite
    return verify_suite(ArtifactCatalog(args.catalog), ref_id, version)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dataforge")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output")
    algorithms = commands.add_parser("algorithms").add_subparsers(dest="action", required=True)
    algorithms.add_parser("list")
    algorithms.add_parser("describe").add_argument("id")
    commands.add_parser("validate").add_argument("manifest")
    benchmark = commands.add_parser("benchmark").add_subparsers(dest="action", required=True)
    benchmark.add_parser("build").add_argument("--config", required=True)
    benchmark_show = benchmark.add_parser("show")
    benchmark_show.add_argument("--catalog", required=True)
    benchmark_show.add_argument("--suite", required=True, help="ref_id@version")
    benchmark_show.add_argument("--approved-only", action="store_true")
    benchmark_verify = benchmark.add_parser("verify")
    benchmark_verify.add_argument("--catalog")
    benchmark_verify.add_argument("--suite", help="ref_id@version")
    benchmark_verify.add_argument("--package", help="dataforge-benchmark-v1 package directory")
    export = commands.add_parser("export")
    export.add_argument("host", choices=("gym", "evaluator"))
    export.add_argument("--config", required=True)
    from data_designer.cli.rubric_criteria import register as register_rubric_criteria
    register_rubric_criteria(commands)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            from data_designer.api import generate
            from data_designer.config import load_config
            config = load_config(args.config)
            if args.output:
                from pathlib import Path
                config["output"]["uri"] = str(Path(args.output).resolve())
            result = generate(config=config)
            value = result.report
        elif args.command == "algorithms":
            from data_designer.generation.algorithms.registry import AlgorithmRegistry
            registry = AlgorithmRegistry()
            value = [asdict(d) for d in registry.list()] if args.action == "list" else asdict(registry.describe(args.id))
        elif args.command == "export":
            from pathlib import Path
            from data_designer.config import object_schema, validate_config
            path = Path(args.config).resolve()
            if path.suffix == ".json":
                request = json.loads(path.read_text(encoding="utf-8"))
            else:
                import yaml
                request = yaml.safe_load(path.read_text(encoding="utf-8"))
            request = validate_config(request, object_schema({
                "manifest": {"type": "string"}, "output": {"type": "string"},
                "digest": {"type": "string"}, "requirements": {"type": "object"},
                "python_executable": {"type": "string"},
                "config_paths": {"type": "array", "items": {"type": "string"}},
            }, ("manifest", "output")))
            for key in ("manifest", "output"):
                request[key] = str((path.parent / request[key]).resolve())
            if "config_paths" in request:
                request["config_paths"] = [str((path.parent / item).resolve()) for item in request["config_paths"]]
            if args.host == "gym":
                from data_designer.integrations.gym.export import export_gym
                if "requirements" in request or "python_executable" in request:
                    raise ValueError("Evaluator-only options cannot be used for Gym")
                value = export_gym(**request)
            else:
                from data_designer.integrations.evaluator.export import export_evaluator
                if "config_paths" in request:
                    raise ValueError("Gym-only config_paths cannot be used for Evaluator")
                value = export_evaluator(**request)
        elif args.command == "benchmark":
            if args.action == "build":
                from data_designer.pipelines.benchmarks.api import build_benchmark
                package = build_benchmark(args.config)
                if "manifest_uri" in package:
                    value = {"manifest_uri": package["manifest_uri"], "digest": package["digest"]}
                else:
                    value = {"splits": {split: {"manifest_uri": split_package["manifest_uri"],
                                                "digest": split_package["digest"]}
                                        for split, split_package in package.items()}}
            elif args.action == "show":
                ref_id, version = _parse_suite_ref(args.suite)
                from data_designer.pipelines.benchmarks.suite import show_suite
                from data_designer.storage.catalog import ArtifactCatalog
                value = show_suite(
                    ArtifactCatalog(args.catalog), ref_id, version,
                    require_approved=args.approved_only,
                )
            else:
                value = _verify_benchmark(args)
        elif args.command == "rubric-criteria":
            from data_designer.cli.rubric_criteria import run as run_rubric_criteria
            value = run_rubric_criteria(args)
        else:
            from pathlib import Path
            path = Path(args.manifest)
            if path.is_dir():
                path /= "manifest.json"
            version = json.loads(path.read_text(encoding="utf-8")).get("protocol_version")
            if version == "dataforge-benchmark-v1":
                from data_designer.pipelines.benchmarks.reader import read_package
                package = read_package(path, verify_resources=True)
                value = {"manifest_uri": package["manifest_uri"], "digest": package["digest"]}
            else:
                from data_designer.storage.local import validate_package
                value = asdict(validate_package(args.manifest))
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"dataforge: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
