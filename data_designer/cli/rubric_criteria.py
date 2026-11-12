"""Demonstration-library CLI (migrated from RuACT criteria.py main; subcommands attached
to the dataforge entry point). - `extract`: reviewed good cases ->
demonstrations.json (verbatim evidence_quote citations; --exclude removes heldout
overlap). - `extract-error`: reviewed error cases -> negative rubric library
error_patterns.json (prompt-layer variant; payload marks library_polarity=negative).
- `generate`: demonstration library + target query/parent -> generated_unscored
rubric candidates. - `validate`: validate a demonstration-library file
(schema/hash/citations). - `export-seed`: freeze a seed library -> evolution seed
rows (frozen_seed_only evidence scope). - `evolve`: section 6.3 single-round
evolution orchestration (run_evolution_round) -- grouped observations + current
rubric library -> evolution_result.json (noreplace) + overlaid library + dcd_* score-
adjustment evidence. Endpoint credentials come from the
`HEALTHBENCH_GENERATOR_API_BASE/KEY/MODEL` environment variables (design document
section 6.4); they never enter configuration files or outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_designer.rubric.generator.client import WorkerConfig
from data_designer.rubric.generator.demonstrations import DemonstrationLibrary


def _generator_config(args) -> WorkerConfig:
    import os

    config = WorkerConfig(
        base_url=os.environ.get("HEALTHBENCH_GENERATOR_API_BASE", ""),
        api_key=os.environ.get("HEALTHBENCH_GENERATOR_API_KEY", ""),
        model=os.environ.get("HEALTHBENCH_GENERATOR_MODEL", ""),
        max_tokens=args.max_tokens,
        max_attempts=3,
    )
    config.validate()
    return config


def _require_new_output(args) -> None:
    if getattr(args, "output", None) is not None and Path(args.output).exists():
        raise FileExistsError("Choose a new output file")


def register(subparsers) -> argparse.ArgumentParser:
    """Attach the `rubric-criteria` subcommand to the dataforge main parser."""
    command = subparsers.add_parser(
        "rubric-criteria", help="Demo library extraction/generation/validation/seed export"
    )
    action = command.add_subparsers(dest="action", required=True)
    for name in ("extract", "extract-error", "generate"):
        sub = action.add_parser(name)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--max-tokens", type=int, default=4096)
        sub.add_argument("--max-input-characters", type=int, default=48000)
        if name in ("extract", "extract-error"):
            sub.add_argument("--cases", type=Path, required=True)
            sub.add_argument("--exclude", type=Path, action="append", default=[])
        else:
            sub.add_argument("--targets", type=Path, required=True)
            sub.add_argument("--demonstrations", type=Path, required=True)
    export_seed = action.add_parser("export-seed")
    export_seed.add_argument("--seed-library", type=Path, required=True)
    export_seed.add_argument("--output", type=Path, required=True)
    validate = action.add_parser("validate")
    validate.add_argument("--demonstrations", type=Path, required=True)
    evolve = action.add_parser("evolve", help="§6.3 single-round evolution orchestration")
    evolve.add_argument("--responses", type=Path, required=True)
    evolve.add_argument("--rubric-library", type=Path, required=True)
    evolve.add_argument("--registry-dir", type=Path, required=True)
    evolve.add_argument("--result", type=Path, required=True)
    evolve.add_argument("--seed-library", type=Path)
    evolve.add_argument("--overlaid-output", type=Path)
    evolve.add_argument("--score-adjustments-output", type=Path)
    evolve.add_argument("--run-id", required=True)
    evolve.add_argument("--global-step", type=int, required=True)
    evolve.add_argument("--max-tokens", type=int, default=4096)
    evolve.add_argument("--keep-constant-children", action="store_true")
    evolve.add_argument("--include-negative-weight", action="store_true")
    return command


def run(args, *, judge_transport=None) -> dict:
    """Execute the rubric-criteria subcommand; return a JSON-serializable result
    summary. `judge_transport` is a test injection point: by default construct
    `JudgeTransport` per section 6.4 (credentials from HEALTHBENCH_JUDGE_*
    environment variables, resolved at request_once time).
    """
    if args.action == "evolve":
        return _run_evolve(args, judge_transport=judge_transport)
    from data_designer.rubric.generator.extract import (
        extract_demonstrations,
        extract_error_patterns,
        generate_criteria,
        read_records,
        write_new,
    )
    from data_designer.rubric.evolve.seed_library import (
        PositiveRefinementSeedLibrary,
    )

    _require_new_output(args)
    if args.action in ("extract", "extract-error"):
        rows = read_records(args.cases)
        excluded = {
            row["query"]
            for path in args.exclude
            for row in read_records(path)
        }
        extract = (
            extract_demonstrations if args.action == "extract" else extract_error_patterns
        )
        payload = extract(
            rows,
            config=_generator_config(args),
            exclude_queries=excluded,
            max_input_characters=args.max_input_characters,
        )
        write_new(args.output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        summary = {"examples": len(payload["examples"]), "output": str(args.output)}
        if args.action == "extract-error":
            summary["library_polarity"] = payload["library_polarity"]
        return summary
    if args.action == "generate":
        library = DemonstrationLibrary(args.demonstrations)
        rows = read_records(args.targets)
        generated = generate_criteria(
            rows,
            library,
            config=_generator_config(args),
            max_input_characters=args.max_input_characters,
        )
        write_new(
            args.output,
            "".join((json.dumps(row, ensure_ascii=False) + "\n" for row in generated)),
        )
        return {"targets": len(generated), "output": str(args.output)}
    if args.action == "export-seed":
        PositiveRefinementSeedLibrary(args.seed_library)
        data = Path(args.seed_library).read_bytes()
        payload = json.loads(data)
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        output = []
        for entry in payload["entries"]:
            parent = {
                key: value
                for key, value in entry.items()
                if key not in ("children", "candidates")
            }
            children = entry.get("children") or entry.get("candidates") or []
            for index, child in enumerate(children):
                output.append(
                    {
                        "parent": parent,
                        "child_index": index,
                        "child": child,
                        "source_sha256": digest,
                        "evidence_scope": "frozen_seed_only",
                    }
                )
        write_new(
            args.output,
            "".join((json.dumps(row, ensure_ascii=False) + "\n" for row in output)),
        )
        return {
            "parents": len(payload["entries"]),
            "children": len(output),
            "source_sha256": digest,
        }
    library = DemonstrationLibrary(args.demonstrations)
    return {"examples": len(library.examples), "sha256": library.sha256}


def _run_evolve(args, *, judge_transport=None) -> dict:
    """Section 6.3 single-round evolution orchestration: grouped observations + current
    rubric library -> evolution_result.json. The discipline follows
    run_evolution_round: result files are noreplace (re-export to the same path
    raises FileExistsError); the overlaid library and dcd_* score adjustments are
    optional sidecar outputs (jsonl, also noreplace), pre-checked before the run to
    avoid partial artifacts; credentials come from HEALTHBENCH_GENERATOR_* /
    HEALTHBENCH_JUDGE_* environment variables and never enter outputs.
    """
    from data_designer.rubric.evolve.registry import PositiveRefinementRegistry
    from data_designer.rubric.evolve.round import run_evolution_round
    from data_designer.rubric.evolve.seed_library import (
        PositiveRefinementSeedLibrary,
    )
    from data_designer.rubric.generator.extract import read_records, write_new
    from data_designer.rubric.judge.transport import JudgeTransport

    output_paths = [
        path for path in (args.overlaid_output, args.score_adjustments_output) if path
    ]
    for path in (*output_paths, args.result):
        if Path(path).exists():
            raise FileExistsError(f"evolution round artifacts are never overwritten: {path}")

    rows = read_records(args.responses)
    rubrics = read_records(args.rubric_library)
    seed_library = (
        PositiveRefinementSeedLibrary(args.seed_library) if args.seed_library else None
    )
    payload, effects = run_evolution_round(
        rows,
        rubrics=rubrics,
        registry=PositiveRefinementRegistry(args.registry_dir),
        config=_generator_config(args),
        judge_transport=judge_transport if judge_transport is not None else JudgeTransport(),
        run_id=args.run_id,
        global_step=args.global_step,
        seed_library=seed_library,
        result_path=args.result,
        keep_constant_children=bool(args.keep_constant_children),
        include_negative_weight=bool(args.include_negative_weight),
    )
    if args.overlaid_output:
        write_new(
            args.overlaid_output,
            "".join(
                (
                    json.dumps(
                        {"query_id": query_id, "rubrics": library}, ensure_ascii=False
                    )
                    + "\n"
                )
                for query_id, library in sorted(effects["overlaid_libraries"].items())
            ),
        )
    if args.score_adjustments_output:
        write_new(
            args.score_adjustments_output,
            "".join(
                (
                    json.dumps({"response_id": response_id, **correction}, ensure_ascii=False)
                    + "\n"
                )
                for response_id, correction in sorted(
                    effects["score_adjustments"].items()
                )
            ),
        )
    audit = payload["audit"]
    return {
        "status": payload["status"],
        "run_id": payload["run_id"],
        "global_step": payload["global_step"],
        "job_count": audit["job_count"],
        "committed_count": audit["committed_count"],
        "selected_parent_count": len(payload["selected_parents"]),
        "decomposition_count": len(payload["decompositions"]),
        "skipped_parent_count": len(payload["skipped_parents"]),
        "result": str(args.result),
        "overlaid_queries": sorted(effects["overlaid_libraries"]),
        "adjusted_response_count": len(effects["score_adjustments"]),
    }
