"""C-WK-0 file-protocol worker: user examples → seed-import bundle.

Primary v0.1/v0.2 entry point::

    python -m data_designer.pipelines.envsynth.cli.source_import <input.json> <outputDir>

The output directory must be absolute and must not already exist.  Exit codes
follow the repository worker red lines: 0 success, 1 build failure, 2 envelope
invalid (build not started).
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_designer.pipelines.envsynth.cli.common import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    SCHEMA_VERSION,
    WorkerError,
    absolute_path,
    load_json,
)
from data_designer.pipelines.envsynth.seed_import import (
    SeedImportError,
    build_seed_import_from_source,
    sha256_file,
)
from data_designer.pipelines.envsynth.domains.user_examples_seed_adapter import FAMILY


@dataclass
class SourceInput:
    source_paths: List[str]
    regulation_paths: List[str]
    family: str
    anchor_holdout_ratio: float
    max_questions: int
    parse_options: Dict[str, Any]
    ocr_options: Dict[str, Any]


def parse_input(path: str) -> SourceInput:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise WorkerError("Input must be a JSON object")
    if raw.get("schemaVersion") != SCHEMA_VERSION:
        raise WorkerError(f"schemaVersion must be {SCHEMA_VERSION}")
    request = raw.get("request")
    if not isinstance(request, dict):
        raise WorkerError("request must be an object")

    source_paths = request.get("sourcePaths")
    if not isinstance(source_paths, list) or not source_paths:
        raise WorkerError("request.sourcePaths must be a non-empty array")
    source_paths = [absolute_path(item, "sourcePaths[]") for item in source_paths]
    regulation_paths = request.get("regulationPaths") or []
    if not isinstance(regulation_paths, list):
        raise WorkerError("request.regulationPaths must be an array")
    regulation_paths = [
        absolute_path(item, "regulationPaths[]") for item in regulation_paths
    ]
    family = request.get("family", FAMILY)
    if not isinstance(family, str) or not family:
        raise WorkerError("request.family must be a non-empty string")
    try:
        ratio = float(request.get("anchorHoldoutRatio", 0.3))
    except (TypeError, ValueError) as error:
        raise WorkerError("request.anchorHoldoutRatio must be numeric") from error
    if not 0.0 < ratio < 1.0:
        raise WorkerError("request.anchorHoldoutRatio must be in (0,1)")
    max_questions = request.get("maxQuestions", 20)
    if not isinstance(max_questions, int) or isinstance(max_questions, bool) or max_questions < 1:
        raise WorkerError("request.maxQuestions must be a positive integer")
    parse_options = request.get("parseOptions") or {}
    if not isinstance(parse_options, dict):
        raise WorkerError("request.parseOptions must be an object")
    ocr_options = request.get("ocrOptions") or {}
    if not isinstance(ocr_options, dict):
        raise WorkerError("request.ocrOptions must be an object")
    return SourceInput(
        source_paths=source_paths,
        regulation_paths=regulation_paths,
        family=family,
        anchor_holdout_ratio=ratio,
        max_questions=max_questions,
        parse_options=parse_options,
        ocr_options=ocr_options,
    )


def run_worker(argv: List[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: python -m data_designer.pipelines.envsynth.cli.source_import "
            "<input.json> <outputDir>",
            file=sys.stderr,
        )
        return EXIT_USAGE
    output_dir = Path(argv[1])
    if not output_dir.is_absolute() or output_dir.exists():
        print(
            "[env-source-worker] outputDir must be an absolute path that does not yet exist",
            file=sys.stderr,
        )
        return EXIT_USAGE
    try:
        input_data = parse_input(argv[0])
    except WorkerError as error:
        print(f"[env-source-worker] Invalid input (build not started): {error}", file=sys.stderr)
        return EXIT_USAGE
    try:
        manifest = build_seed_import_from_source(
            input_data.source_paths,
            str(output_dir),
            regulation_paths=input_data.regulation_paths,
            family=input_data.family,
            anchor_holdout_ratio=input_data.anchor_holdout_ratio,
            max_questions=input_data.max_questions,
            parse_options=input_data.parse_options,
            ocr_options=input_data.ocr_options,
        )
        manifest_path = output_dir / "seed_manifest.json"
        summary = {
            "outputDir": str(output_dir),
            "seedManifestPath": str(manifest_path),
            "seedManifestSha256": sha256_file(manifest_path),
            "questionCount": manifest.get("questionCount", 0),
            "blockingQuestionCount": manifest.get("blockingQuestionCount", 0),
            "domainSchemaGuess": manifest.get("domainSchemaGuess"),
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return EXIT_OK
    except SeedImportError as error:
        print(f"[env-source-worker] Build failed: {error}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as error:  # never emit a usable-looking result on surprise
        print(f"[env-source-worker] Unexpected exception: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return EXIT_FAILED


def main(argv: Optional[List[str]] = None) -> int:
    return run_worker(sys.argv[1:] if argv is None else list(argv))


if __name__ == "__main__":
    sys.exit(main())
