from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curator.utils.jsonl import read_jsonl, write_jsonl  # noqa: E402


DEFAULT_KG_PATHS = Path("data/inaturalist_root_concepts/audit_200_kg_paths.jsonl")
DEFAULT_INATURALIST_VAL_ARCHIVE_URL = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz"
)
DEFAULT_USER_AGENT = "DataForge-iNaturalist-ReasonVQA/0.1"
DEFAULT_CATEGORIES = [{"id": 8, "name": "knowledge"}]
DEFAULT_CATEGORY_IDS_BY_PROPERTY = {
    "P141": [8],
    "P1843": [8],
    "P225": [8],
}
DEFAULT_NEGATIVE_STRATEGY = "wikidata_taxonomy_neighbors"
DEFAULT_MAX_TAXONOMY_DISTANCE = "family"


@dataclass(frozen=True)
class ExtractionReport:
    requested: int
    extracted: int
    missing: list[str]
    scanned_members: int


@dataclass(frozen=True)
class ImageAvailabilityReport:
    requested: int
    already_present: int
    extracted: int
    missing: list[str]
    invalid: list[str]
    scanned_members: int
    image_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an iNaturalist ReasonVQA export by generating QA rows from "
            "Wikidata KG paths, binding local image paths, and extracting the "
            "corresponding images from the iNaturalist 2021 val archive."
        )
    )
    parser.add_argument("--kg-paths", type=Path, default=DEFAULT_KG_PATHS)
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("REASONVQA_OUTPUT_DIR"))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--archive-source",
        default=DEFAULT_INATURALIST_VAL_ARCHIVE_URL,
        help="Local val.tar.gz path or remote archive URL.",
    )
    parser.add_argument(
        "--archive-url",
        default=DEFAULT_INATURALIST_VAL_ARCHIVE_URL,
        help="URI recorded in image_url/provenance; may be the same as archive-source.",
    )
    parser.add_argument("--limit-qa", type=int, default=None)
    parser.add_argument(
        "--negative-strategy",
        default=DEFAULT_NEGATIVE_STRATEGY,
        choices=[
            "wikidata_taxonomy_neighbors",
            "taxonomy_hard",
            "typed_taxonomic_neighbors",
            "fine_grained_taxonomy",
            "same_route_property",
        ],
        help=(
            "MC distractor strategy. The default uses strict Wikidata/taxonomy "
            "neighbors instead of same-route random answers."
        ),
    )
    parser.add_argument(
        "--max-taxonomy-distance",
        default=DEFAULT_MAX_TAXONOMY_DISTANCE,
        help=(
            "Largest taxonomy-neighbor tier accepted for hard negatives: "
            "genus, family, order, class, phylum, kingdom, or an integer distance."
        ),
    )
    parser.add_argument(
        "--neighbor-candidates",
        type=Path,
        default=None,
        help=(
            "Optional JSONL cache of Wikidata taxon-neighbor choice candidates. "
            "Rows are matched by root_qid, property_id, and route_properties."
        ),
    )
    parser.add_argument("--no-download-images", action="store_true")
    parser.add_argument("--no-validate-images", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument(
        "--max-answer-prior-fraction",
        type=float,
        default=0.5,
        help=(
            "Required shortcut-control filter: drop answer groups whose majority "
            "answer fraction is above this value."
        ),
    )
    parser.add_argument(
        "--min-answer-prior-group-size",
        type=int,
        default=20,
        help="Only apply answer-prior filtering to groups at least this large.",
    )
    args = parser.parse_args()
    if not args.output_dir:
        parser.error("--output-dir or REASONVQA_OUTPUT_DIR is required")
    args.output_dir = Path(args.output_dir)
    return args


def default_recipe_config(
    export_dir: Path,
    *,
    negative_strategy: str = DEFAULT_NEGATIVE_STRATEGY,
    max_taxonomy_distance: str | int = DEFAULT_MAX_TAXONOMY_DISTANCE,
) -> dict[str, Any]:
    return {
        "export_dir": str(export_dir),
        "image_dir": str(export_dir / "images"),
        "image_source": "inaturalist_2021_val",
        "split": "train",
        "categories": DEFAULT_CATEGORIES,
        "category_ids_by_property": DEFAULT_CATEGORY_IDS_BY_PROPERTY,
        "filter_template_usable_paths": True,
        "template_version": "inat-reasonvqa-template-v1",
        "difficulty_policy": {
            "negative_strategy": negative_strategy,
            "max_taxonomy_distance": max_taxonomy_distance,
            "drop_rows_without_hard_negatives": True,
        },
    }


def generate_reasonvqa_rows(
    *,
    path_records: list[dict[str, Any]],
    export_dir: Path,
    split: str,
    negative_strategy: str = DEFAULT_NEGATIVE_STRATEGY,
    max_taxonomy_distance: str | int = DEFAULT_MAX_TAXONOMY_DISTANCE,
    limit_qa: int | None = None,
) -> list[dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data_designer.generation.algorithms.builtin.reasonvqa_paths import (
        generate_reasonvqa_rows_from_wikidata_paths,
    )

    config = default_recipe_config(
        export_dir,
        negative_strategy=negative_strategy,
        max_taxonomy_distance=max_taxonomy_distance,
    )
    config["split"] = split
    rows = generate_reasonvqa_rows_from_wikidata_paths(
        path_records=path_records,
        config=config,
        recipe="knowledge_reasoning_vqa_inaturalist_wikidata_paths",
        generator_prompt_version="inat-reasonvqa-template-v1",
        schema_version="knowledge_reasoning_vqa.v1",
        params={"filter_versions": ["inaturalist_template_usable_v1"]},
    )
    if limit_qa is not None:
        rows = rows[:limit_qa]
    return rows


def attach_neighbor_candidates_to_path_records(
    path_records: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates_by_key: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = {}
    total_candidates = 0
    for cache_row in cache_rows:
        root_qid = str(cache_row.get("root_qid") or "").strip()
        property_id = str(cache_row.get("property_id") or "").strip()
        route = tuple(str(item) for item in cache_row.get("route_properties") or [])
        candidates = [
            dict(candidate)
            for candidate in cache_row.get("candidates") or []
            if isinstance(candidate, dict)
        ]
        if not root_qid or not property_id or not route or not candidates:
            continue
        candidates_by_key.setdefault((root_qid, property_id, route), []).extend(
            candidates
        )

    enriched_records = []
    enriched_count = 0
    for path_record in path_records:
        updated = copy.deepcopy(path_record)
        route = tuple(str(item) for item in updated.get("route_properties") or [])
        root_qid = _path_root_qid(updated)
        property_id = route[-1] if route else ""
        candidates = candidates_by_key.get((root_qid, property_id, route), [])
        if candidates:
            updated["choice_candidates"] = [
                *updated.get("choice_candidates", []),
                *copy.deepcopy(candidates),
            ]
            enriched_count += 1
            total_candidates += len(candidates)
        enriched_records.append(updated)

    report = {
        "cache_rows": len(cache_rows),
        "path_records_seen": len(path_records),
        "path_records_enriched": enriched_count,
        "candidate_count": total_candidates,
    }
    return enriched_records, report


def load_path_records_with_optional_neighbors(
    *,
    kg_paths: Path,
    neighbor_candidates: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    path_records = read_jsonl(kg_paths)
    report: dict[str, int | bool] = {
        "enabled": False,
        "cache_rows": 0,
        "path_records_seen": len(path_records),
        "path_records_enriched": 0,
        "candidate_count": 0,
    }
    if neighbor_candidates:
        neighbor_cache_rows = read_jsonl(neighbor_candidates)
        path_records, raw_report = attach_neighbor_candidates_to_path_records(
            path_records,
            neighbor_cache_rows,
        )
        report = {"enabled": True, **raw_report}
    return path_records, report


def _path_root_qid(path_record: dict[str, Any]) -> str:
    root = path_record.get("root") or {}
    root_concept = path_record.get("root_concept") or {}
    return str(
        root.get("qid")
        or root_concept.get("wikidata_qid")
        or root_concept.get("qid")
        or ""
    ).strip()


def bind_local_image_paths(
    rows: list[dict[str, Any]],
    *,
    image_root: Path,
    archive_url: str,
) -> list[dict[str, Any]]:
    bound_rows = []
    for row in rows:
        bound = copy.deepcopy(row)
        file_name = image_file_name(bound)
        image_path = image_destination(image_root, file_name)
        image_uri = archive_member_uri(archive_url, file_name)
        bound["image_path"] = str(image_path)
        bound["image_url"] = image_uri
        if isinstance(bound.get("image"), dict):
            bound["image"]["image_path"] = str(image_path)
            bound["image"]["image_url"] = image_uri

        image_ref = bound.get("evidence_refs", {}).get("image")
        if isinstance(image_ref, dict):
            image_ref["image_path"] = str(image_path)
            image_ref["image_url"] = image_uri

        for triplet in bound.get("qae_triplets") or []:
            triplet_image_ref = (
                triplet.get("evidence_refs", {}).get("image")
                if isinstance(triplet, dict)
                else None
            )
            if isinstance(triplet_image_ref, dict):
                triplet_image_ref["image_path"] = str(image_path)
                triplet_image_ref["image_url"] = image_uri

        provenance = bound.get("provenance")
        if isinstance(provenance, dict):
            provenance["source_uri_or_hash"] = image_uri
            image_source = provenance.get("image_source")
            if isinstance(image_source, dict):
                image_source["url"] = image_uri
        bound_rows.append(bound)
    return bound_rows


def filter_answer_prior_shortcuts(
    rows: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...] = ("property_id",),
    max_majority_fraction: float = 0.5,
    min_group_size: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not group_fields:
        raise ValueError("Answer-prior filter requires at least one group field.")
    if not 0 < max_majority_fraction <= 1:
        raise ValueError("max_majority_fraction must be in (0, 1].")
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive.")

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row, group_fields)].append(row)

    drop_keys = set()
    dropped_groups = []
    for key, group_rows in grouped.items():
        answers = [
            _normalize_answer_for_prior(_answer_for_prior(row))
            for row in group_rows
        ]
        counts = Counter(answer for answer in answers if answer)
        if not counts:
            continue
        majority_answer, majority_count = counts.most_common(1)[0]
        majority_fraction = majority_count / len(group_rows)
        if (
            len(group_rows) >= min_group_size
            and majority_fraction > max_majority_fraction
        ):
            drop_keys.add(key)
            dropped_groups.append(
                {
                    "group": dict(zip(group_fields, key, strict=True)),
                    "majority_answer": majority_answer,
                    "majority_fraction": majority_fraction,
                    "rows_dropped": len(group_rows),
                    "total_rows": len(group_rows),
                }
            )

    filtered_rows = [
        row for row in rows if _group_key(row, group_fields) not in drop_keys
    ]
    report = {
        "input_rows": len(rows),
        "output_rows": len(filtered_rows),
        "dropped_rows": len(rows) - len(filtered_rows),
        "group_fields": list(group_fields),
        "max_majority_fraction": max_majority_fraction,
        "min_group_size": min_group_size,
        "dropped_groups": dropped_groups,
    }
    return filtered_rows, report


def _group_key(row: dict[str, Any], group_fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "") for field in group_fields)


def _answer_for_prior(row: dict[str, Any]) -> str:
    answer = row.get("program_answer")
    if answer is not None:
        return str(answer)
    final_answer = row.get("final_answer")
    if isinstance(final_answer, dict):
        value = final_answer.get("label") or final_answer.get("value")
        if value is not None:
            return str(value)
    answers = row.get("answers")
    correct = row.get("correct")
    if isinstance(answers, list) and isinstance(correct, list):
        for candidate, flag in zip(answers, correct, strict=False):
            if flag:
                return str(candidate)
    return ""


def _normalize_answer_for_prior(answer: str) -> str:
    return " ".join(str(answer).casefold().split())


def write_bound_reasonvqa_exports(
    *,
    rows: list[dict[str, Any]],
    export_dir: Path,
    split: str,
    categories: list[dict[str, Any]] | None = None,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    qa_fields = (
        "question_id",
        "question",
        "image_id",
        "image_name",
        "image_path",
        "image_url",
        "source",
        "answers",
        "correct",
        "categories",
    )
    ann_fields = (
        "question_id",
        "hop",
        "property_id",
        "property_label",
        "entity_id",
        "entity_name",
        "route",
        "has_scene_graph",
    )
    write_jsonl(export_dir / f"{split}.jsonl", _project_fields(rows, qa_fields))
    write_jsonl(export_dir / f"{split}_ann.jsonl", _project_fields(rows, ann_fields))
    write_jsonl(export_dir / "knowledge_reasoning_vqa_merged.jsonl", rows)
    write_jsonl(export_dir / "image_manifest.jsonl", [image_manifest_row(row) for row in rows])
    write_jsonl(export_dir / "categories.jsonl", categories or DEFAULT_CATEGORIES)


def image_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": row["question_id"],
        "image_id": row["image_id"],
        "image_name": row["image_name"],
        "image_path": row["image_path"],
        "image_url": row["image_url"],
        "source": row["source"],
        "entity_id": row["entity_id"],
        "entity_name": row["entity_name"],
        "hop": row["hop"],
        "route": row["route"],
    }


def extract_selected_images_from_tar_stream(
    *,
    archive_fileobj: BinaryIO,
    target_file_names: set[str],
    image_root: Path,
    progress_interval: int | None = None,
) -> ExtractionReport:
    targets = {normalize_member_name(name) for name in target_file_names}
    missing = set(targets)
    extracted = 0
    scanned = 0
    image_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=archive_fileobj, mode="r|gz") as archive:
        for member in archive:
            scanned += 1
            member_name = normalize_member_name(member.name)
            if member.isfile() and member_name in missing:
                source = archive.extractfile(member)
                if source is not None:
                    destination = image_destination(image_root, member_name)
                    _write_stream_atomically(source, destination)
                    extracted += 1
                    missing.remove(member_name)
            if progress_interval and scanned % progress_interval == 0:
                print(
                    json.dumps(
                        {
                            "scanned_members": scanned,
                            "remaining_images": len(missing),
                            "extracted_images": extracted,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if not missing:
                break
    return ExtractionReport(
        requested=len(targets),
        extracted=extracted,
        missing=sorted(missing),
        scanned_members=scanned,
    )


def ensure_images_available(
    *,
    rows: list[dict[str, Any]],
    image_root: Path,
    archive_source: str,
    validate_images: bool,
) -> ImageAvailabilityReport:
    file_names = sorted(unique_image_file_names(rows))
    missing_before = [
        file_name
        for file_name in file_names
        if not image_destination(image_root, file_name).is_file()
    ]
    extraction = ExtractionReport(
        requested=len(missing_before),
        extracted=0,
        missing=[],
        scanned_members=0,
    )
    if missing_before:
        with open_archive_stream(archive_source) as archive_fileobj:
            extraction = extract_selected_images_from_tar_stream(
                archive_fileobj=archive_fileobj,
                target_file_names=set(missing_before),
                image_root=image_root,
                progress_interval=5000,
            )
    invalid = validate_image_files(file_names, image_root) if validate_images else []
    return ImageAvailabilityReport(
        requested=len(file_names),
        already_present=len(file_names) - len(missing_before),
        extracted=extraction.extracted,
        missing=extraction.missing,
        invalid=invalid,
        scanned_members=extraction.scanned_members,
        image_bytes=sum(
            image_destination(image_root, file_name).stat().st_size
            for file_name in file_names
            if image_destination(image_root, file_name).is_file()
        ),
    )


def validate_image_files(file_names: list[str], image_root: Path) -> list[str]:
    from PIL import Image

    invalid = []
    for file_name in file_names:
        path = image_destination(image_root, file_name)
        if not path.is_file():
            invalid.append(file_name)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            invalid.append(file_name)
    return invalid


@contextmanager
def open_archive_stream(source: str):
    source_path = Path(source).expanduser()
    if source_path.is_file():
        with source_path.open("rb") as archive_file:
            yield archive_file
        return
    request = Request(source, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(request, timeout=120) as response:
        yield response


def unique_image_file_names(rows: list[dict[str, Any]]) -> set[str]:
    return {image_file_name(row) for row in rows}


def image_file_name(row: dict[str, Any]) -> str:
    image = row.get("image")
    if isinstance(image, dict):
        for key in ("file_name", "image_name"):
            value = str(image.get(key) or "").strip()
            if value:
                return normalize_member_name(value)
    value = str(row.get("image_url") or "").strip()
    if value and "://" not in value and "#" not in value:
        return normalize_member_name(value)
    raise ValueError(f"Could not determine iNaturalist image file_name for row: {row}")


def image_destination(image_root: Path, file_name: str) -> Path:
    relative = safe_relative_member_path(file_name)
    return image_root.joinpath(*relative.parts)


def archive_member_uri(archive_url: str, file_name: str) -> str:
    return f"{archive_url}#{normalize_member_name(file_name)}"


def normalize_member_name(value: str) -> str:
    return str(PurePosixPath(value.strip().replace("\\", "/"))).lstrip("./")


def safe_relative_member_path(file_name: str) -> PurePosixPath:
    relative = PurePosixPath(normalize_member_name(file_name))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe archive member path: {file_name}")
    return relative


def _project_fields(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [{field: row[field] for field in fields} for row in rows]


def _write_stream_atomically(source: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        delete=False,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
    ) as temp_file:
        temp_path = Path(temp_file.name)
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            temp_file.write(chunk)
    temp_path.replace(destination)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    image_root = output_dir / "images"
    path_records, neighbor_candidate_report = load_path_records_with_optional_neighbors(
        kg_paths=args.kg_paths,
        neighbor_candidates=args.neighbor_candidates,
    )
    rows = generate_reasonvqa_rows(
        path_records=path_records,
        export_dir=output_dir,
        split=args.split,
        negative_strategy=args.negative_strategy,
        max_taxonomy_distance=args.max_taxonomy_distance,
        limit_qa=args.limit_qa,
    )
    rows = bind_local_image_paths(
        rows,
        image_root=image_root,
        archive_url=args.archive_url,
    )
    rows, answer_prior_report = filter_answer_prior_shortcuts(
        rows,
        group_fields=("property_id",),
        max_majority_fraction=args.max_answer_prior_fraction,
        min_group_size=args.min_answer_prior_group_size,
    )
    image_report = ImageAvailabilityReport(
        requested=len(unique_image_file_names(rows)),
        already_present=0,
        extracted=0,
        missing=[],
        invalid=[],
        scanned_members=0,
        image_bytes=0,
    )
    if not args.no_download_images:
        image_report = ensure_images_available(
            rows=rows,
            image_root=image_root,
            archive_source=args.archive_source,
            validate_images=not args.no_validate_images,
        )
        if (
            (image_report.missing or image_report.invalid)
            and not args.allow_missing_images
        ):
            raise RuntimeError(
                "Image preparation failed: "
                f"missing={image_report.missing[:10]}, invalid={image_report.invalid[:10]}"
            )

    write_bound_reasonvqa_exports(
        rows=rows,
        export_dir=output_dir,
        split=args.split,
        categories=DEFAULT_CATEGORIES,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "split": args.split,
                "input_path_records": len(path_records),
                "qa_rows": len(rows),
                "unique_images": len(unique_image_file_names(rows)),
                "hard_negative_policy": {
                    "negative_strategy": args.negative_strategy,
                    "max_taxonomy_distance": args.max_taxonomy_distance,
                },
                "neighbor_candidate_cache": neighbor_candidate_report,
                "answer_prior_filter": answer_prior_report,
                "image_report": image_report.__dict__,
                "exports": {
                    "qa": str(output_dir / f"{args.split}.jsonl"),
                    "ann": str(output_dir / f"{args.split}_ann.jsonl"),
                    "merged": str(output_dir / "knowledge_reasoning_vqa_merged.jsonl"),
                    "image_manifest": str(output_dir / "image_manifest.jsonl"),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
