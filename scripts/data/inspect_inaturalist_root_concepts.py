from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curator.utils.jsonl import write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a small iNaturalist annotation archive, inspect its schema, "
            "and link annotation categories to Wikidata root concepts."
        )
    )
    parser.add_argument("--download-dir", default="data/inaturalist_root_concepts")
    parser.add_argument("--split", default="val", choices=["val", "train_mini", "public_test"])
    parser.add_argument("--max-download-mb", type=int, default=100)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--wikidata-limit", type=int, default=3)
    parser.add_argument("--wikidata-timeout", type=int, default=30)
    parser.add_argument("--wikidata-resolved-ip", default=None)
    parser.add_argument("--use-google-doh", action="store_true")
    parser.add_argument("--skip-wikidata", action="store_true")
    parser.add_argument("--use-inat-api", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    module = _load_inaturalist_module()
    max_bytes = args.max_download_mb * 1024 * 1024
    download_dir = Path(args.download_dir)
    archive_path = module.download_inaturalist_annotation_archive(
        download_dir,
        split=args.split,
        max_bytes=max_bytes,
    )
    annotation_doc = module.load_inaturalist_annotation_document(archive_path)
    summary = module.summarize_inaturalist_annotation_document(annotation_doc)
    print(json.dumps({"archive": str(archive_path), "summary": summary}, ensure_ascii=False, indent=2))

    parsed_wikidata_host = "query.wikidata.org"
    wikidata_resolved_ip = args.wikidata_resolved_ip
    if args.use_google_doh and not wikidata_resolved_ip:
        wikidata_resolved_ip = module.resolve_host_via_google_doh(
            parsed_wikidata_host,
            timeout=args.wikidata_timeout,
        )

    records = []
    candidate_records = []
    candidates = module.iter_inaturalist_root_concept_candidates(
        annotation_doc,
        source=f"inaturalist_2021_{args.split}",
        limit=args.limit,
    )
    for candidate in candidates:
        inat_taxon_id = None
        query = module.build_wikidata_taxon_query(
            scientific_name=candidate.scientific_name,
            limit=args.wikidata_limit,
        )
        candidate_records.append(
            {
                "source": candidate.source,
                "image_id": candidate.image_id,
                "category_id": candidate.category_id,
                "scientific_name": candidate.scientific_name,
                "common_name": candidate.common_name,
                "taxonomy_path": candidate.taxonomy_path,
                "image_file_name": candidate.image_file_name,
                "wikidata_lookup_key": candidate.wikidata_lookup_key,
                "wikidata_sparql_query": query,
            }
        )
        if args.skip_wikidata:
            continue
        if args.use_inat_api:
            inat_taxon_id = module.fetch_inaturalist_taxon_id(
                candidate.scientific_name
            )
        try:
            concepts = module.query_wikidata_taxon(
                scientific_name=candidate.scientific_name,
                inaturalist_taxon_id=inat_taxon_id,
                limit=args.wikidata_limit,
                timeout=args.wikidata_timeout,
                resolved_ip=wikidata_resolved_ip,
            )
        except Exception as exc:
            candidate_records[-1]["link_error"] = f"{type(exc).__name__}: {exc}"
            continue
        if concepts:
            records.append(module.root_concept_record(candidate, concepts[0]))

    output_path = Path(args.output) if args.output else download_dir / "root_concepts.jsonl"
    candidate_output_path = download_dir / "root_concept_candidates.jsonl"
    write_jsonl(candidate_output_path, candidate_records)
    write_jsonl(output_path, records)
    print(
        json.dumps(
            {
                "candidate_records": len(candidate_records),
                "candidate_output": str(candidate_output_path),
                "wikidata_resolved_ip": wikidata_resolved_ip,
                "root_concept_records": len(records),
                "output": str(output_path),
                "sample_candidates": candidate_records[: min(3, len(candidate_records))],
                "sample_root_concepts": records[: min(3, len(records))],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _load_inaturalist_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root
        / "curator"
        / "operations"
        / "generation"
        / "recipes"
        / "knowledge_reasoning_vqa_inaturalist.py"
    )
    spec = importlib.util.spec_from_file_location(
        "knowledge_reasoning_vqa_inaturalist", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
