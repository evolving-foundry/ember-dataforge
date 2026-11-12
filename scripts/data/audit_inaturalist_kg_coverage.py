from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_designer.generation.algorithms.builtin.reasonvqa_inaturalist_audit import (  # noqa: E402
    summarize_inaturalist_kg_coverage,
)
from curator.utils.jsonl import write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample iNaturalist root concepts and audit Wikidata KG "
            "link/path coverage under a bounded property whitelist."
        )
    )
    parser.add_argument("--download-dir", default="data/inaturalist_root_concepts")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-download-mb", type=int, default=100)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--wikidata-timeout", type=int, default=60)
    parser.add_argument("--claim-batch-size", type=int, default=50)
    parser.add_argument("--wikidata-resolved-ip", default=None)
    parser.add_argument("--use-google-doh", action="store_true")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    module = _load_inaturalist_module()
    download_dir = Path(args.download_dir)
    output_prefix = args.output_prefix or f"audit_{args.sample_size}"
    archive_path = module.download_inaturalist_annotation_archive(
        download_dir,
        split=args.split,
        max_bytes=args.max_download_mb * 1024 * 1024,
    )
    annotation_doc = module.load_inaturalist_annotation_document(archive_path)
    candidates = _sample_unique_candidates(
        module=module,
        annotation_doc=annotation_doc,
        source=f"inaturalist_2021_{args.split}",
        sample_size=args.sample_size,
        random_seed=args.random_seed,
    )

    resolved_ip = args.wikidata_resolved_ip
    if args.use_google_doh and not resolved_ip:
        resolved_ip = module.resolve_host_via_google_doh(
            "query.wikidata.org",
            timeout=args.wikidata_timeout,
        )

    concepts_by_name = _query_linking_batches(
        module=module,
        scientific_names=[candidate.scientific_name for candidate in candidates],
        timeout=args.wikidata_timeout,
        resolved_ip=resolved_ip,
    )
    linked_records = []
    for candidate in candidates:
        concepts = concepts_by_name.get(candidate.scientific_name) or []
        if concepts:
            linked_records.append(module.root_concept_record(candidate, concepts[0]))

    root_qids = sorted(
        {
            str(record.get("root_concept", {}).get("wikidata_qid") or "").strip()
            for record in linked_records
            if record.get("root_concept", {}).get("wikidata_qid")
        }
    )
    whitelist = module.reasonvqa_claim_property_whitelist()
    claim_cache: dict[str, list[dict[str, Any]]] = {}

    def claim_fetcher(qids: list[str]) -> list[dict[str, Any]]:
        missing_qids = [qid for qid in qids if qid not in claim_cache]
        if missing_qids:
            fetched = _query_claim_batches(
                module=module,
                root_qids=missing_qids,
                whitelist=whitelist,
                timeout=args.wikidata_timeout,
                resolved_ip=resolved_ip,
                batch_size=args.claim_batch_size,
            )
            for qid in missing_qids:
                claim_cache[qid] = []
            for triple in fetched:
                claim_cache.setdefault(triple["root"]["qid"], []).append(triple)
        return [
            triple
            for qid in qids
            for triple in claim_cache.get(qid, [])
        ]

    root_claim_triples = claim_fetcher(root_qids) if root_qids else []
    paths = (
        module.expand_wikidata_claim_paths(
            root_qids=root_qids,
            max_hops=args.max_hops,
            whitelist=whitelist,
            claim_fetcher=claim_fetcher,
        )
        if root_qids
        else []
    )
    root_context_by_qid = {
        str(record.get("root_concept", {}).get("wikidata_qid")): record
        for record in linked_records
    }
    enriched_paths = [
        _with_source_context(path, root_context_by_qid.get(path["root"]["qid"], {}))
        for path in paths
    ]
    extension_claims = (
        _query_claim_batches(
            module=module,
            root_qids=root_qids,
            whitelist=_extension_candidate_properties(module),
            timeout=args.wikidata_timeout,
            resolved_ip=resolved_ip,
            batch_size=args.claim_batch_size,
        )
        if root_qids
        else []
    )

    summary = summarize_inaturalist_kg_coverage(
        sampled_count=len(candidates),
        linked_records=linked_records,
        root_claim_triples=root_claim_triples,
        paths=enriched_paths,
    )
    summary.update(
        {
            "requested_sample_size": args.sample_size,
            "random_seed": args.random_seed,
            "split": args.split,
            "max_hops": args.max_hops,
            "wikidata_resolved_ip": resolved_ip,
            "extension_candidate_hits": _extension_candidate_summary(
                extension_claims,
                linked_count=summary["linked_root_count"],
            ),
            "property_recommendations": _property_recommendations(
                summary,
                extension_claims,
            ),
        }
    )

    output_paths = {
        "candidates": download_dir / f"{output_prefix}_candidates.jsonl",
        "root_concepts": download_dir / f"{output_prefix}_root_concepts.jsonl",
        "root_claim_triples": download_dir / f"{output_prefix}_root_claim_triples.jsonl",
        "kg_paths": download_dir / f"{output_prefix}_kg_paths.jsonl",
        "extension_claims": download_dir / f"{output_prefix}_extension_claims.jsonl",
        "coverage_summary": download_dir / f"{output_prefix}_coverage_summary.json",
    }
    write_jsonl(output_paths["candidates"], [_candidate_record(candidate) for candidate in candidates])
    write_jsonl(output_paths["root_concepts"], linked_records)
    write_jsonl(output_paths["root_claim_triples"], root_claim_triples)
    write_jsonl(output_paths["kg_paths"], enriched_paths)
    write_jsonl(output_paths["extension_claims"], extension_claims)
    output_paths["coverage_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "outputs": {key: str(path) for key, path in output_paths.items()}}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _sample_unique_candidates(
    *,
    module: Any,
    annotation_doc: dict[str, Any],
    source: str,
    sample_size: int,
    random_seed: int,
) -> list[Any]:
    unique_by_name = {}
    for candidate in module.iter_inaturalist_root_concept_candidates(
        annotation_doc,
        source=source,
    ):
        unique_by_name.setdefault(candidate.scientific_name, candidate)
    candidates = list(unique_by_name.values())
    random.Random(random_seed).shuffle(candidates)
    return candidates[:sample_size]


def _query_linking_batches(
    *,
    module: Any,
    scientific_names: list[str],
    timeout: int,
    resolved_ip: str | None,
    batch_size: int = 100,
) -> dict[str, list[Any]]:
    results: dict[str, list[Any]] = {}
    for start in range(0, len(scientific_names), batch_size):
        batch = scientific_names[start : start + batch_size]
        results.update(
            module.query_wikidata_taxon_batch(
                batch,
                timeout=timeout,
                resolved_ip=resolved_ip,
            )
        )
    return results


def _query_claim_batches(
    *,
    module: Any,
    root_qids: list[str],
    whitelist: list[Any],
    timeout: int,
    resolved_ip: str | None,
    batch_size: int = 50,
) -> list[dict[str, Any]]:
    triples: list[dict[str, Any]] = []
    for start in range(0, len(root_qids), batch_size):
        batch = root_qids[start : start + batch_size]
        triples.extend(
            _query_claim_batch_with_502_split(
                module=module,
                root_qids=batch,
                whitelist=whitelist,
                timeout=timeout,
                resolved_ip=resolved_ip,
            )
        )
    return triples


def _query_claim_batch_with_502_split(
    *,
    module: Any,
    root_qids: list[str],
    whitelist: list[Any],
    timeout: int,
    resolved_ip: str | None,
) -> list[dict[str, Any]]:
    try:
        return module.query_wikidata_claim_triples(
            root_qids=root_qids,
            whitelist=whitelist,
            timeout=timeout,
            resolved_ip=resolved_ip,
        )
    except ValueError as exc:
        if "HTTP 502" not in str(exc) or len(root_qids) <= 1:
            raise
        split_at = max(1, len(root_qids) // 2)
        return [
            *_query_claim_batch_with_502_split(
                module=module,
                root_qids=root_qids[:split_at],
                whitelist=whitelist,
                timeout=timeout,
                resolved_ip=resolved_ip,
            ),
            *_query_claim_batch_with_502_split(
                module=module,
                root_qids=root_qids[split_at:],
                whitelist=whitelist,
                timeout=timeout,
                resolved_ip=resolved_ip,
            ),
        ]


def _extension_candidate_properties(module: Any) -> list[Any]:
    return [
        module.ReasonVqaClaimProperty(
            property_id="P2974",
            label="habitat",
            group="ecology",
            answer_type="concept",
            description="Habitat concept for ecology-oriented questions.",
        ),
        module.ReasonVqaClaimProperty(
            property_id="P181",
            label="taxon range map image",
            group="media",
            answer_type="media",
            description="Range-map media, useful for validation but not direct text QA.",
        ),
        module.ReasonVqaClaimProperty(
            property_id="P18",
            label="image",
            group="media",
            answer_type="media",
            description="Wikidata image, not a KG answer target.",
        ),
        module.ReasonVqaClaimProperty(
            property_id="P685",
            label="NCBI taxonomy ID",
            group="literal",
            answer_type="external_id",
            description="External taxonomy ID, useful for linking only.",
        ),
        module.ReasonVqaClaimProperty(
            property_id="P846",
            label="GBIF taxon ID",
            group="literal",
            answer_type="external_id",
            description="External taxonomy ID, useful for linking only.",
        ),
    ]


def _extension_candidate_summary(
    extension_claims: list[dict[str, Any]],
    linked_count: int,
) -> dict[str, Any]:
    property_counts = Counter(
        triple["property"]["id"] for triple in extension_claims
    )
    roots_by_property: dict[str, set[str]] = {}
    for triple in extension_claims:
        roots_by_property.setdefault(triple["property"]["id"], set()).add(
            triple["root"]["qid"]
        )
    return {
        "claim_counts": dict(sorted(property_counts.items())),
        "root_hit_counts": {
            property_id: len(root_qids)
            for property_id, root_qids in sorted(roots_by_property.items())
        },
        "root_hit_rates": {
            property_id: (len(root_qids) / linked_count if linked_count else 0.0)
            for property_id, root_qids in sorted(roots_by_property.items())
        },
    }


def _property_recommendations(
    summary: dict[str, Any],
    extension_claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    extension_hit_summary = _extension_candidate_summary(
        extension_claims,
        linked_count=summary["linked_root_count"],
    )
    hit_rates = extension_hit_summary["root_hit_rates"]
    linked_count = summary["linked_root_count"]
    root_claim_counts = summary["property_hit_distribution"]["root_claim_counts"]
    p141_hit_rate = (
        root_claim_counts.get("P141", 0) / linked_count if linked_count else 0.0
    )
    return [
        {
            "property_id": "P141",
            "label": "IUCN conservation status",
            "decision": "added_to_whitelist_as_optional_final_answer_property",
            "reason": (
                "Domain-relevant categorical answer; use fixed/concept choices "
                f"if coverage is acceptable. Observed root hit rate: {p141_hit_rate:.4f}."
            ),
            "template": "What is the conservation status of this taxon?",
        },
        {
            "property_id": "P2974",
            "label": "habitat",
            "decision": "audit_on_larger_sample_before_adding",
            "reason": (
                "Good ecology reasoning target, but should only be added if "
                f"coverage is not sparse. Observed root hit rate: {hit_rates.get('P2974', 0.0):.4f}."
            ),
            "template": "What habitat is associated with this taxon?",
        },
        {
            "property_id": "P183",
            "label": "endemic to",
            "decision": "keep_but_treat_as_sparse_geography",
            "reason": (
                "Useful for geography constraints, but current geography coverage "
                f"is {summary['rates']['geography_coverage']:.4f}; avoid relying on it as the only geography source."
            ),
        },
        {
            "property_id": "P3151",
            "label": "iNaturalist taxon ID",
            "decision": "keep_for_linking_exclude_from_question_templates",
            "reason": "Stable linking key, but external IDs are weak ReasoningVQA answers.",
        },
        {
            "property_id": "P685/P846",
            "label": "NCBI taxonomy ID / GBIF taxon ID",
            "decision": "do_not_add_as_question_answer_properties",
            "reason": "External IDs may help linking/debugging but would increase non-semantic answer bias.",
        },
        {
            "property_id": "P18/P181",
            "label": "image / taxon range map image",
            "decision": "do_not_add_to_text_qa_whitelist",
            "reason": "Media claims are not literal/concept text answers for the current KG QA pipeline.",
        },
    ]


def _candidate_record(candidate: Any) -> dict[str, Any]:
    return {
        "source": candidate.source,
        "image": {
            "image_id": candidate.image_id,
            "file_name": candidate.image_file_name,
            "metadata": candidate.image_metadata or {},
            "license": candidate.license_metadata or {},
        },
        "inaturalist": {
            "annotation_id": candidate.annotation_id,
            "category_id": candidate.category_id,
            "scientific_name": candidate.scientific_name,
            "common_name": candidate.common_name,
            "taxonomy": candidate.taxonomy,
            "taxonomy_path": candidate.taxonomy_path,
            "wikidata_lookup_key": candidate.wikidata_lookup_key,
        },
    }


def _with_source_context(
    path_record: dict[str, Any],
    root_record: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(path_record)
    enriched["root_concept"] = root_record.get("root_concept", {})
    enriched["inaturalist"] = root_record.get("inaturalist", {})
    enriched["image"] = root_record.get("image", {})
    return enriched


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
        "knowledge_reasoning_vqa_inaturalist",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
