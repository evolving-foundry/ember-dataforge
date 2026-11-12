from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curator.utils.jsonl import write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Expand iNaturalist Wikidata root concepts into whitelisted "
            "ReasonVQA-style KG triples."
        )
    )
    parser.add_argument(
        "--input",
        default="data/inaturalist_root_concepts/root_concepts.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/inaturalist_root_concepts/root_concept_kg_triples.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default="data/inaturalist_root_concepts/root_concept_kg_triples.summary.json",
    )
    parser.add_argument("--wikidata-timeout", type=int, default=30)
    parser.add_argument("--wikidata-resolved-ip", default=None)
    parser.add_argument("--use-google-doh", action="store_true")
    args = parser.parse_args()

    module = _load_inaturalist_module()
    root_records = _load_json_records(Path(args.input))
    root_qids = [
        str(record.get("root_concept", {}).get("wikidata_qid") or "").strip()
        for record in root_records
    ]
    root_qids = [qid for qid in root_qids if qid]
    if not root_qids:
        raise ValueError(f"No root_concept.wikidata_qid found in {args.input}")

    resolved_ip = args.wikidata_resolved_ip
    if args.use_google_doh and not resolved_ip:
        resolved_ip = module.resolve_host_via_google_doh(
            "query.wikidata.org",
            timeout=args.wikidata_timeout,
        )

    whitelist = module.reasonvqa_claim_property_whitelist()
    triples = module.query_wikidata_claim_triples(
        root_qids=sorted(set(root_qids)),
        whitelist=whitelist,
        timeout=args.wikidata_timeout,
        resolved_ip=resolved_ip,
    )
    context_by_qid = {
        str(record.get("root_concept", {}).get("wikidata_qid")): record
        for record in root_records
    }
    enriched_triples = [
        _with_source_context(triple, context_by_qid.get(triple["root"]["qid"], {}))
        for triple in triples
    ]

    write_jsonl(Path(args.output), enriched_triples)
    summary = _summary(
        triples=enriched_triples,
        whitelist=whitelist,
        root_qids=root_qids,
        resolved_ip=resolved_ip,
    )
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                return [json.loads(text)]
    return records


def _with_source_context(
    triple: dict[str, Any],
    root_record: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(triple)
    enriched["root_concept"] = root_record.get("root_concept", {})
    enriched["inaturalist"] = root_record.get("inaturalist", {})
    enriched["image"] = root_record.get("image", {})
    return enriched


def _summary(
    *,
    triples: list[dict[str, Any]],
    whitelist: list[Any],
    root_qids: list[str],
    resolved_ip: str | None,
) -> dict[str, Any]:
    group_counts = Counter(triple["property"]["group"] for triple in triples)
    property_counts = Counter(triple["property"]["id"] for triple in triples)
    return {
        "root_count": len(set(root_qids)),
        "triple_count": len(triples),
        "property_whitelist": [
            {
                "property_id": item.property_id,
                "label": item.label,
                "group": item.group,
                "answer_type": item.answer_type,
            }
            for item in whitelist
        ],
        "group_counts": dict(sorted(group_counts.items())),
        "property_counts": dict(sorted(property_counts.items())),
        "wikidata_resolved_ip": resolved_ip,
        "sample_triples": triples[: min(8, len(triples))],
    }


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
