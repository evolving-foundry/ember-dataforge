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
            "Build a JSONL cache of Wikidata same-genus and same-family taxon "
            "neighbors for iNaturalist ReasonVQA hard MC distractors."
        )
    )
    parser.add_argument(
        "--input",
        default="data/inaturalist_root_concepts/audit_200_root_concepts.jsonl",
        help="Root concept JSONL with root_concept.wikidata_qid fields.",
    )
    parser.add_argument(
        "--output",
        default="data/inaturalist_root_concepts/audit_200_taxon_neighbor_candidates.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "data/inaturalist_root_concepts/"
            "audit_200_taxon_neighbor_candidates.summary.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--per-relation-limit", type=int, default=25)
    parser.add_argument("--wikidata-timeout", type=int, default=60)
    parser.add_argument("--wikidata-resolved-ip", default=None)
    parser.add_argument("--use-google-doh", action="store_true")
    args = parser.parse_args()

    module = _load_inaturalist_module()
    root_records = _load_json_records(Path(args.input))
    root_qids = _root_qids(root_records)
    if not root_qids:
        raise ValueError(f"No root_concept.wikidata_qid found in {args.input}")

    resolved_ip = args.wikidata_resolved_ip
    if args.use_google_doh and not resolved_ip:
        resolved_ip = module.resolve_host_via_google_doh(
            "query.wikidata.org",
            timeout=args.wikidata_timeout,
        )

    rows: list[dict[str, Any]] = []
    for start in range(0, len(root_qids), args.batch_size):
        batch = root_qids[start : start + args.batch_size]
        rows.extend(
            module.query_wikidata_taxon_neighbors(
                batch,
                timeout=args.wikidata_timeout,
                resolved_ip=resolved_ip,
                per_relation_limit=args.per_relation_limit,
            )
        )

    output = Path(args.output)
    write_jsonl(output, rows)
    summary = _summary(
        root_qids=root_qids,
        rows=rows,
        per_relation_limit=args.per_relation_limit,
        resolved_ip=resolved_ip,
    )
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "outputs": {
                    "neighbor_candidates": str(output),
                    "summary": str(summary_output),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _root_qids(root_records: list[dict[str, Any]]) -> list[str]:
    qids = []
    seen = set()
    for record in root_records:
        root_concept = record.get("root_concept") or {}
        qid = str(
            root_concept.get("wikidata_qid") or root_concept.get("qid") or ""
        ).strip()
        if qid and qid not in seen:
            seen.add(qid)
            qids.append(qid)
    return qids


def _summary(
    *,
    root_qids: list[str],
    rows: list[dict[str, Any]],
    per_relation_limit: int,
    resolved_ip: str | None,
) -> dict[str, Any]:
    property_counts = Counter(str(row.get("property_id") or "") for row in rows)
    relation_counts: Counter[str] = Counter()
    candidate_count = 0
    roots_with_candidates = set()
    for row in rows:
        root_qid = str(row.get("root_qid") or "")
        if row.get("candidates"):
            roots_with_candidates.add(root_qid)
        for candidate in row.get("candidates") or []:
            candidate_count += 1
            relation_counts[str(candidate.get("relation") or "")] += 1
    return {
        "root_count": len(root_qids),
        "roots_with_candidates": len(roots_with_candidates),
        "cache_rows": len(rows),
        "candidate_count": candidate_count,
        "property_counts": dict(sorted(property_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "per_relation_limit": per_relation_limit,
        "wikidata_resolved_ip": resolved_ip,
        "sample_rows": rows[: min(5, len(rows))],
    }


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


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
