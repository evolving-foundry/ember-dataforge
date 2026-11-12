from __future__ import annotations

import gzip
import http.client
import json
import os
import socket
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


INATURALIST_2021_ANNOTATION_URLS = {
    "val": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz",
    "train_mini": (
        "https://ml-inat-competition-datasets.s3.amazonaws.com/"
        "2021/train_mini.json.tar.gz"
    ),
    "public_test": (
        "https://ml-inat-competition-datasets.s3.amazonaws.com/"
        "2021/public_test.json.tar.gz"
    ),
}
DEFAULT_MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
DEFAULT_USER_AGENT = "DataForge-iNaturalist-root-concepts/0.1"
DEFAULT_WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_SPARQL_ENDPOINT = os.environ.get(
    "WIKIDATA_SPARQL_ENDPOINT",
    DEFAULT_WIKIDATA_SPARQL_ENDPOINT,
)
WIKIDATA_SPARQL_PREFIXES = """PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>"""
GOOGLE_DOH_ENDPOINT = "https://dns.google/resolve"
INATURALIST_TAXA_ENDPOINT = "https://api.inaturalist.org/v1/taxa/autocomplete"


@dataclass(frozen=True)
class INaturalistRootConceptCandidate:
    source: str
    image_id: int | str
    category_id: int | str
    scientific_name: str
    common_name: str
    taxonomy: dict[str, str]
    taxonomy_path: list[str]
    image_file_name: str
    annotation_id: int | str | None = None
    image_metadata: dict[str, Any] | None = None
    category_metadata: dict[str, Any] | None = None
    license_metadata: dict[str, Any] | None = None
    inaturalist_taxon_id: str | None = None

    @property
    def wikidata_lookup_key(self) -> str:
        return self.scientific_name


@dataclass(frozen=True)
class WikidataRootConcept:
    qid: str
    uri: str
    label: str
    taxon_name: str | None = None
    inaturalist_taxon_id: str | None = None
    taxon_rank: str | None = None
    linking_method: str = "wikidata_taxon"


@dataclass(frozen=True)
class ReasonVqaClaimProperty:
    property_id: str
    label: str
    group: str
    answer_type: str
    description: str


def reasonvqa_claim_property_whitelist() -> list[ReasonVqaClaimProperty]:
    """Whitelisted Wikidata properties for iNaturalist ReasonVQA KG expansion."""

    return [
        ReasonVqaClaimProperty(
            property_id="P105",
            label="taxon rank",
            group="taxonomy",
            answer_type="concept",
            description="Taxonomic rank for the observed taxon.",
        ),
        ReasonVqaClaimProperty(
            property_id="P141",
            label="IUCN conservation status",
            group="conservation",
            answer_type="fixed",
            description="IUCN conservation category, e.g. least concern or endangered.",
        ),
        ReasonVqaClaimProperty(
            property_id="P171",
            label="parent taxon",
            group="taxonomy",
            answer_type="concept",
            description="Immediate parent taxon, e.g. species to genus.",
        ),
        ReasonVqaClaimProperty(
            property_id="P183",
            label="endemic to",
            group="geography",
            answer_type="concept",
            description="Geographic entity where the taxon is endemic.",
        ),
        ReasonVqaClaimProperty(
            property_id="P225",
            label="taxon name",
            group="literal",
            answer_type="literal",
            description="Scientific taxon name.",
        ),
        ReasonVqaClaimProperty(
            property_id="P1843",
            label="taxon common name",
            group="literal",
            answer_type="literal",
            description="English common name for the taxon.",
        ),
        ReasonVqaClaimProperty(
            property_id="P3151",
            label="iNaturalist taxon ID",
            group="literal",
            answer_type="external_id",
            description="iNaturalist taxon identifier used for linking.",
        ),
    ]


def download_inaturalist_annotation_archive(
    output_dir: str | Path,
    split: str = "val",
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    url: str | None = None,
) -> Path:
    """Download a small iNaturalist annotation archive with a hard byte cap."""

    source_url = url or INATURALIST_2021_ANNOTATION_URLS[split]
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / Path(source_url).name
    if archive_path.exists():
        _ensure_size_within_limit(archive_path.stat().st_size, max_bytes, archive_path)
        return archive_path

    request = Request(source_url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(request, timeout=60) as response:
        content_length = response.headers.get("Content-Length")
        if content_length:
            _ensure_size_within_limit(int(content_length), max_bytes, source_url)
        downloaded = 0
        with archive_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                downloaded += len(chunk)
                _ensure_size_within_limit(downloaded, max_bytes, source_url)
                output.write(chunk)
    return archive_path


def load_inaturalist_annotation_document(path: str | Path) -> dict[str, Any]:
    annotation_path = Path(path).expanduser()
    suffixes = "".join(annotation_path.suffixes).lower()
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(annotation_path, "r:gz") as archive:
            json_member = next(
                (
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.endswith(".json")
                ),
                None,
            )
            if json_member is None:
                raise ValueError(f"No JSON member found in archive: {annotation_path}")
            extracted = archive.extractfile(json_member)
            if extracted is None:
                raise ValueError(f"Could not read JSON member: {json_member.name}")
            return json.load(extracted)
    if suffixes.endswith(".json.gz"):
        with gzip.open(annotation_path, "rt", encoding="utf-8") as input_file:
            return json.load(input_file)
    with annotation_path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def summarize_inaturalist_annotation_document(
    annotation_doc: dict[str, Any],
    sample_count: int = 1,
) -> dict[str, Any]:
    return {
        "top_level_keys": sorted(annotation_doc),
        "counts": {
            "images": len(annotation_doc.get("images") or []),
            "categories": len(annotation_doc.get("categories") or []),
            "annotations": len(annotation_doc.get("annotations") or []),
            "licenses": len(annotation_doc.get("licenses") or []),
        },
        "sample_images": list(annotation_doc.get("images") or [])[:sample_count],
        "sample_categories": list(annotation_doc.get("categories") or [])[:sample_count],
        "sample_annotations": list(annotation_doc.get("annotations") or [])[:sample_count],
        "sample_licenses": list(annotation_doc.get("licenses") or [])[:sample_count],
    }


def iter_inaturalist_root_concept_candidates(
    annotation_doc: dict[str, Any],
    source: str = "inaturalist",
    limit: int | None = None,
) -> Iterable[INaturalistRootConceptCandidate]:
    images_by_id = _index_by_id(annotation_doc.get("images") or [])
    categories_by_id = _index_by_id(annotation_doc.get("categories") or [])
    licenses_by_id = _index_by_id(annotation_doc.get("licenses") or [])
    annotations = annotation_doc.get("annotations") or []
    yielded = 0
    for annotation in annotations:
        image = images_by_id.get(annotation.get("image_id"))
        category = categories_by_id.get(annotation.get("category_id"))
        if image is None or category is None:
            raise ValueError(
                "iNaturalist annotation references missing image or category: "
                f"{annotation}"
            )
        license_metadata = licenses_by_id.get(image.get("license"))
        yield _candidate_from_records(source, annotation, image, category, license_metadata)
        yielded += 1
        if limit is not None and yielded >= limit:
            break


def build_wikidata_taxon_query(
    scientific_name: str,
    inaturalist_taxon_id: str | None = None,
    limit: int = 5,
) -> str:
    scientific_literal = _sparql_string_literal(scientific_name)
    clauses = [f"?item wdt:P225 {scientific_literal} ."]
    if inaturalist_taxon_id:
        clauses.append(f"?item wdt:P3151 {_sparql_string_literal(inaturalist_taxon_id)} .")
    union = "\n  UNION\n  ".join(f"{{ {clause} }}" for clause in clauses)
    return f"""{WIKIDATA_SPARQL_PREFIXES}
SELECT ?item ?itemLabel ?taxonName ?inatTaxonId ?taxonRankLabel WHERE {{
  {union}
  OPTIONAL {{ ?item wdt:P225 ?taxonName. }}
  OPTIONAL {{ ?item wdt:P3151 ?inatTaxonId. }}
  OPTIONAL {{ ?item rdfs:label ?itemLabel. FILTER(LANG(?itemLabel) = "en") }}
  OPTIONAL {{
    ?item wdt:P105 ?taxonRank.
    ?taxonRank rdfs:label ?taxonRankLabel.
    FILTER(LANG(?taxonRankLabel) = "en")
  }}
}}
	LIMIT {int(limit)}"""


def build_wikidata_taxon_batch_query(
    scientific_names: list[str],
) -> str:
    names = [name.strip() for name in dict.fromkeys(scientific_names) if name.strip()]
    if not names:
        raise ValueError("At least one scientific name is required for batch linking.")
    taxon_values = " ".join(_sparql_string_literal(name) for name in names)
    return f"""{WIKIDATA_SPARQL_PREFIXES}
SELECT ?requestedTaxonName ?item ?itemLabel ?taxonName ?inatTaxonId ?taxonRankLabel WHERE {{
  VALUES ?requestedTaxonName {{ {taxon_values} }}
  ?item wdt:P225 ?requestedTaxonName.
  OPTIONAL {{ ?item wdt:P225 ?taxonName. }}
  OPTIONAL {{ ?item wdt:P3151 ?inatTaxonId. }}
  OPTIONAL {{ ?item rdfs:label ?itemLabel. FILTER(LANG(?itemLabel) = "en") }}
  OPTIONAL {{
    ?item wdt:P105 ?taxonRank.
    ?taxonRank rdfs:label ?taxonRankLabel.
    FILTER(LANG(?taxonRankLabel) = "en")
  }}
}}
ORDER BY ?requestedTaxonName ?item"""


def query_wikidata_taxon(
    scientific_name: str,
    inaturalist_taxon_id: str | None = None,
    limit: int = 5,
    endpoint: str = WIKIDATA_SPARQL_ENDPOINT,
    timeout: int = 30,
    resolved_ip: str | None = None,
) -> list[WikidataRootConcept]:
    query = build_wikidata_taxon_query(
        scientific_name=scientific_name,
        inaturalist_taxon_id=inaturalist_taxon_id,
        limit=limit,
    )
    return parse_wikidata_root_concepts(
        _query_sparql_json(
            query=query,
            endpoint=endpoint,
            timeout=timeout,
            resolved_ip=resolved_ip,
        )
    )


def query_wikidata_taxon_batch(
    scientific_names: list[str],
    endpoint: str = WIKIDATA_SPARQL_ENDPOINT,
    timeout: int = 30,
    resolved_ip: str | None = None,
) -> dict[str, list[WikidataRootConcept]]:
    query = build_wikidata_taxon_batch_query(scientific_names)
    return parse_wikidata_taxon_batch_results(
        _query_sparql_json(
            query=query,
            endpoint=endpoint,
            timeout=timeout,
            resolved_ip=resolved_ip,
        )
    )


def build_wikidata_claims_query(
    root_qids: list[str],
    whitelist: list[ReasonVqaClaimProperty] | None = None,
) -> str:
    if not root_qids:
        raise ValueError("At least one root QID is required for KG claim expansion.")
    properties = whitelist or reasonvqa_claim_property_whitelist()
    root_values = " ".join(f"wd:{_sparql_qid(qid)}" for qid in root_qids)
    property_values = "\n    ".join(
        (
            f'("{item.property_id}" "{item.group}" "{item.answer_type}")'
        )
        for item in properties
    )
    return f"""{WIKIDATA_SPARQL_PREFIXES}
SELECT ?root ?rootLabel ?property ?propertyId ?propertyLabel ?propertyGroup ?answerType ?value ?valueLabel WHERE {{
  VALUES ?root {{ {root_values} }}
  VALUES (?propertyId ?propertyGroup ?answerType) {{
    {property_values}
  }}
  ?wikidataProperty wikibase:directClaim ?property.
  BIND(STRAFTER(STR(?wikidataProperty), "http://www.wikidata.org/entity/") AS ?actualPropertyId)
  FILTER(?actualPropertyId = ?propertyId)
  ?root ?property ?value.
  FILTER(?propertyId != "P1843" || LANG(?value) = "en")
  OPTIONAL {{
    ?root rdfs:label ?rootLabel.
    FILTER(LANG(?rootLabel) = "en")
  }}
  ?wikidataProperty rdfs:label ?propertyLabel.
  FILTER(LANG(?propertyLabel) = "en")
  OPTIONAL {{
    ?value rdfs:label ?valueLabel.
    FILTER(LANG(?valueLabel) = "en")
  }}
}}
ORDER BY ?root ?propertyId ?value"""


def query_wikidata_claim_triples(
    root_qids: list[str],
    whitelist: list[ReasonVqaClaimProperty] | None = None,
    endpoint: str = WIKIDATA_SPARQL_ENDPOINT,
    timeout: int = 30,
    resolved_ip: str | None = None,
) -> list[dict[str, Any]]:
    query = build_wikidata_claims_query(root_qids, whitelist)
    return parse_wikidata_claim_triples(
        _query_sparql_json(
            query=query,
            endpoint=endpoint,
            timeout=timeout,
            resolved_ip=resolved_ip,
        )
    )


def build_wikidata_taxon_neighbor_query(
    root_qids: list[str],
    per_relation_limit: int = 25,
) -> str:
    if not root_qids:
        raise ValueError("At least one root QID is required for taxon neighbors.")
    root_values = " ".join(f"wd:{_sparql_qid(qid)}" for qid in root_qids)
    limit = int(per_relation_limit)
    if limit < 1:
        raise ValueError("per_relation_limit must be positive.")
    return f"""{WIKIDATA_SPARQL_PREFIXES}
SELECT ?root ?rootLabel ?candidate ?candidateLabel ?candidateTaxonName ?candidateCommonName ?relation WHERE {{
  {{
    SELECT ?root ?candidate ?relation WHERE {{
      VALUES ?root {{ {root_values} }}
      ?root wdt:P171 ?genus.
      ?candidate wdt:P171 ?genus.
      FILTER(?candidate != ?root)
      BIND("same_genus" AS ?relation)
    }}
    LIMIT {limit}
  }}
  UNION
  {{
    SELECT ?root ?candidate ?relation WHERE {{
      VALUES ?root {{ {root_values} }}
      ?root wdt:P171 ?genus.
      ?genus wdt:P171 ?family.
      ?candidate wdt:P171 ?candidateGenus.
      ?candidateGenus wdt:P171 ?family.
      FILTER(?candidate != ?root)
      FILTER(?candidateGenus != ?genus)
      BIND("same_family" AS ?relation)
    }}
    LIMIT {limit}
  }}
  ?candidate wdt:P225 ?candidateTaxonName.
  OPTIONAL {{ ?candidate wdt:P1843 ?candidateCommonName. FILTER(LANG(?candidateCommonName) = "en") }}
  OPTIONAL {{ ?root rdfs:label ?rootLabel. FILTER(LANG(?rootLabel) = "en") }}
  OPTIONAL {{ ?candidate rdfs:label ?candidateLabel. FILTER(LANG(?candidateLabel) = "en") }}
}}
ORDER BY ?root ?relation ?candidateTaxonName"""


def query_wikidata_taxon_neighbors(
    root_qids: list[str],
    endpoint: str = WIKIDATA_SPARQL_ENDPOINT,
    timeout: int = 30,
    resolved_ip: str | None = None,
    per_relation_limit: int = 25,
) -> list[dict[str, Any]]:
    query = build_wikidata_taxon_neighbor_query(
        root_qids,
        per_relation_limit=per_relation_limit,
    )
    return parse_wikidata_taxon_neighbor_candidates(
        _query_sparql_json(
            query=query,
            endpoint=endpoint,
            timeout=timeout,
            resolved_ip=resolved_ip,
        )
    )


def parse_wikidata_taxon_neighbor_candidates(
    sparql_result: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    seen: dict[tuple[str, str], set[str]] = {}
    for binding in sparql_result.get("results", {}).get("bindings", []):
        root_uri = _binding_value(binding, "root")
        candidate_uri = _binding_value(binding, "candidate")
        root_qid = _qid_from_uri(root_uri)
        candidate_qid = _qid_from_uri(candidate_uri)
        if not root_qid or not candidate_qid:
            continue
        root_label = _binding_value(binding, "rootLabel") or root_qid
        relation = _binding_value(binding, "relation") or ""
        taxon_name = _binding_value(binding, "candidateTaxonName")
        common_name = _binding_value(binding, "candidateCommonName")
        _add_taxon_neighbor_candidate(
            grouped,
            seen,
            root_qid=root_qid,
            root_label=root_label,
            property_id="P225",
            label=taxon_name,
            candidate_qid=candidate_qid,
            relation=relation,
        )
        _add_taxon_neighbor_candidate(
            grouped,
            seen,
            root_qid=root_qid,
            root_label=root_label,
            property_id="P1843",
            label=common_name,
            candidate_qid=candidate_qid,
            relation=relation,
        )
    return [
        grouped[key]
        for key in sorted(
            grouped,
            key=lambda item: (item[0], 0 if item[1] == "P225" else 1, item[1]),
        )
    ]


def _add_taxon_neighbor_candidate(
    grouped: dict[tuple[str, str], dict[str, Any]],
    seen: dict[tuple[str, str], set[str]],
    *,
    root_qid: str,
    root_label: str,
    property_id: str,
    label: str | None,
    candidate_qid: str,
    relation: str,
) -> None:
    if not label:
        return
    route = [property_id]
    key = (root_qid, property_id)
    normalized_label = " ".join(str(label).casefold().split())
    seen.setdefault(key, set())
    if not normalized_label or normalized_label in seen[key]:
        return
    seen[key].add(normalized_label)
    grouped.setdefault(
        key,
        {
            "root_qid": root_qid,
            "root_label": root_label,
            "property_id": property_id,
            "route_properties": route,
            "candidates": [],
        },
    )["candidates"].append(
        {
            "label": str(label),
            "qid": candidate_qid,
            "relation": relation,
            "source": "wikidata_sparql_taxon_neighbors",
        }
    )


def expand_wikidata_claim_paths(
    root_qids: list[str],
    max_hops: int = 3,
    whitelist: list[ReasonVqaClaimProperty] | None = None,
    endpoint: str = WIKIDATA_SPARQL_ENDPOINT,
    timeout: int = 30,
    resolved_ip: str | None = None,
    claim_fetcher: Callable[[list[str]], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Traverse whitelisted KG claims from roots until terminal answer claims.

    Concept-valued tails become the next frontier. Literal/date/number-like tails
    stop the route and are emitted as final-answer KG paths.
    """

    if max_hops < 1:
        raise ValueError("max_hops must be at least 1.")
    properties = whitelist or reasonvqa_claim_property_whitelist()
    allowed_property_ids = {item.property_id for item in properties}
    normalized_roots = [_sparql_qid(qid) for qid in root_qids]
    if not normalized_roots:
        raise ValueError("At least one root QID is required for KG path expansion.")

    def fetch_claims(qids: list[str]) -> list[dict[str, Any]]:
        if claim_fetcher is not None:
            return claim_fetcher(qids)
        return query_wikidata_claim_triples(
            qids,
            whitelist=properties,
            endpoint=endpoint,
            timeout=timeout,
            resolved_ip=resolved_ip,
        )

    frontier: dict[str, list[list[dict[str, Any]]]] = {
        qid: [[]] for qid in dict.fromkeys(normalized_roots)
    }
    completed_paths: list[dict[str, Any]] = []
    emitted_path_keys: set[tuple[tuple[str, str, str], ...]] = set()

    for _depth in range(max_hops):
        if not frontier:
            break
        current_qids = list(frontier)
        triples_by_root = _group_claim_triples_by_root(
            fetch_claims(current_qids),
            allowed_property_ids,
        )
        next_frontier: dict[str, list[list[dict[str, Any]]]] = {}
        for current_qid in current_qids:
            prefixes = frontier.get(current_qid, [])
            for prefix in prefixes:
                for triple in triples_by_root.get(current_qid, []):
                    path = [*prefix, triple]
                    path_key = _claim_path_key(path)
                    if path_key in emitted_path_keys:
                        continue
                    emitted_path_keys.add(path_key)
                    if _is_terminal_claim_triple(triple):
                        completed_paths.append(_claim_path_record(path))
                        continue
                    if len(path) >= max_hops:
                        continue
                    tail_qid = str(triple.get("tail", {}).get("qid") or "").strip()
                    if not tail_qid or tail_qid in _route_qids_for_path(path):
                        continue
                    next_frontier.setdefault(tail_qid, []).append(path)
        frontier = next_frontier
    return completed_paths


def parse_wikidata_claim_triples(
    sparql_result: dict[str, Any],
) -> list[dict[str, Any]]:
    triples: list[dict[str, Any]] = []
    for binding in sparql_result.get("results", {}).get("bindings", []):
        root_uri = _binding_value(binding, "root")
        property_uri = _binding_value(binding, "property")
        value = binding.get("value") or {}
        value_uri_or_literal = str(value.get("value") or "")
        root_qid = _qid_from_uri(root_uri)
        tail = _claim_tail(binding, value)
        triples.append(
            {
                "root": {
                    "qid": root_qid,
                    "uri": root_uri,
                    "label": _binding_value(binding, "rootLabel") or root_qid,
                },
                "property": {
                    "id": _binding_value(binding, "propertyId"),
                    "label": _binding_value(binding, "propertyLabel"),
                    "group": _binding_value(binding, "propertyGroup"),
                    "answer_type": _binding_value(binding, "answerType"),
                },
                "tail": tail,
                "triple": {
                    "subject": root_qid,
                    "predicate": _binding_value(binding, "propertyId"),
                    "object": tail.get("qid") or value_uri_or_literal,
                },
                "hop": 1,
                "source": "wikidata_sparql_whitelist",
                "property_uri": property_uri,
            }
        )
    return triples


def resolve_host_via_google_doh(
    host: str,
    timeout: int = 10,
    endpoint: str = GOOGLE_DOH_ENDPOINT,
) -> str:
    payload = urlencode({"name": host, "type": "A"})
    request = Request(
        f"{endpoint}?{payload}",
        headers={
            "Accept": "application/dns-json",
            "User-Agent": DEFAULT_USER_AGENT,
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return parse_dns_google_a_record(json.load(response))


def parse_dns_google_a_record(dns_result: dict[str, Any]) -> str:
    for answer in dns_result.get("Answer") or []:
        if int(answer.get("type") or 0) == 1 and answer.get("data"):
            return str(answer["data"])
    raise ValueError(f"DNS-over-HTTPS response contains no A record: {dns_result}")


def parse_wikidata_root_concepts(
    sparql_result: dict[str, Any],
) -> list[WikidataRootConcept]:
    concepts: list[WikidataRootConcept] = []
    for binding in sparql_result.get("results", {}).get("bindings", []):
        item_uri = _binding_value(binding, "item")
        if not item_uri:
            continue
        qid = item_uri.rstrip("/").rsplit("/", 1)[-1]
        concepts.append(
            WikidataRootConcept(
                qid=qid,
                uri=item_uri,
                label=_binding_value(binding, "itemLabel") or qid,
                taxon_name=_binding_value(binding, "taxonName"),
                inaturalist_taxon_id=_binding_value(binding, "inatTaxonId"),
                taxon_rank=_binding_value(binding, "taxonRankLabel"),
            )
        )
    return concepts


def parse_wikidata_taxon_batch_results(
    sparql_result: dict[str, Any],
) -> dict[str, list[WikidataRootConcept]]:
    concepts_by_name: dict[str, list[WikidataRootConcept]] = {}
    for binding in sparql_result.get("results", {}).get("bindings", []):
        requested_name = _binding_value(binding, "requestedTaxonName")
        item_uri = _binding_value(binding, "item")
        if not requested_name or not item_uri:
            continue
        qid = item_uri.rstrip("/").rsplit("/", 1)[-1]
        concepts_by_name.setdefault(requested_name, []).append(
            WikidataRootConcept(
                qid=qid,
                uri=item_uri,
                label=_binding_value(binding, "itemLabel") or qid,
                taxon_name=_binding_value(binding, "taxonName"),
                inaturalist_taxon_id=_binding_value(binding, "inatTaxonId"),
                taxon_rank=_binding_value(binding, "taxonRankLabel"),
            )
        )
    return concepts_by_name


def fetch_inaturalist_taxon_id(
    scientific_name: str,
    endpoint: str = INATURALIST_TAXA_ENDPOINT,
    timeout: int = 30,
) -> str | None:
    payload = urlencode({"q": scientific_name, "rank": "species", "per_page": "5"})
    request = Request(
        f"{endpoint}?{payload}",
        headers={
            "Accept": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return parse_inaturalist_taxon_id(json.load(response), scientific_name)


def parse_inaturalist_taxon_id(
    api_result: dict[str, Any],
    scientific_name: str,
) -> str | None:
    exact = scientific_name.strip().casefold()
    for result in api_result.get("results") or []:
        name = str(result.get("name") or "").strip()
        if name.casefold() == exact:
            return str(result.get("id"))
    return None


def root_concept_record(
    candidate: INaturalistRootConceptCandidate,
    root_concept: WikidataRootConcept,
) -> dict[str, Any]:
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
            "inaturalist_taxon_id": (
                candidate.inaturalist_taxon_id or root_concept.inaturalist_taxon_id
            ),
        },
        "root_concept": {
            "wikidata_qid": root_concept.qid,
            "wikidata_uri": root_concept.uri,
            "label": root_concept.label,
            "taxon_name": root_concept.taxon_name,
            "inaturalist_taxon_id": root_concept.inaturalist_taxon_id,
            "taxon_rank": root_concept.taxon_rank,
            "linking_method": root_concept.linking_method,
        },
    }


def _candidate_from_records(
    source: str,
    annotation: dict[str, Any],
    image: dict[str, Any],
    category: dict[str, Any],
    license_metadata: dict[str, Any] | None,
) -> INaturalistRootConceptCandidate:
    scientific_name = str(category.get("name") or "").strip()
    if not scientific_name:
        raise ValueError(f"iNaturalist category lacks scientific name: {category}")
    taxonomy = {
        key: str(category.get(key) or "").strip()
        for key in (
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "specific_epithet",
        )
        if category.get(key)
    }
    return INaturalistRootConceptCandidate(
        source=source,
        image_id=image.get("id"),
        category_id=category.get("id"),
        annotation_id=annotation.get("id"),
        scientific_name=scientific_name,
        common_name=str(category.get("common_name") or ""),
        taxonomy=taxonomy,
        taxonomy_path=_taxonomy_path(category, scientific_name),
        image_file_name=str(image.get("file_name") or ""),
        image_metadata=dict(image),
        category_metadata=dict(category),
        license_metadata=license_metadata,
    )


def _taxonomy_path(category: dict[str, Any], scientific_name: str) -> list[str]:
    raw_parts = [
        category.get("kingdom"),
        category.get("phylum"),
        category.get("class"),
        category.get("order"),
        category.get("family"),
        category.get("genus"),
        scientific_name,
    ]
    path: list[str] = []
    for part in raw_parts:
        value = str(part or "").strip()
        if value and value not in path:
            path.append(value)
    return path


def _index_by_id(records: Iterable[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {record.get("id"): record for record in records}


def _binding_value(binding: dict[str, Any], key: str) -> str | None:
    value = binding.get(key, {}).get("value")
    return str(value) if value is not None else None


def _group_claim_triples_by_root(
    triples: Iterable[dict[str, Any]],
    allowed_property_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for triple in triples:
        property_id = str(triple.get("property", {}).get("id") or "").strip()
        root_qid = str(triple.get("root", {}).get("qid") or "").strip()
        if not root_qid or property_id not in allowed_property_ids:
            continue
        grouped.setdefault(root_qid, []).append(triple)
    return grouped


def _is_terminal_claim_triple(triple: dict[str, Any]) -> bool:
    tail = triple.get("tail") or {}
    if tail.get("type") != "wikidata_entity":
        return True
    answer_type = str(triple.get("property", {}).get("answer_type") or "")
    return answer_type in {"literal", "date", "number", "fixed", "external_id"}


def _claim_path_record(path: list[dict[str, Any]]) -> dict[str, Any]:
    final_triple = path[-1]
    route_properties = [
        str(triple.get("property", {}).get("id") or "") for triple in path
    ]
    route_qids = _route_qids_for_path(path)
    return {
        "root": path[0]["root"],
        "path": path,
        "hop": len(path),
        "route_properties": route_properties,
        "route_qids": route_qids,
        "final_answer": final_triple["tail"],
        "source": "wikidata_sparql_whitelist_paths",
    }


def _route_qids_for_path(path: list[dict[str, Any]]) -> list[str]:
    if not path:
        return []
    route_qids = [str(path[0].get("root", {}).get("qid") or "")]
    for triple in path[:-1]:
        tail_qid = str(triple.get("tail", {}).get("qid") or "").strip()
        if tail_qid:
            route_qids.append(tail_qid)
    return [qid for qid in route_qids if qid]


def _claim_path_key(path: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            str(triple.get("triple", {}).get("subject") or ""),
            str(triple.get("triple", {}).get("predicate") or ""),
            str(triple.get("triple", {}).get("object") or ""),
        )
        for triple in path
    )


def _claim_tail(
    binding: dict[str, Any],
    value: dict[str, Any],
) -> dict[str, Any]:
    raw_value = str(value.get("value") or "")
    value_type = str(value.get("type") or "")
    if value_type == "uri" and raw_value.startswith("http://www.wikidata.org/entity/"):
        qid = _qid_from_uri(raw_value)
        return {
            "type": "wikidata_entity",
            "qid": qid,
            "uri": raw_value,
            "label": _binding_value(binding, "valueLabel") or qid,
        }
    return {
        "type": "literal",
        "value": raw_value,
        "label": _binding_value(binding, "valueLabel") or raw_value,
        "datatype": value.get("datatype"),
        "language": value.get("xml:lang"),
    }


def _qid_from_uri(uri: str | None) -> str:
    if not uri:
        return ""
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def _sparql_qid(qid: str) -> str:
    normalized = str(qid).strip()
    if not normalized.startswith("Q") or not normalized[1:].isdigit():
        raise ValueError(f"Invalid Wikidata QID: {qid}")
    return normalized


class _ResolvedIpHttpsConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        resolved_ip: str,
        timeout: int,
    ):
        self._resolved_ip = resolved_ip
        super().__init__(host, timeout=timeout)

    def connect(self) -> None:
        sock = socket.create_connection(
            (self._resolved_ip, self.port),
            self.timeout,
            self.source_address,
        )
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _request_json_via_resolved_ip(
    *,
    host: str,
    path: str,
    headers: dict[str, str],
    timeout: int,
    resolved_ip: str,
) -> dict[str, Any]:
    connection = _ResolvedIpHttpsConnection(
        host=host,
        resolved_ip=resolved_ip,
        timeout=timeout,
    )
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        body = response.read()
        if response.status >= 400:
            raise ValueError(
                f"Resolved-IP HTTPS request failed with HTTP {response.status}: "
                f"{body[:200]!r}"
            )
        return json.loads(body.decode("utf-8"))
    finally:
        connection.close()


def _query_sparql_json(
    *,
    query: str,
    endpoint: str = WIKIDATA_SPARQL_ENDPOINT,
    timeout: int = 30,
    resolved_ip: str | None = None,
) -> dict[str, Any]:
    payload = urlencode({"query": query, "format": "json"})
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if resolved_ip:
        parsed_endpoint = urlparse(endpoint)
        host = parsed_endpoint.hostname
        if not host:
            raise ValueError(f"Wikidata endpoint lacks host: {endpoint}")
        path = parsed_endpoint.path or "/"
        return _request_json_via_resolved_ip(
            host=host,
            path=f"{path}?{payload}",
            headers={**headers, "Host": host},
            timeout=timeout,
            resolved_ip=resolved_ip,
        )

    request = Request(f"{endpoint}?{payload}", headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _sparql_string_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _ensure_size_within_limit(size: int, max_bytes: int, source: object) -> None:
    if size > max_bytes:
        raise ValueError(
            f"Refusing to download/read {source}: {size} bytes exceeds "
            f"the configured limit of {max_bytes} bytes."
        )


__all__ = [
    "DEFAULT_MAX_DOWNLOAD_BYTES",
    "INATURALIST_2021_ANNOTATION_URLS",
    "INaturalistRootConceptCandidate",
    "ReasonVqaClaimProperty",
    "WIKIDATA_SPARQL_ENDPOINT",
    "WikidataRootConcept",
    "build_wikidata_claims_query",
    "build_wikidata_taxon_batch_query",
    "build_wikidata_taxon_query",
    "download_inaturalist_annotation_archive",
    "expand_wikidata_claim_paths",
    "fetch_inaturalist_taxon_id",
    "iter_inaturalist_root_concept_candidates",
    "load_inaturalist_annotation_document",
    "parse_inaturalist_taxon_id",
    "parse_dns_google_a_record",
    "parse_wikidata_taxon_batch_results",
    "parse_wikidata_claim_triples",
    "parse_wikidata_root_concepts",
    "query_wikidata_taxon_batch",
    "query_wikidata_claim_triples",
    "query_wikidata_taxon",
    "reasonvqa_claim_property_whitelist",
    "resolve_host_via_google_doh",
    "root_concept_record",
    "summarize_inaturalist_annotation_document",
]
