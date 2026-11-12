from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from data_designer.generation.algorithms.builtin import (
    reasonvqa_inaturalist_audit,
    reasonvqa_inaturalist_templates,
)
from data_designer.sources import inaturalist


def _inaturalist_module():
    return inaturalist


def _audit_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "data" / "audit_inaturalist_kg_coverage.py"
    spec = importlib.util.spec_from_file_location(
        "audit_inaturalist_kg_coverage", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _inaturalist_audit_module():
    return reasonvqa_inaturalist_audit


def _inaturalist_template_module():
    return reasonvqa_inaturalist_templates


def _sample_inaturalist_doc():
    return {
        "info": {"year": 2021, "description": "iNaturalist Challenge 2021"},
        "images": [
            {
                "id": 0,
                "width": 500,
                "height": 500,
                "file_name": (
                    "val/02912_Animalia_Chordata_Actinopterygii_"
                    "Siluriformes_Ictaluridae_Ameiurus_nebulosus/"
                    "d615f184-8af4-4c60-b9f8-3081c1607644.jpg"
                ),
                "license": 0,
                "rights_holder": "Ken-ichi Ueda",
                "date": "2010-07-14 20:19:00+00:00",
                "latitude": 43.83486,
                "longitude": -71.22231,
                "location_uncertainty": 77,
            }
        ],
        "categories": [
            {
                "id": 2912,
                "name": "Ameiurus nebulosus",
                "common_name": "Brown Bullhead",
                "supercategory": "Actinopterygii",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Actinopterygii",
                "order": "Siluriformes",
                "family": "Ictaluridae",
                "genus": "Ameiurus",
                "specific_epithet": "nebulosus",
                "image_dir_name": (
                    "02912_Animalia_Chordata_Actinopterygii_"
                    "Siluriformes_Ictaluridae_Ameiurus_nebulosus"
                ),
            }
        ],
        "annotations": [{"id": 7, "image_id": 0, "category_id": 2912}],
        "licenses": [{"id": 0, "name": "CC0", "url": "https://example.test"}],
    }


def test_inaturalist_annotations_yield_wikidata_root_concept_candidates():
    module = _inaturalist_module()

    candidates = list(
        module.iter_inaturalist_root_concept_candidates(
            _sample_inaturalist_doc(),
            source="inat2021_val",
            limit=1,
        )
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source == "inat2021_val"
    assert candidate.image_id == 0
    assert candidate.category_id == 2912
    assert candidate.scientific_name == "Ameiurus nebulosus"
    assert candidate.common_name == "Brown Bullhead"
    assert candidate.wikidata_lookup_key == "Ameiurus nebulosus"
    assert candidate.taxonomy_path == [
        "Animalia",
        "Chordata",
        "Actinopterygii",
        "Siluriformes",
        "Ictaluridae",
        "Ameiurus",
        "Ameiurus nebulosus",
    ]
    assert candidate.image_file_name.endswith("d615f184-8af4-4c60-b9f8-3081c1607644.jpg")


def test_wikidata_taxon_query_uses_scientific_name_and_optional_inat_taxon_id():
    module = _inaturalist_module()

    query = module.build_wikidata_taxon_query(
        scientific_name="Ameiurus nebulosus",
        inaturalist_taxon_id="84734",
        limit=3,
    )

    assert 'wdt:P225 "Ameiurus nebulosus"' in query
    assert 'wdt:P3151 "84734"' in query
    assert "LIMIT 3" in query


def test_wikidata_sparql_result_becomes_root_concept_record():
    module = _inaturalist_module()
    [candidate] = list(
        module.iter_inaturalist_root_concept_candidates(_sample_inaturalist_doc())
    )
    sparql_result = {
        "results": {
            "bindings": [
                {
                    "item": {"value": "http://www.wikidata.org/entity/Q2708562"},
                    "itemLabel": {"value": "Ameiurus nebulosus"},
                    "taxonName": {"value": "Ameiurus nebulosus"},
                    "inatTaxonId": {"value": "84734"},
                }
            ]
        }
    }

    root_concepts = module.parse_wikidata_root_concepts(sparql_result)
    record = module.root_concept_record(candidate, root_concepts[0])

    assert root_concepts[0].qid == "Q2708562"
    assert record["root_concept"]["wikidata_qid"] == "Q2708562"
    assert record["root_concept"]["wikidata_uri"].endswith("/Q2708562")
    assert record["root_concept"]["linking_method"] == "wikidata_taxon"
    assert record["inaturalist"]["scientific_name"] == "Ameiurus nebulosus"
    assert record["inaturalist"]["category_id"] == 2912


def test_wikidata_query_can_bypass_polluted_dns_with_resolved_ip(monkeypatch):
    module = _inaturalist_module()
    captured = {}

    def fake_request_json_via_resolved_ip(
        *,
        host,
        path,
        headers,
        timeout,
        resolved_ip,
    ):
        captured.update(
            {
                "host": host,
                "path": path,
                "headers": headers,
                "timeout": timeout,
                "resolved_ip": resolved_ip,
            }
        )
        return {
            "results": {
                "bindings": [
                    {
                        "item": {"value": "http://www.wikidata.org/entity/Q27075261"},
                        "itemLabel": {"value": "Ptilotula penicillata"},
                        "taxonName": {"value": "Ptilotula penicillata"},
                    }
                ]
            }
        }

    monkeypatch.setattr(
        module,
        "_request_json_via_resolved_ip",
        fake_request_json_via_resolved_ip,
        raising=False,
    )

    concepts = module.query_wikidata_taxon(
        scientific_name="Ptilotula penicillata",
        resolved_ip="185.15.58.224",
        timeout=7,
    )

    assert concepts[0].qid == "Q27075261"
    assert captured["host"] == "query.wikidata.org"
    assert captured["resolved_ip"] == "185.15.58.224"
    assert captured["timeout"] == 7
    assert captured["path"].startswith("/sparql?")
    assert captured["headers"]["Host"] == "query.wikidata.org"


def test_wikidata_query_functions_share_sparql_transport(monkeypatch):
    module = _inaturalist_module()
    calls = []

    def fake_query_sparql_json(*, query, endpoint, timeout, resolved_ip):
        calls.append(
            {
                "query": query,
                "endpoint": endpoint,
                "timeout": timeout,
                "resolved_ip": resolved_ip,
            }
        )
        if "requestedTaxonName" in query:
            return {"results": {"bindings": []}}
        if "?candidateTaxonName" in query:
            return {"results": {"bindings": []}}
        if "?propertyId" in query:
            return {"results": {"bindings": []}}
        return {
            "results": {
                "bindings": [
                    {
                        "item": {"value": "http://www.wikidata.org/entity/Q27075261"},
                        "itemLabel": {"value": "Ptilotula penicillata"},
                        "taxonName": {"value": "Ptilotula penicillata"},
                    }
                ]
            }
        }

    monkeypatch.setattr(module, "_query_sparql_json", fake_query_sparql_json)

    assert module.query_wikidata_taxon("Ptilotula penicillata", timeout=9)
    assert module.query_wikidata_taxon_batch(["Ptilotula penicillata"], timeout=9) == {}
    assert module.query_wikidata_claim_triples(["Q27075261"], timeout=9) == []
    assert module.query_wikidata_taxon_neighbors(["Q27075261"], timeout=9) == []

    assert len(calls) == 4
    assert {call["endpoint"] for call in calls} == {module.WIKIDATA_SPARQL_ENDPOINT}
    assert {call["timeout"] for call in calls} == {9}
    assert {call["resolved_ip"] for call in calls} == {None}


def test_google_doh_response_selects_ipv4_a_record():
    module = _inaturalist_module()

    ip_address = module.parse_dns_google_a_record(
        {
            "Answer": [
                {"type": 5, "data": "dyna.wikimedia.org."},
                {"type": 1, "data": "185.15.58.224"},
            ]
        }
    )

    assert ip_address == "185.15.58.224"


def test_claim_expansion_query_is_limited_to_reasonvqa_whitelist():
    module = _inaturalist_module()

    whitelist = module.reasonvqa_claim_property_whitelist()
    query = module.build_wikidata_claims_query(["Q27075261"], whitelist)

    assert [item.property_id for item in whitelist] == [
        "P105",
        "P141",
        "P171",
        "P183",
        "P225",
        "P1843",
        "P3151",
    ]
    assert {item.group for item in whitelist} == {
        "taxonomy",
        "geography",
        "literal",
        "conservation",
    }
    assert '("P141" "conservation" "fixed")' in query
    assert '("P171" "taxonomy" "concept")' in query
    assert '("P183" "geography" "concept")' in query
    assert '("P225" "literal" "literal")' in query
    assert "wd:Q27075261" in query
    assert '"P31"' not in query
    assert "wdt:P31" not in query
    assert 'FILTER(?propertyId != "P1843" || LANG(?value) = "en")' in query


def test_taxon_neighbor_query_fetches_same_genus_and_same_family_candidates():
    module = _inaturalist_module()

    query = module.build_wikidata_taxon_neighbor_query(
        ["Q1029209"],
        per_relation_limit=5,
    )

    assert "wd:Q1029209" in query
    assert '"same_genus"' in query
    assert '"same_family"' in query
    assert "?candidate wdt:P225 ?candidateTaxonName" in query
    assert "OPTIONAL { ?candidate wdt:P1843 ?candidateCommonName" in query
    assert "LIMIT 5" in query


def test_taxon_neighbor_results_become_choice_candidate_cache_rows():
    module = _inaturalist_module()
    sparql_result = {
        "results": {
            "bindings": [
                {
                    "root": {"value": "http://www.wikidata.org/entity/Q1029209"},
                    "rootLabel": {"value": "California buttercup"},
                    "candidate": {"value": "http://www.wikidata.org/entity/Q159584"},
                    "candidateLabel": {"value": "Ranunculus repens"},
                    "candidateTaxonName": {"value": "Ranunculus repens"},
                    "candidateCommonName": {
                        "value": "creeping buttercup",
                        "xml:lang": "en",
                    },
                    "relation": {"value": "same_genus"},
                }
            ]
        }
    }

    rows = module.parse_wikidata_taxon_neighbor_candidates(sparql_result)

    assert rows == [
        {
            "root_qid": "Q1029209",
            "root_label": "California buttercup",
            "property_id": "P225",
            "route_properties": ["P225"],
            "candidates": [
                {
                    "label": "Ranunculus repens",
                    "qid": "Q159584",
                    "relation": "same_genus",
                    "source": "wikidata_sparql_taxon_neighbors",
                }
            ],
        },
        {
            "root_qid": "Q1029209",
            "root_label": "California buttercup",
            "property_id": "P1843",
            "route_properties": ["P1843"],
            "candidates": [
                {
                    "label": "creeping buttercup",
                    "qid": "Q159584",
                    "relation": "same_genus",
                    "source": "wikidata_sparql_taxon_neighbors",
                }
            ],
        },
    ]


def test_wikidata_claim_results_parse_to_typed_kg_triples():
    module = _inaturalist_module()
    sparql_result = {
        "results": {
            "bindings": [
                {
                    "root": {"value": "http://www.wikidata.org/entity/Q27075261"},
                    "rootLabel": {"value": "Ptilotula penicillata"},
                    "property": {"value": "http://www.wikidata.org/prop/direct/P171"},
                    "propertyId": {"value": "P171"},
                    "propertyLabel": {"value": "parent taxon"},
                    "propertyGroup": {"value": "taxonomy"},
                    "answerType": {"value": "concept"},
                    "value": {"type": "uri", "value": "http://www.wikidata.org/entity/Q3758717"},
                    "valueLabel": {"value": "Ptilotula"},
                },
                {
                    "root": {"value": "http://www.wikidata.org/entity/Q27075261"},
                    "rootLabel": {"value": "Ptilotula penicillata"},
                    "property": {"value": "http://www.wikidata.org/prop/direct/P225"},
                    "propertyId": {"value": "P225"},
                    "propertyLabel": {"value": "taxon name"},
                    "propertyGroup": {"value": "literal"},
                    "answerType": {"value": "literal"},
                    "value": {
                        "type": "literal",
                        "value": "Ptilotula penicillata",
                        "datatype": "http://www.w3.org/2001/XMLSchema#string",
                    },
                },
            ]
        }
    }

    triples = module.parse_wikidata_claim_triples(sparql_result)

    assert len(triples) == 2
    assert triples[0]["root"]["qid"] == "Q27075261"
    assert triples[0]["property"] == {
        "id": "P171",
        "label": "parent taxon",
        "group": "taxonomy",
        "answer_type": "concept",
    }
    assert triples[0]["tail"]["type"] == "wikidata_entity"
    assert triples[0]["tail"]["qid"] == "Q3758717"
    assert triples[0]["tail"]["label"] == "Ptilotula"
    assert triples[1]["property"]["id"] == "P225"
    assert triples[1]["tail"]["type"] == "literal"
    assert triples[1]["tail"]["value"] == "Ptilotula penicillata"


def test_batch_taxon_query_links_requested_scientific_names_only():
    module = _inaturalist_module()

    query = module.build_wikidata_taxon_batch_query(
        ["Ameiurus nebulosus", "Ptilotula penicillata"]
    )

    assert 'VALUES ?requestedTaxonName { "Ameiurus nebulosus" "Ptilotula penicillata" }' in query
    assert "?item wdt:P225 ?requestedTaxonName" in query
    assert "wdt:P31 " not in query
    assert "wdt:P31." not in query
    assert "wdt:P279 " not in query
    assert "wdt:P279." not in query


def test_claim_paths_continue_through_concept_tails_until_literal_answer():
    module = _inaturalist_module()

    calls = []
    triples_by_root = {
        "Q100": [
            _claim_triple(
                root_qid="Q100",
                property_id="P171",
                property_group="taxonomy",
                answer_type="concept",
                tail={"type": "wikidata_entity", "qid": "Q101", "label": "Demo genus"},
            )
        ],
        "Q101": [
            _claim_triple(
                root_qid="Q101",
                property_id="P171",
                property_group="taxonomy",
                answer_type="concept",
                tail={"type": "wikidata_entity", "qid": "Q102", "label": "Demo family"},
            )
        ],
        "Q102": [
            _claim_triple(
                root_qid="Q102",
                property_id="P225",
                property_group="literal",
                answer_type="literal",
                tail={"type": "literal", "value": "Demo family", "label": "Demo family"},
            )
        ],
    }

    def claim_fetcher(root_qids):
        calls.append(list(root_qids))
        return [
            triple
            for root_qid in root_qids
            for triple in triples_by_root.get(root_qid, [])
        ]

    paths = module.expand_wikidata_claim_paths(
        ["Q100"],
        max_hops=3,
        claim_fetcher=claim_fetcher,
    )

    assert calls == [["Q100"], ["Q101"], ["Q102"]]
    assert len(paths) == 1
    assert paths[0]["hop"] == 3
    assert paths[0]["route_properties"] == ["P171", "P171", "P225"]
    assert paths[0]["route_qids"] == ["Q100", "Q101", "Q102"]
    assert paths[0]["final_answer"] == {
        "type": "literal",
        "value": "Demo family",
        "label": "Demo family",
    }
    assert [triple["property"]["id"] for triple in paths[0]["path"]] == [
        "P171",
        "P171",
        "P225",
    ]


def test_claim_paths_do_not_expand_after_literal_or_past_max_hops():
    module = _inaturalist_module()

    calls = []
    triples_by_root = {
        "Q100": [
            _claim_triple(
                root_qid="Q100",
                property_id="P225",
                property_group="literal",
                answer_type="literal",
                tail={"type": "literal", "value": "Root literal", "label": "Root literal"},
            ),
            _claim_triple(
                root_qid="Q100",
                property_id="P171",
                property_group="taxonomy",
                answer_type="concept",
                tail={"type": "wikidata_entity", "qid": "Q1", "label": "Level 1"},
            ),
        ],
        "Q1": [
            _claim_triple(
                root_qid="Q1",
                property_id="P171",
                property_group="taxonomy",
                answer_type="concept",
                tail={"type": "wikidata_entity", "qid": "Q2", "label": "Level 2"},
            )
        ],
        "Q2": [
            _claim_triple(
                root_qid="Q2",
                property_id="P171",
                property_group="taxonomy",
                answer_type="concept",
                tail={"type": "wikidata_entity", "qid": "Q3", "label": "Level 3"},
            )
        ],
        "Q3": [
            _claim_triple(
                root_qid="Q3",
                property_id="P225",
                property_group="literal",
                answer_type="literal",
                tail={"type": "literal", "value": "Too deep", "label": "Too deep"},
            )
        ],
    }

    def claim_fetcher(root_qids):
        calls.append(list(root_qids))
        return [
            triple
            for root_qid in root_qids
            for triple in triples_by_root.get(root_qid, [])
        ]

    paths = module.expand_wikidata_claim_paths(
        ["Q100"],
        max_hops=3,
        claim_fetcher=claim_fetcher,
    )

    assert calls == [["Q100"], ["Q1"], ["Q2"]]
    assert [path["hop"] for path in paths] == [1]
    assert paths[0]["final_answer"]["value"] == "Root literal"


def test_fixed_concept_claims_are_terminal_paths():
    module = _inaturalist_module()

    calls = []
    triples_by_root = {
        "Q100": [
            _claim_triple(
                root_qid="Q100",
                property_id="P141",
                property_group="conservation",
                answer_type="fixed",
                tail={
                    "type": "wikidata_entity",
                    "qid": "Q211005",
                    "label": "least concern",
                },
            )
        ],
        "Q211005": [
            _claim_triple(
                root_qid="Q211005",
                property_id="P225",
                property_group="literal",
                answer_type="literal",
                tail={"type": "literal", "value": "should not expand", "label": "should not expand"},
            )
        ],
    }

    def claim_fetcher(root_qids):
        calls.append(list(root_qids))
        return [
            triple
            for root_qid in root_qids
            for triple in triples_by_root.get(root_qid, [])
        ]

    paths = module.expand_wikidata_claim_paths(
        ["Q100"],
        max_hops=3,
        claim_fetcher=claim_fetcher,
    )

    assert calls == [["Q100"]]
    assert len(paths) == 1
    assert paths[0]["hop"] == 1
    assert paths[0]["route_properties"] == ["P141"]
    assert paths[0]["final_answer"]["qid"] == "Q211005"
    assert paths[0]["final_answer"]["label"] == "least concern"


def test_coverage_summary_reports_rates_and_template_usability():
    module = _inaturalist_audit_module()
    root_claims = [
        _claim_triple(
            root_qid="Q1",
            property_id="P183",
            property_group="geography",
            answer_type="concept",
            tail={"type": "wikidata_entity", "qid": "Q408", "label": "Australia"},
        ),
        _claim_triple(
            root_qid="Q1",
            property_id="P225",
            property_group="literal",
            answer_type="literal",
            tail={"type": "literal", "value": "Species one", "label": "Species one"},
        ),
        _claim_triple(
            root_qid="Q2",
            property_id="P171",
            property_group="taxonomy",
            answer_type="concept",
            tail={"type": "wikidata_entity", "qid": "Q20", "label": "Genus two"},
        ),
    ]
    paths = [
        _path_record(
            [
                _claim_triple(
                    root_qid="Q1",
                    property_id="P225",
                    property_group="literal",
                    answer_type="literal",
                    tail={"type": "literal", "value": "Repeated", "label": "Repeated"},
                )
            ]
        ),
        _path_record(
            [
                _claim_triple(
                    root_qid="Q1",
                    property_id="P171",
                    property_group="taxonomy",
                    answer_type="concept",
                    tail={"type": "wikidata_entity", "qid": "Q10", "label": "Genus one"},
                ),
                _claim_triple(
                    root_qid="Q10",
                    property_id="P225",
                    property_group="literal",
                    answer_type="literal",
                    tail={"type": "literal", "value": "Repeated", "label": "Repeated"},
                ),
            ]
        ),
        _path_record(
            [
                _claim_triple(
                    root_qid="Q2",
                    property_id="P171",
                    property_group="taxonomy",
                    answer_type="concept",
                    tail={"type": "wikidata_entity", "qid": "Q20", "label": "Genus two"},
                ),
                _claim_triple(
                    root_qid="Q20",
                    property_id="P171",
                    property_group="taxonomy",
                    answer_type="concept",
                    tail={"type": "wikidata_entity", "qid": "Q30", "label": "Family two"},
                ),
                _claim_triple(
                    root_qid="Q30",
                    property_id="P3151",
                    property_group="literal",
                    answer_type="external_id",
                    tail={"type": "literal", "value": "12345", "label": "12345"},
                ),
            ]
        ),
    ]

    summary = module.summarize_inaturalist_kg_coverage(
        sampled_count=3,
        linked_records=[
            {"root_concept": {"wikidata_qid": "Q1"}},
            {"root_concept": {"wikidata_qid": "Q2"}},
        ],
        root_claim_triples=root_claims,
        paths=paths,
    )

    assert summary["rates"]["link_success_rate"] == 2 / 3
    assert summary["rates"]["one_hop_path_rate"] == 1 / 2
    assert summary["rates"]["two_hop_path_rate"] == 1 / 2
    assert summary["rates"]["three_hop_path_rate"] == 1 / 2
    assert summary["rates"]["geography_coverage"] == 1 / 2
    assert summary["rates"]["duplicate_answer_rate"] == 2 / 3
    assert summary["rates"]["question_template_usable_rate"] == 2 / 3
    assert summary["property_hit_distribution"]["root_claim_counts"]["P183"] == 1
    assert summary["terminal_answer_type_distribution"] == {
        "external_id": 1,
        "literal": 2,
    }
    assert summary["question_template_policy"]["final_property_ids"] == ["P141", "P1843", "P225"]


def test_inaturalist_template_policy_is_derived_from_template_defaults():
    module = _inaturalist_template_module()

    policy = module.inaturalist_question_template_policy()

    assert policy == {
        "intermediate_property_ids": ["P171", "P183"],
        "final_property_ids": ["P141", "P1843", "P225"],
        "linking_only_property_ids": ["P3151"],
    }
    assert module.is_inaturalist_question_template_usable_path(
        {"route_properties": ["P171", "P225"]}
    )
    assert not module.is_inaturalist_question_template_usable_path(
        {"route_properties": ["P3151"]}
    )


def test_audit_claim_queries_are_batched_to_avoid_large_wdqs_requests():
    audit = _audit_script_module()
    calls = []

    class FakeModule:
        @staticmethod
        def query_wikidata_claim_triples(
            *,
            root_qids,
            whitelist,
            timeout,
            resolved_ip,
        ):
            calls.append(list(root_qids))
            return [
                {
                    "root": {"qid": qid},
                    "property": {"id": "P225"},
                    "tail": {"type": "literal", "value": qid},
                }
                for qid in root_qids
            ]

    triples = audit._query_claim_batches(
        module=FakeModule,
        root_qids=["Q1", "Q2", "Q3", "Q4", "Q5"],
        whitelist=["P225"],
        timeout=30,
        resolved_ip="185.15.58.224",
        batch_size=2,
    )

    assert calls == [["Q1", "Q2"], ["Q3", "Q4"], ["Q5"]]
    assert [triple["root"]["qid"] for triple in triples] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
    ]


def test_audit_claim_batches_split_after_wdqs_502():
    audit = _audit_script_module()
    calls = []

    class FakeModule:
        @staticmethod
        def query_wikidata_claim_triples(
            *,
            root_qids,
            whitelist,
            timeout,
            resolved_ip,
        ):
            calls.append(list(root_qids))
            if len(root_qids) > 1:
                raise ValueError("Resolved-IP HTTPS request failed with HTTP 502")
            return [
                {
                    "root": {"qid": root_qids[0]},
                    "property": {"id": "P225"},
                    "tail": {"type": "literal", "value": root_qids[0]},
                }
            ]

    triples = audit._query_claim_batches(
        module=FakeModule,
        root_qids=["Q1", "Q2", "Q3"],
        whitelist=["P225"],
        timeout=30,
        resolved_ip="185.15.58.224",
        batch_size=3,
    )

    assert calls == [["Q1", "Q2", "Q3"], ["Q1"], ["Q2", "Q3"], ["Q2"], ["Q3"]]
    assert [triple["root"]["qid"] for triple in triples] == ["Q1", "Q2", "Q3"]


def _claim_triple(
    *,
    root_qid,
    property_id,
    property_group,
    answer_type,
    tail,
):
    return {
        "root": {"qid": root_qid, "uri": f"http://www.wikidata.org/entity/{root_qid}", "label": root_qid},
        "property": {
            "id": property_id,
            "label": property_id,
            "group": property_group,
            "answer_type": answer_type,
        },
        "tail": tail,
        "triple": {
            "subject": root_qid,
            "predicate": property_id,
            "object": tail.get("qid") or tail.get("value"),
        },
        "hop": 1,
        "source": "test",
        "property_uri": f"http://www.wikidata.org/prop/direct/{property_id}",
    }


def _path_record(path):
    final_triple = path[-1]
    route_qids = [path[0]["root"]["qid"]]
    for triple in path[:-1]:
        route_qids.append(triple["tail"]["qid"])
    return {
        "root": path[0]["root"],
        "path": path,
        "hop": len(path),
        "route_properties": [triple["property"]["id"] for triple in path],
        "route_qids": route_qids,
        "final_answer": final_triple["tail"],
    }


def test_inaturalist_taxon_api_result_can_supply_optional_conversion_key():
    module = _inaturalist_module()
    api_result = {
        "results": [
            {"id": 123, "name": "Ameiurus", "rank": "genus"},
            {"id": 84734, "name": "Ameiurus nebulosus", "rank": "species"},
        ]
    }

    taxon_id = module.parse_inaturalist_taxon_id(
        api_result,
        scientific_name="Ameiurus nebulosus",
    )

    assert taxon_id == "84734"
