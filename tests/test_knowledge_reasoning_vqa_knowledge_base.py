from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


def test_legacy_literal_distractors_use_other_values_from_same_property(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )
    paths = [
        _path_answer("P225", "Ptilotula penicillata"),
        _path_answer("P225", "Meliphagidae"),
        _path_answer("P225", "Animalia"),
        _path_answer("P225", "Chordata"),
        _path_answer("P1843", "White-plumed Honeyeater"),
    ]

    options = choices.typed_answer_options(
        answer="Ptilotula penicillata",
        answer_type="literal",
        property_id="P225",
        kg_paths=paths,
        negative_strategy="same_route_property",
    )

    assert options == [
        "Ptilotula penicillata",
        "Meliphagidae",
        "Animalia",
        "Chordata",
    ]


def test_taxonomy_hard_distractors_do_not_use_far_same_route_fallback(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )
    reference = _inat_route_path_answer(
        ["P225"],
        "Ranunculus californicus",
        root_qid="Q1",
        genus="Ranunculus",
        family="Ranunculaceae",
        order="Ranunculales",
    )
    paths = [
        reference,
        _inat_route_path_answer(
            ["P225"],
            "Clematis vitalba",
            root_qid="Q2",
            genus="Clematis",
            family="Ranunculaceae",
            order="Ranunculales",
        ),
        _inat_route_path_answer(
            ["P225"],
            "Quercus falcata",
            root_qid="Q3",
            genus="Quercus",
            family="Fagaceae",
            order="Fagales",
        ),
        _inat_route_path_answer(
            ["P225"],
            "Cocos nucifera",
            root_qid="Q4",
            genus="Cocos",
            family="Arecaceae",
            order="Arecales",
        ),
    ]

    try:
        choices.typed_answer_options(
            answer="Ranunculus californicus",
            answer_type="literal",
            property_id="P225",
            kg_paths=paths,
            reference_path_record=reference,
            negative_strategy="wikidata_taxonomy_neighbors",
            max_taxonomy_distance="family",
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected strict taxonomy-hard choices to reject far fallback.")

    assert "Not enough typed distractors" in message


def test_taxonomy_hard_distractors_use_external_wikidata_neighbor_candidates(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )
    reference = _inat_route_path_answer(
        ["P225"],
        "Ranunculus californicus",
        root_qid="Q1",
        genus="Ranunculus",
        family="Ranunculaceae",
        order="Ranunculales",
    )

    options = choices.typed_answer_options(
        answer="Ranunculus californicus",
        answer_type="literal",
        property_id="P225",
        kg_paths=[reference],
        reference_path_record=reference,
        candidate_values=[
            {"label": "Ranunculus repens", "relation": "same_genus"},
            {"label": "Ranunculus acris", "relation": "same_genus"},
            {"label": "Clematis vitalba", "relation": "same_family"},
            {"label": "Quercus falcata", "relation": "far"},
        ],
        negative_strategy="wikidata_taxonomy_neighbors",
        max_taxonomy_distance="family",
    )

    assert options == [
        "Ranunculus californicus",
        "Ranunculus repens",
        "Ranunculus acris",
        "Clematis vitalba",
    ]


def test_typed_literal_distractors_can_match_reference_route(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )
    paths = [
        _path_answer("P225", "Ranunculus californicus"),
        _path_answer("P225", "Quercus falcata"),
        _route_path_answer(["P171", "P225"], "Ranunculus"),
        _route_path_answer(["P171", "P225"], "Quercus"),
        _route_path_answer(["P171", "P171", "P225"], "Ranunculeae"),
        _route_path_answer(["P171", "P171", "P225"], "Fagaceae"),
        _route_path_answer(["P171", "P171", "P225"], "Noctuidae"),
        _route_path_answer(["P171", "P171", "P225"], "Geometridae"),
    ]

    options = choices.typed_answer_options(
        answer="Ranunculeae",
        answer_type="literal",
        property_id="P225",
        kg_paths=paths,
        reference_path_record=paths[4],
    )

    assert options[0] == "Ranunculeae"
    assert set(options[1:]) == {"Fagaceae", "Noctuidae", "Geometridae"}


def test_typed_common_name_distractors_match_answer_shape_when_possible(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )
    route = ["P171", "P171", "P1843"]
    paths = [
        _route_path_answer(route, "Beech Family"),
        _route_path_answer(route, "Storksbill family"),
        _route_path_answer(route, "viola family"),
        _route_path_answer(route, "Cypress family"),
        _route_path_answer(route, "Grass moth"),
        _route_path_answer(route, "Typical pigeons"),
    ]

    options = choices.typed_answer_options(
        answer="Beech Family",
        answer_type="literal",
        property_id="P1843",
        kg_paths=paths,
        reference_path_record=paths[0],
    )

    assert options[0] == "Beech Family"
    assert all("family" in option.casefold() for option in options[1:])


def test_typed_literal_distractors_prioritize_same_genus_and_family(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )
    reference = _inat_route_path_answer(
        ["P225"],
        "Ranunculus californicus",
        root_qid="Q1",
        genus="Ranunculus",
        family="Ranunculaceae",
        order="Ranunculales",
    )
    paths = [
        reference,
        _inat_route_path_answer(
            ["P225"],
            "Ranunculus repens",
            root_qid="Q2",
            genus="Ranunculus",
            family="Ranunculaceae",
            order="Ranunculales",
        ),
        _inat_route_path_answer(
            ["P225"],
            "Ranunculus acris",
            root_qid="Q3",
            genus="Ranunculus",
            family="Ranunculaceae",
            order="Ranunculales",
        ),
        _inat_route_path_answer(
            ["P225"],
            "Clematis vitalba",
            root_qid="Q4",
            genus="Clematis",
            family="Ranunculaceae",
            order="Ranunculales",
        ),
        _inat_route_path_answer(
            ["P225"],
            "Quercus falcata",
            root_qid="Q5",
            genus="Quercus",
            family="Fagaceae",
            order="Fagales",
        ),
        _inat_route_path_answer(
            ["P225"],
            "Cocos nucifera",
            root_qid="Q6",
            genus="Cocos",
            family="Arecaceae",
            order="Arecales",
        ),
    ]

    options = choices.typed_answer_options(
        answer="Ranunculus californicus",
        answer_type="literal",
        property_id="P225",
        kg_paths=paths,
        reference_path_record=reference,
    )

    assert options[0] == "Ranunculus californicus"
    assert set(options[1:]) == {
        "Ranunculus repens",
        "Ranunculus acris",
        "Clematis vitalba",
    }


def test_typed_common_name_distractors_exclude_same_final_target_aliases(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )
    reference = _inat_route_path_answer(
        ["P1843"],
        "Southern Red Oak",
        root_qid="Q1",
        genus="Quercus",
        family="Fagaceae",
        order="Fagales",
    )
    paths = [
        reference,
        _inat_route_path_answer(
            ["P1843"],
            "Spanish Oak",
            root_qid="Q1",
            genus="Quercus",
            family="Fagaceae",
            order="Fagales",
        ),
        _inat_route_path_answer(
            ["P1843"],
            "Shumard Oak",
            root_qid="Q2",
            genus="Quercus",
            family="Fagaceae",
            order="Fagales",
        ),
        _inat_route_path_answer(
            ["P1843"],
            "Black Oak",
            root_qid="Q3",
            genus="Quercus",
            family="Fagaceae",
            order="Fagales",
        ),
        _inat_route_path_answer(
            ["P1843"],
            "White Oak",
            root_qid="Q4",
            genus="Quercus",
            family="Fagaceae",
            order="Fagales",
        ),
        _inat_route_path_answer(
            ["P1843"],
            "Coconut Palm",
            root_qid="Q5",
            genus="Cocos",
            family="Arecaceae",
            order="Arecales",
        ),
    ]

    options = choices.typed_answer_options(
        answer="Southern Red Oak",
        answer_type="literal",
        property_id="P1843",
        kg_paths=paths,
        reference_path_record=reference,
    )

    assert "Spanish Oak" not in options
    assert set(options[1:]) == {"Shumard Oak", "Black Oak", "White Oak"}


def test_typed_fixed_distractors_use_closed_iucn_status_set(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )

    options = choices.typed_answer_options(
        answer="least concern",
        answer_type="fixed",
        property_id="P141",
    )

    assert options == [
        "least concern",
        "near threatened",
        "vulnerable",
        "endangered",
    ]


def test_typed_number_distractors_perturb_numeric_answer(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )

    options = choices.typed_answer_options(
        answer="42",
        answer_type="number",
        property_id="P2048",
    )

    assert options == ["42", "41", "43", "44"]


def test_typed_date_distractors_perturb_nearby_years(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )

    options = choices.typed_answer_options(
        answer="2020",
        answer_type="date",
        property_id="P571",
    )

    assert options == ["2020", "2019", "2021", "2018"]


def test_external_id_answers_are_not_used_for_typed_qa(monkeypatch):
    choices = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_choices"
    )

    try:
        choices.typed_answer_options(
            answer="508976",
            answer_type="external_id",
            property_id="P3151",
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected external-id answers to be rejected.")

    assert "external-id answers are not used for QA" in message


def test_path_generator_renders_recursive_question_from_wikidata_path(monkeypatch, tmp_path):
    paths_module = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_paths"
    )
    paths = [
        _path_answer("P225", "Ameiurus nebulosus"),
        _path_answer("P225", "Ptilotula penicillata"),
        _path_answer("P225", "Ranunculus californicus"),
        _path_answer("P225", "Corvus albus"),
        _route_path_answer(["P171", "P171", "P225"], "Sturnidae"),
        _route_path_answer(["P171", "P171", "P225"], "Passeridae"),
        _route_path_answer(["P171", "P171", "P225"], "Accipitridae"),
        _three_hop_path(),
    ]

    row = paths_module.generate_reasonvqa_row_from_wikidata_path(
        path_record=paths[-1],
        all_path_records=paths,
        row_index=4,
        config={
            "export_dir": str(tmp_path),
            "image_source": "iNaturalist2021",
            "template_version": "inat-reasonvqa-template-v1",
            "difficulty_policy": {"negative_strategy": "same_route_property"},
            "category_ids_by_property": {"P225": [8]},
            "visual_anchor_template": "this {root_entity_type}",
            "question_templates": {
                "P171": {
                    "main": "What is the parent taxon of {subject}?",
                    "sub_clause": "the parent taxon of {subject}",
                },
                "P225": {
                    "main": "What is the scientific name of {subject}?",
                    "sub_clause": "the scientific name of {subject}",
                },
            },
        },
        recipe="knowledge_reasoning_vqa",
        generator_prompt_version="reasonvqa-template-v1",
        schema_version="synthetic-reasonvqa-v1",
        params={"filter_versions": ["wikidata-path-v1"], "intended_split": "eval"},
    )

    assert row["question"] == (
        "What is the scientific name of the taxonomic group two levels above "
        "this species?"
    )
    assert row["program_answer"] == "Corvidae"
    assert row["answers"][row["correct"].index(1)] == "Corvidae"
    assert row["categories"] == [8]
    assert row["route"] == ["Pied Crow", "Corvus", "Corvidae", "Corvidae"]
    assert row["provenance"]["filter_versions"] == ["wikidata-path-v1"]
    assert row["provenance"]["intended_split"] == "eval"


def test_path_generator_uses_inaturalist_default_templates(monkeypatch, tmp_path):
    paths_module = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa_paths"
    )
    paths = [
        _path_answer("P225", "Ameiurus nebulosus"),
        _path_answer("P225", "Ptilotula penicillata"),
        _path_answer("P225", "Ranunculus californicus"),
        _path_answer("P225", "Corvus albus"),
    ]

    row = paths_module.generate_reasonvqa_row_from_wikidata_path(
        path_record=paths[0],
        all_path_records=paths,
        row_index=0,
        config={
            "export_dir": str(tmp_path),
            "image_source": "iNaturalist2021",
            "category_ids_by_property": {"P225": [8]},
            "difficulty_policy": {"negative_strategy": "same_route_property"},
        },
        recipe="knowledge_reasoning_vqa",
        generator_prompt_version="reasonvqa-template-v1",
        schema_version="synthetic-reasonvqa-v1",
        params={},
    )

    assert row["question"] == (
        "What is the scientific name of the species shown in the image?"
    )
    assert row["provenance"]["template_version"] == "inat-reasonvqa-template-v1"



def _path_answer(property_id: str, answer: str) -> dict[str, object]:
    root = {
        "qid": f"Q_{answer}",
        "uri": f"http://www.wikidata.org/entity/Q_{answer}",
        "label": answer,
    }
    triple = _claim_triple(
        root=root,
        property_id=property_id,
        property_label="taxon name",
        property_group="literal",
        answer_type="literal",
        tail={"type": "literal", "value": answer, "label": answer},
    )
    return _path_record(root=root, path=[triple], image_id=1, scientific_name=answer)


def _route_path_answer(route_properties: list[str], answer: str) -> dict[str, object]:
    root = {
        "qid": f"Q_root_{answer}",
        "uri": f"http://www.wikidata.org/entity/Q_root_{answer}",
        "label": f"{answer} species",
    }
    path = []
    current_root = root
    for index, property_id in enumerate(route_properties):
        is_final = index == len(route_properties) - 1
        tail = (
            {"type": "literal", "value": answer, "label": answer}
            if is_final
            else {
                "type": "wikidata_entity",
                "qid": f"Q_{answer}_{index}",
                "label": f"{answer} parent {index}",
            }
        )
        path.append(
            _claim_triple(
                root=current_root,
                property_id=property_id,
                property_label=(
                    "taxon name"
                    if property_id == "P225"
                    else "taxon common name"
                    if property_id == "P1843"
                    else "parent taxon"
                ),
                property_group="literal" if is_final else "taxonomy",
                answer_type="literal" if is_final else "concept",
                tail=tail,
            )
        )
        if not is_final:
            current_root = {
                "qid": str(tail["qid"]),
                "uri": f"http://www.wikidata.org/entity/{tail['qid']}",
                "label": str(tail["label"]),
            }
    return _path_record(
        root=root,
        path=path,
        image_id=len(answer) + len(route_properties),
        scientific_name=f"{answer} species",
    )


def _inat_route_path_answer(
    route_properties: list[str],
    answer: str,
    *,
    root_qid: str,
    genus: str,
    family: str,
    order: str,
) -> dict[str, object]:
    record = _route_path_answer(route_properties, answer)
    record["root"] = {
        "qid": root_qid,
        "uri": f"http://www.wikidata.org/entity/{root_qid}",
        "label": answer,
    }
    record["root_concept"]["wikidata_qid"] = root_qid
    record["root_concept"]["wikidata_uri"] = f"http://www.wikidata.org/entity/{root_qid}"
    record["root_concept"]["label"] = answer
    record["root_concept"]["taxon_name"] = (
        answer if route_properties == ["P225"] else f"{genus} example"
    )
    record["inaturalist"]["scientific_name"] = record["root_concept"]["taxon_name"]
    record["inaturalist"]["taxonomy"] = {
        "kingdom": "Plantae",
        "phylum": "Tracheophyta",
        "class": "Magnoliopsida",
        "order": order,
        "family": family,
        "genus": genus,
    }
    return record


def _three_hop_path() -> dict[str, object]:
    root = {"qid": "Q200", "uri": "http://www.wikidata.org/entity/Q200", "label": "Pied Crow"}
    path = [
        _claim_triple(
            root=root,
            property_id="P171",
            property_label="parent taxon",
            property_group="taxonomy",
            answer_type="concept",
            tail={"type": "wikidata_entity", "qid": "Q201", "label": "Corvus"},
        ),
        _claim_triple(
            root={"qid": "Q201", "uri": "http://www.wikidata.org/entity/Q201", "label": "Corvus"},
            property_id="P171",
            property_label="parent taxon",
            property_group="taxonomy",
            answer_type="concept",
            tail={"type": "wikidata_entity", "qid": "Q202", "label": "Corvidae"},
        ),
        _claim_triple(
            root={"qid": "Q202", "uri": "http://www.wikidata.org/entity/Q202", "label": "Corvidae"},
            property_id="P225",
            property_label="taxon name",
            property_group="literal",
            answer_type="literal",
            tail={"type": "literal", "value": "Corvidae", "label": "Corvidae"},
        ),
    ]
    return _path_record(
        root=root,
        path=path,
        image_id=200,
        scientific_name="Corvus albus",
    )


def _claim_triple(
    *,
    root: dict[str, str],
    property_id: str,
    property_label: str,
    property_group: str,
    answer_type: str,
    tail: dict[str, str],
) -> dict[str, object]:
    return {
        "root": root,
        "property": {
            "id": property_id,
            "label": property_label,
            "group": property_group,
            "answer_type": answer_type,
        },
        "tail": tail,
        "triple": {
            "subject": root["qid"],
            "predicate": property_id,
            "object": tail.get("qid") or tail.get("value") or tail.get("label"),
        },
    }


def _path_record(
    *,
    root: dict[str, str],
    path: list[dict[str, object]],
    image_id: int,
    scientific_name: str,
) -> dict[str, object]:
    return {
        "root": root,
        "path": path,
        "hop": len(path),
        "route_properties": [triple["property"]["id"] for triple in path],
        "route_qids": [root["qid"]],
        "final_answer": path[-1]["tail"],
        "source": "wikidata_sparql_whitelist_paths",
        "root_concept": {
            "wikidata_qid": root["qid"],
            "wikidata_uri": root["uri"],
            "label": root["label"],
            "taxon_name": scientific_name,
            "taxon_rank": "species",
            "linking_method": "wikidata_taxon",
        },
        "inaturalist": {
            "annotation_id": image_id,
            "scientific_name": scientific_name,
            "common_name": scientific_name,
        },
        "image": {
            "image_id": image_id,
            "file_name": f"val/demo/{image_id}.jpg",
            "url": f"https://example.test/val/demo/{image_id}.jpg",
        },
    }
