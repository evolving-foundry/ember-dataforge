from __future__ import annotations

from tests.generation.config_helpers import job_for_recipe

import importlib
import json
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")

from base.dataset import MultimodalDataset  # noqa: E402
from data_designer.operators import (  # noqa: E402
    SyntheticDataGenerationOperation,
)

QA_FIELDS = {
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
}
ANN_FIELDS = {
    "question_id",
    "hop",
    "property_id",
    "property_label",
    "entity_id",
    "entity_name",
    "route",
    "has_scene_graph",
}
INTERNAL_FIELDS = {
    "reasoning_units",
    "qae_triplets",
    "evidence_refs",
    "quality_scores",
    "audit_status",
    "provenance",
}


def _dataset(rows):
    return MultimodalDataset(
        ray.data.from_items(rows),
        modalities=["image", "text"],
    )


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _correct_answer(row):
    return row["answers"][row["correct"].index(1)]


def _recipe_config(tmp_path: Path, **overrides):
    config = {
        "export_dir": str(tmp_path),
        "export_enabled": False,
        "image_source": "iNaturalist2021",
        "template_version": "inat-reasonvqa-template-v1",
        "difficulty_policy": {
            "hop_distribution": {1: 1, 2: 1, 3: 1},
            "hard_negative_min_count": 3,
            "negative_strategy": "typed_same_property",
        },
        "categories": [
            {"category_id": 8, "category_name": "taxonomy"},
            {"category_id": 9, "category_name": "conservation"},
        ],
        "category_ids_by_property": {
            "P141": [9],
            "P1843": [8],
            "P225": [8],
        },
        "visual_anchor_template": "this {root_entity_type}",
        "question_templates": {
            "P141": {
                "main": "What is the conservation status of {subject}?",
                "sub_clause": "the conservation status of {subject}",
            },
            "P171": {
                "main": "What is the parent taxon of {subject}?",
                "sub_clause": "the parent taxon of {subject}",
            },
            "P1843": {
                "main": "What common name is used for {subject}?",
                "sub_clause": "the common name of {subject}",
            },
            "P225": {
                "main": "What is the scientific name of {subject}?",
                "sub_clause": "the scientific name of {subject}",
            },
        },
    }
    config.update(overrides)
    return config


def _wikidata_path_rows():
    return [
        _one_hop_taxon_name_path("Q100", "Ameiurus nebulosus", 100),
        _one_hop_taxon_name_path("Q101", "Ptilotula penicillata", 101),
        _one_hop_taxon_name_path("Q102", "Ranunculus californicus", 102),
        _one_hop_taxon_name_path("Q103", "Corvus albus", 103),
        _three_hop_taxon_name_distractor_path(
            root_qid="Q210",
            root_label="Common Starling",
            genus_qid="Q211",
            genus_label="Sturnus",
            family_qid="Q212",
            family_label="Sturnidae",
            image_id=210,
        ),
        _three_hop_taxon_name_distractor_path(
            root_qid="Q220",
            root_label="House Sparrow",
            genus_qid="Q221",
            genus_label="Passer",
            family_qid="Q222",
            family_label="Passeridae",
            image_id=220,
        ),
        _three_hop_taxon_name_distractor_path(
            root_qid="Q230",
            root_label="Red-tailed Hawk",
            genus_qid="Q231",
            genus_label="Buteo",
            family_qid="Q232",
            family_label="Accipitridae",
            image_id=230,
        ),
        _three_hop_taxon_name_path(),
    ]


def _one_hop_taxon_name_path(qid: str, taxon_name: str, image_id: int):
    root = {"qid": qid, "uri": f"http://www.wikidata.org/entity/{qid}", "label": taxon_name}
    triple = _claim_triple(
        root=root,
        property_id="P225",
        property_label="taxon name",
        property_group="literal",
        answer_type="literal",
        tail={"type": "literal", "value": taxon_name, "label": taxon_name},
    )
    return _path_record(
        root=root,
        path=[triple],
        image_id=image_id,
        scientific_name=taxon_name,
        common_name=f"common {image_id}",
    )


def _three_hop_taxon_name_path():
    root = {"qid": "Q200", "uri": "http://www.wikidata.org/entity/Q200", "label": "Pied Crow"}
    genus = {"type": "wikidata_entity", "qid": "Q201", "uri": "http://www.wikidata.org/entity/Q201", "label": "Corvus"}
    family = {"type": "wikidata_entity", "qid": "Q202", "uri": "http://www.wikidata.org/entity/Q202", "label": "Corvidae"}
    path = [
        _claim_triple(
            root=root,
            property_id="P171",
            property_label="parent taxon",
            property_group="taxonomy",
            answer_type="concept",
            tail=genus,
        ),
        _claim_triple(
            root={"qid": "Q201", "uri": "http://www.wikidata.org/entity/Q201", "label": "Corvus"},
            property_id="P171",
            property_label="parent taxon",
            property_group="taxonomy",
            answer_type="concept",
            tail=family,
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
        common_name="Pied Crow",
    )


def _three_hop_taxon_name_distractor_path(
    *,
    root_qid: str,
    root_label: str,
    genus_qid: str,
    genus_label: str,
    family_qid: str,
    family_label: str,
    image_id: int,
):
    root = {
        "qid": root_qid,
        "uri": f"http://www.wikidata.org/entity/{root_qid}",
        "label": root_label,
    }
    genus = {
        "type": "wikidata_entity",
        "qid": genus_qid,
        "uri": f"http://www.wikidata.org/entity/{genus_qid}",
        "label": genus_label,
    }
    family = {
        "type": "wikidata_entity",
        "qid": family_qid,
        "uri": f"http://www.wikidata.org/entity/{family_qid}",
        "label": family_label,
    }
    path = [
        _claim_triple(
            root=root,
            property_id="P171",
            property_label="parent taxon",
            property_group="taxonomy",
            answer_type="concept",
            tail=genus,
        ),
        _claim_triple(
            root={
                "qid": genus_qid,
                "uri": f"http://www.wikidata.org/entity/{genus_qid}",
                "label": genus_label,
            },
            property_id="P171",
            property_label="parent taxon",
            property_group="taxonomy",
            answer_type="concept",
            tail=family,
        ),
        _claim_triple(
            root={
                "qid": family_qid,
                "uri": f"http://www.wikidata.org/entity/{family_qid}",
                "label": family_label,
            },
            property_id="P225",
            property_label="taxon name",
            property_group="literal",
            answer_type="literal",
            tail={"type": "literal", "value": family_label, "label": family_label},
        ),
    ]
    return _path_record(
        root=root,
        path=path,
        image_id=image_id,
        scientific_name=root_label,
        common_name=root_label,
    )


def _linking_only_path():
    root = {"qid": "Q300", "uri": "http://www.wikidata.org/entity/Q300", "label": "Skip species"}
    triple = _claim_triple(
        root=root,
        property_id="P3151",
        property_label="iNaturalist taxon ID",
        property_group="literal",
        answer_type="external_id",
        tail={"type": "literal", "value": "12345", "label": "12345"},
    )
    return _path_record(
        root=root,
        path=[triple],
        image_id=300,
        scientific_name="Skip species",
        common_name="skip",
    )


def _claim_triple(
    *,
    root: dict[str, str],
    property_id: str,
    property_label: str,
    property_group: str,
    answer_type: str,
    tail: dict[str, str],
):
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
        "hop": 1,
        "source": "wikidata_sparql_whitelist",
    }


def _path_record(
    *,
    root: dict[str, str],
    path: list[dict[str, object]],
    image_id: int,
    scientific_name: str,
    common_name: str,
):
    file_name = f"val/demo_taxon/{image_id}.jpg"
    return {
        "root": root,
        "path": path,
        "hop": len(path),
        "route_properties": [triple["property"]["id"] for triple in path],
        "route_qids": [root["qid"], *[
            triple["tail"]["qid"]
            for triple in path[:-1]
            if triple["tail"].get("qid")
        ]],
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
            "category_id": image_id,
            "scientific_name": scientific_name,
            "common_name": common_name,
            "taxonomy_path": ["Animalia", "Chordata", scientific_name],
        },
        "image": {
            "image_id": image_id,
            "file_name": file_name,
            "url": f"https://example.test/{file_name}",
            "metadata": {
                "id": image_id,
                "file_name": file_name,
                "width": 500,
                "height": 375,
            },
        },
    }


def test_knowledge_reasoning_vqa_recipe_is_registered():
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "knowledge_reasoning_vqa",
            "recipe_config": {"export_enabled": False},
        })
    )

    assert op.config["recipe"] == "knowledge_reasoning_vqa"


def test_knowledge_reasoning_vqa_no_longer_exports_seed_row_kg_mode():
    module = importlib.import_module(
        "data_designer.generation.algorithms.builtin.reasonvqa"
    )

    assert not hasattr(module, "build_reasoning_vqa_knowledge_base")
    assert "build_reasoning_vqa_knowledge_base" not in getattr(module, "__all__", ())


def test_knowledge_reasoning_vqa_generates_from_wikidata_path_rows(tmp_path):
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "knowledge_reasoning_vqa",
            "recipe_config": _recipe_config(tmp_path),
        })
    )

    rows = sorted(
        op.execute(_dataset(_wikidata_path_rows())).dataset.take_all(),
        key=lambda item: item["question_id"],
    )
    row = rows[0]

    assert row["entity_id"] == "Q100"
    assert row["entity_name"] == "Ameiurus nebulosus"
    assert row["program_answer"] == "Ameiurus nebulosus"
    assert row["property_id"] == "P225"
    assert row["property_label"] == "taxon name"
    assert row["categories"] == [8]
    assert row["image_id"] == "100"
    assert row["image_name"] == "100.jpg"
    assert row["image_url"] == "https://example.test/val/demo_taxon/100.jpg"
    assert row["source"] == "iNaturalist2021"
    assert row["question"] == "What is the scientific name of this species?"
    assert row["provenance"]["kg_adapter"] == "wikidata_sparql_whitelist_paths"
    assert row["provenance"]["template_version"] == "inat-reasonvqa-template-v1"
    assert row["provenance"]["difficulty_policy"]["hard_negative_min_count"] == 3
    assert row["evidence_refs"]["kg_property_route"] == ["P225"]
    assert row["evidence_refs"]["kg_path"][0]["triple"]["subject"] == "Q100"
    assert row["answers"][row["correct"].index(1)] == "Ameiurus nebulosus"


def test_knowledge_reasoning_vqa_filters_linking_only_paths(tmp_path):
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "knowledge_reasoning_vqa",
            "recipe_config": _recipe_config(tmp_path),
        })
    )

    rows = op.execute(
        _dataset([_linking_only_path(), *_wikidata_path_rows()])
    ).dataset.take_all()

    assert rows
    assert all(row["property_id"] != "P3151" for row in rows)


def test_knowledge_reasoning_vqa_requires_explicit_category_mapping(tmp_path):
    config = _recipe_config(tmp_path)
    config.pop("category_ids_by_property")
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "knowledge_reasoning_vqa",
            "recipe_config": config,
        })
    )

    with pytest.raises(ValueError, match="category_ids_by_property"):
        op.execute(_dataset(_wikidata_path_rows()))


def test_knowledge_reasoning_vqa_recursively_renders_multihop_path_question(tmp_path):
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "knowledge_reasoning_vqa",
            "recipe_config": _recipe_config(tmp_path),
        })
    )

    rows = sorted(
        op.execute(_dataset(_wikidata_path_rows())).dataset.take_all(),
        key=lambda item: (item["hop"], item["question_id"]),
    )
    row = rows[-1]

    assert row["hop"] == 3
    assert row["question"] == (
        "What is the scientific name of the taxonomic group two levels above "
        "this species?"
    )
    assert row["route"] == ["Pied Crow", "Corvus", "Corvidae", "Corvidae"]
    assert row["program_answer"] == "Corvidae"


def test_knowledge_reasoning_vqa_route_template_covers_known_multihop(tmp_path):
    config = _recipe_config(tmp_path)
    config["question_templates"]["P171"].pop("sub_clause")
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "knowledge_reasoning_vqa",
            "recipe_config": config,
        })
    )

    rows = op.execute(_dataset(_wikidata_path_rows())).dataset.take_all()

    assert any(
        row["question"]
        == "What is the scientific name of the taxonomic group two levels above this species?"
        for row in rows
    )


def test_knowledge_reasoning_vqa_generates_reasonvqa_rows_and_exports(tmp_path):
    op = SyntheticDataGenerationOperation(
        job_for_recipe({
            "recipe": "knowledge_reasoning_vqa",
            "schema_version": "synthetic-reasonvqa-v1",
            "generator_prompt_version": "reasonvqa-template-v1",
            "recipe_config": _recipe_config(
                tmp_path,
                export_enabled=True,
            ),
        })
    )

    result = op.execute(_dataset(_wikidata_path_rows()))
    rows = sorted(result.dataset.take_all(), key=lambda item: item["question_id"])

    assert len(rows) == 8
    assert [row["hop"] for row in rows] == [1, 1, 1, 1, 3, 3, 3, 3]

    for row in rows:
        assert QA_FIELDS <= set(row)
        assert ANN_FIELDS <= set(row)
        assert INTERNAL_FIELDS <= set(row)
        assert str(tmp_path / "images") in row["image_path"]
        assert Path(row["image_path"]).name == row["image_name"]
        assert row["source"] == "iNaturalist2021"
        assert len(row["answers"]) == len(row["correct"])
        assert sum(row["correct"]) == 1
        assert row["answers"].count(_correct_answer(row)) == 1
        assert row["program_answer"] == _correct_answer(row)
        assert row["route_verified"] is True
        assert row["quality_scores"]["generation_success"] == 1.0
        assert row["quality_scores"]["route_verified"] == 1.0
        assert row["audit_status"] == "auto_pass"
        assert row["provenance"]["recipe"] == "knowledge_reasoning_vqa"
        assert row["provenance"]["template_version"] == "inat-reasonvqa-template-v1"
        assert row["provenance"]["kg_adapter"] == "wikidata_sparql_whitelist_paths"
        assert row["provenance"]["image_source"]["source"] == row["source"]
        assert row["provenance"]["image_source"]["url"] == row["image_url"]
        assert len(row["route"]) == row["hop"] + 1
        assert row["qae_triplets"][0]["question"] == row["question"]
        assert row["qae_triplets"][0]["answer"] == _correct_answer(row)
        assert row["qae_triplets"][0]["choices"] == row["answers"]
        assert row["reasoning_units"][0]["route"] == row["route"]
        assert row["evidence_refs"]["kg_route"] == row["route"]

    qa_rows = _read_jsonl(tmp_path / "train.jsonl")
    ann_rows = _read_jsonl(tmp_path / "train_ann.jsonl")
    merged_rows = _read_jsonl(tmp_path / "knowledge_reasoning_vqa_merged.jsonl")
    manifest_rows = _read_jsonl(tmp_path / "image_manifest.jsonl")
    categories = _read_jsonl(tmp_path / "categories.jsonl")

    assert len(qa_rows) == len(ann_rows) == len(merged_rows) == 8
    assert len(manifest_rows) == 8
    assert {row["question_id"] for row in qa_rows} == {
        row["question_id"] for row in ann_rows
    }
    assert {row["question_id"] for row in qa_rows} == {
        row["question_id"] for row in manifest_rows
    }
    assert QA_FIELDS == set(qa_rows[0])
    assert ANN_FIELDS == set(ann_rows[0])
    assert QA_FIELDS <= set(merged_rows[0])
    assert ANN_FIELDS <= set(merged_rows[0])
    assert INTERNAL_FIELDS <= set(merged_rows[0])
    assert {
        "question_id",
        "image_id",
        "image_name",
        "image_path",
        "image_url",
        "entity_id",
        "entity_name",
    } <= set(manifest_rows[0])
    assert categories == [
        {"category_id": 8, "category_name": "taxonomy"},
        {"category_id": 9, "category_name": "conservation"},
    ]
