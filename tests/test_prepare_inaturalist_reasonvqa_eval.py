from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest


def _prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "data" / "prepare_inaturalist_reasonvqa_eval.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_inaturalist_reasonvqa_eval",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_bind_local_image_paths_updates_top_level_and_nested_references(tmp_path):
    module = _prepare_module()
    row = {
        "image": {"file_name": "val/species/example.jpg"},
        "image_path": "old/path.jpg",
        "image_url": "val/species/example.jpg",
        "evidence_refs": {
            "image": {
                "image_path": "old/path.jpg",
                "image_url": "val/species/example.jpg",
            }
        },
        "qae_triplets": [
            {
                "evidence_refs": {
                    "image": {
                        "image_path": "old/path.jpg",
                        "image_url": "val/species/example.jpg",
                    }
                }
            }
        ],
        "provenance": {
            "source_uri_or_hash": "val/species/example.jpg",
            "image_source": {"url": "val/species/example.jpg"},
        },
    }

    [bound] = module.bind_local_image_paths(
        [row],
        image_root=tmp_path / "images",
        archive_url="https://example.test/val.tar.gz",
    )

    expected_path = str(tmp_path / "images" / "val" / "species" / "example.jpg")
    expected_uri = "https://example.test/val.tar.gz#val/species/example.jpg"
    assert bound["image_path"] == expected_path
    assert bound["image_url"] == expected_uri
    assert bound["evidence_refs"]["image"]["image_path"] == expected_path
    assert bound["evidence_refs"]["image"]["image_url"] == expected_uri
    assert bound["qae_triplets"][0]["evidence_refs"]["image"]["image_path"] == expected_path
    assert bound["qae_triplets"][0]["evidence_refs"]["image"]["image_url"] == expected_uri
    assert bound["provenance"]["source_uri_or_hash"] == expected_uri
    assert bound["provenance"]["image_source"]["url"] == expected_uri


def test_extract_selected_images_from_tar_stream_writes_only_requested_members(tmp_path):
    module = _prepare_module()
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        _add_tar_member(archive, "val/species/keep.jpg", b"keep-bytes")
        _add_tar_member(archive, "val/species/skip.jpg", b"skip-bytes")
    archive_bytes.seek(0)

    report = module.extract_selected_images_from_tar_stream(
        archive_fileobj=archive_bytes,
        target_file_names={"val/species/keep.jpg", "val/species/missing.jpg"},
        image_root=tmp_path / "images",
    )

    assert report.requested == 2
    assert report.extracted == 1
    assert report.missing == ["val/species/missing.jpg"]
    assert (tmp_path / "images" / "val" / "species" / "keep.jpg").read_bytes() == b"keep-bytes"
    assert not (tmp_path / "images" / "val" / "species" / "skip.jpg").exists()


def test_write_reasonvqa_exports_uses_bound_image_paths(tmp_path):
    module = _prepare_module()
    rows = module.bind_local_image_paths(
        [
            {
                "question_id": 7,
                "question": "What is the scientific name of the species shown in the image?",
                "image_id": "image-7",
                "image_name": "example.jpg",
                "image": {"file_name": "val/species/example.jpg"},
                "image_path": "old/path.jpg",
                "image_url": "val/species/example.jpg",
                "source": "inaturalist_2021_val",
                "answers": ["Ameiurus nebulosus", "Puma concolor", "Corvus albus", "Acer rubrum"],
                "correct": [1, 0, 0, 0],
                "categories": [8],
                "hop": 1,
                "property_id": "P225",
                "property_label": "taxon name",
                "entity_id": "Q1",
                "entity_name": "Brown Bullhead",
                "route": ["Brown Bullhead"],
                "has_scene_graph": False,
            }
        ],
        image_root=tmp_path / "export" / "images",
        archive_url="https://example.test/val.tar.gz",
    )

    module.write_bound_reasonvqa_exports(
        rows=rows,
        export_dir=tmp_path / "export",
        split="train",
    )

    train_text = (tmp_path / "export" / "train.jsonl").read_text(encoding="utf-8")
    manifest_text = (tmp_path / "export" / "image_manifest.jsonl").read_text(
        encoding="utf-8"
    )
    assert json.loads(train_text)["image_path"] == str(
        tmp_path / "export" / "images" / "val" / "species" / "example.jpg"
    )
    assert "https://example.test/val.tar.gz#val/species/example.jpg" in manifest_text


def test_filter_answer_prior_shortcuts_drops_skewed_property_groups():
    module = _prepare_module()
    rows = [
        {"property_id": "P141", "program_answer": "least concern"},
        {"property_id": "P141", "program_answer": "Least Concern"},
        {"property_id": "P141", "program_answer": "least concern"},
        {"property_id": "P141", "program_answer": "near threatened"},
        {"property_id": "P225", "program_answer": "Ameiurus nebulosus"},
        {"property_id": "P225", "program_answer": "Ptilotula penicillata"},
        {"property_id": "P225", "program_answer": "Ranunculus californicus"},
        {"property_id": "P225", "program_answer": "Corvus albus"},
    ]

    filtered, report = module.filter_answer_prior_shortcuts(
        rows,
        group_fields=("property_id",),
        max_majority_fraction=0.5,
        min_group_size=4,
    )

    assert [row["property_id"] for row in filtered] == ["P225", "P225", "P225", "P225"]
    assert report["dropped_groups"] == [
        {
            "group": {"property_id": "P141"},
            "majority_answer": "least concern",
            "majority_fraction": 0.75,
            "rows_dropped": 4,
            "total_rows": 4,
        }
    ]


def test_default_recipe_config_uses_wikidata_taxonomy_hard_negatives(tmp_path):
    module = _prepare_module()

    config = module.default_recipe_config(tmp_path / "export")

    assert config["difficulty_policy"] == {
        "negative_strategy": "wikidata_taxonomy_neighbors",
        "max_taxonomy_distance": "family",
        "drop_rows_without_hard_negatives": True,
    }


def test_attach_neighbor_candidates_matches_root_property_and_route():
    module = _prepare_module()
    path_records = [
        {
            "root": {"qid": "Q1", "label": "California buttercup"},
            "route_properties": ["P225"],
        },
        {
            "root": {"qid": "Q1", "label": "California buttercup"},
            "route_properties": ["P1843"],
        },
    ]
    cache_rows = [
        {
            "root_qid": "Q1",
            "property_id": "P225",
            "route_properties": ["P225"],
            "candidates": [
                {"label": "Ranunculus repens", "relation": "same_genus"},
                {"label": "Ranunculus acris", "relation": "same_genus"},
            ],
        },
        {
            "root_qid": "Q2",
            "property_id": "P225",
            "route_properties": ["P225"],
            "candidates": [{"label": "Quercus rubra", "relation": "same_genus"}],
        },
    ]

    enriched, report = module.attach_neighbor_candidates_to_path_records(
        path_records,
        cache_rows,
    )

    assert enriched[0]["choice_candidates"] == [
        {"label": "Ranunculus repens", "relation": "same_genus"},
        {"label": "Ranunculus acris", "relation": "same_genus"},
    ]
    assert "choice_candidates" not in enriched[1]
    assert report == {
        "cache_rows": 2,
        "path_records_seen": 2,
        "path_records_enriched": 1,
        "candidate_count": 2,
    }


def test_main_always_applies_answer_prior_filter(monkeypatch, tmp_path, capsys):
    module = _prepare_module()
    output_dir = tmp_path / "export"
    rows = [
        _qa_row(question_id=1, property_id="P141", answer="least concern"),
        _qa_row(question_id=2, property_id="P141", answer="Least Concern"),
        _qa_row(question_id=3, property_id="P141", answer="least concern"),
        _qa_row(question_id=4, property_id="P141", answer="near threatened"),
        _qa_row(question_id=5, property_id="P225", answer="Ameiurus nebulosus"),
        _qa_row(question_id=6, property_id="P225", answer="Ptilotula penicillata"),
        _qa_row(question_id=7, property_id="P225", answer="Ranunculus californicus"),
        _qa_row(question_id=8, property_id="P225", answer="Corvus albus"),
    ]

    monkeypatch.setattr(sys, "argv", [
        "prepare_inaturalist_reasonvqa_eval.py",
        "--kg-paths",
        str(tmp_path / "paths.jsonl"),
        "--output-dir",
        str(output_dir),
        "--no-download-images",
        "--max-answer-prior-fraction",
        "0.5",
        "--min-answer-prior-group-size",
        "4",
    ])
    monkeypatch.setattr(module, "read_jsonl", lambda _path: [{"path": []}])
    monkeypatch.setattr(module, "generate_reasonvqa_rows", lambda **_kwargs: rows)

    assert module.main() == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["qa_rows"] == 4
    assert summary["answer_prior_filter"]["dropped_rows"] == 4
    merged_rows = [
        json.loads(line)
        for line in (output_dir / "knowledge_reasoning_vqa_merged.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["property_id"] for row in merged_rows} == {"P225"}


def test_cli_rejects_answer_prior_shortcut_escape_hatch(monkeypatch):
    module = _prepare_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_inaturalist_reasonvqa_eval.py",
            "--allow-answer-prior-shortcuts",
        ],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_cli_rejects_model_confusion_distractor_injection(monkeypatch):
    module = _prepare_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_inaturalist_reasonvqa_eval.py",
            "--confusion-predictions",
            "predictions.jsonl",
        ],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def _add_tar_member(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def _qa_row(question_id: int, property_id: str, answer: str) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": "What is the answer?",
        "image_id": f"image-{question_id}",
        "image_name": f"example-{question_id}.jpg",
        "image": {"file_name": f"val/species/example-{question_id}.jpg"},
        "image_path": "",
        "image_url": f"val/species/example-{question_id}.jpg",
        "source": "inaturalist_2021_val",
        "answers": [answer, "choice-b", "choice-c", "choice-d"],
        "correct": [1, 0, 0, 0],
        "program_answer": answer,
        "categories": [8],
        "hop": 1,
        "property_id": property_id,
        "property_label": "property",
        "entity_id": "Q1",
        "entity_name": "Example species",
        "route": [property_id],
        "has_scene_graph": False,
    }
