"""Fixtures for C-IR-2 rows, seed bundles and C-WK-1 worker inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from data_designer.pipelines.envsynth.seed_import import (
    load_seed_bundle,
    sha256_file,
    write_seed_import_from_rows,
)
from data_designer.pipelines.envsynth.cli import env_compile as worker


def _cell(row: int, col: int, text: str) -> Dict[str, Any]:
    return {
        "row": row,
        "col": col,
        "row_span": 1,
        "col_span": 1,
        "text": text,
        "cell_status": "value" if text else "empty",
        "bbox": None,
        "covered_by": None,
        "ocr_confidence": None,
    }


def _table_block(rows: List[List[str]], block_id: str = "b-000001") -> Dict[str, Any]:
    n_rows = len(rows)
    n_cols = len(rows[0])
    cells = [
        _cell(row_index, col_index, text)
        for row_index, row in enumerate(rows)
        for col_index, text in enumerate(row)
    ]
    return {
        "block_id": block_id,
        "type": "table",
        "sheet": {"name": "ticket", "index": 0, "state": "visible"},
        "bbox": None,
        "text": "",
        "table": {
            "structure": "grid",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "cells": cells,
        },
        "kv_pair": None,
        "style": None,
        "ocr_confidence": None,
        "page": None,
    }


def parsed_row(
    tickets: Iterable[Tuple[str, str, str]],
    *,
    sha256: Optional[str] = None,
    doc_id: Optional[str] = None,
    media_type: str = "xlsx",
    block_id: str = "b-000001",
) -> Dict[str, Any]:
    tickets = list(tickets)
    sha = sha256 or hashlib.sha256(
        json.dumps(tickets, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    rows = [["Ticket ID", "标题", "状态"]] + [list(ticket) for ticket in tickets]
    document = {
        "schema_version": "parsed-doc-v2",
        "doc_id": doc_id or "pd-" + sha[:16],
        "source": {
            "media_type": media_type,
            "sha256": sha,
            "byte_size": 123,
            "page_count": None,
        },
        "language": "zh",
        "provenance": {
            "extractor": "openpyxl",
            "extractor_version": "1",
            "operation": "document_parsing",
            "op_version": "1",
            "reading_order_version": "ro-v1",
            "routing_reason": None,
            "ranking_version": None,
            "ranking_evaluation_set": None,
            "backend": None,
            "backend_version": None,
            "device": None,
            "model_id": None,
        },
        "blocks": [_table_block(rows, block_id=block_id)],
    }
    return {
        "input_index": 0,
        "doc_id": document["doc_id"],
        "parsed_doc": document,
        "error": None,
    }


def build_seed(
    tmp_path: Path,
    tickets: Iterable[Tuple[str, str, str]],
    *,
    seed_name: str = "seed",
    include_path_ref: bool = True,
    anchor_ratio: float = 0.3,
):
    row = parsed_row(tickets)
    sha = row["parsed_doc"]["source"]["sha256"]
    records = None
    if include_path_ref:
        records = [
            {
                "pathRef": str(tmp_path / "source.xlsx"),
                "relativeRef": "source.xlsx",
                "sha256": sha,
                "mediaType": "xlsx",
                "byteSize": 123,
                "role": "seed",
            }
        ]
    seed_dir = tmp_path / seed_name
    write_seed_import_from_rows(
        str(seed_dir),
        [row],
        file_records=records,
        anchor_holdout_ratio=anchor_ratio,
    )
    bundle = load_seed_bundle(str(seed_dir / "seed_manifest.json"))
    return bundle, seed_dir


def worker_request(
    tmp_path: Path,
    seed_dir: Path,
    *,
    asset_name: str = "assets",
    **overrides: Any,
) -> Dict[str, Any]:
    seed_manifest = seed_dir / "seed_manifest.json"
    request = {
        "refId": "office-power",
        "family": "power-ticket",
        "split": "train",
        "seedManifestPath": str(seed_manifest),
        "seedManifestSha256": sha256_file(seed_manifest),
        "envProfile": "office",
        "synthesis": {
            "mode": "template",
            "maxRounds": 3,
            "grounding": {"embedModel": "char-ngram-v1", "topK": 4},
        },
        "curationAnswersPath": None,
        "qualityThreshold": 0.5,
        "anchorHoldoutRatio": 0.3,
        "expertReview": {"mode": "waived", "sampleRate": 0.0},
        "seed": 42,
        "assetRoot": str(tmp_path / asset_name),
    }
    request.update(overrides)
    return {"schemaVersion": 1, "request": request}


def write_input(tmp_path: Path, payload: Dict[str, Any], name: str = "input.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def run_template_build(
    tmp_path: Path,
    tickets: Iterable[Tuple[str, str, str]],
    *,
    seed_name: str = "seed",
    asset_name: str = "assets",
    snapshot_name: str = "snapshot.json",
    request_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], Path]:
    _, seed_dir = build_seed(tmp_path, tickets, seed_name=seed_name)
    request = worker_request(
        tmp_path,
        seed_dir,
        asset_name=asset_name,
        **(request_overrides or {}),
    )
    input_path = write_input(tmp_path, request, name=seed_name + "-input.json")
    snapshot_path = tmp_path / snapshot_name
    code = worker.run_worker([input_path, str(snapshot_path)])
    assert code == 0, "worker failed"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return request["request"]["assetRoot"], snapshot, snapshot_path
