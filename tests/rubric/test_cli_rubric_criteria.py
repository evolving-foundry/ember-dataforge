"""rubric-criteria CLI `evolve` subcommand: argument wiring, noreplace preflight,
sidecar output persistence (overlaid library / dcd_* score corrections) as jsonl, and
empty-round summary without overlay.
"""
from __future__ import annotations

import argparse
import json
import threading

import pytest

from data_designer.cli.rubric_criteria import register
from data_designer.rubric.evolve.registry import PositiveRefinementRegistry
from tests.rubric.evolve_fixture import group_rows, mock_generator_server
from tests.rubric.test_evolve_round import BASE_RUBRICS, _ColumnPatternTransport


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register(subparsers)
    return parser


@pytest.fixture()
def generator_server():
    server = mock_generator_server("ok")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


@pytest.fixture()
def generator_env(monkeypatch, generator_server):
    base = f"http://127.0.0.1:{generator_server.server_address[1]}"
    monkeypatch.setenv("HEALTHBENCH_GENERATOR_API_BASE", base)
    monkeypatch.setenv("HEALTHBENCH_GENERATOR_API_KEY", "k")
    monkeypatch.setenv("HEALTHBENCH_GENERATOR_MODEL", "mock-model")


def _write_inputs(tmp_path, rows):
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    library = tmp_path / "rubric_library.json"
    library.write_text(
        json.dumps(BASE_RUBRICS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return responses, library


def test_evolve_subcommand_end_to_end(tmp_path, generator_env):
    parser = _parser()
    rows = group_rows()
    responses, library = _write_inputs(tmp_path, rows)
    result = tmp_path / "evolution_result.json"
    overlaid = tmp_path / "overlaid_libraries.jsonl"
    adjustments = tmp_path / "score_adjustments.jsonl"
    args = parser.parse_args(
        [
            "rubric-criteria",
            "evolve",
            "--responses", str(responses),
            "--rubric-library", str(library),
            "--registry-dir", str(tmp_path / "registry"),
            "--result", str(result),
            "--overlaid-output", str(overlaid),
            "--score-adjustments-output", str(adjustments),
            "--run-id", "rsi-1",
            "--global-step", "1",
        ]
    )
    from data_designer.cli.rubric_criteria import run as run_cli

    summary = run_cli(args, judge_transport=_ColumnPatternTransport())
    # Summary can be printed as JSON (main.py contract).
    json.dumps(summary, allow_nan=False)
    assert summary["status"] == "ok"
    assert summary["selected_parent_count"] == 1
    assert summary["decomposition_count"] == 1
    assert summary["overlaid_queries"] == ["q1"]
    assert summary["adjusted_response_count"] == len(rows)

    # Result file and sidecar outputs are persisted and equivalent to running the round
    # directly.
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["schema"] == "rubric-evolution-result"
    overlaid_rows = [
        json.loads(line) for line in overlaid.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["query_id"] for row in overlaid_rows] == ["q1"]
    parent_entry = next(
        rubric for rubric in overlaid_rows[0]["rubrics"] if rubric["rubric_key"] == "r0"
    )
    assert parent_entry["weight"] == pytest.approx(0.6 * 0.5)
    adjustment_rows = [
        json.loads(line)
        for line in adjustments.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["response_id"] for row in adjustment_rows} == {
        row["response_id"] for row in rows
    }
    assert all("dcd_reward_delta" in row for row in adjustment_rows)

    # noreplace: rerunning with the same parameters fails explicitly; result file and
    # sidecar outputs are not overwritten.
    from data_designer.cli.rubric_criteria import run as run_cli_again

    with pytest.raises(FileExistsError):
        run_cli_again(args, judge_transport=_ColumnPatternTransport())


def test_evolve_subcommand_empty_round(tmp_path, generator_env):
    parser = _parser()
    rows = group_rows(scores_matrix=[[0, i % 2] for i in range(8)])
    responses, library = _write_inputs(tmp_path, rows)
    args = parser.parse_args(
        [
            "rubric-criteria",
            "evolve",
            "--responses", str(responses),
            "--rubric-library", str(library),
            "--registry-dir", str(tmp_path / "registry"),
            "--result", str(tmp_path / "evolution_result.json"),
            "--run-id", "rsi-1",
            "--global-step", "1",
        ]
    )
    from data_designer.cli.rubric_criteria import run as run_cli

    summary = run_cli(args, judge_transport=_ColumnPatternTransport())
    assert summary["status"] == "ok"
    assert summary["selected_parent_count"] == 0
    assert summary["overlaid_queries"] == []
    assert summary["adjusted_response_count"] == 0
    # When sidecar outputs are not specified, no sidecar files are written.
    assert not (tmp_path / "overlaid_libraries.jsonl").exists()


def test_evolve_precheck_rejects_existing_artifacts_before_running(
    tmp_path, generator_env
):
    parser = _parser()
    rows = group_rows()
    responses, library = _write_inputs(tmp_path, rows)
    adjustments = tmp_path / "score_adjustments.jsonl"
    adjustments.write_text("", encoding="utf-8")
    args = parser.parse_args(
        [
            "rubric-criteria",
            "evolve",
            "--responses", str(responses),
            "--rubric-library", str(library),
            "--registry-dir", str(tmp_path / "registry"),
            "--result", str(tmp_path / "evolution_result.json"),
            "--score-adjustments-output", str(adjustments),
            "--run-id", "rsi-1",
            "--global-step", "1",
        ]
    )
    from data_designer.cli.rubric_criteria import run as run_cli

    with pytest.raises(FileExistsError):
        run_cli(args, judge_transport=_ColumnPatternTransport())
    # Preflight rejects before starting: no result file should be produced.
    assert not (tmp_path / "evolution_result.json").exists()
    # registry was not touched by this round (no pending / no decomposition).
    registry = PositiveRefinementRegistry(tmp_path / "registry")
    assert registry.blocked_query_ids() == frozenset()
    assert registry.snapshot("q1")["decompositions"] == []
