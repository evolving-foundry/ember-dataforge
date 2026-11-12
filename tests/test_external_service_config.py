"""External service entry points must not depend on a user's home directory."""

import runpy
import sys
from pathlib import Path

import pytest


def test_reasonvqa_model_and_data_paths_come_from_environment(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluation" / "evaluate_reasonvqa_probe_qwen.py"
    parse_args = runpy.run_path(str(script))["parse_args"]
    model = tmp_path / "model"
    data = tmp_path / "data"
    output = tmp_path / "output"
    monkeypatch.setenv("REASONVQA_MODEL_PATH", str(model))
    monkeypatch.setenv("REASONVQA_DATA_DIR", str(data))
    monkeypatch.setenv("REASONVQA_OUTPUT_DIR", str(output))
    monkeypatch.setattr(sys, "argv", [str(script)])

    args = parse_args()
    assert (args.model, args.data_dir, args.output_dir) == (model, data, output)

    monkeypatch.delenv("REASONVQA_MODEL_PATH")
    monkeypatch.delenv("REASONVQA_DATA_DIR")
    with pytest.raises(SystemExit, match="2"):
        parse_args()
