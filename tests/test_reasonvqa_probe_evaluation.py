from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_evaluation_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluation"
        / "evaluate_reasonvqa_probe_qwen.py"
    )
    spec = importlib.util.spec_from_file_location("reasonvqa_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_blank_mc_uses_multiple_choice_prompt_without_open_answer_instruction():
    row = {
        "question_id": 0,
        "question": "Which country is home to the pictured landmark?",
        "answers": ["France", "Germany", "Italy", "Spain"],
        "correct": [1, 0, 0, 0],
    }

    prompt = _load_evaluation_module().build_prompt(row, "blank_mc")

    assert "Choices:" in prompt
    assert "A. France" in prompt
    assert "Respond with only the option letter" in prompt
    assert "Give only the short answer" not in prompt


def test_choice_view_balances_correct_answer_position_when_shuffling():
    module = _load_evaluation_module()
    rows = [
        {
            "question_id": question_id,
            "answers": ["France", "Germany", "Italy", "Spain"],
            "correct": [1, 0, 0, 0],
        }
        for question_id in range(4)
    ]

    shuffled = [module.choice_view(row, shuffle_choices=True) for row in rows]

    assert [row["correct"].index(1) for row in shuffled] == [0, 1, 2, 3]
    assert all(set(row["answers"]) == {"France", "Germany", "Italy", "Spain"} for row in shuffled)
