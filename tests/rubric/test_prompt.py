"""Judge prompt construction and response parsing tests (matching source
verl_reward.py:167-364 semantics).
"""

from __future__ import annotations

import pytest

from data_designer.rubric.judge.prompt import (
    JudgeProtocolError,
    RUACT_JUDGE_INSTRUCTIONS,
    build_judge_prompt,
    candidate_final_answer,
    parse_judge_reply,
    require_nonempty_text,
)

DESCRIPTIONS = ("criterion A", "criterion B")


def test_build_judge_prompt_shape():
    prompt = build_judge_prompt(
        query="user query", final_answer="answer", descriptions=DESCRIPTIONS
    )
    assert RUACT_JUDGE_INSTRUCTIONS in prompt
    assert '"1", "2"' in prompt
    assert "<Prompt>\nuser query\n</Prompt>" in prompt
    assert "<Response>\nanswer\n</Response>" in prompt
    assert "1. criterion A" in prompt
    assert "2. criterion B" in prompt


def test_build_judge_prompt_rejects_empty():
    with pytest.raises(JudgeProtocolError):
        build_judge_prompt(query=" ", final_answer="a", descriptions=DESCRIPTIONS)
    with pytest.raises(JudgeProtocolError):
        build_judge_prompt(query="q", final_answer="", descriptions=DESCRIPTIONS)
    with pytest.raises(JudgeProtocolError):
        build_judge_prompt(query="q", final_answer="a", descriptions=())


def test_parse_judge_reply_strict():
    reply = '{"1": "PRESENT", "2": "NOT_PRESENT"}'
    assert parse_judge_reply(reply, rubric_count=2) == (1, 0)


def test_parse_judge_reply_lenient_recovery():
    noisy = 'Sure! {"1": "PRESENT", "2": "not present"} hope that helps'
    assert parse_judge_reply(noisy, rubric_count=2) == (1, 0)
    # Lenient recovery for missing keys: missing counts as 0
    partial = '{"1": "PRESENT"}'
    assert parse_judge_reply(partial, rubric_count=2) == (1, 0)


def test_parse_judge_reply_value_variants():
    reply = '{"1": true, "2": "absent"}'
    assert parse_judge_reply(reply, rubric_count=2) == (1, 0)
    reply = '{"1": false, "2": "not_applicable"}'
    assert parse_judge_reply(reply, rubric_count=2) == (0, 0)


def test_parse_judge_reply_abstain_all_zero():
    assert parse_judge_reply("not json at all", rubric_count=2) == (0, 0)
    assert parse_judge_reply(None, rubric_count=2) == (0, 0)


def test_candidate_final_answer_extraction():
    assert candidate_final_answer("</think>final text") == ("final text", "after_think_close")
    assert candidate_final_answer("plain") == ("plain", "plain_final_answer")
    assert candidate_final_answer("<think>truncated") == (None, "missing_final_answer")
    assert candidate_final_answer(
        "<think>x\n答案: the answer"
    ) == ("the answer", "explicit_final_after_malformed_thinking")
    assert candidate_final_answer("") == (None, "missing_final_answer")


def test_require_nonempty_text():
    assert require_nonempty_text("x", "f") == "x"
    with pytest.raises(JudgeProtocolError):
        require_nonempty_text("", "f")
