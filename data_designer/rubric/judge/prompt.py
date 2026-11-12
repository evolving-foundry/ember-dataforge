"""Judge prompt construction and response parsing (migrated from the RuACT
verl_reward.py kernel, decoupled from verl). Protocol points: - One Judge call lists
all rubrics by number; Judge returns a strict JSON object (keys are rubric-number
strings, values are exactly PRESENT / NOT_PRESENT). - Strict parse failure allows
lenient recovery: extract a JSON object from the response and score missing keys as 0
(record a JUDGE_LENIENT_RECOVERED audit row); if completely unparseable, score all
zero and record JUDGE_ABSTAIN_ALL -- the transport layer decides whether to retry;
this layer does not hide degradation. - `candidate_final_answer` extracts the final
answer after the think tag; when the reasoning segment is missing or truncated,
return None explicitly instead of guessing.
"""
from __future__ import annotations

import json
import re
from typing import Any

RUACT_JUDGE_INSTRUCTIONS = (
    "Given a user prompt, a generated response, and a list of quality rubrics, e"
    'valuate the response against EACH rubric.\nFor each rubric:\n- Mark "PRESENT"'
    ' if the criterion is satisfied, or "NOT_PRESENT" if it is not. For example,'
    ' given the response "Apples are red", the rubric "Mentions apples" is PRESE'
    'NT, "Does not mention strawberries" is also PRESENT since the response does'
    ' not mention strawberries, and "Mentions oranges" is NOT_PRESENT. Likewise,'
    ' "Avoids mentioning strawberries" is PRESENT, while "Avoids mentioning appl'
    'es" is NOT_PRESENT.\n- If a rubric contains multiple sentences or criteria, '
    "consider all of them. If any required part is not met, mark it NOT_PRESENT."
    " Return PRESENT only if the entire rubric is met.\n- When a rubric gives exa"
    'mples with wording such as "such as", "for example", or "including", the re'
    "sponse does not need to include every example. Satisfying one valid listed "
    "example can satisfy the criterion. There is no partial credit.\nReturn only "
    "one strict JSON object. Its keys must be the rubric numbers and its values "
    'must be exactly "PRESENT" or "NOT_PRESENT". Do not return Markdown, reasons'
    ", or extra keys."
)


class JudgeProtocolError(RuntimeError):
    pass


class JudgeGiveUpError(JudgeProtocolError):
    pass


def require_nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JudgeProtocolError(f"{field} must be a non-empty string")
    return value


def candidate_final_answer(solution: Any) -> tuple[str | None, str]:
    text = str(solution or "").strip()
    if not text:
        return (None, "missing_final_answer")
    close_matches = tuple(re.finditer("</think>", text, flags=re.IGNORECASE))
    if close_matches:
        final_answer = text[close_matches[-1].end() :].strip()
        if final_answer:
            return (final_answer, "after_think_close")
        return (None, "missing_final_answer")
    if re.search("<think>", text, flags=re.IGNORECASE):
        explicit_final = tuple(
            re.finditer(
                "(?:^|\\n)\\s*(?:#{1,6}\\s*)?(?:final\\s+answer|最终答案|答案)\\s*[:：]\\s*",
                text,
                flags=re.IGNORECASE,
            )
        )
        if explicit_final:
            final_answer = text[explicit_final[-1].end() :].strip()
            if final_answer:
                return (final_answer, "explicit_final_after_malformed_thinking")
        return (None, "missing_final_answer")
    return (text, "plain_final_answer")


def build_judge_prompt(
    *,
    query: str,
    final_answer: str,
    descriptions: tuple[str, ...],
) -> str:
    query = require_nonempty_text(query, "query")
    final_answer = require_nonempty_text(final_answer, "final answer")
    if not descriptions:
        raise JudgeProtocolError("judge prompt requires rubrics")
    rubrics_text = "\n".join(
        (
            f"{index}. {description}"
            for index, description in enumerate(descriptions, start=1)
        )
    )
    exact_keys = ", ".join((f'"{index}"' for index in range(1, len(descriptions) + 1)))
    return f"{RUACT_JUDGE_INSTRUCTIONS}\n\nThe output object must contain exactly these keys, with no omissions or extras: {exact_keys}. Verify the key count before responding.\n\n<Prompt>\n{query}\n</Prompt>\n\n<Response>\n{final_answer}\n</Response>\n\n<Rubrics>\n{rubrics_text}\n</Rubrics>"


def _score_value(value: Any) -> int | None:
    if isinstance(value, str):
        normalized = re.sub("\\s+", "_", value.strip().lower())
        if (
            "not_present" in normalized
            or normalized.startswith("not_")
            or "absent" in normalized
        ):
            return 0
        if "present" in normalized:
            return 1
        return None
    if value is True:
        return 1
    if value is False or value is None:
        return 0
    if isinstance(value, int) and value in (0, 1):
        return value
    return None


def parse_judge_reply(
    content: Any, *, rubric_count: int, prompt_sha256: str | None = None
) -> tuple[int, ...]:
    """Parse a Judge response into a tuple of binary scores. Strict path: complete JSON
    with an exactly matching key set and values exactly PRESENT/NOT_PRESENT. Lenient
    path: extract the first JSON object from the text; missing/undecidable keys score
    0. Completely unparseable: all zero (abstain), handled by the caller's retry
    policy.
    """
    expected_keys = tuple((str(index) for index in range(1, rubric_count + 1)))

    strict_payload: Any = None
    if isinstance(content, str) and content.strip():
        try:
            strict_payload = json.loads(content)
        except json.JSONDecodeError:
            strict_payload = None
        if isinstance(strict_payload, dict) and set(strict_payload) == set(
            expected_keys
        ):
            values = [strict_payload[key] for key in expected_keys]
            if all((v == "PRESENT" or v == "NOT_PRESENT" for v in values)):
                return tuple((1 if v == "PRESENT" else 0 for v in values))
    payload: Any = None
    if isinstance(content, str):
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            try:
                payload = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, dict):
        scores: list[int] = []
        read = 0
        for key in expected_keys:
            value = _score_value(payload.get(key))
            scores.append(1 if value == 1 else 0)
            read += 1 if value is not None else 0
        if prompt_sha256:
            print(
                f"JUDGE_LENIENT_RECOVERED prompt_sha256={prompt_sha256} expected_count={rubric_count} abstained={rubric_count - read}",
                flush=True,
            )
        return tuple(scores)
    if prompt_sha256:
        print(
            f"JUDGE_ABSTAIN_ALL prompt_sha256={prompt_sha256} expected_count={rubric_count} reason=unparseable_content",
            flush=True,
        )
    return tuple([0] * rubric_count)
