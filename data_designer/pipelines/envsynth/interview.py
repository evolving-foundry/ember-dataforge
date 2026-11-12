"""Offline curation questions and answers (decision 2/3).

The synthesizer never fills under-specified fields from model priors.  It emits
a bounded ``curation-questions-v1`` file (exit 1), and a later request may
carry an answers file that resolves those questions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from data_designer.pipelines.envsynth.domains.user_examples_seed_adapter import (
    CANONICAL_STATUSES,
    normalize_text,
    normalize_status,
)


CURATION_QUESTIONS_SCHEMA_VERSION = "curation-questions-v1"
CURATION_ANSWERS_SCHEMA_VERSION = 1


class CurationError(RuntimeError):
    """Curation answers are invalid or cannot be applied."""


def blocking_questions(domain: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        question
        for question in domain.get("questions") or []
        if question.get("blocking")
    ]


def questions_payload(
    domain: Dict[str, Any], *, seed_manifest_sha256: str = ""
) -> Dict[str, Any]:
    questions = list(domain.get("questions") or [])
    return {
        "schemaVersion": CURATION_QUESTIONS_SCHEMA_VERSION,
        "seedManifestSha256": seed_manifest_sha256,
        "questionCount": len(questions),
        "blockingCount": len([q for q in questions if q.get("blocking")]),
        "questions": questions,
    }


def write_curation_questions(
    path: str, domain: Dict[str, Any], *, seed_manifest_sha256: str = ""
) -> Dict[str, Any]:
    payload = questions_payload(domain, seed_manifest_sha256=seed_manifest_sha256)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def load_answers(path: str) -> Dict[str, Any]:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CurationError(f"curation answers is unreadable or not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise CurationError("curation answers must be an object")
    if raw.get("schemaVersion") != CURATION_ANSWERS_SCHEMA_VERSION:
        raise CurationError(
            "curation answers schemaVersion must be {}".format(
                CURATION_ANSWERS_SCHEMA_VERSION
            )
        )
    answers = raw.get("answers")
    if isinstance(answers, dict):
        return {str(key): value for key, value in answers.items()}
    if isinstance(answers, list):
        result: Dict[str, Any] = {}
        for item in answers:
            if not isinstance(item, dict) or not item.get("questionId"):
                raise CurationError("answers list entries must contain questionId")
            result[str(item["questionId"])] = item.get("value")
        return result
    raise CurationError("curation answers.answers must be an object or array")


def _question_refs(question: Dict[str, Any]) -> Set[str]:
    return {str(ref) for ref in question.get("contextRefs") or [] if ref}


def _ticket_refs(ticket: Dict[str, Any]) -> Set[str]:
    source = ticket.get("source") or {}
    refs = {
        str(source.get("context_ref")),
        "{}#{}".format(source.get("doc_id"), source.get("block_id")),
        str(source.get("doc_id")),
    }
    refs.update(str(ref) for ref in ticket.get("sourceRefs") or [])
    return {ref for ref in refs if ref and ref != "None"}


def _tickets_for_question(
    domain: Dict[str, Any], question: Dict[str, Any]
) -> List[Dict[str, Any]]:
    refs = _question_refs(question)
    if not refs:
        return []
    return [
        ticket
        for ticket in domain.get("tickets") or []
        if _ticket_refs(ticket) & refs
    ]


def _normalize_answer_status(value: Any) -> str:
    status = normalize_status(value)
    if status not in CANONICAL_STATUSES:
        raise CurationError(
            "Status answer must be one of {}; received {!r}".format(
                list(CANONICAL_STATUSES), value
            )
        )
    return status


def apply_answers(
    domain: Dict[str, Any], answers: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """Apply answers; return (domain, unresolved_questions, applied_ids)."""
    questions = list(domain.get("questions") or [])
    known_ids = {str(question.get("id")) for question in questions}
    unknown = sorted(set(answers) - known_ids)
    if unknown:
        raise CurationError(
            "curation answers contain an unknown questionId: {}".format(", ".join(unknown))
        )

    applied: List[str] = []
    unresolved: List[Dict[str, Any]] = []
    for question in questions:
        qid = str(question.get("id"))
        if qid not in answers:
            unresolved.append(question)
            continue
        value = answers[qid]
        code = question.get("code")
        tickets = _tickets_for_question(domain, question)
        if code in ("unknown_status", "missing_status", "conflicting_status"):
            status = _normalize_answer_status(value)
            if not tickets:
                unresolved.append(question)
                continue
            for ticket in tickets:
                ticket["status"] = status
            applied.append(qid)
        elif code == "missing_ticket_id":
            ticket_id = normalize_text(value)
            if not ticket_id:
                raise CurationError(f"{qid}: ticket_id answer must not be empty")
            if not tickets:
                unresolved.append(question)
                continue
            for ticket in tickets:
                ticket["ticket_id"] = ticket_id
            applied.append(qid)
        elif code == "title":
            title = normalize_text(value)
            if not title:
                raise CurationError(f"{qid}: title answer must not be empty")
            if not tickets:
                unresolved.append(question)
                continue
            for ticket in tickets:
                ticket["title"] = title
            applied.append(qid)
        else:
            # Document-level failures (no_tickets/parse_error) need a new source
            # import rather than a field answer; keep them blocking.
            unresolved.append(question)

    domain = dict(domain)
    domain["questions"] = unresolved
    domain["curationAnswersApplied"] = sorted(applied)
    # Validate the post-answer domain; duplicate ids can surface here.
    from data_designer.pipelines.envsynth.domains.user_examples_seed_adapter import validate_domain

    validate_domain(domain)
    return domain, unresolved, sorted(applied)


def apply_answers_file(
    domain: Dict[str, Any], answers_path: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    return apply_answers(domain, load_answers(answers_path))


__all__ = [
    "CURATION_ANSWERS_SCHEMA_VERSION",
    "CURATION_QUESTIONS_SCHEMA_VERSION",
    "CurationError",
    "apply_answers",
    "apply_answers_file",
    "blocking_questions",
    "load_answers",
    "questions_payload",
    "write_curation_questions",
]
