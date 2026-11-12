"""Semantic mapping: Curator ``parsed-doc-v2`` rows → power-ticket domain data.

This is domain recipe knowledge, not framework code.  The first supported
family is the power-office ticket ledger.  The mapper is deliberately strict:
unknown statuses, missing ticket ids and empty documents become explicit
curation questions; the mapper never silently invents a status.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from data_designer.pipelines.envsynth.domains.power_ticket_profile import (
    CANONICAL_STATUSES,
    DEFAULT_GLOSSARY,
    DOMAIN_SCHEMA_VERSION,
    FAMILY,
)

STATUS_ALIASES: Dict[str, str] = {
    "草稿": "draft",
    "暂存": "draft",
    "draft": "draft",
    "待提交": "draft",
    "已提交": "submitted",
    "待审批": "submitted",
    "待审核": "submitted",
    "待处理": "submitted",
    "submitted": "submitted",
    "pending": "submitted",
    "审核中": "reviewing",
    "审批中": "reviewing",
    "处理中": "reviewing",
    "复核中": "reviewing",
    "reviewing": "reviewing",
    "in_progress": "reviewing",
    "已审批": "approved",
    "审批通过": "approved",
    "审核通过": "approved",
    "通过": "approved",
    "同意": "approved",
    "approved": "approved",
    "已驳回": "rejected",
    "驳回": "rejected",
    "不通过": "rejected",
    "拒绝": "rejected",
    "rejected": "rejected",
    "已关闭": "closed",
    "关闭": "closed",
    "已办结": "closed",
    "办结": "closed",
    "归档": "closed",
    "closed": "closed",
}

HEADER_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "ticket_id": ("工单编号", "工单号", "编号", "ticket id", "ticket_id", "id"),
    "title": ("工单标题", "标题", "事项", "工作内容", "任务", "title"),
    "status": ("状态", "审批状态", "当前状态", "status"),
    "applicant": ("申请人", "发起人", "填报人", "申请单位", "applicant"),
    "assignee": ("负责人", "处理人", "承办人", "执行人", "assignee"),
    "region": ("区域", "地区", "班组", "单位", "region"),
    "priority": ("优先级", "紧急程度", "priority"),
    "submitted_at": ("提交时间", "申请时间", "创建时间", "submitted_at"),
    "due_at": ("时限", "截止时间", "要求完成时间", "due_at"),
}

REGULATION_NAME_HINTS = ("规程", "制度", "规范", "办法", "细则", "policy", "regulation")

_WS_RE = re.compile(r"\s+")
_KEY_DROP_RE = re.compile(r"[\s_\-:：/\\()（）\[\]【】.。]+")
_TICKET_ID_RE = re.compile(
    r'(?:工单编号|工单号|编号|ticket[_ ]?id)\s*[:：]\s*'
    r'([A-Za-z0-9][A-Za-z0-9._/-]{2,63}(?:\s+\d{1,})?)',
    re.IGNORECASE,
)
_HEADER_ALIAS_PATTERN = re.compile(
    r'^\s*({})\s+(.+)$'.format(
        '|'.join(
            re.escape(alias)
            for alias in sorted(
                {alias for aliases in HEADER_SYNONYMS.values() for alias in aliases},
                key=len,
                reverse=True,
            )
        )
    ),
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return _WS_RE.sub(" ", text).strip()


def normalize_key(value: Any) -> str:
    text = normalize_text(value).lower()
    return _KEY_DROP_RE.sub("", text)


_NORMALIZED_HEADERS: Dict[str, str] = {}
for _field, _aliases in HEADER_SYNONYMS.items():
    for _alias in _aliases:
        _NORMALIZED_HEADERS[normalize_key(_alias)] = _field

_HEADER_PATTERNS = {
    field: [
        re.compile(
            r"{}\s*[:：]\s*([^\s；;|]+)".format(re.escape(alias)),
            re.IGNORECASE,
        )
        for alias in aliases
    ]
    for field, aliases in HEADER_SYNONYMS.items()
    if field != "ticket_id"
}


def _field_for_header(header: str) -> Optional[str]:
    key = normalize_key(header)
    if not key:
        return None
    exact = _NORMALIZED_HEADERS.get(key)
    if exact:
        return exact
    # A header such as "approval status (latest)" still maps to status, but only when the
    # synonym is a contiguous prefix/suffix to avoid broad substring guesses.
    for alias, field in _NORMALIZED_HEADERS.items():
        if len(alias) >= 2 and (key.startswith(alias) or key.endswith(alias)):
            return field
    return None


def _normalize_ocr_key_value_line(text: str) -> str:
    """Accept OCR lines such as status approved without punctuation."""
    if ':' in text or '：' in text:
        return text
    match = _HEADER_ALIAS_PATTERN.match(text)
    if match is None:
        return text
    return '{}: {}'.format(match.group(1), match.group(2))


def _normalize_status(value: Any) -> Optional[str]:
    key = normalize_key(value)
    if not key:
        return None
    return STATUS_ALIASES.get(key)


# Public alias: mapping recipes, curation answers and tests all normalize via
# the same table.  Keep the private name for existing internal callers.
normalize_status = _normalize_status


def _cell_text(cell: Dict[str, Any]) -> str:
    if cell.get("cell_status") == "covered":
        return ""
    return normalize_text(cell.get("text", ""))


def _grid_values(block: Dict[str, Any]) -> List[List[str]]:
    table = block.get("table") or {}
    if table.get("structure") != "grid":
        return []
    n_rows = int(table.get("n_rows") or 0)
    n_cols = int(table.get("n_cols") or 0)
    cells = table.get("cells") or []
    grid: Dict[Tuple[int, int], str] = {}
    for cell in cells:
        grid[(int(cell["row"]), int(cell["col"]))] = _cell_text(cell)
    return [
        [grid.get((row, col), "") for col in range(n_cols)]
        for row in range(n_rows)
    ]


def _find_header(grid: Sequence[Sequence[str]]) -> Tuple[int, Dict[int, str]]:
    best_index = -1
    best_map: Dict[int, str] = {}
    best_score = 0
    for index, row in enumerate(grid[:8]):
        mapping: Dict[int, str] = {}
        for col, value in enumerate(row):
            field = _field_for_header(value)
            if field and field not in mapping.values():
                mapping[col] = field
        score = len(mapping)
        if "ticket_id" in mapping.values() and score > best_score:
            best_index, best_map, best_score = index, mapping, score
    if "ticket_id" not in best_map.values():
        return -1, {}
    if not ({"status", "title"} & set(best_map.values())):
        return -1, {}
    return best_index, best_map


def _question(
    questions: List[Dict[str, Any]],
    *,
    code: str,
    field: str,
    question: str,
    context_refs: Sequence[str],
    value: Any = None,
    options: Optional[Sequence[str]] = None,
    blocking: bool = True,
    max_questions: int = 20,
) -> None:
    if len(questions) >= max_questions:
        return
    context_key = tuple(sorted({str(ref) for ref in context_refs if ref}))
    key = (code, field, normalize_text(value), context_key)
    for existing in questions:
        if (
            existing.get("code"),
            existing.get("field"),
            normalize_text(existing.get("observedValue")),
            tuple(sorted(str(ref) for ref in existing.get("contextRefs") or [] if ref)),
        ) == key:
            return
    item: Dict[str, Any] = {
        "id": "q-{:04d}".format(len(questions) + 1),
        "code": code,
        "field": field,
        "question": question,
        "contextRefs": list(context_refs),
        "blocking": bool(blocking),
    }
    if value is not None:
        item["observedValue"] = value
    if options:
        item["options"] = list(options)
    questions.append(item)


def _source_ref(
    document: Dict[str, Any], block: Dict[str, Any], file_record: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    source: Dict[str, Any] = {
        "doc_id": document.get("doc_id"),
        "block_id": block.get("block_id"),
        "media_type": (document.get("source") or {}).get("media_type"),
        "file_sha256": (document.get("source") or {}).get("sha256"),
    }
    if block.get("page") is not None:
        source["page"] = block["page"]
    if block.get("sheet") is not None:
        source["sheet"] = block["sheet"].get("name")
    if file_record:
        source["role"] = file_record.get("role")
    return source


def _new_ticket(
    values: Dict[str, str], source: Dict[str, Any], document: Dict[str, Any]
) -> Dict[str, Any]:
    ticket_id = normalize_text(values.get("ticket_id"))
    title = normalize_text(values.get("title"))
    status = _normalize_status(values.get("status"))
    extra = {
        key: normalize_text(value)
        for key, value in values.items()
        if key not in HEADER_SYNONYMS and normalize_text(value)
    }
    return {
        "ticket_id": ticket_id or None,
        "title": title or ("工单 {}".format(ticket_id) if ticket_id else ""),
        "status": status,
        "applicant": normalize_text(values.get("applicant")) or None,
        "assignee": normalize_text(values.get("assignee")) or None,
        "region": normalize_text(values.get("region")) or None,
        "priority": normalize_text(values.get("priority")) or None,
        "submitted_at": normalize_text(values.get("submitted_at")) or None,
        "due_at": normalize_text(values.get("due_at")) or None,
        "extra": extra,
        "source": dict(source),
        "rawStatus": normalize_text(values.get("status")),
        "docId": document.get("doc_id"),
        "language": document.get("language"),
    }


def _table_tickets(
    document: Dict[str, Any],
    block: Dict[str, Any],
    file_record: Optional[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    max_questions: int,
) -> List[Dict[str, Any]]:
    grid = _grid_values(block)
    if not grid:
        return []
    header_index, header_map = _find_header(grid)
    if header_index < 0:
        return []
    source = _source_ref(document, block, file_record)
    tickets: List[Dict[str, Any]] = []
    for row_index in range(header_index + 1, len(grid)):
        row = grid[row_index]
        if not any(normalize_text(value) for value in row):
            continue
        values: Dict[str, str] = {}
        for col, field in header_map.items():
            if col < len(row):
                values[field] = row[col]
        ref = "{}#{}:r{}".format(
            document.get("doc_id"), block.get("block_id"), row_index
        )
        if not normalize_text(values.get("ticket_id")):
            _question(
                questions,
                code="missing_ticket_id",
                field="ticket_id",
                question="Table row is missing a ticket ID; provide the ticket ID for this row or delete the row.",
                context_refs=[ref],
                value=" | ".join(normalize_text(v) for v in row if normalize_text(v)),
                max_questions=max_questions,
            )
        status_raw = normalize_text(values.get("status"))
        if status_raw and not _normalize_status(status_raw):
            _question(
                questions,
                code="unknown_status",
                field="status",
                question="Ticket status cannot be mapped; provide a canonical status.",
                context_refs=[ref],
                value=status_raw,
                options=list(CANONICAL_STATUSES),
                max_questions=max_questions,
            )
        if not status_raw:
            _question(
                questions,
                code="missing_status",
                field="status",
                question="Table row is missing a status; provide a canonical status.",
                context_refs=[ref],
                options=list(CANONICAL_STATUSES),
                max_questions=max_questions,
            )
        ticket = _new_ticket(values, source, document)
        ticket["source"]["row"] = row_index
        ticket["source"]["context_ref"] = ref
        tickets.append(ticket)
    return tickets


def _paragraph_tickets(
    document: Dict[str, Any],
    file_record: Optional[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    max_questions: int,
) -> List[Dict[str, Any]]:
    """Fallback for key-value paragraphs; never invents ids."""
    blocks = [block for block in document.get("blocks") or []]
    records: List[Dict[str, Any]] = []
    current: Dict[str, str] = {}
    current_source: Optional[Dict[str, Any]] = None
    current_ref: Optional[str] = None

    def flush() -> None:
        nonlocal current, current_source, current_ref
        if current.get("ticket_id"):
            ticket = _new_ticket(current, current_source or {}, document)
            if current_ref:
                ticket["source"]["context_ref"] = current_ref
            records.append(ticket)
        current = {}
        current_source = None
        current_ref = None

    for block in blocks:
        values: Dict[str, str] = {}
        ref = "{}#{}".format(document.get("doc_id"), block.get("block_id"))
        if block.get("type") == "kv_pair" and block.get("kv_pair"):
            pair = block["kv_pair"]
            field = _field_for_header(pair.get("key", ""))
            if field:
                values[field] = pair.get("value", "")
        elif block.get("type") in ("paragraph", "list"):
            text = normalize_text(block.get("text", ""))
            if not text:
                continue
            for line in re.split(r"[\n；;]+", text):
                line = _normalize_ocr_key_value_line(line)
                match = _TICKET_ID_RE.search(line)
                if match:
                    if current.get("ticket_id") and current["ticket_id"] != match.group(1):
                        flush()
                    values["ticket_id"] = match.group(1)
                    current_source = _source_ref(document, block, file_record)
                    current_ref = ref
                for field, patterns in _HEADER_PATTERNS.items():
                    for pattern in patterns:
                        found = pattern.search(line)
                        if found:
                            values[field] = found.group(1)
                            break
        if not values:
            continue
        current.update({key: value for key, value in values.items() if value})
        if current_source is None:
            current_source = _source_ref(document, block, file_record)
            current_ref = ref
    flush()

    for ticket in records:
        ref = ticket.get("source", {}).get("context_ref", document.get("doc_id"))
        if not ticket.get("status") and ticket.get("rawStatus"):
            _question(
                questions,
                code="unknown_status",
                field="status",
                question="Ticket status cannot be mapped; provide a canonical status.",
                context_refs=[ref],
                value=ticket.get("rawStatus"),
                options=list(CANONICAL_STATUSES),
                max_questions=max_questions,
            )
        elif not ticket.get("status"):
            _question(
                questions,
                code="missing_status",
                field="status",
                question="Ticket is missing a status; provide a canonical status.",
                context_refs=[ref],
                options=list(CANONICAL_STATUSES),
                max_questions=max_questions,
            )
    return records


def _chunk_text(text: str, *, size: int = 600, overlap: int = 80) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _regulation_chunks(
    document: Dict[str, Any], file_record: Optional[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    regulations: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    paragraphs = [
        block
        for block in document.get("blocks") or []
        if block.get("type") in ("paragraph", "list") and normalize_text(block.get("text"))
    ]
    title = ""
    if paragraphs:
        title = normalize_text(paragraphs[0].get("text"))[:80]
    if file_record:
        title = title or str(file_record.get("relativeRef") or "").split("/")[-1]
    for block in paragraphs:
        text = normalize_text(block.get("text"))
        for part in _chunk_text(text):
            chunk_id = "chunk-{:06d}".format(len(chunks) + 1)
            source = _source_ref(document, block, file_record)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": part,
                    "role": "regulation",
                    "source": source,
                }
            )
            regulations.append(
                {
                    "regulation_id": "reg-{:06d}".format(len(regulations) + 1),
                    "title": title or "规程文档",
                    "text": part,
                    "source": source,
                }
            )
    return regulations, chunks


def _seed_chunks(
    document: Dict[str, Any],
    file_record: Optional[Dict[str, Any]],
    tickets: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for block in document.get("blocks") or []:
        if block.get("type") not in ("paragraph", "list"):
            continue
        text = normalize_text(block.get("text"))
        if not text:
            continue
        source = _source_ref(document, block, file_record)
        key = (text, str(source.get("doc_id")), str(source.get("block_id")))
        if key in seen:
            continue
        seen.add(key)
        chunks.append(
            {
                "chunk_id": "chunk-{:06d}".format(len(chunks) + 1),
                "text": text,
                "role": "seed",
                "source": source,
            }
        )
    for ticket in tickets:
        text = ticket_to_text(ticket)
        source = dict(ticket.get("source") or {})
        key = (text, str(source.get("doc_id")), str(source.get("block_id")))
        if key in seen:
            continue
        seen.add(key)
        chunks.append(
            {
                "chunk_id": "chunk-{:06d}".format(len(chunks) + 1),
                "text": text,
                "role": "seed",
                "source": source,
            }
        )
    return chunks


def ticket_to_text(ticket: Dict[str, Any]) -> str:
    fields = [
        ("工单编号", ticket.get("ticket_id")),
        ("标题", ticket.get("title")),
        ("状态", ticket.get("status")),
        ("申请人", ticket.get("applicant")),
        ("负责人", ticket.get("assignee")),
        ("区域", ticket.get("region")),
        ("优先级", ticket.get("priority")),
        ("提交时间", ticket.get("submitted_at")),
        ("时限", ticket.get("due_at")),
    ]
    parts = ["{}：{}".format(label, value) for label, value in fields if value]
    for key, value in sorted((ticket.get("extra") or {}).items()):
        if value:
            parts.append("{}：{}".format(key, value))
    return "；".join(parts)


def compute_anchor_split(
    tickets: Sequence[Dict[str, Any]], ratio: float
) -> Dict[str, Any]:
    ids = sorted(
        str(ticket.get("ticket_id")) for ticket in tickets if ticket.get("ticket_id")
    )
    if not ids:
        return {
            "ratio": float(ratio),
            "strategy": "ticket-stratified",
            "anchorTicketIds": [],
            "groundingTicketIds": [],
            "overlap": False,
        }
    if len(ids) == 1 or ratio <= 0:
        return {
            "ratio": float(ratio),
            "strategy": "ticket-stratified",
            "anchorTicketIds": list(ids),
            "groundingTicketIds": list(ids),
            "overlap": True,
        }
    count = max(1, min(len(ids) - 1, int(round(len(ids) * float(ratio)))))
    indexes = sorted(
        {
            min(len(ids) - 1, max(0, int((index + 1) * len(ids) / (count + 1))))
            for index in range(count)
        }
    )
    anchor_ids = [ids[index] for index in indexes]
    grounding_ids = [ticket_id for ticket_id in ids if ticket_id not in set(anchor_ids)]
    return {
        "ratio": float(ratio),
        "strategy": "ticket-stratified",
        "anchorTicketIds": anchor_ids,
        "groundingTicketIds": grounding_ids,
        "overlap": False,
    }


def _merge_tickets(
    tickets: List[Dict[str, Any]], questions: List[Dict[str, Any]], max_questions: int
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    result: List[Dict[str, Any]] = []
    for ticket in tickets:
        ticket_id = ticket.get("ticket_id")
        if not ticket_id:
            result.append(ticket)
            continue
        existing = merged.get(ticket_id)
        if existing is None:
            merged[ticket_id] = ticket
            result.append(ticket)
            continue
        if existing.get("status") != ticket.get("status"):
            _question(
                questions,
                code="conflicting_status",
                field="status",
                question="The same ticket has conflicting statuses in the source data; confirm one status.",
                context_refs=[
                    str((existing.get("source") or {}).get("context_ref")),
                    str((ticket.get("source") or {}).get("context_ref")),
                ],
                value="{} vs {}".format(existing.get("status"), ticket.get("status")),
                options=list(CANONICAL_STATUSES),
                max_questions=max_questions,
            )
        for key, value in ticket.items():
            if key in ("source", "extra", "rawStatus"):
                continue
            if not existing.get(key) and value:
                existing[key] = value
        existing.setdefault("extra", {}).update(ticket.get("extra") or {})
        refs = set(existing.get("sourceRefs") or [])
        refs.add(str((existing.get("source") or {}).get("context_ref")))
        refs.add(str((ticket.get("source") or {}).get("context_ref")))
        existing["sourceRefs"] = sorted(ref for ref in refs if ref and ref != "None")
    return result


def domain_schema_guess(domain: Dict[str, Any]) -> Dict[str, Any]:
    tickets = domain.get("tickets") or []
    detected = set()
    for ticket in tickets:
        for key, value in ticket.items():
            if key in ("source", "extra", "rawStatus", "sourceRefs", "docId", "language"):
                continue
            if value not in (None, ""):
                detected.add(key)
    required = {"ticket_id", "title", "status"}
    missing = sorted(required - detected)
    confidence = 0.0
    if tickets:
        confidence = round(max(0.0, min(1.0, 1.0 - len(missing) / len(required))), 4)
    return {
        "schemaVersion": DOMAIN_SCHEMA_VERSION,
        "family": domain.get("family", FAMILY),
        "confidence": confidence,
        "fieldsDetected": sorted(detected),
        "missingFields": missing,
        "ticketCount": len(tickets),
    }


def validate_domain(domain: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(domain, dict):
        raise ValueError("domain dataset must be an object")
    if domain.get("schemaVersion") != DOMAIN_SCHEMA_VERSION:
        raise ValueError(
            "domain dataset schemaVersion must be {}".format(DOMAIN_SCHEMA_VERSION)
        )
    tickets = domain.get("tickets")
    if not isinstance(tickets, list):
        raise ValueError("domain.tickets must be an array")
    seen: Set[str] = set()
    for index, ticket in enumerate(tickets):
        if not isinstance(ticket, dict):
            raise ValueError("domain.tickets[{}] must be an object".format(index))
        ticket_id = ticket.get("ticket_id")
        if ticket_id:
            if ticket_id in seen:
                raise ValueError("domain.tickets contains duplicate ticket_id: {}".format(ticket_id))
            seen.add(ticket_id)
        status = ticket.get("status")
        if status is not None and status not in CANONICAL_STATUSES:
            raise ValueError("ticket {} has a non-canonical status: {}".format(ticket_id, status))
        if not isinstance(ticket.get("source"), dict):
            raise ValueError("ticket {} source must be an object".format(ticket_id))
    return domain


def map_parsed_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    file_records: Optional[Iterable[Dict[str, Any]]] = None,
    family: str = FAMILY,
    regulation_doc_ids: Optional[Iterable[str]] = None,
    anchor_holdout_ratio: float = 0.3,
    max_questions: int = 20,
) -> Dict[str, Any]:
    """Map C-IR-2 rows into ``domain-power-ticket-v1`` with curation questions."""
    files_by_sha: Dict[str, Dict[str, Any]] = {}
    for record in file_records or []:
        if record.get("sha256"):
            files_by_sha[str(record["sha256"])] = dict(record)

    regulation_ids = set(regulation_doc_ids or [])
    questions: List[Dict[str, Any]] = []
    tickets: List[Dict[str, Any]] = []
    regulations: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []
    assumptions: List[str] = []

    for row in rows:
        if row.get("error"):
            document = None
        else:
            document = row.get("parsed_doc")
        if not document:
            _question(
                questions,
                code="parse_error",
                field="document",
                question="Source file parsing failed; fix or remove the file and retry.",
                context_refs=[str(row.get("doc_id") or "input")],
                value=(row.get("error") or {}).get("message"),
                max_questions=max_questions,
            )
            continue

        doc_id = str(document.get("doc_id"))
        sha = str((document.get("source") or {}).get("sha256"))
        file_record = files_by_sha.get(sha)
        role = "regulation" if doc_id in regulation_ids else "seed"
        if file_record and file_record.get("role") in ("seed", "regulation"):
            role = str(file_record["role"])
        else:
            filename = str((file_record or {}).get("relativeRef") or "").lower()
            if any(hint in filename for hint in REGULATION_NAME_HINTS):
                role = "regulation"
        documents.append(
            {
                "doc_id": doc_id,
                "role": role,
                "media_type": (document.get("source") or {}).get("media_type"),
                "sha256": sha,
                "language": document.get("language"),
            }
        )

        if role == "regulation":
            doc_regulations, doc_chunks = _regulation_chunks(document, file_record)
            regulations.extend(doc_regulations)
            chunks.extend(doc_chunks)
            if not doc_chunks:
                _question(
                    questions,
                    code="empty_regulation",
                    field="regulation",
                    question="No usable text was parsed from the regulation document; check whether it is a scan or provide a text version.",
                    context_refs=[doc_id],
                    blocking=False,
                    max_questions=max_questions,
                )
            continue

        doc_tickets: List[Dict[str, Any]] = []
        for block in document.get("blocks") or []:
            if block.get("type") == "table":
                doc_tickets.extend(
                    _table_tickets(
                        document, block, file_record, questions, max_questions
                    )
                )
        if not doc_tickets:
            doc_tickets.extend(
                _paragraph_tickets(
                    document, file_record, questions, max_questions
                )
            )
        if not doc_tickets:
            _question(
                questions,
                code="no_tickets",
                field="tickets",
                question="No ticket table or key-value ticket fields were recognized in the user samples; check the file type or add samples.",
                context_refs=[doc_id],
                max_questions=max_questions,
            )
        tickets.extend(doc_tickets)
        chunks.extend(_seed_chunks(document, file_record, doc_tickets))

    tickets = _merge_tickets(tickets, questions, max_questions)
    for ticket in tickets:
        if not ticket.get("title") and ticket.get("ticket_id"):
            ticket["title"] = "工单 {}".format(ticket["ticket_id"])
            if "title_from_template" not in assumptions:
                assumptions.append("title_from_template")

    split = compute_anchor_split(tickets, anchor_holdout_ratio)
    anchor_ids = set(split["anchorTicketIds"])
    for ticket in tickets:
        if ticket.get("ticket_id") in anchor_ids:
            ticket["split"] = "anchor"
        else:
            ticket["split"] = "grounding"

    # Keep chunk ids stable and unique after all documents are processed.
    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = "chunk-{:06d}".format(index + 1)

    domain: Dict[str, Any] = {
        "schemaVersion": DOMAIN_SCHEMA_VERSION,
        "family": family,
        "documents": documents,
        "tickets": sorted(
            tickets,
            key=lambda item: (
                str(item.get("ticket_id") or ""),
                str((item.get("source") or {}).get("doc_id") or ""),
            ),
        ),
        "regulations": regulations,
        "chunks": chunks,
        "glossary": list(DEFAULT_GLOSSARY),
        "anchorSplit": split,
        "questions": questions,
        "assumptions": assumptions,
        "coverage": {
            "documentCount": len(documents),
            "ticketCount": len(tickets),
            "regulationChunkCount": len(regulations),
            "questionCount": len(questions),
        },
    }
    domain["schemaGuess"] = domain_schema_guess(domain)
    validate_domain(domain)
    return domain


__all__ = [
    "CANONICAL_STATUSES",
    "DOMAIN_SCHEMA_VERSION",
    "FAMILY",
    "compute_anchor_split",
    "domain_schema_guess",
    "map_parsed_rows",
    "normalize_status",
    "ticket_to_text",
    "validate_domain",
]
