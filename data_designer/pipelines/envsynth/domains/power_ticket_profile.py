"""Single source of truth for the first power-ticket domain profile."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple


DOMAIN_SCHEMA_VERSION = "domain-power-ticket-v1"
FAMILY = "power-ticket"
PROFILE = "office"

CANONICAL_STATUSES = (
    "draft",
    "submitted",
    "reviewing",
    "approved",
    "rejected",
    "closed",
)

DEFAULT_GLOSSARY = (
    "工单",
    "审批",
    "驳回",
    "工作票",
    "操作票",
    "两票三制",
    "安全",
    "调度",
    "班组",
    "时限",
    "复核",
    "验收",
    "归档",
    "停电",
    "送电",
    "检修",
    "缺陷",
    "风险",
)

TICKET_STATE_SCHEMA: Dict[str, Any] = {
    "ticket": {
        "ticket_id": "string",
        "title": "string",
        "status": {"enum": list(CANONICAL_STATUSES)},
        "applicant": "string|null",
        "assignee": "string|null",
        "region": "string|null",
        "priority": "string|null",
        "submitted_at": "string|null",
        "due_at": "string|null",
    }
}

TICKET_ACTIONS: List[Dict[str, Any]] = [
    {
        "name": "submit",
        "allowed_roles": ["applicant"],
        "from": ["draft"],
        "to": "submitted",
    },
    {
        "name": "approve",
        "allowed_roles": ["reviewer"],
        "from": ["submitted"],
        "to": "reviewing",
    },
    {
        "name": "approve",
        "allowed_roles": ["dispatcher"],
        "from": ["reviewing"],
        "to": "approved",
    },
    {
        "name": "reject",
        "allowed_roles": ["reviewer", "dispatcher"],
        "from": ["submitted", "reviewing"],
        "to": "rejected",
    },
    {
        "name": "revise",
        "allowed_roles": ["applicant"],
        "from": ["rejected"],
        "to": "draft",
    },
    {
        "name": "close",
        "allowed_roles": ["records_clerk"],
        "from": ["approved", "rejected"],
        "to": "closed",
    },
    {
        "name": "cancel",
        "allowed_roles": ["applicant"],
        "from": ["draft"],
        "to": "closed",
    },
]

DANGEROUS_ACTIONS = [
    "bypass_approval",
    "delete_ticket",
    "execute_operation_ticket_without_work_ticket",
]

TICKET_TRANSITIONS: List[Dict[str, str]] = [
    {"from": "draft", "action": "submit", "to": "submitted", "by_role": "applicant"},
    {
        "from": "submitted",
        "action": "approve",
        "to": "reviewing",
        "by_role": "reviewer",
    },
    {
        "from": "reviewing",
        "action": "approve",
        "to": "approved",
        "by_role": "dispatcher",
    },
    {
        "from": "submitted",
        "action": "reject",
        "to": "rejected",
        "by_role": "reviewer",
    },
    {
        "from": "reviewing",
        "action": "reject",
        "to": "rejected",
        "by_role": "dispatcher",
    },
    {
        "from": "rejected",
        "action": "revise",
        "to": "draft",
        "by_role": "applicant",
    },
    {
        "from": "approved",
        "action": "close",
        "to": "closed",
        "by_role": "records_clerk",
    },
    {
        "from": "rejected",
        "action": "close",
        "to": "closed",
        "by_role": "records_clerk",
    },
    {
        "from": "draft",
        "action": "cancel",
        "to": "closed",
        "by_role": "applicant",
    },
]

NPC_CARDS: Dict[str, Dict[str, Any]] = {
    "applicant": {
        "display_name": "申请人",
        "persona": "提交并跟踪本班组工单的一线员工；不掌握审批权限。",
        "permissions": [
            "create_ticket",
            "submit_ticket",
            "revise_ticket",
            "cancel_ticket",
            "read_ticket",
        ],
        "allowed_actions": ["create", "submit", "revise", "cancel", "read"],
        "latency_seconds": 15,
    },
    "reviewer": {
        "display_name": "初审员",
        "persona": "负责合规与资料完整性初审；不得审批本人提交的工单。",
        "permissions": [
            "read_ticket",
            "approve_ticket",
            "reject_ticket",
            "request_changes",
        ],
        "allowed_actions": ["approve", "reject", "read"],
        "latency_seconds": 45,
    },
    "dispatcher": {
        "display_name": "调度复核员",
        "persona": "负责调度与安全复核，执行第二级审批。",
        "permissions": [
            "read_ticket",
            "approve_ticket",
            "reject_ticket",
            "escalate_ticket",
        ],
        "allowed_actions": ["approve", "reject", "escalate", "read"],
        "latency_seconds": 60,
    },
    "safety_officer": {
        "display_name": "安全员",
        "persona": "关注两票三制与危险操作，可阻断不合规动作。",
        "permissions": [
            "read_ticket",
            "block_action",
            "escalate_ticket",
            "record_violation",
        ],
        "allowed_actions": ["block", "escalate", "read"],
        "latency_seconds": 30,
    },
    "records_clerk": {
        "display_name": "归档员",
        "persona": "负责办结、归档与审计记录完整性。",
        "permissions": ["read_ticket", "close_ticket", "archive_ticket"],
        "allowed_actions": ["close", "archive", "read"],
        "latency_seconds": 20,
    },
}


def ticket_rules() -> List[Dict[str, Any]]:
    return [
        {
            "id": "ticket_approval_chain",
            "kind": "fsm",
            "entity": "ticket",
            "source": "domain_template",
            "initial": "draft",
            "states": list(CANONICAL_STATUSES),
            "transitions": copy.deepcopy(TICKET_TRANSITIONS),
            "timeouts": {
                "submitted": {"sim_seconds": 86400, "on_expire": "escalate"},
                "reviewing": {"sim_seconds": 172800, "on_expire": "escalate"},
            },
        },
        {
            "id": "two_ticket_compliance",
            "kind": "assertion",
            "source": "domain_template",
            "check": "操作票执行前工作票必须已审批",
            "on_violation": "block",
        },
        {
            "id": "no_self_approval",
            "kind": "assertion",
            "source": "domain_template",
            "check": "申请人不得审批自己的工单",
            "on_violation": "block",
        },
        {
            "id": "audit_trail_required",
            "kind": "assertion",
            "source": "domain_template",
            "check": "每次状态变更必须记录操作者与时间",
            "on_violation": "block",
        },
    ]


def npc_cards() -> Dict[str, Any]:
    return copy.deepcopy(NPC_CARDS)


def event_model(seed: int, tickets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    starts = sorted(
        str(ticket.get("submitted_at"))
        for ticket in tickets
        if ticket.get("submitted_at")
    )
    return {
        "schemaVersion": "event-model-v1",
        "seed": int(seed),
        "clock": {
            "start": starts[0] if starts else "2026-01-05T09:00:00Z",
            "tick_seconds": 3600,
            "horizon_hours": 72,
        },
        "event_types": [
            {"name": "ticket_submitted", "source": "applicant", "weight": 3},
            {"name": "review_requested", "source": "reviewer", "weight": 2},
            {"name": "approval_decided", "source": "dispatcher", "weight": 2},
            {"name": "timeout_escalated", "source": "system", "weight": 1},
            {"name": "comment_added", "source": "any", "weight": 4},
        ],
        "arrivalRatePerHour": 0.25,
        "deterministic": True,
    }


def simulator_prompt(domain: Dict[str, Any], rules: Sequence[Dict[str, Any]]) -> str:
    from data_designer.pipelines.envsynth.domains.user_examples_seed_adapter import ticket_to_text

    tickets = domain.get("tickets") or []
    lines = [
        "# 冻结办公环境模拟器",
        "",
        "你是一个电力办公场景的确定性业务模拟器。只依据下方冻结状态、规则与 grounding 引用回答；",
        "不得编造未给出的审批层级、时限或合规约束。若信息不足，明确输出 `needs_curation`。",
        "不得联网，不得调用外部工具，不得改变规则本身。",
        "",
        "## 当前工单",
    ]
    for ticket in tickets[:20]:
        lines.append("- " + ticket_to_text(ticket))
    lines.extend(["", "## 规则"])
    for rule in rules:
        if rule.get("kind") == "fsm":
            lines.append(
                "- FSM {}：状态 {}；超时 {}。".format(
                    rule.get("id"),
                    "/".join(rule.get("states") or []),
                    rule.get("timeouts"),
                )
            )
        else:
            lines.append(
                "- 断言 {}：{}（违规={}）。".format(
                    rule.get("id"),
                    rule.get("check"),
                    rule.get("on_violation"),
                )
            )
    lines.extend(
        [
            "",
            "## 输出纪律",
            "- 回答业务问题时给出 `ticket_id`、状态、允许动作与引用来源。",
            "- 涉及危险操作（停电/送电/检修）时先检查工作票与操作票审批状态。",
            "- 规则未覆盖的细节必须标记为待澄清，不得用常识补全。",
        ]
    )
    return "\n".join(lines) + "\n"


def transition_for(status: Optional[str]) -> Tuple[str, str]:
    mapping = {
        "draft": ("submit", "submitted"),
        "submitted": ("approve", "reviewing"),
        "reviewing": ("approve", "approved"),
        "rejected": ("revise", "draft"),
        "approved": ("close", "closed"),
        "closed": ("read", "closed"),
    }
    return mapping.get(status or "", ("read", status or "draft"))


__all__ = [
    "CANONICAL_STATUSES",
    "DANGEROUS_ACTIONS",
    "DEFAULT_GLOSSARY",
    "DOMAIN_SCHEMA_VERSION",
    "FAMILY",
    "NPC_CARDS",
    "PROFILE",
    "TICKET_ACTIONS",
    "TICKET_STATE_SCHEMA",
    "TICKET_TRANSITIONS",
    "event_model",
    "npc_cards",
    "simulator_prompt",
    "ticket_rules",
    "transition_for",
]
