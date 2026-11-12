"""Environment synthesizer: deterministic template mode + bounded agent loop.

Template mode is the CI/M1 baseline and uses zero LLM calls.  Agent mode is an
amplifier: it may revise rules/prompts/tasks through a whitelisted tool set, but
the final draft always passes the same deterministic validator.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from data_designer.pipelines.envsynth import grounding
from data_designer.pipelines.envsynth.domains.user_examples_seed_adapter import (
    ticket_to_text,
)
from data_designer.pipelines.envsynth.domains.power_ticket_profile import (
    CANONICAL_STATUSES,
    event_model as profile_event_model,
    npc_cards as profile_npc_cards,
    simulator_prompt as profile_simulator_prompt,
    ticket_rules,
    transition_for as profile_transition_for,
)
from data_designer.verifiers import validate_spec


TOOL_WHITELIST = (
    "read_seed",
    "read_regulation",
    "draft_artifact",
    "self_check",
    "ask_user",
)
MAX_AGENT_ROUNDS = 3
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DRAFT_ARTIFACT_KINDS = (
    "rules",
    "simulator_prompt",
    "tasks",
    "npc_cards",
    "event_model",
    "curation_questions",
)


class SynthesisError(RuntimeError):
    """Synthesis failed or produced an invalid draft."""


@dataclass
class SynthesisResult:
    status: str
    draft: Optional[Dict[str, Any]] = None
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ChatClient(Protocol):
    def complete(self, messages: Sequence[Dict[str, str]]) -> str:
        """Return the assistant message content."""


class OpenAICompatibleChatClient:
    """Minimal OpenAI-compatible chat client using only the standard library."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        temperature: float = 0.2,
        timeout_s: float = 60.0,
    ):
        if not base_url or not model or not api_key_env:
            raise SynthesisError("model config is missing baseUrl/model/apiKeyEnv")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)

    def complete(self, messages: Sequence[Dict[str, str]]) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise SynthesisError(
                f"Environment variable {self.api_key_env} is not set (credentials are never persisted; follow the DATA_FORGE_JUDGE_* convention)"
            )
        body = json.dumps(
            {
                "model": self.model,
                "messages": list(messages),
                "temperature": self.temperature,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise SynthesisError(f"Synthesis model call failed: {error}") from error
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise SynthesisError("Synthesis model response is missing choices[0].message.content") from error


def _ticket_summary(ticket: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: ticket.get(key)
        for key in (
            "ticket_id",
            "title",
            "status",
            "applicant",
            "assignee",
            "region",
            "priority",
            "submitted_at",
            "due_at",
        )
        if ticket.get(key) not in (None, "")
    }


def _rules(seed: int) -> List[Dict[str, Any]]:
    return ticket_rules()


def _npc_cards() -> Dict[str, Any]:
    return profile_npc_cards()


def _event_model(seed: int, tickets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return profile_event_model(seed, tickets)


def _simulator_prompt(domain: Dict[str, Any], rules: Sequence[Dict[str, Any]]) -> str:
    return profile_simulator_prompt(domain, rules)


def _transition_for(status: Optional[str]) -> Tuple[str, str]:
    return profile_transition_for(status)


def _task(
    *,
    index: int,
    ticket: Dict[str, Any],
    anchor_id: str,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ticket_id = str(ticket.get("ticket_id"))
    status = ticket.get("status")
    anchor = {
        "anchor_id": anchor_id,
        "ticket_id": ticket_id,
        "text": ticket_to_text(ticket),
        "source": ticket.get("source") or {},
        "expected": {"ticket_id": ticket_id, "status": status},
        "negative": {
            "ticket_id": ticket_id,
            "status": "rejected" if status != "rejected" else "approved",
        },
    }
    action, next_status = _transition_for(status)
    if action == "read":
        instruction = (
            "查询工单 {} 的当前状态。只输出 JSON："
            '{{"ticket_id": "...", "status": "..."}}。'.format(ticket_id)
        )
        gold = {"ticket_id": ticket_id, "status": status}
        negative = dict(anchor["negative"])
    else:
        instruction = (
            "工单 {} 当前状态为 {}。按冻结规则执行 `{}` 动作，并输出 JSON："
            '{{"ticket_id": "...", "action": "...", "next_status": "..."}}。'
        ).format(ticket_id, status, action)
        gold = {"ticket_id": ticket_id, "action": action, "next_status": next_status}
        other = "rejected" if next_status != "rejected" else "approved"
        negative = {"ticket_id": ticket_id, "action": action, "next_status": other}
    from data_designer.pipelines.envsynth.package import generator_ref

    task = {
        "task_id": "env-task-{:04d}".format(index),
        "anchor_id": anchor_id,
        "ticket_id": ticket_id,
        "instruction": instruction,
        "initial_state": {"ticket": _ticket_summary(ticket)},
        "verifier_spec": {
            "kind": "artifact_compare",
            "payload": {
                "artifact": {"path": "output/result.json", "format": "json"},
                "gold": gold,
            },
            "generator_ref": generator_ref(seed),
        },
        "gold_solution": gold,
        "negative_solution": negative,
        "provenance": {
            "generator": "envsynth",
            "seed": int(seed),
            "source": ticket.get("source") or {},
        },
    }
    return task, anchor


def template_draft(
    domain: Dict[str, Any], *, seed: int = 42, anchor_ratio: float = 0.3
) -> Dict[str, Any]:
    tickets = list(domain.get("tickets") or [])
    anchor_ids = set((domain.get("anchorSplit") or {}).get("anchorTicketIds") or [])
    anchors = [
        ticket for ticket in tickets if str(ticket.get("ticket_id")) in anchor_ids
    ]
    if not anchors:
        anchors = tickets[:1]
    tasks: List[Dict[str, Any]] = []
    anchor_rows: List[Dict[str, Any]] = []
    for index, ticket in enumerate(anchors, start=1):
        anchor_id = "anchor-{:04d}".format(index)
        task, anchor = _task(
            index=len(tasks) + 1, ticket=ticket, anchor_id=anchor_id, seed=seed
        )
        tasks.append(task)
        anchor_rows.append(anchor)
        if ticket.get("status") not in (None, "closed"):
            extra_task, _ = _task(
                index=len(tasks) + 1,
                ticket=ticket,
                anchor_id=anchor_id + "-transition",
                seed=seed,
            )
            if extra_task["gold_solution"] != task["gold_solution"]:
                tasks.append(extra_task)
    rules = _rules(seed)
    return {
        "rules": rules,
        "simulator_prompt": _simulator_prompt(domain, rules),
        "npc_cards": _npc_cards(),
        "event_model": _event_model(seed, tickets),
        "tasks": tasks,
        "anchors": anchor_rows,
        "assumptions": list(domain.get("assumptions") or []),
        "curation_questions": list(domain.get("questions") or []),
    }


def validate_task(task: Any, *, index: int = 0) -> Dict[str, Any]:
    if not isinstance(task, dict):
        raise SynthesisError(f"task[{index}] must be an object")
    for key in (
        "task_id",
        "anchor_id",
        "instruction",
        "verifier_spec",
        "gold_solution",
        "negative_solution",
    ):
        if key not in task:
            raise SynthesisError(f"task[{index}] is missing field {key}")
    for key in ("task_id", "anchor_id", "instruction"):
        if not isinstance(task[key], str) or not task[key].strip():
            raise SynthesisError(f"task[{index}].{key} must be a non-empty string")
    if not isinstance(task["verifier_spec"], dict):
        raise SynthesisError(f"task[{index}].verifier_spec must be an object")
    try:
        validate_spec(task["verifier_spec"])
    except Exception as error:
        raise SynthesisError(f"task[{index}].verifier_spec is invalid: {error}") from error
    if task.get("negative_solution") == task.get("gold_solution"):
        raise SynthesisError(f"task[{index}] negative_solution must not equal gold_solution")
    return task


def validate_draft(draft: Any) -> Dict[str, Any]:
    if not isinstance(draft, dict):
        raise SynthesisError("draft must be an object")
    rules = draft.get("rules")
    if not isinstance(rules, list) or not rules:
        raise SynthesisError("draft.rules must be a non-empty array")
    seen: set = set()
    for rule in rules:
        if not isinstance(rule, dict) or not rule.get("id"):
            raise SynthesisError("Each rule must contain an id")
        if rule.get("kind") not in ("fsm", "assertion"):
            raise SynthesisError("rule.kind must be fsm or assertion: {}".format(rule.get("id")))
        if rule["id"] in seen:
            raise SynthesisError("Duplicate rule.id: {}".format(rule["id"]))
        seen.add(rule["id"])
        if rule.get("kind") == "fsm":
            if not rule.get("states") or not isinstance(rule.get("transitions"), list):
                raise SynthesisError("fsm rule is missing states/transitions: {}".format(rule["id"]))
            states = set(rule.get("states") or [])
            if not states.issubset(set(CANONICAL_STATUSES)):
                raise SynthesisError(
                    "fsm rule contains unknown states: {}".format(rule["id"])
                )
            for transition in rule.get("transitions") or []:
                if not isinstance(transition, dict):
                    raise SynthesisError(
                        "fsm transition must be an object: {}".format(rule["id"])
                    )
                for key in ("from", "action", "to", "by_role"):
                    if not isinstance(transition.get(key), str) or not transition[
                        key
                    ].strip():
                        raise SynthesisError(
                            "fsm transition is missing {}: {}".format(key, rule["id"])
                        )
                if transition["from"] not in states or transition["to"] not in states:
                    raise SynthesisError(
                        "fsm transition references an unknown state: {}".format(rule["id"])
                    )
    tasks = draft.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SynthesisError("draft.tasks must be a non-empty array")
    seen_task_ids: set = set()
    for index, task in enumerate(tasks):
        validate_task(task, index=index)
        task_id = str(task["task_id"])
        if task_id in seen_task_ids:
            raise SynthesisError("Duplicate draft.tasks task_id: {}".format(task_id))
        seen_task_ids.add(task_id)
    if not isinstance(draft.get("simulator_prompt"), str) or not draft["simulator_prompt"].strip():
        raise SynthesisError("draft.simulator_prompt must be a non-empty string")
    if not isinstance(draft.get("npc_cards"), dict) or not draft["npc_cards"]:
        raise SynthesisError("draft.npc_cards must be a non-empty object")
    for role, card in draft["npc_cards"].items():
        if not isinstance(role, str) or not _SAFE_COMPONENT_RE.fullmatch(role):
            raise SynthesisError("draft.npc_cards role must be a safe file-path component")
        if not isinstance(card, dict):
            raise SynthesisError("draft.npc_cards.{} must be an object".format(role))
    if not isinstance(draft.get("event_model"), dict):
        raise SynthesisError("draft.event_model must be an object")
    if not isinstance(draft.get("anchors"), list) or not draft["anchors"]:
        raise SynthesisError("draft.anchors must be a non-empty array")
    anchor_ids: set = set()
    for index, anchor in enumerate(draft["anchors"]):
        if not isinstance(anchor, dict):
            raise SynthesisError("draft.anchors[{}] must be an object".format(index))
        anchor_id = anchor.get("anchor_id")
        if not isinstance(anchor_id, str) or not anchor_id.strip():
            raise SynthesisError("draft.anchors[{}].anchor_id must be non-empty".format(index))
        if anchor_id in anchor_ids:
            raise SynthesisError("Duplicate draft.anchors anchor_id: {}".format(anchor_id))
        anchor_ids.add(anchor_id)
    missing_anchor_tasks = sorted(
        {str(task["anchor_id"]) for task in tasks} - anchor_ids
    )
    if missing_anchor_tasks:
        raise SynthesisError(
            "tasks reference nonexistent anchor_id values: {}".format(", ".join(missing_anchor_tasks))
        )
    return draft


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    match = _FENCE_RE.search(text or "")
    candidate = match.group(1) if match else (text or "").strip()
    try:
        return json.loads(candidate)
    except ValueError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except ValueError:
                pass
    raise SynthesisError("Model output is not a valid JSON object")


def _agent_system_prompt(domain: Dict[str, Any], max_rounds: int) -> str:
    return (
        "You are the DataForge environment synthesizer. Revise the environment draft using the frozen power-office domain data.\n"
        "Each round, output exactly one JSON object in the format {{\"action\": <tool>, ...}} or "
        "{{\"action\": \"final\", \"draft\": {{...}}}}.\n"
        "Allowed tools (allowlist; no other tools): {}.\n"
        "Fields that draft_artifact may revise: {}.\n"
        "Observe these constraints: do not invent approval levels or deadlines absent from the source data; call ask_user when uncertain; "
        "do not change the semantics of task verifier_spec; do not delete anchors.\n"
        "At most {} rounds; wait for the tool result after each round.\n"
        "Domain: {}; ticket count: {}."
    ).format(
        ", ".join(TOOL_WHITELIST),
        ", ".join(DRAFT_ARTIFACT_KINDS),
        int(max_rounds),
        domain.get("family"),
        len(domain.get("tickets") or []),
    )


class AgentRunner:
    """Bounded tool loop; every tool call is validated against the whitelist."""

    def __init__(
        self,
        *,
        domain: Dict[str, Any],
        draft: Dict[str, Any],
        client: ChatClient,
        max_rounds: int = MAX_AGENT_ROUNDS,
        grounding_index: Optional[Dict[str, Any]] = None,
        backend: Optional[grounding.EmbeddingBackend] = None,
        profile: str = "office",
        quality_threshold: float = 0.6,
        expert_review: Optional[Dict[str, Any]] = None,
        expert_evidence_path: Optional[str] = None,
        anchor_ratio: float = 0.3,
        seed: int = 42,
    ):
        if not 1 <= int(max_rounds) <= MAX_AGENT_ROUNDS:
            raise SynthesisError(
                "maxRounds must be an integer in 1..{}".format(MAX_AGENT_ROUNDS)
            )
        self.domain = domain
        self.draft = draft
        self.client = client
        self.max_rounds = int(max_rounds)
        self.profile = profile
        self.quality_threshold = float(quality_threshold)
        self.expert_review = dict(
            expert_review or {"mode": "waived", "sampleRate": 0.0}
        )
        self.expert_evidence_path = expert_evidence_path
        self.anchor_ratio = float(anchor_ratio)
        self.seed = int(seed)
        self._protected_anchors = copy.deepcopy(list(draft.get("anchors") or []))
        self._protected_tasks = {
            str(task.get("task_id")): copy.deepcopy(task)
            for task in draft.get("tasks") or []
            if isinstance(task, dict)
        }
        self.index = grounding_index or grounding.build_index(
            grounding.chunks_from_domain(domain), backend=backend
        )
        self.backend = backend
        self.rounds: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.questions: List[Dict[str, Any]] = list(draft.get("curation_questions") or [])
        self.failed_checks: List[str] = []
        self.warnings: List[str] = []
        self._gate_bundle_cache: Optional[Tuple[str, Any]] = None

    def _observation(self, payload: Any) -> str:
        return "TOOL_RESULT " + json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _retrieve(self, query: str, role: Optional[str], top_k: Any) -> Dict[str, Any]:
        try:
            limit = int(top_k)
        except (TypeError, ValueError):
            limit = 4
        limit = max(1, min(limit, 8))
        results = grounding.retrieve(
            self.index, str(query or ""), top_k=limit, role=role, backend=self.backend
        )
        return {"query": query, "role": role, "results": results}

    def _ask_user(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        question = str(payload.get("question") or payload.get("prompt") or "").strip()
        if not question:
            raise SynthesisError("ask_user is missing question")
        item = {
            "id": "q-agent-{:04d}".format(len(self.questions) + 1),
            "code": "agent_question",
            "field": str(payload.get("field") or "unspecified"),
            "question": question,
            "contextRefs": list(payload.get("contextRefs") or []),
            "blocking": bool(payload.get("blocking", True)),
        }
        if payload.get("options"):
            item["options"] = list(payload["options"])
        self.questions.append(item)
        return {"questionId": item["id"], "recorded": True}

    def _enforce_immutable_contract(self, draft: Dict[str, Any]) -> None:
        if list(draft.get("anchors") or []) != self._protected_anchors:
            raise SynthesisError("agent must not modify or delete anchors")
        tasks = draft.get("tasks")
        if not isinstance(tasks, list):
            raise SynthesisError("draft.tasks must be an array")
        current_ids = [str(task.get("task_id")) for task in tasks if isinstance(task, dict)]
        if len(current_ids) != len(self._protected_tasks) or set(current_ids) != set(
            self._protected_tasks
        ):
            raise SynthesisError("agent must not delete, duplicate, or add task_id values")
        for task in tasks:
            original = self._protected_tasks[str(task.get("task_id"))]
            for key in (
                "verifier_spec",
                "gold_solution",
                "negative_solution",
                "anchor_id",
            ):
                if task.get(key) != original.get(key):
                    raise SynthesisError(
                        "agent must not change task {} field {}".format(
                            task.get("task_id"), key
                        )
                    )

    def _candidate_draft(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        candidate = copy.deepcopy(self.draft)
        candidate.update(copy.deepcopy(updates))
        self._enforce_immutable_contract(candidate)
        validate_draft(candidate)
        return candidate

    def _gate_bundle(self):
        from data_designer.pipelines.envsynth.package import build_env_package

        cache_key = json.dumps(self.draft, ensure_ascii=False, sort_keys=True)
        if self._gate_bundle_cache is not None and self._gate_bundle_cache[0] == cache_key:
            return self._gate_bundle_cache[1]

        bundle = build_env_package(
            self.domain,
            self.draft,
            ref_id="agent-self-check",
            version=1,
            profile=self.profile,
            seed=self.seed,
            mode="agent",
            max_rounds=self.max_rounds,
            anchor_ratio=self.anchor_ratio,
            backend=self.backend,
            expert_review=self.expert_review,
        )
        self._gate_bundle_cache = (cache_key, bundle)
        return bundle

    def _self_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from data_designer.pipelines.envsynth.gates import run_gate, run_gates

        kind = str(
            payload.get("gate") or payload.get("kind") or payload.get("check") or "all"
        )
        bundle = self._gate_bundle()
        if kind == "all":
            report = run_gates(
                bundle,
                quality_threshold=self.quality_threshold,
                expert_review=self.expert_review,
                expert_evidence_path=self.expert_evidence_path,
            )
            ok = bool(report.get("passed"))
            detail = report
        else:
            if kind not in ("anchor_replay", "distribution", "expert_review"):
                raise SynthesisError(
                    "self_check.gate must be anchor_replay/distribution/expert_review/all"
                )
            detail = run_gate(
                kind,
                bundle,
                quality_threshold=self.quality_threshold,
                expert_review=self.expert_review,
                expert_evidence_path=self.expert_evidence_path,
            )
            ok = bool(detail.get("passed"))
        if not ok:
            self.failed_checks.append("self_check:{}".format(kind))
        return {
            "ok": ok,
            "gate": kind,
            "taskCount": len(self.draft.get("tasks") or []),
            "detail": detail,
        }

    def _draft_artifact(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        kind = payload.get("kind")
        if kind not in DRAFT_ARTIFACT_KINDS:
            raise SynthesisError("invalid draft_artifact.kind: {!r}".format(kind))
        value = payload.get("value")
        if kind == "curation_questions":
            if not isinstance(value, list):
                raise SynthesisError("curation_questions must be an array")
            self.questions = list(value)
        else:
            candidate = self._candidate_draft({kind: value})
            self.draft = candidate
        if kind in ("rules", "tasks", "simulator_prompt", "npc_cards", "event_model"):
            validate_draft(self.draft)
        return {"kind": kind, "accepted": True}

    def _execute(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if action not in TOOL_WHITELIST:
            raise SynthesisError("Tool is not in the allowlist: {!r}".format(action))
        if action == "read_seed":
            return self._retrieve(payload.get("query", ""), "seed", payload.get("topK", 4))
        if action == "read_regulation":
            return self._retrieve(
                payload.get("query", ""), "regulation", payload.get("topK", 4)
            )
        if action == "draft_artifact":
            return self._draft_artifact(payload)
        if action == "self_check":
            return self._self_check(payload)
        return self._ask_user(payload)

    def run(self) -> SynthesisResult:
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": _agent_system_prompt(self.domain, self.max_rounds),
            },
            {
                "role": "user",
                "content": "Initial draft summary:\n" + json.dumps(
                    {
                        "rules": [rule.get("id") for rule in self.draft.get("rules") or []],
                        "tasks": [task.get("task_id") for task in self.draft.get("tasks") or []],
                        "anchors": [anchor.get("anchor_id") for anchor in self.draft.get("anchors") or []],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        final = False
        for round_index in range(1, self.max_rounds + 1):
            try:
                content = self.client.complete(messages)
            except Exception as error:
                self.failed_checks.append("model_call_failed")
                self.rounds.append({"round": round_index, "error": str(error)})
                break
            messages.append({"role": "assistant", "content": content})
            try:
                payload = _extract_json(content)
            except SynthesisError as error:
                self.failed_checks.append("invalid_model_json")
                self.rounds.append({"round": round_index, "error": str(error)})
                messages.append(
                    {
                        "role": "user",
                        "content": "Output must be a JSON object; please retry.",
                    }
                )
                continue
            if not isinstance(payload, dict):
                self.failed_checks.append("model_json_not_object")
                self.rounds.append({"round": round_index, "error": "not object"})
                continue
            action = payload.get("action") or payload.get("tool")
            if action == "final":
                updates = payload.get("draft")
                if updates is None:
                    updates = {}
                if not isinstance(updates, dict):
                    self.failed_checks.append("invalid_final_draft")
                    self.rounds.append(
                        {"round": round_index, "action": "final", "ok": False}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": self._observation(
                                {"error": "final.draft must be an object"}
                            ),
                        }
                    )
                    continue
                try:
                    candidate = self._candidate_draft(updates)
                except SynthesisError as error:
                    self.failed_checks.append("invalid_final_draft")
                    self.rounds.append(
                        {"round": round_index, "action": "final", "ok": False}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": self._observation({"error": str(error)}),
                        }
                    )
                    continue
                self.draft = candidate
                final = True
                self.rounds.append({"round": round_index, "action": "final"})
                break
            if action not in TOOL_WHITELIST:
                if any(key in payload for key in DRAFT_ARTIFACT_KINDS):
                    action = "draft_artifact"
                    payload = {"kind": next(k for k in DRAFT_ARTIFACT_KINDS if k in payload), "value": payload[next(k for k in DRAFT_ARTIFACT_KINDS if k in payload)]}
                else:
                    self.failed_checks.append("tool_not_whitelisted")
                    self.tool_calls.append(
                        {"round": round_index, "tool": action, "ok": False}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": self._observation(
                                {"error": "tool_not_whitelisted", "allowed": list(TOOL_WHITELIST)}
                            ),
                        }
                    )
                    self.rounds.append({"round": round_index, "action": action, "ok": False})
                    continue
            try:
                result = self._execute(str(action), payload)
                ok = True
            except SynthesisError as error:
                result = {"error": str(error)}
                ok = False
                if "not_whitelisted" in str(error):
                    self.failed_checks.append("tool_not_whitelisted")
            self.tool_calls.append(
                {"round": round_index, "tool": action, "ok": ok, "result": result if ok else None}
            )
            self.rounds.append({"round": round_index, "action": action, "ok": ok})
            messages.append({"role": "user", "content": self._observation(result)})
        if not final:
            self.warnings.append("max_rounds_reached_without_final")
        self.draft["curation_questions"] = list(self.questions)
        if [q for q in self.questions if q.get("blocking")]:
            return SynthesisResult(
                status="needs_curation",
                draft=self.draft,
                rounds=self.rounds,
                questions=self.questions,
                failed_checks=self.failed_checks + ["needs_curation"],
                tool_calls=self.tool_calls,
                warnings=self.warnings,
            )
        if not final:
            return SynthesisResult(
                status="synthesis_failed",
                draft=self.draft,
                rounds=self.rounds,
                questions=self.questions,
                failed_checks=self.failed_checks
                + ["max_rounds_reached_without_final"],
                tool_calls=self.tool_calls,
                warnings=self.warnings,
            )
        if self.failed_checks:
            return SynthesisResult(
                status="synthesis_failed",
                draft=self.draft,
                rounds=self.rounds,
                questions=self.questions,
                failed_checks=self.failed_checks,
                tool_calls=self.tool_calls,
                warnings=self.warnings,
            )
        try:
            validate_draft(self.draft)
        except SynthesisError as error:
            self.failed_checks.append(str(error))
            return SynthesisResult(
                status="synthesis_failed",
                draft=self.draft,
                rounds=self.rounds,
                questions=self.questions,
                failed_checks=self.failed_checks,
                tool_calls=self.tool_calls,
                warnings=self.warnings,
            )
        return SynthesisResult(
            status="synthesized",
            draft=self.draft,
            rounds=self.rounds,
            questions=self.questions,
            failed_checks=self.failed_checks,
            tool_calls=self.tool_calls,
            warnings=self.warnings,
        )


def synthesize(
    domain: Dict[str, Any],
    *,
    mode: str = "template",
    max_rounds: int = MAX_AGENT_ROUNDS,
    client: Optional[ChatClient] = None,
    seed: int = 42,
    anchor_ratio: float = 0.3,
    backend: Optional[grounding.EmbeddingBackend] = None,
    profile: str = "office",
    quality_threshold: float = 0.6,
    expert_review: Optional[Dict[str, Any]] = None,
    expert_evidence_path: Optional[str] = None,
) -> SynthesisResult:
    draft = template_draft(domain, seed=seed, anchor_ratio=anchor_ratio)
    if not 1 <= int(max_rounds) <= MAX_AGENT_ROUNDS:
        raise SynthesisError(
            "maxRounds must be an integer in 1..{}".format(MAX_AGENT_ROUNDS)
        )
    if mode == "template":
        try:
            validate_draft(draft)
        except SynthesisError as error:
            return SynthesisResult(
                status="synthesis_failed", draft=draft, failed_checks=[str(error)]
            )
        return SynthesisResult(status="synthesized", draft=draft)
    if mode != "agent":
        raise SynthesisError(f"Unknown synthesis mode: {mode!r}")
    if client is None:
        raise SynthesisError("mode=agent requires an injected ChatClient")
    runner = AgentRunner(
        domain=domain,
        draft=draft,
        client=client,
        max_rounds=max_rounds,
        backend=backend,
        profile=profile,
        quality_threshold=quality_threshold,
        expert_review=expert_review,
        expert_evidence_path=expert_evidence_path,
        anchor_ratio=anchor_ratio,
        seed=seed,
    )
    return runner.run()


__all__ = [
    "AgentRunner",
    "ChatClient",
    "DRAFT_ARTIFACT_KINDS",
    "MAX_AGENT_ROUNDS",
    "OpenAICompatibleChatClient",
    "SynthesisError",
    "SynthesisResult",
    "TOOL_WHITELIST",
    "synthesize",
    "template_draft",
    "validate_draft",
    "validate_task",
]
