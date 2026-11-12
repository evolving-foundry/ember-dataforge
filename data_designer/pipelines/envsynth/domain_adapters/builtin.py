"""Built-in domain adapters shipped with the environment synthesis recipe."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from data_designer.pipelines.envsynth.domain_adapters.base import (
    DomainAdapterError,
    register_domain_adapter,
)
from data_designer.pipelines.envsynth.environment_model import (
    ActionDefinition,
    ActorDefinition,
    EnvironmentModel,
    EventModel,
    METADATA_DANGEROUS_ACTIONS,
    METADATA_DOMAIN_SCHEMA_VERSION,
    METADATA_GLOSSARY,
    ReplayEntry,
    RuleSet,
    SimulatorDefinition,
    StateCollection,
    TaskDefinition,
    Transition,
    WorkspaceFile,
    mapping_tuple,
    state_rows,
)
from data_designer.pipelines.envsynth.domains.power_ticket_profile import (
    DANGEROUS_ACTIONS,
    DOMAIN_SCHEMA_VERSION,
    TICKET_STATE_SCHEMA,
    TICKET_TRANSITIONS,
)
from data_designer.pipelines.envsynth.domains.user_examples_seed_adapter import ticket_to_text


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DomainAdapterError(f"{label} must be an object")
    return dict(value)


def _virtual_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise DomainAdapterError(f"{label} must be an absolute virtual path")
    if value != "/" and any(part in ("", ".", "..") for part in value.split("/")[1:]):
        raise DomainAdapterError(f"{label} must not contain empty or '..' components")
    return value


def _workspace_files(value: Any) -> tuple[WorkspaceFile, ...]:
    if value is None:
        return ()
    result: list[WorkspaceFile] = []
    if isinstance(value, Mapping):
        items: Sequence[Any] = [
            {"path": path, "content": content} for path, content in value.items()
        ]
    elif isinstance(value, list):
        items = value
    else:
        raise DomainAdapterError("workspace must be an object or an array")
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise DomainAdapterError(f"workspace[{index}] must be an object")
        path = _virtual_path(item.get("path"), f"workspace[{index}].path")
        content = item.get("content", item.get("text", ""))
        if isinstance(content, Mapping):
            content = content.get("content", content.get("text", ""))
        if not isinstance(content, str):
            raise DomainAdapterError(f"workspace[{index}].content must be text")
        result.append(WorkspaceFile(path=path, content=content))
    return tuple(result)


def _normalize_command(command: str) -> str:
    return " ".join(command.strip().split())


def _replay_entries(value: Any) -> tuple[ReplayEntry, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        entries = value.get("entries", value.get("rows", []))
    else:
        entries = value
    if not isinstance(entries, list):
        raise DomainAdapterError("replay must contain an entries array")
    result: list[ReplayEntry] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise DomainAdapterError(f"replay[{index}] must be an object")
        command = entry.get("cmd", entry.get("command"))
        if not isinstance(command, str) or not command.strip():
            raise DomainAdapterError(f"replay[{index}].cmd must be non-empty")
        cwd = _virtual_path(entry.get("cwd", "/"), f"replay[{index}].cwd")
        outcome = {
            key: value
            for key, value in entry.items()
            if key not in ("cmd", "command", "cwd")
        }
        result.append(ReplayEntry(cwd=cwd, command=command, outcome=outcome))
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(result):
        key = (entry.cwd, _normalize_command(entry.command))
        if key in seen:
            raise DomainAdapterError(
                f"replay[{index}] duplicates (cwd, cmd) after whitespace normalization"
            )
        seen.add(key)
    return tuple(result)


def _expected_files(task: Mapping[str, Any]) -> dict[str, str]:
    candidates: list[Any] = [
        task.get("expected_files"),
        _mapping(task.get("metadata"), "task.metadata").get("expected_files"),
        _mapping(task.get("gold_solution"), "task.gold_solution").get("expect_files"),
        _mapping(task.get("gold_solution"), "task.gold_solution").get("expected_files"),
        _mapping(task.get("expected"), "task.expected").get("files"),
    ]
    verifier = _mapping(task.get("verifier_spec"), "task.verifier_spec")
    payload = _mapping(verifier.get("payload"), "task.verifier_spec.payload")
    candidates.append(payload.get("expect_files"))
    candidates.append(payload.get("expected_files"))
    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, Mapping):
            raise DomainAdapterError("expected_files must be an object")
        if not candidate:
            continue
        result: dict[str, str] = {}
        for path, content in candidate.items():
            if not isinstance(path, str) or not path:
                raise DomainAdapterError("expected_files keys must be non-empty paths")
            if not isinstance(content, str):
                raise DomainAdapterError(
                    f"expected_files[{path!r}] must contain text"
                )
            result[path] = content
        return result
    return {}


_TASK_FIELDS = {
    "task_id",
    "instruction",
    "goal",
    "initial_state",
    "gold_solution",
    "negative_solution",
    "verifier_spec",
    "max_steps",
    "seed",
    "anchor_id",
    "expected_files",
    "metadata",
    "provenance",
}


def _tasks_from_draft(draft: Mapping[str, Any]) -> tuple[TaskDefinition, ...]:
    raw_tasks = draft.get("tasks") or []
    if not isinstance(raw_tasks, list):
        raise DomainAdapterError("draft.tasks must be an array")
    tasks: list[TaskDefinition] = []
    for index, task in enumerate(raw_tasks):
        if not isinstance(task, Mapping):
            raise DomainAdapterError(f"draft.tasks[{index}] must be an object")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise DomainAdapterError(f"draft.tasks[{index}].task_id must be non-empty")
        instruction = task.get("instruction")
        if instruction is None or instruction == "":
            instruction = task.get("goal", "")
        if not isinstance(instruction, str):
            raise DomainAdapterError(f"draft.tasks[{index}].instruction must be text")
        metadata = {
            key: value for key, value in task.items() if key not in _TASK_FIELDS
        }
        if isinstance(task.get("provenance"), Mapping):
            metadata["provenance"] = dict(task["provenance"])
        tasks.append(
            TaskDefinition(
                task_id=task_id,
                instruction=instruction,
                initial_state=_mapping(task.get("initial_state"), "task.initial_state"),
                gold_solution=_mapping(task.get("gold_solution"), "task.gold_solution"),
                negative_solution=_mapping(
                    task.get("negative_solution"), "task.negative_solution"
                ),
                verifier_spec=_mapping(task.get("verifier_spec"), "task.verifier_spec"),
                max_steps=task.get("max_steps"),
                seed=task.get("seed"),
                anchor_id=str(task.get("anchor_id") or ""),
                expected_files=_expected_files(task),
                metadata=metadata,
            )
        )
    return tuple(tasks)


def _actors_from_draft(draft: Mapping[str, Any]) -> tuple[ActorDefinition, ...]:
    cards = draft.get("npc_cards") or {}
    if not isinstance(cards, Mapping):
        raise DomainAdapterError("draft.npc_cards must be an object")
    actors: list[ActorDefinition] = []
    for role, card in cards.items():
        if not isinstance(role, str) or not role.strip():
            raise DomainAdapterError("draft.npc_cards keys must be non-empty roles")
        if not isinstance(card, Mapping):
            raise DomainAdapterError(f"draft.npc_cards[{role!r}] must be an object")
        actors.append(ActorDefinition(role=role, card=dict(card)))
    return tuple(sorted(actors, key=lambda actor: actor.role))


def _rules_from_draft(draft: Mapping[str, Any]) -> RuleSet:
    raw = draft.get("rules") or []
    if not isinstance(raw, list):
        raise DomainAdapterError("draft.rules must be an array")
    parsed: list[dict[str, Any]] = []
    for index, rule in enumerate(raw):
        if not isinstance(rule, Mapping):
            raise DomainAdapterError(f"draft.rules[{index}] must be an object")
        parsed.append(dict(rule))
    assertions = tuple(rule for rule in parsed if rule.get("kind") == "assertion")
    return RuleSet(raw=tuple(parsed), assertions=assertions)


def _events_from_draft(draft: Mapping[str, Any]) -> EventModel:
    raw = _mapping(draft.get("event_model"), "draft.event_model")
    clock = _mapping(raw.get("clock"), "draft.event_model.clock")
    tick_seconds = clock.get("tick_seconds", 3600)
    if isinstance(tick_seconds, bool) or not isinstance(tick_seconds, int) or tick_seconds < 1:
        raise DomainAdapterError(
            "draft.event_model.clock.tick_seconds must be a positive integer"
        )
    raw_events = raw.get("events")
    if raw_events is None:
        raw_events = []
    if not isinstance(raw_events, list):
        raise DomainAdapterError("draft.event_model.events must be an array")
    events: list[dict[str, Any]] = []
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            raise DomainAdapterError(
                f"draft.event_model.events[{index}] must be an object"
            )
        events.append(dict(event))
    return EventModel(step_minutes=max(1, tick_seconds // 60), events=tuple(events))


def _simulator_from_draft(draft: Mapping[str, Any]) -> SimulatorDefinition:
    prompt = draft.get("simulator_prompt")
    if prompt is None:
        prompt = ""
    if not isinstance(prompt, str):
        raise DomainAdapterError("draft.simulator_prompt must be text")
    return SimulatorDefinition(prompt=prompt)


class PowerTicketDomainAdapter:
    """First business recipe: the power-office ticket ledger."""

    family = "power-ticket"

    def supports(self, domain: Mapping[str, Any], draft: Mapping[str, Any]) -> bool:
        del draft
        return (
            domain.get("family") == self.family
            or domain.get("schemaVersion") == DOMAIN_SCHEMA_VERSION
        )

    def extract(
        self,
        domain: Mapping[str, Any],
        draft: Mapping[str, Any],
        *,
        grounding_chunks: Sequence[Mapping[str, Any]] = (),
    ) -> EnvironmentModel:
        tickets = domain.get("tickets") or []
        if not isinstance(tickets, list):
            raise DomainAdapterError("domain.tickets must be an array")
        for index, ticket in enumerate(tickets):
            if not isinstance(ticket, Mapping):
                raise DomainAdapterError(f"domain.tickets[{index}] must be an object")
        grouped: dict[str, list[Transition]] = {}
        for row in TICKET_TRANSITIONS:
            grouped.setdefault(str(row["action"]), []).append(
                Transition(
                    from_state=str(row["from"]),
                    to_state=str(row["to"]),
                    role=str(row["by_role"]),
                )
            )
        actions = tuple(
            ActionDefinition(
                name=name,
                description=f"Execute ticket action {name}",
                parameters={
                    "type": "object",
                    "properties": {
                        "ticket_id": {"type": "string", "description": "Ticket ID"}
                    },
                    "required": ["ticket_id"],
                },
                effect={
                    "type": "state_transition",
                    "collection": "tickets",
                    "key_from": "ticket_id",
                },
                transitions=tuple(transitions),
            )
            for name, transitions in grouped.items()
        )
        state = StateCollection(
            name="tickets",
            key="ticket_id",
            fields=dict(TICKET_STATE_SCHEMA["ticket"]),
            rows=state_rows(tickets),
            required_fields=("ticket_id", "status"),
            audit_name="ticket",
            label="ticket",
            state_field="status",
        )
        return EnvironmentModel(
            family=str(domain.get("family") or self.family),
            state_collections=(state,),
            actions=actions,
            rules=_rules_from_draft(draft),
            events=_events_from_draft(draft),
            actors=_actors_from_draft(draft),
            tasks=_tasks_from_draft(draft),
            workspace=_workspace_files(domain.get("workspace", draft.get("workspace"))),
            replay=_replay_entries(domain.get("replay", draft.get("replay"))),
            grounding=mapping_tuple(grounding_chunks),
            simulator=_simulator_from_draft(draft),
            metadata={
                METADATA_DOMAIN_SCHEMA_VERSION: domain.get(
                    "schemaVersion", DOMAIN_SCHEMA_VERSION
                ),
                METADATA_GLOSSARY: list(domain.get("glossary") or []),
                METADATA_DANGEROUS_ACTIONS: list(DANGEROUS_ACTIONS),
            },
            profile_hints={"office": {"command_max_length": 4000}},
        )

    def render_seed_text(self, ticket: Mapping[str, Any]) -> str:
        return ticket_to_text(dict(ticket))


class TerminalDomainAdapter:
    """Terminal workspace/replay recipe for pre-expressed terminal tasks."""

    family = "terminal"

    def supports(self, domain: Mapping[str, Any], draft: Mapping[str, Any]) -> bool:
        del draft
        return (
            domain.get("family") == self.family
            or domain.get("schemaVersion") == "domain-terminal-v1"
        )

    def extract(
        self,
        domain: Mapping[str, Any],
        draft: Mapping[str, Any],
        *,
        grounding_chunks: Sequence[Mapping[str, Any]] = (),
    ) -> EnvironmentModel:
        terminal = _mapping(domain.get("terminal"), "domain.terminal")
        workspace = terminal.get("workspace", domain.get("workspace", draft.get("workspace")))
        replay = terminal.get("replay", domain.get("replay", draft.get("replay")))
        danger_patterns = terminal.get("danger_patterns") or []
        if not isinstance(danger_patterns, list) or any(
            not isinstance(item, str) or not item for item in danger_patterns
        ):
            raise DomainAdapterError(
                "terminal.danger_patterns must be an array of non-empty strings"
            )
        return EnvironmentModel(
            family=str(domain.get("family") or self.family),
            tasks=_tasks_from_draft(draft),
            workspace=_workspace_files(workspace),
            replay=_replay_entries(replay),
            grounding=mapping_tuple(grounding_chunks),
            rules=RuleSet(),
            metadata={
                METADATA_DOMAIN_SCHEMA_VERSION: domain.get(
                    "schemaVersion", "domain-terminal-v1"
                ),
                METADATA_GLOSSARY: list(domain.get("glossary") or []),
                METADATA_DANGEROUS_ACTIONS: list(
                    terminal.get("dangerous_actions") or []
                ),
            },
            profile_hints={
                "terminal": {
                    "command_max_length": terminal.get("command_max_length", 2000),
                    "danger_patterns": list(danger_patterns),
                    "dynamics": terminal.get("dynamics", "replay"),
                    "real_exec_allowlist": list(
                        terminal.get("real_exec_allowlist") or []
                    ),
                    "exec_timeout_s": terminal.get("exec_timeout_s", 5.0),
                    "exec_max_output_chars": terminal.get(
                        "exec_max_output_chars", 65536
                    ),
                    "hybrid_miss_policy": terminal.get(
                        "hybrid_miss_policy", "replay"
                    ),
                }
            },
        )


register_domain_adapter(PowerTicketDomainAdapter(), replace=True)
register_domain_adapter(TerminalDomainAdapter(), replace=True)


__all__ = ["PowerTicketDomainAdapter", "TerminalDomainAdapter"]
