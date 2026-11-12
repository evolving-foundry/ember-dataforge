"""Gym office profile compiler.

This module owns the Gym-specific vocabulary (state_assert, tickets, fsm, NPC
cards). The domain adapter supplies the profile-neutral EnvironmentModel.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence, Tuple

from data_designer.pipelines.envsynth.environment_model import (
    EnvironmentModel,
    StateCollection,
)
from data_designer.pipelines.envsynth.profile_compilers.base import (
    CompileContext,
    ProfileCompileError,
)
from data_designer.pipelines.envsynth.redaction import redact_text
from data_designer.pipelines.envsynth.serialization import (
    json_bytes as _json_bytes,
    jsonl_bytes as _jsonl_bytes,
    yaml_bytes as _yaml_bytes,
)


READ_ACTION = "read"
WRITE_ACTION = "write_file"


class GymProjectionError(ProfileCompileError):
    """The semantic model cannot be represented by Gym's office profile."""


def _field_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        base = spec.split("|", 1)[0]
        if base not in ("string", "number", "boolean"):
            raise GymProjectionError(f"unsupported state field type {spec!r}")
        return {"type": base}
    if isinstance(spec, dict) and spec.get("enum"):
        return {"type": "enum", "enum": list(spec["enum"])}
    raise GymProjectionError(f"unsupported state field spec {spec!r}")


def _primary_collection(model: EnvironmentModel) -> StateCollection:
    try:
        return model.collection()
    except Exception as error:
        raise GymProjectionError(str(error)) from error


def _label(collection: StateCollection) -> str:
    return collection.label or collection.audit_name or collection.name


def _office_hint(model: EnvironmentModel, key: str, default: Any) -> Any:
    hints = model.profile_hints.get("office") or {}
    if not isinstance(hints, Mapping):
        raise GymProjectionError("office profile_hints must be an object")
    return hints.get(key, default)


def _default_parameters(collection: StateCollection) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            collection.key: {
                "type": "string",
                "description": f"{_label(collection)} ID",
            }
        },
        "required": [collection.key],
    }


def _office_effect(action: Any, collection: StateCollection) -> dict[str, Any]:
    effect = action.effect if action is not None else {}
    if not effect:
        return {
            "type": "fsm",
            "collection": collection.name,
            "key_from": collection.key,
        }
    if not isinstance(effect, Mapping):
        raise GymProjectionError("action effect must be an object")
    kind = effect.get("type")
    if kind != "state_transition":
        raise GymProjectionError(f"unsupported action effect type: {kind!r}")
    projected_collection = effect.get("collection") or collection.name
    key_from = effect.get("key_from") or collection.key
    if not isinstance(projected_collection, str) or not projected_collection:
        raise GymProjectionError("action effect collection must be non-empty")
    if not isinstance(key_from, str) or not key_from:
        raise GymProjectionError("action effect key_from must be non-empty")
    return {
        "type": "fsm",
        "collection": projected_collection,
        "key_from": key_from,
    }


def _state_field(collection: StateCollection) -> str:
    if collection.state_field:
        return collection.state_field
    for name, spec in collection.fields.items():
        if isinstance(spec, dict) and spec.get("enum"):
            return name
    raise GymProjectionError(
        f"state collection {collection.name!r} has no enum transition field"
    )


def _transition_rows(model: EnvironmentModel) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for action in model.actions:
        for transition in action.transitions:
            rows.append(
                {
                    "from": transition.from_state,
                    "action": action.name,
                    "to": transition.to_state,
                    "by_role": transition.role,
                }
            )
    return rows


def project_state_schema(collection: StateCollection) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, spec in collection.fields.items():
        field = _field_spec(spec)
        if name in collection.required_fields:
            field["required"] = True
        fields[name] = field
    return {collection.name: {"key": collection.key, "fields": fields}}


def project_state_seed(collection: StateCollection) -> list[dict[str, Any]]:
    schema = project_state_schema(collection)[collection.name]["fields"]
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(collection.rows):
        key = source.get(collection.key)
        if not isinstance(key, str) or not key.strip():
            raise GymProjectionError(
                f"{collection.name}[{index}] missing {collection.key}; cannot project state seed"
            )
        obj: dict[str, Any] = {}
        for name in schema:
            value = source.get(name)
            if name in collection.required_fields and (value is None or value == ""):
                raise GymProjectionError(
                    f"{collection.name}[{index}] missing required field {name!r}; "
                    "cannot project state seed"
                )
            if value is None or value == "":
                continue
            obj[name] = value
        rows.append({"kind": collection.name, "obj": obj})
    return rows


def _action_name(
    row: Mapping[str, Any], groups: Mapping[str, Sequence[Mapping[str, Any]]]
) -> str:
    action = str(row["action"])
    peers = groups[action]
    roles = {str(item.get("by_role")) for item in peers}
    if len(peers) > 1 and len(roles) > 1:
        return f"{action}_{row['by_role']}"
    return action


def project_actions(
    model: EnvironmentModel,
) -> Tuple[list[dict[str, Any]], dict[Tuple[str, str], str], dict[str, set]]:
    rows = _transition_rows(model)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["action"]), []).append(row)
    projected_by_transition: dict[Tuple[str, str], str] = {}
    action_names_by_role: dict[str, set] = {}
    definitions: dict[str, dict[str, Any]] = {}
    actions_by_name = {action.name: action for action in model.actions}
    collection = _primary_collection(model)
    label = _label(collection)
    for row in rows:
        name = _action_name(row, groups)
        key = (str(row["from"]), str(row["action"]))
        existing = projected_by_transition.get(key)
        if existing is not None and existing != name:
            raise GymProjectionError(
                f"one state/action maps to multiple Gym action names: {key} -> {existing}/{name}"
            )
        projected_by_transition[key] = name
        action_names_by_role.setdefault(str(row["by_role"]), set()).add(name)
        if name not in definitions:
            role_hint = (
                f" ({row['by_role']})" if name != row["action"] else ""
            )
            action = actions_by_name.get(str(row["action"]))
            parameters = (
                copy.deepcopy(dict(action.parameters))
                if action is not None and action.parameters
                else _default_parameters(collection)
            )
            definitions[name] = {
                "name": name,
                "description": f"Execute {label} action {row['action']}{role_hint}",
                "parameters": parameters,
                "effect": _office_effect(action, collection),
            }
    definitions[READ_ACTION] = {
        "name": READ_ACTION,
        "description": f"Read current {label} status",
        "parameters": {
            "type": "object",
            "properties": {
                collection.key: {"type": "string", "description": f"{label} ID"}
            },
            "required": [collection.key],
        },
        "effect": {"type": "none"},
    }
    definitions[WRITE_ACTION] = {
        "name": WRITE_ACTION,
        "description": "Write a workspace file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute workspace path"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
        "effect": {"type": "fs_write", "path_from": "path", "content_from": "content"},
    }
    return list(definitions.values()), projected_by_transition, action_names_by_role


def project_rules(
    model: EnvironmentModel,
    projected_by_transition: Mapping[Tuple[str, str], str],
) -> dict[str, Any]:
    collection = _primary_collection(model)
    transitions = []
    for row in _transition_rows(model):
        transitions.append(
            {
                "from": row["from"],
                "to": row["to"],
                "action": projected_by_transition[(str(row["from"]), str(row["action"]))],
                "roles": sorted({"agent", str(row["by_role"])}),
            }
        )
    fsm_name = f"{collection.name}_lifecycle"
    return {
        "fsms": {
            fsm_name: {
                "collection": collection.name,
                "field": _state_field(collection),
                "transitions": transitions,
            }
        },
        "assertions": [],
    }


def project_events(model: EnvironmentModel) -> dict[str, Any]:
    return {
        "step_minutes": model.events.step_minutes,
        "events": copy.deepcopy(list(model.events.events)),
    }


def _npc_action_name(
    name: str, role: str, action_names_by_role: Mapping[str, set]
) -> str:
    available = action_names_by_role.get(role, set())
    if name in available:
        return name
    matches = sorted(item for item in available if item.startswith(name + "_"))
    if len(matches) == 1:
        return matches[0]
    return name


def _normalize_npc_auto_actions(
    actions: Any,
    role: str,
    action_names_by_role: Mapping[str, set],
    collection: StateCollection,
) -> list[dict[str, Any]]:
    if actions is None:
        return []
    if not isinstance(actions, list):
        raise GymProjectionError(f"npc card auto_actions must be an array: {role}")
    normalized: list[dict[str, Any]] = []
    aliases = {"ticket": collection.name}
    if collection.audit_name:
        aliases[collection.audit_name] = collection.name
    for index, rule in enumerate(actions):
        if not isinstance(rule, dict):
            raise GymProjectionError(
                f"npc card auto_actions[{index}] must be an object: {role}"
            )
        item = copy.deepcopy(rule)
        when = item.get("when")
        if when is not None:
            if not isinstance(when, dict):
                raise GymProjectionError(
                    f"npc card auto_actions[{index}].when must be an object: {role}"
                )
            if when.get("collection") in aliases:
                when["collection"] = aliases[when["collection"]]
            if "id" in when and collection.key not in when:
                when[collection.key] = when.pop("id")
        action = item.get("action")
        if action is not None:
            if not isinstance(action, dict):
                raise GymProjectionError(
                    f"npc card auto_actions[{index}].action must be an object: {role}"
                )
            name = action.get("name")
            if isinstance(name, str) and name:
                action["name"] = _npc_action_name(name, str(role), action_names_by_role)
            args = action.get("args")
            if isinstance(args, dict):
                for key, value in list(args.items()):
                    if value == "{obj.id}":
                        args[key] = f"{{obj.{collection.key}}}"
        normalized.append(item)
    return normalized


def project_npcs(
    model: EnvironmentModel,
    action_names_by_role: Mapping[str, set],
) -> dict[str, Any]:
    collection = _primary_collection(model)
    npcs = []
    for actor in model.actors:
        card = actor.card
        allowed = set(card.get("allowed_actions") or [])
        projected = set(action_names_by_role.get(str(actor.role), set()))
        if READ_ACTION in allowed:
            projected.add(READ_ACTION)
        npcs.append(
            {
                "id": str(actor.role),
                "role": str(actor.role),
                "mode": "template",
                "allowed_actions": sorted(projected),
                "template": redact_text(
                    str(card.get("template") or "{outcome}"),
                    allowed_paths=tuple(
                        sorted(item.path for item in model.workspace)
                    ),
                ),
                "auto_actions": _normalize_npc_auto_actions(
                    card.get("auto_actions"),
                    str(actor.role),
                    action_names_by_role,
                    collection,
                ),
            }
        )
    return {"npcs": npcs}


def project_grounding(
    chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set = set()
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.get("text")
        if not isinstance(text, str) or not text:
            continue
        raw_id = chunk.get("chunk_id") or chunk.get("id") or f"chunk-{index:06d}"
        identifier = str(raw_id)
        while identifier in seen:
            identifier = f"{identifier}-{index}"
        seen.add(identifier)
        role = chunk.get("role")
        rows.append(
            {
                "id": identifier,
                "text": text,
                "tags": [str(role)] if role else [],
            }
        )
    return rows


def project_llm_sim(model: EnvironmentModel) -> dict[str, Any]:
    system = model.simulator.prompt
    if not system.strip():
        system = "You are a constrained business-environment simulator. Output only content that conforms to the schema."
    system = redact_text(
        system, allowed_paths=tuple(sorted(item.path for item in model.workspace))
    )
    return {
        "system": system,
        "prompt_template": (
            "Current state:\n{state}\n\nAction:\n{action}\n\nReference material:\n{context}\n\n"
            "Generate the scenario text for this action (fill only the content field)."
        ),
        "output_schema": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
        "max_output_tokens": 400,
    }


def projection_report(model: EnvironmentModel) -> dict[str, Any]:
    collection = _primary_collection(model)
    timeouts = []
    assertions = []
    for rule in model.rules.raw:
        if rule.get("timeouts"):
            timeouts.append(str(rule.get("id")))
        if rule.get("kind") == "assertion":
            assertions.append(str(rule.get("id")))
    explicit_events = model.events.events
    return {
        "schemaVersion": "gym-projection-report-v1",
        "gymSchema": "Gym package_loader/task_data (PR #7)",
        "projected": {
            "state": "{} -> {} (key={})".format(
                collection.audit_name or collection.name,
                collection.name,
                collection.key,
            ),
            "actions": "fsm transitions + read + write_file",
            "verifier": "gold action -> state_assert",
            "grounding": "chunk id/text/tags",
            "npc": "template mode + normalized auto_actions",
        },
        "auditOnly": [
            "fsm_timeouts",
            "natural_language_assertions",
            "dd_event_types_and_arrival_rate",
            "rich_task_instruction_and_gold_output_format",
            "original_role_permission_matrix",
        ],
        "details": {
            "fsmTimeoutRuleIds": timeouts,
            "assertionRuleIds": assertions,
            "explicitGymEvents": len(explicit_events)
            if isinstance(explicit_events, list)
            else 0,
            "rolePolicy": "Gym agent role is fixed to 'agent'; original by_role is kept for NPC turns",
        },
    }


def project_env_spec(
    actions: Sequence[Mapping[str, Any]], model: EnvironmentModel
) -> dict[str, Any]:
    collection = _primary_collection(model)
    command_max_length = _office_hint(model, "command_max_length", 4000)
    if (
        isinstance(command_max_length, bool)
        or not isinstance(command_max_length, int)
        or command_max_length < 1
    ):
        raise GymProjectionError("office command_max_length must be a positive integer")
    return {
        "action_schema": {
            "name": "office",
            "parameters": {"type": "object", "properties": {}},
        },
        "command_max_length": command_max_length,
        "danger_patterns": [],
        "state_schema": project_state_schema(collection),
        "actions": list(actions),
        "dynamics": "replay",
    }


def project_task_goal(
    task: Any,
    projected_by_transition: Mapping[Tuple[str, str], str],
    collection: StateCollection,
) -> Tuple[str, str, dict[str, Any]]:
    initial = task.initial_state if isinstance(task.initial_state, Mapping) else {}
    candidates = [collection.audit_name, collection.name]
    obj = next(
        (
            initial.get(name)
            for name in candidates
            if isinstance(initial.get(name), Mapping)
        ),
        None,
    )
    if obj is None:
        obj = next(
            (
                value
                for value in initial.values()
                if isinstance(value, Mapping) and collection.key in value
            ),
            None,
        )
    obj = obj or {}
    object_id = obj.get(collection.key)
    if not isinstance(object_id, str) or not object_id:
        raise GymProjectionError(f"{task.task_id}: missing {collection.key}")
    state_field = _state_field(collection)
    status = obj.get(state_field)
    gold = task.gold_solution
    if not isinstance(gold, Mapping):
        raise GymProjectionError(f"{task.task_id}: gold_solution must be an object")
    action = gold.get("action")
    label = _label(collection)
    if isinstance(action, str) and action:
        projected = projected_by_transition.get((str(status), action))
        if projected is None:
            raise GymProjectionError(
                f"{task.task_id}: no Gym action for ({status}, {action})"
            )
        next_status = gold.get("next_status")
        if not isinstance(next_status, str) or not next_status:
            raise GymProjectionError(f"{task.task_id}: gold is missing next_status")
        goal = f"Process {label} {object_id} (current status {status}): call `{projected}` action."
        return goal, projected, {
            "kind": "state_assert",
            "payload": {
                "checks": [
                    {
                        "collection": collection.name,
                        "match": {collection.key: object_id},
                        "expect": {state_field: next_status},
                    }
                ]
            },
        }
    expected_status = gold.get(state_field)
    if not isinstance(expected_status, str) or not expected_status:
        raise GymProjectionError(
            f"{task.task_id}: gold has neither action nor {state_field}"
        )
    goal = f"Check {label} {object_id} current status."
    return goal, READ_ACTION, {
        "kind": "state_assert",
        "payload": {
            "checks": [
                {
                    "collection": collection.name,
                    "match": {collection.key: object_id},
                    "expect": {state_field: expected_status},
                }
            ]
        },
    }


def project_tasks(
    model: EnvironmentModel,
    context: CompileContext,
    *,
    fingerprint: str,
) -> list[dict[str, Any]]:
    collection = _primary_collection(model)
    _, projected_by_transition, _ = project_actions(model)
    rows: list[dict[str, Any]] = []
    for task in model.tasks:
        goal, _, verifier = project_task_goal(task, projected_by_transition, collection)
        row_max_steps = context.max_steps if task.max_steps is None else task.max_steps
        row_seed = context.seed if task.seed is None else task.seed
        rows.append(
            {
                "task_id": str(task.task_id),
                "env_release": {
                    "ref_id": context.ref_id,
                    "version": int(context.version),
                    "fingerprint": fingerprint,
                },
                "goal": goal,
                "success_check": {"verifier_spec": verifier},
                "max_steps": row_max_steps,
                "seed": row_seed,
            }
        )
    return rows


class OfficeProfileCompiler:
    profile = "office"

    def agent_prompt(self, name: str) -> dict[str, str]:
        return {
            "system": (
                "You are an office operator inside the synthetic environment "
                "'{}'. Work only inside the declared action set and visible state. "
                "Treat the ticket lifecycle and approval rules as hard constraints."
            ).format(name),
            "user": (
                "Each turn issue exactly one action from the available action list as "
                "a tool call. Observations contain action results, event notices and "
                "NPC replies; hidden state is never shown. Never assume an action "
                "succeeded unless the observation says so."
            ),
        }

    def build_files(
        self, model: EnvironmentModel, context: CompileContext
    ) -> dict[str, bytes]:
        del context
        collection = _primary_collection(model)
        actions, projected_by_transition, action_names_by_role = project_actions(model)
        rules = project_rules(model, projected_by_transition)
        env_spec = project_env_spec(actions, model)
        snapshot = [
            {"path": item.path, "content": item.content} for item in model.workspace
        ]
        return {
            "env_spec.json": _json_bytes(env_spec),
            "fs/snapshot.jsonl": _jsonl_bytes(snapshot),
            "state/seed.jsonl": _jsonl_bytes(project_state_seed(collection)),
            "rules.yaml": _yaml_bytes(rules),
            "events.yaml": _yaml_bytes(project_events(model)),
            "npcs.yaml": _yaml_bytes(project_npcs(model, action_names_by_role)),
            "grounding/index.jsonl": _jsonl_bytes(project_grounding(model.grounding)),
            "llm_sim.yaml": _yaml_bytes(project_llm_sim(model)),
        }

    def build_tasks(
        self,
        model: EnvironmentModel,
        context: CompileContext,
        *,
        fingerprint: str,
    ) -> list[dict[str, Any]]:
        return project_tasks(model, context, fingerprint=fingerprint)

    def projection_report(self, model: EnvironmentModel) -> dict[str, Any]:
        return projection_report(model)


__all__ = [
    "GymProjectionError",
    "OfficeProfileCompiler",
    "READ_ACTION",
    "WRITE_ACTION",
    "project_actions",
    "project_env_spec",
    "project_events",
    "project_grounding",
    "project_llm_sim",
    "project_npcs",
    "project_rules",
    "project_state_schema",
    "project_state_seed",
    "project_task_goal",
    "project_tasks",
    "projection_report",
]
