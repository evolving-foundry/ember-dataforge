"""Profile-neutral environment model shared by every Gym compiler.

The model is the boundary between domain knowledge and Gym package schemas:

* a DomainAdapter extracts semantic state, actions, tasks, workspace files and
  replay entries from a user-data domain;
* a ProfileCompiler maps that model to one concrete Gym profile (office,
  terminal, or a future profile registered by another repo).

Nothing in this module knows about tickets, state_assert or any other
Gym-specific vocabulary. Those belong to the profile compilers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


class EnvironmentModelError(ValueError):
    """The semantic model is malformed or cannot be compiled safely."""


METADATA_DOMAIN_SCHEMA_VERSION = "domainSchemaVersion"
METADATA_GLOSSARY = "glossary"
METADATA_DANGEROUS_ACTIONS = "dangerousActions"


def _valid_virtual_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("/"):
        return False
    if value == "/":
        return True
    return all(part not in ("", ".", "..") for part in value.split("/")[1:])


@dataclass(frozen=True)
class StateCollection:
    """One semantic object collection (for example tickets, orders, files)."""

    name: str
    key: str
    fields: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...] = ()
    required_fields: tuple[str, ...] = ()
    audit_name: str = ""
    label: str = ""
    state_field: str = ""

    def validate(self) -> None:
        if not self.name or not self.key:
            raise EnvironmentModelError("state collection name/key must be non-empty")
        if not isinstance(self.fields, Mapping):
            raise EnvironmentModelError(
                f"state collection {self.name!r} fields must be an object"
            )
        if self.key not in self.fields:
            raise EnvironmentModelError(
                f"state collection {self.name!r} key {self.key!r} is absent from fields"
            )
        if self.state_field and self.state_field not in self.fields:
            raise EnvironmentModelError(
                f"state collection {self.name!r} state_field "
                f"{self.state_field!r} is absent from fields"
            )
        for field_name in self.required_fields:
            if field_name not in self.fields:
                raise EnvironmentModelError(
                    f"state collection {self.name!r} required field "
                    f"{field_name!r} is absent from fields"
                )
        for index, row in enumerate(self.rows):
            if not isinstance(row, Mapping):
                raise EnvironmentModelError(
                    f"state collection {self.name!r} row {index} must be an object"
                )


@dataclass(frozen=True)
class Transition:
    from_state: str
    to_state: str
    role: str = "agent"

    def validate(self, action_name: str) -> None:
        if not self.from_state or not self.to_state:
            raise EnvironmentModelError(
                f"action {action_name!r} transition states must be non-empty"
            )
        if not self.role:
            raise EnvironmentModelError(
                f"action {action_name!r} transition role must be non-empty"
            )


@dataclass(frozen=True)
class ActionDefinition:
    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    effect: Mapping[str, Any] = field(default_factory=dict)
    transitions: tuple[Transition, ...] = ()

    def validate(self) -> None:
        if not self.name:
            raise EnvironmentModelError("action name must be non-empty")
        if not isinstance(self.parameters, Mapping):
            raise EnvironmentModelError(
                f"action {self.name!r} parameters must be an object"
            )
        if not isinstance(self.effect, Mapping):
            raise EnvironmentModelError(
                f"action {self.name!r} effect must be an object"
            )
        if not isinstance(self.transitions, (tuple, list)):
            raise EnvironmentModelError(
                f"action {self.name!r} transitions must be an array"
            )
        for transition in self.transitions:
            if not isinstance(transition, Transition):
                raise EnvironmentModelError(
                    f"action {self.name!r} transitions must contain Transition values"
                )
            transition.validate(self.name)


@dataclass(frozen=True)
class RuleSet:
    raw: tuple[Mapping[str, Any], ...] = ()
    assertions: tuple[Mapping[str, Any], ...] = ()

    def validate(self) -> None:
        for label, rows in (("raw", self.raw), ("assertions", self.assertions)):
            if not isinstance(rows, (tuple, list)):
                raise EnvironmentModelError(f"rules.{label} must be an array")
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise EnvironmentModelError(
                        f"rules.{label}[{index}] must be an object"
                    )


@dataclass(frozen=True)
class EventModel:
    step_minutes: int = 60
    events: tuple[Mapping[str, Any], ...] = ()

    def validate(self) -> None:
        if (
            isinstance(self.step_minutes, bool)
            or not isinstance(self.step_minutes, int)
            or self.step_minutes < 1
        ):
            raise EnvironmentModelError("event step_minutes must be a positive integer")
        if not isinstance(self.events, (tuple, list)):
            raise EnvironmentModelError("event events must be an array")
        for index, event in enumerate(self.events):
            if not isinstance(event, Mapping):
                raise EnvironmentModelError(
                    f"event events[{index}] must be an object"
                )


@dataclass(frozen=True)
class ActorDefinition:
    role: str
    card: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.role:
            raise EnvironmentModelError("actor role must be non-empty")
        if not isinstance(self.card, Mapping):
            raise EnvironmentModelError(f"actor {self.role!r} card must be an object")


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    instruction: str = ""
    initial_state: Mapping[str, Any] = field(default_factory=dict)
    gold_solution: Mapping[str, Any] = field(default_factory=dict)
    negative_solution: Mapping[str, Any] = field(default_factory=dict)
    verifier_spec: Mapping[str, Any] = field(default_factory=dict)
    max_steps: Optional[int] = None
    seed: Optional[int] = None
    anchor_id: str = ""
    expected_files: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.task_id:
            raise EnvironmentModelError("task_id must be non-empty")
        for label, value in (
            ("initial_state", self.initial_state),
            ("gold_solution", self.gold_solution),
            ("negative_solution", self.negative_solution),
            ("verifier_spec", self.verifier_spec),
            ("metadata", self.metadata),
        ):
            if not isinstance(value, Mapping):
                raise EnvironmentModelError(f"task {self.task_id!r} {label} must be an object")
        if self.max_steps is not None:
            if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
                raise EnvironmentModelError(f"task {self.task_id!r} max_steps must be an integer")
            if self.max_steps < 1:
                raise EnvironmentModelError(f"task {self.task_id!r} max_steps must be >= 1")
        if not isinstance(self.instruction, str):
            raise EnvironmentModelError(f"task {self.task_id!r} instruction must be text")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise EnvironmentModelError(f"task {self.task_id!r} seed must be an integer")
        for path, content in self.expected_files.items():
            if not isinstance(path, str) or not path:
                raise EnvironmentModelError(
                    f"task {self.task_id!r} expected_files path must be non-empty"
                )
            if not isinstance(content, str):
                raise EnvironmentModelError(
                    f"task {self.task_id!r} expected_files[{path!r}] must be a string"
                )


@dataclass(frozen=True)
class WorkspaceFile:
    path: str
    content: str

    def validate(self) -> None:
        if not _valid_virtual_path(self.path):
            raise EnvironmentModelError(
                f"workspace path must be an absolute non-escaping path: {self.path!r}"
            )
        if not isinstance(self.content, str):
            raise EnvironmentModelError(f"workspace file {self.path!r} content must be text")


@dataclass(frozen=True)
class ReplayEntry:
    cwd: str
    command: str
    outcome: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not _valid_virtual_path(self.cwd):
            raise EnvironmentModelError(
                f"replay cwd must be an absolute non-escaping path: {self.cwd!r}"
            )
        if not isinstance(self.command, str) or not self.command.strip():
            raise EnvironmentModelError("replay command must be non-empty")
        if not isinstance(self.outcome, Mapping):
            raise EnvironmentModelError("replay outcome must be an object")


@dataclass(frozen=True)
class SimulatorDefinition:
    prompt: str = ""

    def validate(self) -> None:
        if not isinstance(self.prompt, str):
            raise EnvironmentModelError("simulator prompt must be text")


@dataclass(frozen=True)
class EnvironmentModel:
    """The profile-neutral semantic environment."""

    family: str
    state_collections: tuple[StateCollection, ...] = ()
    actions: tuple[ActionDefinition, ...] = ()
    rules: RuleSet = field(default_factory=RuleSet)
    events: EventModel = field(default_factory=EventModel)
    actors: tuple[ActorDefinition, ...] = ()
    tasks: tuple[TaskDefinition, ...] = ()
    workspace: tuple[WorkspaceFile, ...] = ()
    replay: tuple[ReplayEntry, ...] = ()
    grounding: tuple[Mapping[str, Any], ...] = ()
    simulator: SimulatorDefinition = field(default_factory=SimulatorDefinition)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    profile_hints: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def validate(self) -> "EnvironmentModel":
        if not isinstance(self.family, str) or not self.family.strip():
            raise EnvironmentModelError("family must be non-empty")
        collection_names: set[str] = set()
        for collection in self.state_collections:
            collection.validate()
            if collection.name in collection_names:
                raise EnvironmentModelError(
                    f"duplicate state collection {collection.name!r}"
                )
            collection_names.add(collection.name)
        action_names: set[str] = set()
        for action in self.actions:
            action.validate()
            if action.name in action_names:
                raise EnvironmentModelError(f"duplicate action {action.name!r}")
            action_names.add(action.name)
        self.rules.validate()
        self.events.validate()
        self.simulator.validate()
        task_ids: set[str] = set()
        for task in self.tasks:
            task.validate()
            if task.task_id in task_ids:
                raise EnvironmentModelError(f"duplicate task_id {task.task_id!r}")
            task_ids.add(task.task_id)
        workspace_paths: set[str] = set()
        for item in self.workspace:
            item.validate()
            if item.path in workspace_paths:
                raise EnvironmentModelError(f"duplicate workspace path {item.path!r}")
            workspace_paths.add(item.path)
        for actor in self.actors:
            actor.validate()
        replay_keys: set[tuple[str, str]] = set()
        for entry in self.replay:
            entry.validate()
            key = (entry.cwd, " ".join(entry.command.strip().split()))
            if key in replay_keys:
                raise EnvironmentModelError(
                    "duplicate replay entry for cwd={!r}, command={!r}".format(*key)
                )
            replay_keys.add(key)
        if not isinstance(self.grounding, (tuple, list)):
            raise EnvironmentModelError("grounding must be an array")
        for index, chunk in enumerate(self.grounding):
            if not isinstance(chunk, Mapping):
                raise EnvironmentModelError(
                    f"grounding[{index}] must be an object"
                )
        if not isinstance(self.metadata, Mapping):
            raise EnvironmentModelError("metadata must be an object")
        if not isinstance(self.profile_hints, Mapping):
            raise EnvironmentModelError("profile_hints must be an object")
        for profile, hints in self.profile_hints.items():
            if not isinstance(profile, str) or not profile:
                raise EnvironmentModelError("profile_hints keys must be non-empty strings")
            if not isinstance(hints, Mapping):
                raise EnvironmentModelError(
                    f"profile_hints[{profile!r}] must be an object"
                )
        return self

    def collection(self, name: Optional[str] = None) -> StateCollection:
        if name is not None:
            for collection in self.state_collections:
                if collection.name == name:
                    return collection
            raise EnvironmentModelError(f"unknown state collection {name!r}")
        if len(self.state_collections) != 1:
            raise EnvironmentModelError(
                "profile requires exactly one state collection; got {}".format(
                    len(self.state_collections)
                )
            )
        return self.state_collections[0]


def state_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(row) for row in rows)


def mapping_tuple(
    values: Optional[Sequence[Mapping[str, Any]]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(value) for value in (values or ()))


__all__ = [
    "ActionDefinition",
    "ActorDefinition",
    "EnvironmentModel",
    "EnvironmentModelError",
    "METADATA_DANGEROUS_ACTIONS",
    "METADATA_DOMAIN_SCHEMA_VERSION",
    "METADATA_GLOSSARY",
    "EventModel",
    "ReplayEntry",
    "RuleSet",
    "SimulatorDefinition",
    "StateCollection",
    "TaskDefinition",
    "Transition",
    "WorkspaceFile",
    "mapping_tuple",
    "state_rows",
]
