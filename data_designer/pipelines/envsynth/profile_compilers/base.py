"""Profile compiler registry and shared compile context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from data_designer.pipelines.envsynth.environment_model import EnvironmentModel
from data_designer.pipelines.envsynth.package_identity import DEFAULT_MAX_STEPS


class ProfileCompileError(ValueError):
    """The EnvironmentModel cannot be represented by the selected Gym profile."""


class UnsupportedProfileError(ProfileCompileError):
    """No compiler is registered for the requested Gym profile."""


@dataclass(frozen=True)
class CompileContext:
    ref_id: str
    version: int
    seed: int
    max_steps: int = DEFAULT_MAX_STEPS


class ProfileCompiler(Protocol):
    profile: str

    def build_files(
        self, model: EnvironmentModel, context: CompileContext
    ) -> dict[str, bytes]:
        """Build Gym package files except manifest.json and tasks.jsonl."""

    def build_tasks(
        self,
        model: EnvironmentModel,
        context: CompileContext,
        *,
        fingerprint: str,
    ) -> list[dict[str, Any]]:
        """Build the strict six-field Gym task rows."""

    def projection_report(self, model: EnvironmentModel) -> dict[str, Any]:
        """Describe projected versus audit-only semantics."""

    def agent_prompt(self, name: str) -> dict[str, str]:
        """Build the Gym agent prompt for this profile."""


_COMPILERS: dict[str, ProfileCompiler] = {}


def register_profile_compiler(
    compiler: ProfileCompiler, *, replace: bool = False
) -> None:
    name = getattr(compiler, "profile", None)
    if not isinstance(name, str) or not name:
        raise UnsupportedProfileError("profile compiler name must be non-empty")
    if name in _COMPILERS and not replace:
        raise UnsupportedProfileError(f"profile compiler already registered: {name}")
    if not callable(getattr(compiler, "agent_prompt", None)):
        raise UnsupportedProfileError(
            f"profile compiler {name!r} must implement agent_prompt()"
        )
    _COMPILERS[name] = compiler


def unregister_profile_compiler(profile: str) -> bool:
    """Remove one compiler; useful for extension tests and controlled reloads."""
    return _COMPILERS.pop(profile, None) is not None


def get_profile_compiler(profile: str) -> ProfileCompiler:
    compiler = _COMPILERS.get(profile)
    if compiler is None:
        raise UnsupportedProfileError(
            f"unsupported Gym profile {profile!r}; registered: {supported_profiles()}"
        )
    return compiler


def supported_profiles() -> list[str]:
    return sorted(_COMPILERS)


__all__ = [
    "CompileContext",
    "ProfileCompileError",
    "ProfileCompiler",
    "UnsupportedProfileError",
    "get_profile_compiler",
    "register_profile_compiler",
    "supported_profiles",
    "unregister_profile_compiler",
]
