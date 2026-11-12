"""Domain adapter protocol and registry.

Adapters convert business-domain rows (tickets, terminal recordings, future
domains) into the profile-neutral EnvironmentModel. Business fields must not
emit Gym profile vocabulary; that belongs to ProfileCompiler. A profile may be
selected through an explicit ProfileCompiler, and adapters may pass narrowly
scoped compilation parameters through EnvironmentModel.profile_hints.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

from data_designer.pipelines.envsynth.environment_model import EnvironmentModel


class DomainAdapter(Protocol):
    family: str

    def supports(self, domain: Mapping[str, Any], draft: Mapping[str, Any]) -> bool:
        """Return whether this adapter owns the supplied domain."""

    def extract(
        self,
        domain: Mapping[str, Any],
        draft: Mapping[str, Any],
        *,
        grounding_chunks: Sequence[Mapping[str, Any]] = (),
    ) -> EnvironmentModel:
        """Extract semantic state, actions and tasks from the domain."""


_ADAPTERS: list[DomainAdapter] = []


class DomainAdapterError(ValueError):
    """No registered adapter can safely model the domain."""


def register_domain_adapter(adapter: DomainAdapter, *, replace: bool = False) -> None:
    family = getattr(adapter, "family", None)
    if not isinstance(family, str) or not family:
        raise DomainAdapterError("domain adapter family must be non-empty")
    existing = [item for item in _ADAPTERS if item.family == family]
    if existing and not replace:
        raise DomainAdapterError(f"domain adapter already registered: {family}")
    if existing:
        _ADAPTERS[:] = [item for item in _ADAPTERS if item.family != family]
    _ADAPTERS.append(adapter)


def unregister_domain_adapter(family: str) -> bool:
    """Remove one adapter; useful for extension tests and controlled reloads."""
    before = len(_ADAPTERS)
    _ADAPTERS[:] = [item for item in _ADAPTERS if item.family != family]
    return len(_ADAPTERS) != before


def resolve_domain_adapter(
    domain: Mapping[str, Any], draft: Mapping[str, Any]
) -> DomainAdapter:
    for adapter in reversed(_ADAPTERS):
        if adapter.supports(domain, draft):
            return adapter
    raise DomainAdapterError(
        "no domain adapter supports family {!r} (registered: {})".format(
            domain.get("family"), sorted(item.family for item in _ADAPTERS)
        )
    )


def adapt_environment(
    domain: Mapping[str, Any],
    draft: Mapping[str, Any],
    *,
    grounding_chunks: Sequence[Mapping[str, Any]] = (),
    adapter: Optional[DomainAdapter] = None,
) -> EnvironmentModel:
    selected = adapter or resolve_domain_adapter(domain, draft)
    try:
        model = selected.extract(domain, draft, grounding_chunks=grounding_chunks)
    except Exception as error:
        if isinstance(error, DomainAdapterError):
            raise
        raise DomainAdapterError(f"{selected.family}: {error}") from error
    if not isinstance(model, EnvironmentModel):
        raise DomainAdapterError(
            f"{selected.family}: extract() must return EnvironmentModel"
        )
    try:
        return model.validate()
    except ValueError as error:
        raise DomainAdapterError(f"{selected.family}: {error}") from error


__all__ = [
    "DomainAdapter",
    "DomainAdapterError",
    "adapt_environment",
    "register_domain_adapter",
    "resolve_domain_adapter",
    "unregister_domain_adapter",
]
