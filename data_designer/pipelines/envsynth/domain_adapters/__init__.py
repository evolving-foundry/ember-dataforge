"""Public domain-adapter API and built-in registrations."""

from data_designer.pipelines.envsynth.domain_adapters.base import (
    DomainAdapter,
    DomainAdapterError,
    adapt_environment,
    register_domain_adapter,
    resolve_domain_adapter,
    unregister_domain_adapter,
)
from data_designer.pipelines.envsynth.domain_adapters.builtin import (
    PowerTicketDomainAdapter,
    TerminalDomainAdapter,
)


__all__ = [
    "DomainAdapter",
    "DomainAdapterError",
    "PowerTicketDomainAdapter",
    "TerminalDomainAdapter",
    "adapt_environment",
    "register_domain_adapter",
    "resolve_domain_adapter",
    "unregister_domain_adapter",
]
