"""Compile-time environment synthesis (user examples → frozen env package)."""

from data_designer.pipelines.envsynth.domain_adapters import (
    DomainAdapter,
    DomainAdapterError,
    adapt_environment,
    register_domain_adapter,
    resolve_domain_adapter,
    unregister_domain_adapter,
)
from data_designer.pipelines.envsynth.environment_model import (
    EnvironmentModel,
    EnvironmentModelError,
)
from data_designer.pipelines.envsynth.gates import GateError, run_gates
from data_designer.pipelines.envsynth.grounding import (
    CharNgramEmbeddingBackend,
    EmbeddingBackend,
    build_index,
    retrieve,
)
from data_designer.pipelines.envsynth.interview import (
    CurationError,
    apply_answers,
    blocking_questions,
    write_curation_questions,
)
from data_designer.pipelines.envsynth.package import (
    EnvPackageBundle,
    EnvPackageError,
    build_env_package,
    finalize_bundle,
    load_env_package,
    write_release,
)
from data_designer.pipelines.envsynth.profile_compilers import (
    CompileContext,
    ProfileCompileError,
    ProfileCompiler,
    UnsupportedProfileError,
    get_profile_compiler,
    register_profile_compiler,
    supported_profiles,
    unregister_profile_compiler,
)
from data_designer.pipelines.envsynth.seed_import import (
    SeedBundle,
    SeedImportError,
    build_seed_import_from_source,
    load_seed_bundle,
    write_seed_import_from_rows,
)
from data_designer.pipelines.envsynth.synthesizer import (
    AgentRunner,
    ChatClient,
    OpenAICompatibleChatClient,
    SynthesisError,
    SynthesisResult,
    synthesize,
    template_draft,
)


__all__ = [
    "AgentRunner",
    "CharNgramEmbeddingBackend",
    "ChatClient",
    "CompileContext",
    "CurationError",
    "DomainAdapter",
    "DomainAdapterError",
    "EmbeddingBackend",
    "EnvPackageBundle",
    "EnvPackageError",
    "EnvironmentModel",
    "EnvironmentModelError",
    "GateError",
    "OpenAICompatibleChatClient",
    "ProfileCompileError",
    "ProfileCompiler",
    "SeedBundle",
    "SeedImportError",
    "SynthesisError",
    "SynthesisResult",
    "UnsupportedProfileError",
    "adapt_environment",
    "apply_answers",
    "blocking_questions",
    "build_env_package",
    "build_index",
    "build_seed_import_from_source",
    "finalize_bundle",
    "get_profile_compiler",
    "load_env_package",
    "load_seed_bundle",
    "register_domain_adapter",
    "register_profile_compiler",
    "resolve_domain_adapter",
    "retrieve",
    "run_gates",
    "supported_profiles",
    "synthesize",
    "template_draft",
    "unregister_domain_adapter",
    "unregister_profile_compiler",
    "write_curation_questions",
    "write_release",
    "write_seed_import_from_rows",
]
