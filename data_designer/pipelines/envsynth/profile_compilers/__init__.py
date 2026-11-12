"""Built-in Gym profile compilers.

Importing this package registers the compilers used by the package builder.
External profiles can call register_profile_compiler() before building.
"""

from data_designer.pipelines.envsynth.profile_compilers.base import (
    CompileContext,
    ProfileCompileError,
    ProfileCompiler,
    UnsupportedProfileError,
    get_profile_compiler,
    register_profile_compiler,
    supported_profiles,
    unregister_profile_compiler,
)
from data_designer.pipelines.envsynth.profile_compilers.office import (
    OfficeProfileCompiler,
)
from data_designer.pipelines.envsynth.profile_compilers.terminal import (
    TerminalProfileCompiler,
)


register_profile_compiler(OfficeProfileCompiler(), replace=True)
register_profile_compiler(TerminalProfileCompiler(), replace=True)


__all__ = [
    "CompileContext",
    "OfficeProfileCompiler",
    "ProfileCompileError",
    "ProfileCompiler",
    "TerminalProfileCompiler",
    "UnsupportedProfileError",
    "get_profile_compiler",
    "register_profile_compiler",
    "supported_profiles",
    "unregister_profile_compiler",
]
