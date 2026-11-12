"""Generation operator exports loaded on demand to avoid recipe import cycles."""

from importlib import import_module

__all__ = [
    "GenerationOperation",
    "SyntheticDataGenerationOperation",
]

_MODULES = {
    "GenerationOperation": "generation_operator",
    "SyntheticDataGenerationOperation": "synthetic_data_generation_operator",
}


def __getattr__(name):
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{module}"), name)
