"""
DataForge: synthetic data generation package.
(Standalone generation, extensions, and optional benchmark export; the framework depends on curator.)
"""

__all__ = [
    "GenerationOperation",
    "SyntheticDataGenerationOperation",
]

__version__ = "0.2.0"


def __getattr__(name):
    """Load Curator operators only when explicitly requested."""
    if name in __all__:
        from . import operators
        return getattr(operators, name)
    raise AttributeError(name)
