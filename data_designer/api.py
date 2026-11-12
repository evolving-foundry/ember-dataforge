"""Independent SDK. Importing this module does not load Curator or models."""
from data_designer.generation.service import generate as _generate
from data_designer.generation.result import GenerationResult

__all__ = ["benchmark", "generate", "GenerationResult"]


def __getattr__(name):
    if name == "benchmark":
        from data_designer.pipelines.benchmarks import api as benchmark
        return benchmark
    raise AttributeError(name)


def generate(input=None, config=None, context=None, *, base_directory=None):
    from data_designer.config import load_config
    from data_designer.generation.context import AlgorithmContext
    from data_designer.generation.registry import RecipeRegistry
    from data_designer.generation.algorithms.registry import AlgorithmRegistry
    job = load_config(config, base_directory=base_directory)
    context = context or AlgorithmContext(seed=job["execution"]["seed"])
    registry = context.registry or AlgorithmRegistry()
    context.registry = registry
    selection = RecipeRegistry().resolve(job["recipe"], job["recipe_config"])
    spec = registry.describe(selection["id"]).spec
    if "curator.recipe" in spec.resources.get("providers", ()) and "curator.recipe" not in context.providers:
        from data_designer.integrations.curator.recipes import CuratorRecipeProvider
        context.providers["curator.recipe"] = CuratorRecipeProvider(context.providers)
    return _generate(input, job, context)
