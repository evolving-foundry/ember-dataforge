"""Explicit migration of test scenarios to the current job schema."""
def job_for_recipe(parameters):
    params = dict(parameters)
    recipe = params.pop("recipe", "dense_caption")
    roles = {"dense_caption": "caption", "video_temporal_qa": "video", "knowledge_reasoning_vqa": "qa"}
    settings = params.pop("recipe_config", {})
    annotation_schema = params.pop("schema_version", None)
    if annotation_schema:
        params["annotation_schema_version"] = annotation_schema
    if recipe == "knowledge_reasoning_vqa":
        params = settings
    elif settings:
        params["steps"] = settings
    return {"recipe": recipe, "recipe_config": {"algorithms": {
        roles.get(recipe, "main"): {"params": params}}}}


def executor_context(op):
    """Build the RecipeContext a CuratorRecipeProvider would hand the executor."""
    from data_designer.integrations.curator.recipes import RecipeContext

    selection = next(iter(op.config["recipe_config"]["algorithms"].values()))
    config = dict(selection["params"])
    steps = config.pop("steps", {})
    return RecipeContext(
        op.config["recipe"], config, steps,
        schema_version=config.get("annotation_schema_version", "synthetic-data-v1"),
        generator_prompt_version=config.get("generator_prompt_version", "v1"),
        on_error=config.get("on_error", "raise"),
    )
