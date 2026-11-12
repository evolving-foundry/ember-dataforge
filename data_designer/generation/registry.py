"""Recipe composition descriptors, separate from algorithm registration."""
from dataclasses import dataclass

from data_designer.config import object_schema, validate_config


@dataclass(frozen=True)
class RecipeSpec:
    id: str
    algorithm: str
    role: str


class RecipeRegistry:
    def __init__(self):
        self._specs = {}
        for spec in (RecipeSpec("agent_task", "native.agent_task", "normalize"),
                     RecipeSpec("multihop_fusion", "deepfusion.dag_fusion", "fusion"),
                     RecipeSpec("dense_caption", "native.dense_caption", "caption"),
                     RecipeSpec("video_temporal_qa", "native.video_temporal_qa", "video"),
                     RecipeSpec("workbuddy_task", "native.workbuddy", "generation"),
                     RecipeSpec("knowledge_reasoning_vqa", "native.reasonvqa", "qa"),
                     RecipeSpec("algorithm", "", "main")):
            self.register(spec)

    def register(self, spec):
        if spec.id in self._specs:
            raise ValueError(f"duplicate recipe id: {spec.id}")
        self._specs[spec.id] = spec

    def list(self):
        return tuple(self._specs[key] for key in sorted(self._specs))

    def resolve(self, name, config):
        try:
            spec = self._specs[name]
        except KeyError as error:
            raise ValueError(f"unknown recipe: {name}") from error
        selection_schema = object_schema({
            "id": {"type": "string", "minLength": 1, **({"default": spec.algorithm} if spec.algorithm else {})},
            "implementation_version": {"type": "string", "minLength": 1},
            "adapter": {"enum": ["python", "subprocess", "service"], "default": "python"},
            "params": {"type": "object", "default": {}},
        }, required=("id",))
        schema = object_schema({"algorithms": {
            **object_schema({spec.role: {**selection_schema, "default": {}}}), "default": {}}})
        parsed = validate_config(config, schema)
        return parsed["algorithms"][spec.role]
