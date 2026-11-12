from __future__ import annotations

import copy


class _Processor:
    """Internal transformations, not an operator framework or registration API."""
    def __init__(self, params=None):
        from data_designer.config import object_schema, validate_config
        types = {"int": "integer", "str": "string", "bool": "boolean", "list": "array"}
        fields = {name: {"type": types[spec["type"]], "default": copy.deepcopy(spec["default"])}
                  for name, spec in self.PARAMS_SCHEMA.get("params", {}).items()}
        self.params = validate_config(params or {}, object_schema(fields))
