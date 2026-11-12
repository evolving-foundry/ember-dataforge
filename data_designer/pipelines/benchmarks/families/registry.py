"""Family-owned hooks. Adding a family does not change the build pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class FamilyRules:
    id: str
    module: str
    source_verifier: str = ""
    derivation_key: str = ""
    derive: str = ""
    runtime_assets: str = ""
    assets_with_case: bool = False
    build_assertion: str = ""
    state_patch_allowed: bool = True
    native_pins: str = ""

    def _call(self, name, *args):
        if name:
            return getattr(import_module(self.module), name)(*args)

    def verify_source(self, source):
        if source.get("source_snapshot") is not None:
            self._call(self.source_verifier, source)

    def materialize(self, source, source_case, variant):
        if self.derivation_key and self.derivation_key in variant:
            return {**variant, "private": self._call(self.derive, source, source_case, variant[self.derivation_key])}
        return variant

    def check_build(self, source, source_case, runtime, private):
        snapshot = source.get("source_snapshot")
        if snapshot is not None and self.runtime_assets:
            args = (runtime, snapshot, source_case) if self.assets_with_case else (runtime, snapshot)
            self._call(self.runtime_assets, *args)
        self._call(self.build_assertion, runtime, private, source_case)


_RULES = {}


def register_family(rules):
    if rules.id in _RULES:
        raise ValueError(f"duplicate benchmark family: {rules.id}")
    _RULES[rules.id] = rules


def get_family(name):
    try:
        return _RULES[name]
    except KeyError as error:
        raise ValueError(f"unknown benchmark family: {name}") from error


for family in ("swe", "osworld", "tau", "deepswe"):
    register_family(FamilyRules(
        family, f"data_designer.pipelines.benchmarks.families.{family}",
        f"verify_{family}_source_manifest", f"{family}_offline", f"derive_{family}_offline_spec",
        f"assert_{family}_runtime_assets" if family in ("osworld", "tau") else "",
        assets_with_case=family == "osworld", build_assertion=f"assert_{family}_offline_build",
        state_patch_allowed=family not in ("swe", "osworld", "deepswe")))
for family in ("reasoning_qa", "knowledge_qa", "code_gen", "health_qa"):
    register_family(FamilyRules(
        family, "data_designer.pipelines.benchmarks.families.hf_single_turn",
        "verify_hf_single_turn_source_manifest", "hf_single_turn", "derive_hf_single_turn_spec",
        build_assertion="assert_hf_single_turn_build"))
for family in ("gaia", "webarena", "appworld", "terminal", "workbuddy"):
    register_family(FamilyRules(
        family, f"data_designer.pipelines.benchmarks.families.{family}",
        "" if family == "workbuddy" else f"verify_{family}_source_manifest",
        runtime_assets=f"assert_{family}_runtime_assets" if family in ("webarena", "appworld") else "",
        assets_with_case=family == "appworld",
        build_assertion="assert_terminal_build" if family == "terminal" else "",
        native_pins="verify_native_ingest_pins" if family == "workbuddy" else ""))
