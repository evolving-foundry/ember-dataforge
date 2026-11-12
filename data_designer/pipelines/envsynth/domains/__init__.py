"""Domain-specific adaptation layer for the envsynth pipeline.

These modules adapt external user examples (power-ticket domain) into the
neutral structures the synthesizer consumes.  Generic envsynth layers
(``environment_model``, ``grounding``, ``package``, ``gates``,
``profile_compilers``, ``domain_adapters.base``) MUST NOT import from this
package (enforced by ``tests/envsynth/test_architecture.py``).
"""
