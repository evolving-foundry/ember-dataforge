"""Vertical product pipelines composing the horizontal capability layers.

Layout convention:

- Capability layers (``data_designer.contracts/sources/storage/verifiers/
  generation/operators``) are domain-agnostic and MUST NOT import from this
  package (enforced by ``tests/test_architecture_layers.py``).
- Pipelines compose layers into end-to-end products; ``integrations`` and
  ``cli`` are bridges/entry points and may import from here.
"""
