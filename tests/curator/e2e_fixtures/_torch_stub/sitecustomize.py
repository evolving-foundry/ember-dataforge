"""Stub ``torch`` in Ray worker interpreters.

Ray deserializes user code (e.g. ``ExactDuplicateRemovalOperation``) into fresh
worker processes. Curator's ``augmentation/__init__.py`` chain pulls in
``video_temporal_structure_operation`` which does ``import torch`` at module
top — even though we don't use any video augmentation here, the import explodes.

This module is shipped to workers via Ray ``runtime_env`` and auto-loaded as a
``sitecustomize`` (Ray prepends the configured ``PYTHONPATH`` entry containing
this directory to its worker interpreters), so ``torch`` exists before any
Curator/DataForge import is attempted.
"""
import sys
import types

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.float32 = "float32"
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.int64 = "int64"
    torch_stub.int32 = "int32"
    sys.modules["torch"] = torch_stub
