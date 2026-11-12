"""Stub ``torch`` in Ray worker interpreters (usercustomize variant).

Ray's ``runtime_env`` supports a ``py_modules`` list whose directories are added
to ``PYTHONPATH`` inside worker processes. When a worker interpreter starts,
CPython auto-imports ``usercustomize`` from any directory on ``PYTHONPATH``,
giving us a hook to register ``torch`` *before* Curator's augmentation chain
runs ``import torch`` at module load.
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
