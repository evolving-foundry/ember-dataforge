from __future__ import annotations

import os
from typing import Iterable, Sequence


DEFAULT_LOCAL_MODEL_ROOTS = (
    "/data/llm_models",
)


def local_model_roots_from_params(value: object = None) -> list[str]:
    """Return local model search roots from params/env plus project defaults."""
    roots: list[str] = []
    if isinstance(value, str):
        roots.extend(part for part in value.split(os.pathsep) if part)
    elif isinstance(value, Iterable):
        roots.extend(str(part) for part in value if part)

    env_value = os.environ.get("MODEL_ROOTS")
    if env_value:
        roots.extend(part for part in env_value.split(os.pathsep) if part)

    roots.extend(DEFAULT_LOCAL_MODEL_ROOTS)
    return _unique_existing_dirs(roots)


def resolve_local_model_path(
    model_path: str,
    local_roots: Sequence[str] | None = None,
) -> str:
    """Resolve a Hugging Face model id to an existing local directory when present."""
    raw_model_path = str(model_path or "").strip()
    if not raw_model_path:
        return raw_model_path
    if os.path.exists(raw_model_path):
        return raw_model_path

    roots = list(local_roots or local_model_roots_from_params())
    normalized = raw_model_path.strip("/")
    candidate_suffixes = [normalized]
    if "/" in normalized:
        candidate_suffixes.append(normalized.split("/")[-1])

    for root in roots:
        for suffix in candidate_suffixes:
            candidate = os.path.join(root, suffix)
            if os.path.isdir(candidate):
                return candidate
    return raw_model_path


def _unique_existing_dirs(paths: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = os.path.abspath(os.path.expanduser(str(path)))
        if normalized in seen or not os.path.isdir(normalized):
            continue
        unique.append(normalized)
        seen.add(normalized)
    return unique
