"""Central redaction for user-data paths before freezing or agent retrieval."""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence

_REDACTED_PATH = "<redacted-path>"

PATH_KEYS = frozenset(
    {
        "absolute_path",
        "absolutePath",
        "assetDir",
        "assetRoot",
        "expertEvidencePath",
        "inputPath",
        "manifestOutDir",
        "outputDir",
        "path_ref",
        "pathRef",
        "regulationPaths",
        "relativeRef",
        "seedManifestPath",
        "source_file",
        "source_path",
        "sourceFile",
        "sourcePath",
        "sourcePaths",
        "storePath",
    }
)

# Host filesystem spellings are redacted even when they appear in free text.
# Virtual-environment paths such as /report.md are intentionally left alone;
# callers that know their virtual workspace can additionally pass allowed_paths.
_BACKSLASH = re.escape("\\")
_HOST_PATH_PATTERNS = (
    re.compile(
        r"(?<![\w])[A-Za-z]:" + "[" + _BACKSLASH + "/]" + r"[^\s\"'<>|]+"
    ),
    re.compile(r"(?<![\w])" + _BACKSLASH + _BACKSLASH + r"[^\s\"'<>|]+"),
    re.compile(
        r"(?<![\w:])/(?:home|Users|root|mnt|tmp|var|opt|etc|srv|private|data|Volumes)/"
        r"(?:[^\s\"'<>|]+)"
    ),
)


def _matches_allowed(path: str, allowed_paths: Sequence[str]) -> bool:
    for allowed in allowed_paths:
        if not isinstance(allowed, str) or not allowed.startswith("/"):
            continue
        if allowed == "/":
            # The virtual root is a cwd, not a blanket allow-list.
            continue
        if path == allowed or path.startswith(allowed.rstrip("/") + "/"):
            return True
    return False


def redact_text(value: str, *, allowed_paths: Sequence[str] = ()) -> str:
    """Redact host-looking paths while preserving declared virtual paths."""
    if not isinstance(value, str):
        return value
    result = value
    for pattern in _HOST_PATH_PATTERNS:
        def replace(match: re.Match[str]) -> str:
            path = match.group(0)
            return path if _matches_allowed(path, allowed_paths) else _REDACTED_PATH

        result = pattern.sub(replace, result)
    return result


def redact(value: Any, *, allowed_paths: Sequence[str] = ()) -> Any:
    """Recursively drop path-carrying keys and redact host paths in strings."""
    if isinstance(value, dict):
        return {
            key: redact(item, allowed_paths=allowed_paths)
            for key, item in value.items()
            if key not in PATH_KEYS and item is not None and item != ""
        }
    if isinstance(value, list):
        return [redact(item, allowed_paths=allowed_paths) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item, allowed_paths=allowed_paths) for item in value)
    if isinstance(value, str):
        return redact_text(value, allowed_paths=allowed_paths)
    return value


def sanitize_source(source: Any, *, allowed_paths: Sequence[str] = ()) -> Dict[str, Any]:
    return (
        redact(source, allowed_paths=allowed_paths)
        if isinstance(source, dict)
        else {}
    )


def sanitize_record(record: Any, *, allowed_paths: Sequence[str] = ()) -> Dict[str, Any]:
    return (
        redact(record, allowed_paths=allowed_paths)
        if isinstance(record, dict)
        else {}
    )


def sanitize_draft(draft: Any, *, allowed_paths: Sequence[str] = ()) -> Dict[str, Any]:
    return (
        redact(draft, allowed_paths=allowed_paths)
        if isinstance(draft, dict)
        else {}
    )


__all__ = [
    "PATH_KEYS",
    "redact",
    "redact_text",
    "sanitize_draft",
    "sanitize_record",
    "sanitize_source",
]
