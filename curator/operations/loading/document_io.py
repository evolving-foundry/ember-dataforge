"""Input discovery, path validation and immutable source snapshots."""
from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from .document_common import DocumentParseError

DEFAULT_DOCUMENT_EXTENSIONS = {".docx": "docx", ".xlsx": "xlsx", ".pdf": "pdf"}


class SnapshotOptions(Protocol):
    max_file_mb: float
    allowed_roots: Tuple[str, ...]


@dataclass(frozen=True)
class InputItem:
    path: str
    error: Optional[DocumentParseError] = None
    explicit: bool = False


@dataclass(frozen=True)
class Snapshot:
    path: str
    sha256: str
    byte_size: int
    doc_id: str


def _invalid(message: str, stage: str = "validate") -> DocumentParseError:
    return DocumentParseError("INVALID_PARAMETER", message, stage=stage)


def _within_root(path: str, roots: Sequence[str]) -> bool:
    for root in roots:
        try:
            path_key = os.path.normcase(path)
            root_key = os.path.normcase(root)
            if os.path.commonpath([path_key, root_key]) == root_key:
                return True
        except ValueError:
            continue
    return False


def canonicalize_roots(
    roots: Iterable[str], *, require_exists: bool = True
) -> Tuple[str, ...]:
    result: List[str] = []
    for root in roots:
        if not isinstance(root, str) or not root:
            raise _invalid("allowed_roots entries must be non-empty strings")
        real = os.path.realpath(os.path.abspath(root))
        if require_exists and not os.path.isdir(real):
            raise _invalid("allowed_roots entries must be directories")
        result.append(real)
    return tuple(dict.fromkeys(result))


def default_roots(data_paths: Sequence[str]) -> Tuple[str, ...]:
    roots: List[str] = []
    for raw in data_paths:
        path = os.path.realpath(os.path.abspath(raw))
        roots.append(path if os.path.isdir(path) else os.path.dirname(path))
    return tuple(dict.fromkeys(roots))


def _check_file(path: str, roots: Sequence[str]) -> str:
    real = os.path.realpath(path)
    try:
        st = os.stat(real)
    except OSError as exc:
        raise DocumentParseError(
            "FILE_UNREADABLE", "source file is not readable", stage="read"
        ) from exc
    if not stat.S_ISREG(st.st_mode):
        raise DocumentParseError(
            "FILE_UNREADABLE", "source is not a regular file", stage="read"
        )
    if roots and not _within_root(real, roots):
        raise DocumentParseError(
            "PATH_OUTSIDE_ALLOWED_ROOT",
            "source path is outside allowed roots",
            stage="validate",
        )
    return real


def validate_source_file(path: str, roots: Sequence[str]) -> str:
    """Validate one file for op-mode reuse."""
    return _check_file(path, roots)


def discover_inputs(
    data_paths: Sequence[str],
    roots: Sequence[str],
    *,
    supported_extensions: Optional[Mapping[str, str]] = None,
) -> List[InputItem]:
    extensions = (
        DEFAULT_DOCUMENT_EXTENSIONS
        if supported_extensions is None
        else supported_extensions
    )
    if not isinstance(data_paths, (list, tuple)) or not data_paths:
        raise _invalid("data_paths must be a non-empty array")
    if not all(isinstance(item, str) and item for item in data_paths):
        raise _invalid("data_paths entries must be non-empty strings")
    effective_roots = canonicalize_roots(roots) if roots else default_roots(data_paths)
    found: List[InputItem] = []
    seen: set[str] = set()
    for raw in data_paths:
        path = os.path.abspath(raw)
        if not os.path.exists(path):
            found.append(
                InputItem(
                    path,
                    DocumentParseError(
                        "FILE_UNREADABLE", "source does not exist", stage="read"
                    ),
                    True,
                )
            )
            continue
        if os.path.isdir(path):
            for base, dirs, files in os.walk(path, followlinks=False):
                dirs[:] = [
                    name
                    for name in dirs
                    if not os.path.islink(os.path.join(base, name))
                ]
                for name in sorted(files):
                    candidate = os.path.join(base, name)
                    if os.path.splitext(name)[1].lower() not in extensions:
                        continue
                    real = os.path.realpath(candidate)
                    if real in seen:
                        continue
                    seen.add(real)
                    try:
                        found.append(InputItem(_check_file(real, effective_roots)))
                    except DocumentParseError as exc:
                        found.append(InputItem(real, exc))
            continue
        ext = os.path.splitext(path)[1].lower()
        real = os.path.realpath(path)
        if ext not in extensions:
            found.append(
                InputItem(
                    real,
                    DocumentParseError(
                        "UNSUPPORTED_FORMAT",
                        "unsupported file extension",
                        stage="validate",
                    ),
                    True,
                )
            )
            continue
        if real in seen:
            continue
        seen.add(real)
        try:
            found.append(InputItem(_check_file(real, effective_roots), explicit=True))
        except DocumentParseError as exc:
            found.append(InputItem(real, exc, explicit=True))
    found.sort(key=lambda item: item.path)
    return found


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_file(
    path: str,
    options: SnapshotOptions,
    work_dir: Optional[str] = None,
) -> Snapshot:
    real = os.path.realpath(path)
    roots = (
        canonicalize_roots(options.allowed_roots, require_exists=False)
        if options.allowed_roots
        else ()
    )
    if roots and not _within_root(real, roots):
        raise DocumentParseError(
            "PATH_OUTSIDE_ALLOWED_ROOT",
            "source path is outside allowed roots",
            stage="validate",
        )
    try:
        fd = os.open(real, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError as exc:
        raise DocumentParseError(
            "FILE_UNREADABLE", "source file is not readable", stage="read"
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise DocumentParseError(
                "FILE_UNREADABLE", "source is not a regular file", stage="read"
            )
        try:
            path_stat = os.stat(real)
        except FileNotFoundError:
            path_stat = None
        except OSError as exc:
            raise DocumentParseError(
                "FILE_UNREADABLE", "source file is not readable", stage="read"
            ) from exc
        if path_stat is not None and (path_stat.st_dev, path_stat.st_ino) != (
            st.st_dev,
            st.st_ino,
        ):
            raise DocumentParseError(
                "FILE_UNREADABLE",
                "source file changed during validation",
                stage="read",
                retryable=True,
            )
        limit = int(options.max_file_mb * 1024 * 1024)
        if st.st_size > limit:
            raise DocumentParseError(
                "FILE_TOO_LARGE", "source file exceeds max_file_mb", stage="validate"
            )
        temp_path = ""
        copied = 0
        digest = hashlib.sha256()
        try:
            suffix = os.path.splitext(path)[1].lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,16}", suffix):
                suffix = ""
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".data-forge-parse-",
                suffix=suffix,
                dir=work_dir,
                delete=False,
            ) as target:
                temp_path = target.name
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > limit:
                        raise DocumentParseError(
                            "FILE_TOO_LARGE",
                            "source file exceeds max_file_mb",
                            stage="validate",
                        )
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        except Exception:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise
        sha = digest.hexdigest()
        return Snapshot(temp_path, sha, copied, f"pd-{sha[:16]}")
    finally:
        os.close(fd)


__all__ = [
    "DEFAULT_DOCUMENT_EXTENSIONS",
    "InputItem",
    "Snapshot",
    "canonicalize_roots",
    "default_roots",
    "discover_inputs",
    "sha256_file",
    "snapshot_file",
    "validate_source_file",
]
