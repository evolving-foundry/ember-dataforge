"""Source import for env synthesis: inventory + C-IR-2 parse + seed manifest.

Raw user data stays outside the repository.  This module writes only a
deterministic ``seed-manifest-v1`` bundle (parsed rows + semantic domain data)
into a caller-owned output directory that must not already exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from data_designer.pipelines.envsynth import curator_contract
from data_designer.pipelines.envsynth.redaction import sanitize_source
from data_designer.pipelines.envsynth.serialization import (
    json_bytes as _json_bytes,
    jsonl_bytes as _jsonl_bytes,
)
from data_designer.pipelines.envsynth.domains.user_examples_seed_adapter import (
    FAMILY,
    map_parsed_rows,
    validate_domain,
)


SEED_MANIFEST_SCHEMA_VERSION = "seed-manifest-v1"
IMAGE_EXTENSIONS = {".png": "png", ".jpg": "jpg", ".jpeg": "jpg"}
SUPPORTED_EXTENSIONS = {
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".pdf": "pdf",
    **IMAGE_EXTENSIONS,
}
IMAGE_MEDIA_TYPES = frozenset(IMAGE_EXTENSIONS.values())


class SeedImportError(RuntimeError):
    """Source import failed or the seed manifest is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sort_key(item: Dict[str, Any]) -> tuple:
    return (
        str(item.get("sha256") or ""),
        str(item.get("role") or ""),
        str(item.get("mediaType") or ""),
        int(item.get("byteSize") or 0),
    )


def _require_absolute_new_dir(path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        raise SeedImportError("outputDir must be an absolute path")
    if target.exists():
        raise SeedImportError(f"outputDir already exists; refusing to overwrite: {target}")
    return target


def _walk_files(root: Path, *, include_extensions: bool = True) -> List[Path]:
    files: List[Path] = []
    for base, dirs, names in os.walk(root, followlinks=False):
        base_path = Path(base)
        dirs[:] = sorted(
            name for name in dirs if not (base_path / name).is_symlink()
        )
        for name in sorted(names):
            path = base_path / name
            if path.is_symlink() or not path.is_file():
                continue
            if include_extensions and path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            files.append(path)
    return files


def _resolved_regulation_paths(paths: Sequence[str]) -> set:
    resolved: set = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            resolved.update(file.resolve() for file in _walk_files(path))
        elif path.is_file():
            resolved.add(path)
        else:
            raise SeedImportError(f"Regulation path does not exist: {raw}")
    return resolved


def discover_source_files(
    source_paths: Sequence[str],
    *,
    regulation_paths: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Deterministic file inventory with per-file SHA-256 and source role."""
    if not source_paths:
        raise SeedImportError("sourcePaths must not be empty")
    regulation_files = _resolved_regulation_paths(list(regulation_paths or []))
    records: Dict[str, Dict[str, Any]] = {}

    def record_file(path: Path, base: Path, role: str) -> None:
        real = path.resolve()
        key = str(real)
        if key in records:
            if role == "regulation":
                records[key]["role"] = "regulation"
            return
        try:
            relative = real.relative_to(base.resolve())
        except ValueError:
            relative = Path(real.name)
        records[key] = {
            "pathRef": str(real),
            "relativeRef": str(relative).replace("\\", "/"),
            "sha256": sha256_file(real),
            "mediaType": SUPPORTED_EXTENSIONS.get(real.suffix.lower(), ""),
            "byteSize": real.stat().st_size,
            "role": role,
        }

    for raw in source_paths:
        root = Path(raw).expanduser().resolve()
        if not root.exists():
            raise SeedImportError(f"Source path does not exist: {raw}")
        if root.is_file():
            candidates = [root]
            base = root.parent
        elif root.is_dir():
            candidates = _walk_files(root)
            base = root
        else:
            raise SeedImportError(f"Source path is neither a file nor a directory: {raw}")
        for path in candidates:
            real = path.resolve()
            role = "regulation" if real in regulation_files else "seed"
            record_file(path, base, role)

    # regulationPaths may point at files outside sourcePaths; include them as
    # independent regulation inputs instead of only tagging already-discovered
    # source files.
    for raw in regulation_paths or []:
        root = Path(raw).expanduser().resolve()
        is_file = root.is_file()
        candidates = [root] if is_file else _walk_files(root)
        base = root.parent if is_file else root
        for path in candidates:
            record_file(path, base, "regulation")

    files = sorted(records.values(), key=_file_sort_key)
    if not any(item["role"] == "seed" for item in files):
        raise SeedImportError("User samples (L1) are empty: sourcePaths contains only regulation documents or no parsable files")
    return files


def _allowed_roots(paths: Sequence[str]) -> List[str]:
    roots: set = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        roots.add(str(path if path.is_dir() else path.parent))
    return sorted(roots)


def _resolve_manifest_ref(manifest_dir: Path, ref: str) -> Path:
    if not isinstance(ref, str) or not ref:
        raise SeedImportError("seed manifest ref must be a non-empty string")
    candidate = (manifest_dir / ref).resolve()
    try:
        candidate.relative_to(manifest_dir.resolve())
    except ValueError as error:
        raise SeedImportError(f"seed manifest ref is out of bounds: {ref}") from error
    if not candidate.is_file():
        raise SeedImportError(f"seed manifest ref does not exist: {ref}")
    return candidate


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SeedImportError(f"Unable to read JSON {path}: {error}") from error


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise SeedImportError(
                f"{path}:{line_number} is not valid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise SeedImportError(f"{path}:{line_number} line must be an object")
        rows.append(row)
    return rows


@dataclass
class SeedBundle:
    manifest_path: Path
    manifest_sha256: str
    manifest: Dict[str, Any]
    parsed_rows: List[Dict[str, Any]]
    domain: Dict[str, Any]
    files: List[Dict[str, Any]]


def write_seed_import_from_rows(
    output_dir: str,
    parsed_rows: Sequence[Dict[str, Any]],
    file_records: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    family: str = FAMILY,
    anchor_holdout_ratio: float = 0.3,
    max_questions: int = 20,
) -> Dict[str, Any]:
    """Write a deterministic seed-import bundle from already parsed C-IR-2 rows."""
    target = _require_absolute_new_dir(output_dir)
    rows = curator_contract.validate_parsed_rows(parsed_rows)
    files = [dict(item) for item in (file_records or [])]
    doc_id_by_sha = {
        str((row.get("parsed_doc") or {}).get("source", {}).get("sha256")): row.get(
            "doc_id"
        )
        for row in rows
        if row.get("parsed_doc")
    }
    known = {str(item.get("sha256")) for item in files if item.get("sha256")}
    for sha, doc_id in doc_id_by_sha.items():
        if sha not in known:
            files.append(
                {
                    "sha256": sha,
                    "mediaType": "",
                    "byteSize": 0,
                    "role": "seed",
                }
            )
            known.add(sha)
    for item in files:
        sha = str(item.get("sha256") or "")
        if sha and not item.get("docId") and sha in doc_id_by_sha:
            item["docId"] = doc_id_by_sha[sha]
    files.sort(key=_file_sort_key)

    domain = map_parsed_rows(
        rows,
        file_records=files,
        family=family,
        anchor_holdout_ratio=anchor_holdout_ratio,
        max_questions=max_questions,
    )
    parsed_bytes = _jsonl_bytes(rows)
    domain_bytes = _json_bytes(domain)
    manifest_files = [sanitize_source(item) for item in files]
    manifest: Dict[str, Any] = {
        "schemaVersion": SEED_MANIFEST_SCHEMA_VERSION,
        "family": family,
        "curatorSchema": curator_contract.PARSED_DOC_SCHEMA_VERSION,
        "files": manifest_files,
        "parsedRefs": ["parsed/documents.jsonl"],
        "parsedSha256": _sha256_bytes(parsed_bytes),
        "domainRef": "domain_dataset.json",
        "domainSha256": _sha256_bytes(domain_bytes),
        "domainSchemaGuess": domain.get("schemaGuess"),
        "anchorSplit": domain.get("anchorSplit"),
        "questionCount": len(domain.get("questions") or []),
        "blockingQuestionCount": len(
            [q for q in domain.get("questions") or [] if q.get("blocking")]
        ),
    }
    manifest_bytes = _json_bytes(manifest)

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(dir=str(target.parent), prefix=".envseed-import-")
    )
    try:
        (temp_dir / "parsed").mkdir()
        (temp_dir / "parsed" / "documents.jsonl").write_bytes(parsed_bytes)
        (temp_dir / "domain_dataset.json").write_bytes(domain_bytes)
        (temp_dir / "seed_manifest.json").write_bytes(manifest_bytes)
        os.replace(str(temp_dir), str(target))
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    return manifest


def load_seed_bundle(
    manifest_path: str, *, expected_sha256: Optional[str] = None
) -> SeedBundle:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise SeedImportError(f"seed manifest does not exist: {manifest_path}")
    raw = path.read_bytes()
    actual_sha = _sha256_bytes(raw)
    if expected_sha256 and actual_sha != expected_sha256:
        raise SeedImportError(
            "seed manifest SHA-256 mismatch: expected {}, actual {}".format(
                expected_sha256, actual_sha
            )
        )
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise SeedImportError(f"seed manifest is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise SeedImportError("seed manifest must be an object")
    if manifest.get("schemaVersion") != SEED_MANIFEST_SCHEMA_VERSION:
        raise SeedImportError(
            "seed manifest schemaVersion must be {}".format(
                SEED_MANIFEST_SCHEMA_VERSION
            )
        )
    manifest_dir = path.parent
    parsed_refs = manifest.get("parsedRefs")
    if not isinstance(parsed_refs, list) or not parsed_refs:
        raise SeedImportError("seed manifest parsedRefs must be a non-empty array")
    rows: List[Dict[str, Any]] = []
    for ref in parsed_refs:
        parsed_path = _resolve_manifest_ref(manifest_dir, ref)
        if parsed_path.suffix.lower() == ".jsonl":
            rows.extend(_load_jsonl(parsed_path))
        else:
            payload = _load_json(parsed_path)
            if not isinstance(payload, list):
                raise SeedImportError(f"parsed ref must contain an array: {ref}")
            rows.extend(payload)
        expected_parsed_sha = manifest.get("parsedSha256")
        if expected_parsed_sha and _sha256_bytes(parsed_path.read_bytes()) != (
            expected_parsed_sha
        ):
            raise SeedImportError(f"parsed file SHA-256 mismatch: {ref}")
    rows = curator_contract.validate_parsed_rows(rows)

    domain_ref = manifest.get("domainRef")
    if not isinstance(domain_ref, str) or not domain_ref:
        raise SeedImportError("seed manifest is missing domainRef")
    domain_path = _resolve_manifest_ref(manifest_dir, domain_ref)
    if manifest.get("domainSha256") and _sha256_bytes(
        domain_path.read_bytes()
    ) != manifest.get("domainSha256"):
        raise SeedImportError("domain file SHA-256 mismatch")
    domain = _load_json(domain_path)
    validate_domain(domain)

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise SeedImportError("seed manifest files must be a non-empty array")
    row_shas = {
        str((row.get("parsed_doc") or {}).get("source", {}).get("sha256"))
        for row in rows
        if row.get("parsed_doc")
    }
    for item in files:
        if not isinstance(item, dict) or not item.get("sha256"):
            raise SeedImportError("seed manifest files entries must contain sha256")
        if str(item["sha256"]) not in row_shas:
            raise SeedImportError(
                "Source file has no corresponding C-IR-2 row: {}".format(item.get("sha256"))
            )
    return SeedBundle(
        manifest_path=path,
        manifest_sha256=actual_sha,
        manifest=manifest,
        parsed_rows=rows,
        domain=domain,
        files=files,
    )


def _merge_parsed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = [dict(row) for row in rows]
    for index, row in enumerate(merged):
        row["input_index"] = index
    return curator_contract.validate_parsed_rows(merged)


def build_seed_import_from_source(
    source_paths: Sequence[str],
    output_dir: str,
    *,
    regulation_paths: Optional[Sequence[str]] = None,
    family: str = FAMILY,
    anchor_holdout_ratio: float = 0.3,
    max_questions: int = 20,
    parse_options: Optional[Dict[str, Any]] = None,
    ocr_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover, parse documents/images through Curator, and write a bundle."""
    files = discover_source_files(source_paths, regulation_paths=regulation_paths)
    document_files = [
        item for item in files if item.get("mediaType") not in IMAGE_MEDIA_TYPES
    ]
    image_files = [
        item for item in files if item.get("mediaType") in IMAGE_MEDIA_TYPES
    ]
    effective_roots = _allowed_roots(
        list(source_paths) + list(regulation_paths or [])
    )
    rows: List[Dict[str, Any]] = []
    if document_files:
        rows.extend(
            curator_contract.parse_documents(
                [str(item["pathRef"]) for item in document_files],
                allowed_roots=effective_roots,
                parse_options=parse_options,
            )
        )
    if image_files:
        rows.extend(
            curator_contract.parse_images(
                [str(item["pathRef"]) for item in image_files],
                allowed_roots=effective_roots,
                ocr_options=ocr_options,
            )
        )
    if not rows:
        raise SeedImportError("No C-IR-2 rows were parsed")
    manifest = write_seed_import_from_rows(
        output_dir,
        _merge_parsed_rows(rows),
        files,
        family=family,
        anchor_holdout_ratio=anchor_holdout_ratio,
        max_questions=max_questions,
    )
    return manifest


__all__ = [
    "SEED_MANIFEST_SCHEMA_VERSION",
    "SeedBundle",
    "IMAGE_EXTENSIONS",
    "SeedImportError",
    "build_seed_import_from_source",
    "discover_source_files",
    "load_seed_bundle",
    "sha256_file",
    "write_seed_import_from_rows",
]
