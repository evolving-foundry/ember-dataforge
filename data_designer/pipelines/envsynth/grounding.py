"""Frozen grounding index shared by synthesis and Gym runtime (decision 6).

The default backend is deterministic and dependency-free so template-mode CI
never needs an embedding service.  Production may inject a bge backend that
implements ``EmbeddingBackend``; its version must be pinned in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from data_designer.pipelines.envsynth.redaction import sanitize_record


GROUNDING_SCHEMA_VERSION = "grounding-index-v1"
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


class EmbeddingBackend:
    version: str = "unknown"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


def _tokens(text: str) -> List[str]:
    text = str(text or "").lower()
    tokens: List[str] = [match.group(0) for match in _WORD_RE.finditer(text)]
    for match in _CJK_RE.finditer(text):
        run = match.group(0)
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


class CharNgramEmbeddingBackend(EmbeddingBackend):
    """Hashing bag-of-tokens backend with CJK unigrams and bigrams."""

    version = "char-ngram-v1"

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _tokens(text):
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                value = int(digest, 16)
                index = value % self.dim
                sign = 1.0 if (value >> 128) % 2 == 0 else -1.0
                vec[index] += sign
            norm = math.sqrt(sum(item * item for item in vec)) or 1.0
            vectors.append([round(item / norm, 8) for item in vec])
        return vectors


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return numerator / (norm_left * norm_right)


def chunks_from_domain(
    domain: Dict[str, Any],
    *,
    renderer: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> List[Dict[str, Any]]:
    """Build REDACTED grounding chunks without importing any domain recipe.

    Domain-specific text rendering is supplied by the selected DomainAdapter via
    renderer. Without a renderer, records fall back to deterministic JSON text.
    """
    chunks = domain.get("chunks")
    if isinstance(chunks, list) and chunks:
        return [sanitize_record(dict(chunk)) for chunk in chunks]
    result: List[Dict[str, Any]] = []
    for regulation in domain.get("regulations") or []:
        result.append(
            {
                "text": regulation.get("text", ""),
                "role": "regulation",
                "source": regulation.get("source"),
            }
        )
    for ticket in domain.get("tickets") or []:
        clean_ticket = sanitize_record(ticket)
        if renderer is not None:
            text = renderer(clean_ticket)
        else:
            text = json.dumps(clean_ticket, ensure_ascii=False, sort_keys=True)
        result.append(
            {
                "text": text,
                "role": "seed",
                "source": clean_ticket.get("source"),
            }
        )
    return [sanitize_record(chunk) for chunk in result]


def _validate_vector(value: Any, label: str) -> List[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("{} must be a non-empty numeric array".format(label))
    result: List[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError("{} may contain only numeric values".format(label))
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("{} must not contain NaN/Inf".format(label))
        result.append(number)
    return result


def build_index(
    chunks: Sequence[Dict[str, Any]],
    *,
    backend: Optional[EmbeddingBackend] = None,
) -> Dict[str, Any]:
    backend = backend or CharNgramEmbeddingBackend()
    if not isinstance(backend.version, str) or not backend.version.strip():
        raise ValueError("embedding backend.version must be a non-empty string")
    normalized: List[Dict[str, Any]] = []
    seen: set = set()
    seen_ids: set = set()
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError("grounding chunk must be an object")
        text = str(chunk.get("text") or "")
        if not text.strip():
            continue
        chunk_id = str(chunk.get("chunk_id") or "chunk-{:06d}".format(index + 1))
        if chunk_id in seen_ids:
            raise ValueError("Duplicate grounding chunk_id: {}".format(chunk_id))
        seen_ids.add(chunk_id)
        source = dict(chunk.get("source") or {})
        key = (chunk_id, text, json.dumps(source, sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "role": chunk.get("role") or "seed",
                "source": source,
            }
        )
    raw_vectors = (
        backend.embed([chunk["text"] for chunk in normalized]) if normalized else []
    )
    if not isinstance(raw_vectors, list) or len(raw_vectors) != len(normalized):
        raise ValueError("embedding backend must return one vector per chunk")
    vectors = [
        _validate_vector(vector, "embedding backend vector")
        for vector in raw_vectors
    ]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) > 1 or (dimensions and 0 in dimensions):
        raise ValueError("embedding backend returned vectors with inconsistent dimensions or no vectors")
    for chunk, vector in zip(normalized, vectors):
        chunk["vector"] = vector
    dim = len(vectors[0]) if vectors else 0
    return {
        "schemaVersion": GROUNDING_SCHEMA_VERSION,
        "embeddingModel": backend.version,
        "dim": dim,
        "chunkCount": len(normalized),
        "chunks": normalized,
    }


def index_chunks(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(index, dict):
        raise ValueError("grounding index must be an object")
    if index.get("schemaVersion") != GROUNDING_SCHEMA_VERSION:
        raise ValueError("invalid grounding index schemaVersion")
    chunks = index.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("grounding index chunks must be an array")
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("grounding chunk must be an object")
    return list(chunks)


def retrieve(
    index: Dict[str, Any],
    query: str,
    *,
    top_k: int = 4,
    role: Optional[str] = None,
    backend: Optional[EmbeddingBackend] = None,
) -> List[Dict[str, Any]]:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer")
    if top_k <= 0:
        return []
    chunks = index_chunks(index)
    if not chunks:
        return []
    try:
        dimension = int(index.get("dim") or 0)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid grounding index dim") from error
    if dimension <= 0:
        raise ValueError("grounding index dim must be a positive integer")
    expected_model = index.get("embeddingModel")
    if not isinstance(expected_model, str) or not expected_model:
        raise ValueError("grounding index embeddingModel is missing")
    if backend is None:
        if expected_model != CharNgramEmbeddingBackend.version:
            raise ValueError(
                "grounding index was generated by {}; a matching backend must be injected explicitly".format(
                    expected_model
                )
            )
        backend = CharNgramEmbeddingBackend(dim=dimension)
    elif backend.version != expected_model:
        raise ValueError(
            "grounding backend {} does not match index embeddingModel {}".format(
                backend.version, expected_model
            )
        )
    query_vectors = backend.embed([query])
    if not isinstance(query_vectors, list) or len(query_vectors) != 1:
        raise ValueError("embedding backend must return a single query vector")
    query_vector = _validate_vector(query_vectors[0], "grounding query vector")
    if len(query_vector) != dimension:
        raise ValueError(
            "grounding query vector dimension {} does not match index dim {}".format(
                len(query_vector), dimension
            )
        )
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for chunk in chunks:
        if role and chunk.get("role") != role:
            continue
        vector = _validate_vector(
            chunk.get("vector"),
            "grounding chunk {} vector".format(chunk.get("chunk_id")),
        )
        if len(vector) != dimension:
            raise ValueError(
                "grounding chunk {} vector dimension does not match index dim".format(
                    chunk.get("chunk_id")
                )
            )
        score = cosine_similarity(query_vector, vector)
        scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("chunk_id"))))
    return [
        {
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
            "role": chunk.get("role"),
            "source": chunk.get("source"),
            "score": round(score, 6),
        }
        for score, chunk in scored[: int(top_k)]
    ]


def write_grounding(
    root: str, chunks: Sequence[Dict[str, Any]], index: Dict[str, Any]
) -> List[str]:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    chunks_path = target / "chunks.jsonl"
    chunks_path.write_text(
        "".join(
            json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )
    written.append(str(chunks_path))
    index_path = target / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    written.append(str(index_path))
    return written


def load_grounding(root: str) -> Dict[str, Any]:
    index_path = Path(root) / "index.json"
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to read grounding index {index_path}: {error}") from error


__all__ = [
    "CharNgramEmbeddingBackend",
    "EmbeddingBackend",
    "GROUNDING_SCHEMA_VERSION",
    "build_index",
    "chunks_from_domain",
    "cosine_similarity",
    "index_chunks",
    "load_grounding",
    "retrieve",
    "write_grounding",
]
