"""
Data loader factory for dynamically selecting the appropriate Ray data reading method
"""
import logging
import json

logger = logging.getLogger(__name__)
import mimetypes
import os
from typing import Dict, Any, List, Union, Callable

import ray
from ray.data import Dataset


class DataLoaderFactory:
    """Data loader factory class that dynamically selects a loading method based on file type and modality.

    load_dataset() execution flow:
      _normalize_paths → _group_by_modality → _load_each_group → _merge_datasets
    """

    # Mapping from file extensions to modality types
    EXTENSION_TO_MODALITY = {
        # Image formats
        ".jpg": "image", ".jpeg": "image", ".png": "image",
        ".bmp": "image", ".tiff": "image", ".tif": "image",
        ".webp": "image", ".gif": "image", ".svg": "image",
        # Audio formats
        ".mp3": "audio", ".wav": "audio", ".aac": "audio",
        ".flac": "audio", ".ogg": "audio", ".m4a": "audio",
        ".wma": "audio", ".opus": "audio", ".aiff": "audio",
        ".alac": "audio", ".amr": "audio", ".ra": "audio",
        ".rm": "audio", ".au": "audio", ".mid": "audio",
        ".midi": "audio", ".pcm": "audio", ".caf": "audio",
        # Video formats
        ".mp4": "video", ".avi": "video", ".mov": "video",
        ".mkv": "video", ".webm": "video", ".flv": "video",
        ".wmv": "video", ".m4v": "video", ".3gp": "video",
        ".mpeg": "video", ".mpg": "video", ".ts": "video",
        ".ogv": "video", ".rmvb": "video", ".vob": "video",
        ".asf": "video", ".f4v": "video", ".m2ts": "video",
        ".mts": "video", ".divx": "video", ".xvid": "video",
        ".mxf": "video",
        # Text formats
        ".txt": "text", ".json": "text", ".jsonl": "text",
        ".csv": "text", ".tsv": "text", ".xml": "text",
        ".html": "text", ".htm": "text", ".md": "text",
        ".rst": "text", ".rtf": "text",
    }

    # Mapping from modality to Ray read method
    MODALITY_TO_READER = {
        "image": "read_images",
        "audio": "read_audio",
        "video": "read_videos",
        "text": "read_text_or_structured",
    }

    # ── Directory expansion ───────────────────────────────────────────────────

    @classmethod
    def _expand_directory(cls, path: str) -> List[str]:
        """Recursively expand a directory path into a list of data file paths within it.

        Args:
            path: Directory path

        Returns:
            List of file paths for all recognized modalities under the directory
        """
        import os
        expanded: List[str] = []
        for root, _, files in os.walk(path):
            for fname in files:
                full = os.path.join(root, fname)
                if cls.detect_modality(full) != "unknown":
                    expanded.append(full)
        return expanded

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def detect_modality(cls, path: str) -> str:
        """Detect the data modality type from a file path."""
        _, ext = os.path.splitext(path.lower())
        return cls.EXTENSION_TO_MODALITY.get(ext, "unknown")

    @classmethod
    def get_reader_for_modality(cls, modality: str) -> Callable:
        """Get the read method for the specified modality."""
        reader_method = cls.MODALITY_TO_READER.get(modality)
        if not reader_method:
            raise ValueError(f"Unsupported modality: {modality}")
        if reader_method == "read_text_or_structured":
            return cls._read_text_or_structured
        try:
            return getattr(ray.data, reader_method)
        except AttributeError:
            logger.warning("Warning: %s not available, falling back to generic reader", reader_method)
            return cls._fallback_reader

    @classmethod
    def load_dataset(cls, paths: Union[str, List[str]]) -> Dataset:
        """Automatically select appropriate read methods based on the path, then load and merge the datasets.

        Supports passing file paths or directory paths directly (directories are expanded automatically).
        """
        import os
        normalized = cls._normalize_paths(paths)

        # Directory expansion: handled uniformly here
        flat_paths: List[str] = []
        for p in normalized:
            if os.path.isdir(p):
                flat_paths.extend(cls._expand_directory(p))
            else:
                flat_paths.append(p)

        if not flat_paths:
            raise ValueError("No valid data paths found")

        groups = cls._group_by_modality(flat_paths)
        datasets = cls._load_each_group(groups)
        return cls._merge_datasets(datasets)

    @classmethod
    def create_multimodal_loader(cls, config: Dict[str, Any]) -> Callable:
        """Create a multimodal loader function from configuration."""
        def multimodal_loader():
            return cls.load_dataset(config.get("data_paths", []))
        return multimodal_loader

    # ── load_dataset decomposition steps ─────────────────────────────────────

    @staticmethod
    def _normalize_paths(paths: Union[str, List[str]]) -> List[str]:
        """Normalize to a list."""
        return [paths] if isinstance(paths, str) else list(paths)

    @classmethod
    def _group_by_modality(cls, paths: List[str]) -> Dict[str, List[str]]:
        """Group paths by modality type, ignoring unrecognized paths."""
        groups: Dict[str, List[str]] = {}
        for path in paths:
            modality = cls.detect_modality(path)
            if modality != "unknown":
                groups.setdefault(modality, []).append(path)
        return groups

    @classmethod
    def _load_each_group(cls, groups: Dict[str, List[str]]) -> List[Dataset]:
        """Use the corresponding read method for each modality group and return a list of Datasets."""
        datasets = []
        for modality, modality_paths in groups.items():
            datasets.append(cls._load_modality_group(modality, modality_paths))
        return datasets

    @classmethod
    def _load_modality_group(cls, modality: str, paths: List[str]) -> Dataset:
        """Load all paths in a single modality group and merge them into one Dataset."""
        reader = cls.get_reader_for_modality(modality)
        if modality in ("image", "audio", "video"):
            try:
                return reader(paths)
            except Exception:
                # Read each path with fallback, then merge
                datasets = [cls._fallback_reader(p) for p in paths]
                return datasets[0].union(*datasets[1:]) if len(datasets) > 1 else datasets[0]
        else:
            # Text/structured data: read each path then merge
            datasets = [reader(p) for p in paths]
            return datasets[0].union(*datasets[1:]) if len(datasets) > 1 else datasets[0]

    @staticmethod
    def _merge_datasets(datasets: List[Dataset]) -> Dataset:
        """Merge multiple Datasets into one; raise ValueError when the list is empty."""
        if not datasets:
            raise ValueError("No valid data paths found")
        if len(datasets) == 1:
            return datasets[0]
        return datasets[0].union(*datasets[1:])

    # ── Concrete read implementations ─────────────────────────────────────────

    @staticmethod
    def _read_text_or_structured(path: str) -> Dataset:
        """Read text or structured data, automatically recognizing JSON / JSONL / CSV / Parquet / TXT."""
        if path.endswith(".jsonl"):
            return ray.data.read_json(path)
        if path.endswith(".json"):
            return DataLoaderFactory._read_json_file(path)
        if path.endswith(".csv"):
            return ray.data.read_csv(path)
        if path.endswith(".parquet"):
            return ray.data.read_parquet(path)
        if path.endswith(".txt"):
            return ray.data.read_text(path)
        return ray.data.read_json(path)

    @staticmethod
    def _read_json_file(path: str) -> Dataset:
        """Detect the JSON file format (array vs NDJSON) and select the corresponding read method."""
        if DataLoaderFactory._is_json_array(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ray.data.from_items(data)
        return ray.data.read_json(path)

    @staticmethod
    def _is_json_array(path: str) -> bool:
        """Check whether the JSON file root structure is an array (read the first non-whitespace character)."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for chunk in iter(lambda: f.read(64), ""):
                    stripped = chunk.lstrip()
                    if stripped:
                        return stripped[0] == "["
        except Exception:
            pass
        return False

    @staticmethod
    def _fallback_reader(path: str) -> Dataset:
        """Generic fallback read method that tries binary or text reading based on MIME type."""
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type and mime_type.startswith(("image/", "audio/", "video/")):
            return ray.data.read_binary_files(path)
        try:
            return ray.data.read_text(path)
        except Exception:
            return ray.data.read_json(path)
