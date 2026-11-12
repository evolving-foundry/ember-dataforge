"""Media processing acceleration backend (optional MultimodalSDK enhancement).

Design points (implementation is in this module):
- Operators do not directly depend on MultimodalSDK; they only call ``get_media_backend()``;
- If the SDK is available (linux-aarch64 + smoke probe passes), use the C++ acceleration path; otherwise silently fall back to
  the existing PIL / cv2 / librosa implementation with zero behavior changes;
- ``MMSDKMediaBackend`` performs per-item format gate checks and fallback: within the same dataset,
  JPEG uses acceleration while PNG uses the old path, transparent to users and configuration.

This module introduces no new dependencies; ``mm`` is a purely optional external dependency imported lazily inside method calls.
"""

import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
from typing import List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Valid values for GlobalConfig.media_backend / the ADF_MEDIA_BACKEND environment variable
VALID_MEDIA_BACKENDS = ("auto", "mm_sdk", "default")

_ENV_MEDIA_BACKEND = "ADF_MEDIA_BACKEND"

# Maximum files per batch for mm.load_audio (SDK constraint 1-128)
MM_AUDIO_BATCH_LIMIT = 128


class BackendUnavailableError(RuntimeError):
    """MMSDK backend call failure (SDK failure at runtime or per-item decode error).

    In auto mode, ``MMSDKMediaBackend`` converts this exception into a fallback to the default implementation;
    in forced ``mm_sdk`` mode it is re-raised so users who pin acceleration can observe the failure.
    """


class MediaBackend(Protocol):
    """Media processing backend protocol (organized with the backend interface separated from concrete implementations).

    Conventions:
    - ``decode_image`` returns an RGB, uint8 ndarray with shape=(H, W, 3);
    - ``decode_video_frames`` returns a list of RGB ndarrays; when ``frame_indices`` is passed,
      the returned list corresponds one-to-one with the order of ``sorted(frame_indices)``;
    - ``load_audio`` returns (amplitude, sample_rate), where amplitude is float32 with
      shape=(channels, samples), aligned with the row schema of ray.data.read_audio;
    - ``size`` parameters are all (height, width).
    """

    name: str

    def decode_image(self, path: str) -> np.ndarray:
        """Decode a single image into an RGB ndarray."""
        ...

    def resize_image(
        self,
        img: np.ndarray,
        size: Tuple[int, int],
        interpolation: str = "bicubic",
    ) -> np.ndarray:
        """Resize an image to (height, width)."""
        ...

    def decode_video_frames(
        self,
        path: str,
        sample_num: Optional[int] = None,
        frame_indices: Optional[Set[int]] = None,
    ) -> List[np.ndarray]:
        """Decode video frames: explicit frame indices, uniform sampling of N frames, or all frames (choose one)."""
        ...

    def load_audio(self, path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load a single audio file."""
        ...

    def load_audio_batch(
        self, paths: Sequence[str], sr: Optional[int] = None
    ) -> List[Tuple[np.ndarray, int]]:
        """Batch-load audio files (results correspond one-to-one with input path order)."""
        ...


class DefaultMediaBackend:
    """Default backend: ports the existing inline PIL / cv2 / librosa logic with zero behavior changes."""

    name = "default"

    # ── Images ────────────────────────────────────────────────────────────────

    def decode_image(self, path: str) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required. Please install Pillow>=8.0.0.") from exc

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def resize_image(
        self,
        img: np.ndarray,
        size: Tuple[int, int],
        interpolation: str = "bicubic",
    ) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required. Please install Pillow>=8.0.0.") from exc

        pil_filters = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
        }
        key = str(interpolation).lower()
        if key not in pil_filters:
            raise ValueError(f"Unsupported interpolation: {interpolation}")

        height, width = int(size[0]), int(size[1])
        image = Image.fromarray(np.asarray(img))
        resized = image.resize((width, height), pil_filters[key])
        return np.asarray(resized)

    # ── Video ─────────────────────────────────────────────────────────────────

    def decode_video_frames(
        self,
        path: str,
        sample_num: Optional[int] = None,
        frame_indices: Optional[Set[int]] = None,
    ) -> List[np.ndarray]:
        from .lazy_imports import get_cv2_backend

        cv2_backend = get_cv2_backend()
        cv2 = cv2_backend.get_cv2()
        if cv2 is None:
            raise ImportError("OpenCV is required. Please install opencv-python>=4.5.0.")

        capture = cv2_backend.VideoCapture(str(path))
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise ValueError(f"Cannot open video: {path}")

        try:
            total_frames = int(capture.get(cv2_backend.CAP_PROP_FRAME_COUNT) or 0)

            if frame_indices is not None:
                indices = sorted({int(index) for index in frame_indices})
                for index in indices:
                    if index < 0 or (total_frames > 0 and index >= total_frames):
                        raise ValueError(
                            f"Frame index out of range for video_path={path}, "
                            f"frame_index={index}, total_frames={total_frames}"
                        )
                return [
                    self._read_frame_at(capture, cv2, index, path)
                    for index in indices
                ]

            if sample_num is not None:
                sample_num = int(sample_num)
                if sample_num <= 0:
                    raise ValueError(f"sample_num must be positive, got {sample_num}")
                if total_frames <= 0:
                    raise ValueError(
                        f"Invalid video metadata for {path}: total_frames={total_frames}"
                    )
                if sample_num > total_frames:
                    sample_num = total_frames
                picked = np.linspace(0, total_frames - 1, sample_num)
                indices = sorted({int(round(float(value))) for value in picked})
                return [
                    self._read_frame_at(capture, cv2, index, path)
                    for index in indices
                ]

            # All frames: sequential grab/retrieve, aligned with the per-frame output of ray.data.read_videos
            frames = []
            while True:
                ok = capture.grab()
                if not ok:
                    break
                ok, frame_bgr = capture.retrieve()
                if not ok or frame_bgr is None:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            return frames
        finally:
            capture.release()

    @staticmethod
    def _read_frame_at(capture, cv2, index: int, path: str) -> np.ndarray:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame_bgr = capture.read()
        if not success or frame_bgr is None:
            raise ValueError(
                f"Failed to read frame for video_path={path}, frame_index={index}"
            )
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ── Audio ─────────────────────────────────────────────────────────────────

    def load_audio(self, path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        amplitude, sample_rate = self._read_audio(str(path))
        if sr is not None and int(sr) != sample_rate:
            try:
                import librosa
            except ImportError as exc:
                raise ImportError(
                    "librosa is required for audio resampling. Please install librosa."
                ) from exc
            amplitude = librosa.resample(
                amplitude.T, orig_sr=sample_rate, target_sr=int(sr)
            ).T
            sample_rate = int(sr)
        return amplitude, sample_rate

    @staticmethod
    def _read_audio(path: str) -> Tuple[np.ndarray, int]:
        """Read audio using the same source as ray.data.read_audio (soundfile, float32, (channels, samples))."""
        try:
            import soundfile
        except ImportError:
            return DefaultMediaBackend._read_audio_librosa(path)

        amplitude, sample_rate = soundfile.read(path, always_2d=True, dtype="float32")
        # (amplitude, channels) -> (channels, amplitude)
        return amplitude.transpose((1, 0)), int(sample_rate)

    @staticmethod
    def _read_audio_librosa(path: str) -> Tuple[np.ndarray, int]:
        try:
            import librosa
        except ImportError as exc:
            raise ImportError(
                "soundfile or librosa is required for audio loading. "
                "Please install soundfile (preferred) or librosa."
            ) from exc
        amplitude, sample_rate = librosa.load(path, sr=None, mono=False, dtype=np.float32)
        if amplitude.ndim == 1:
            amplitude = amplitude[np.newaxis, :]
        return amplitude, int(sample_rate)

    def load_audio_batch(
        self, paths: Sequence[str], sr: Optional[int] = None
    ) -> List[Tuple[np.ndarray, int]]:
        return [self.load_audio(path, sr=sr) for path in paths]


class MMSDKMediaBackend:
    """MultimodalSDK acceleration backend (the only genuinely new logic).

    - Each method first checks the format gate; if outside the mm support matrix, it directly delegates to the default implementation;
    - By default, mm runtime exceptions fall back per item to the default implementation (``allow_fallback=True``, auto mode);
      in forced ``mm_sdk`` mode it is constructed with ``allow_fallback=False``, and exceptions are re-raised unchanged;
    - All ``import mm`` statements are deferred until method calls.
    """

    name = "mm_sdk"

    IMAGE_EXTS = (".jpg", ".jpeg")
    VIDEO_EXTS = (".mp4",)
    AUDIO_EXTS = (".wav",)

    def __init__(
        self,
        default_backend: Optional[DefaultMediaBackend] = None,
        allow_fallback: bool = True,
        fallback_stats_file: Optional[str] = None,
    ):
        self._default = default_backend if default_backend is not None else DefaultMediaBackend()
        self._allow_fallback = allow_fallback
        # Freeze the aggregate file path at construction time: Ray workers do not inherit driver environment variables, and
        # the backend instance resolved on the driver side is pickled into the worker through a closure; instance attributes
        # are the only deterministic propagation path (environment variables are only a configuration entry point at construction time).
        self._fallback_stats_file = fallback_stats_file or os.environ.get(FALLBACK_STATS_ENV)
        self._mm = None
        self.fallback_count = 0

    # ── Internal utilities ────────────────────────────────────────────────────

    def _get_mm(self):
        if self._mm is None:
            try:
                import mm
            except Exception as exc:  # import mm failure may be due to missing CANN / dependency conflicts
                raise BackendUnavailableError(
                    f"MultimodalSDK (mm) import failed: {exc}"
                ) from exc
            self._mm = mm
        return self._mm

    def _fallback(self, operation: str, path: str, exc: Exception) -> None:
        """Per-item fallback decision: in auto mode, log DEBUG and delegate to the default implementation; in forced mode, re-raise."""
        if not self._allow_fallback:
            if isinstance(exc, BackendUnavailableError):
                raise exc
            raise BackendUnavailableError(
                f"mm {operation} failed for {path}: {exc}"
            ) from exc
        self.fallback_count += 1
        _record_fallback_event(operation, path, exc, sink=self._fallback_stats_file)
        logger.debug(
            "mm %s failed for %s, falling back to default backend: %s",
            operation, path, exc,
        )

    # ── Images ────────────────────────────────────────────────────────────────

    def decode_image(self, path: str) -> np.ndarray:
        if not str(path).lower().endswith(self.IMAGE_EXTS):
            return self._default.decode_image(path)
        try:
            mm = self._get_mm()
            result = mm.Image.open(str(path)).numpy()
        except Exception as exc:
            self._fallback("decode_image", path, exc)
            return self._default.decode_image(path)
        return self._as_rgb_image(result)

    def resize_image(
        self,
        img: np.ndarray,
        size: Tuple[int, int],
        interpolation: str = "bicubic",
    ) -> np.ndarray:
        if str(interpolation).lower() != "bicubic":
            # The mm kernel supports only BICUBIC
            return self._default.resize_image(img, size, interpolation)
        try:
            mm = self._get_mm()
            if hasattr(mm.Image, "from_numpy"):
                image = mm.Image.from_numpy(np.asarray(img))
            else:
                image = mm.Image(np.asarray(img))
            resized = image.resize((int(size[0]), int(size[1])), mm.Interpolation.BICUBIC)
            result = resized.numpy() if hasattr(resized, "numpy") else np.asarray(resized)
        except Exception as exc:
            self._fallback("resize_image", "<ndarray>", exc)
            return self._default.resize_image(img, size, interpolation)
        return self._as_rgb_image(result)

    # ── Video ─────────────────────────────────────────────────────────────────

    def decode_video_frames(
        self,
        path: str,
        sample_num: Optional[int] = None,
        frame_indices: Optional[Set[int]] = None,
    ) -> List[np.ndarray]:
        if not str(path).lower().endswith(self.VIDEO_EXTS):
            return self._default.decode_video_frames(
                path, sample_num=sample_num, frame_indices=frame_indices
            )
        try:
            mm = self._get_mm()
            # mm hard constraint: resolution must be in [480x480, 4096x4096] (CheckVideoResolution
            # rejects smaller videos, such as 32x24 → per-item fallback to cv2); encoding must be H.264 (mp4v is unsupported)
            kwargs = {}
            if frame_indices is not None:
                # mm requires a set type, and the returned order matches the input index set (pass after sorting)
                kwargs["frame_indices"] = {int(index) for index in frame_indices}
            elif sample_num is not None:
                kwargs["sample_num"] = int(sample_num)
            else:
                # mm.video_decode must explicitly provide frame indices or a sample count (CheckTargetFrameIndices
                # validation), full decoding must first obtain the total frame count and then pass the full set of frame indices
                total = self._video_frame_count(path)
                kwargs["frame_indices"] = set(range(total))
            decoded = mm.video_decode(str(path), "cpu", **kwargs)
            frames = [
                item.numpy() if hasattr(item, "numpy") else np.asarray(item)
                for item in decoded
            ]
        except Exception as exc:
            self._fallback("decode_video_frames", path, exc)
            return self._default.decode_video_frames(
                path, sample_num=sample_num, frame_indices=frame_indices
            )
        return [self._as_rgb_image(frame) for frame in frames]

    @staticmethod
    def _video_frame_count(path: str) -> int:
        """Probe the total frame count with cv2 (mm.video_decode full decoding requires an explicit full frame-index set)."""
        from .lazy_imports import get_cv2_backend

        cv2 = get_cv2_backend().get_cv2()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if count <= 0:
            raise ValueError(f"Cannot determine frame count: {path}")
        return count

    # ── Audio ─────────────────────────────────────────────────────────────────

    def load_audio(self, path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        results = self.load_audio_batch([path], sr=sr)
        return results[0]

    def load_audio_batch(
        self, paths: Sequence[str], sr: Optional[int] = None
    ) -> List[Tuple[np.ndarray, int]]:
        paths = [str(path) for path in paths]
        wav_indices = [
            index for index, path in enumerate(paths)
            if path.lower().endswith(self.AUDIO_EXTS)
        ]
        if not wav_indices:
            return self._default.load_audio_batch(paths, sr=sr)

        results: List[Optional[Tuple[np.ndarray, int]]] = [None] * len(paths)
        wav_paths = [paths[index] for index in wav_indices]

        try:
            mm = self._get_mm()
            mm_results = []
            # mm.load_audio has a per-batch limit of 128 files; process the remainder in chunks
            for start in range(0, len(wav_paths), MM_AUDIO_BATCH_LIMIT):
                chunk = wav_paths[start:start + MM_AUDIO_BATCH_LIMIT]
                mm_results.extend(mm.load_audio(chunk, sr=sr))
            for index, item in zip(wav_indices, mm_results):
                tensor, sample_rate = item
                amplitude = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
                amplitude = np.asarray(amplitude, dtype=np.float32)
                if amplitude.ndim == 1:
                    amplitude = amplitude[np.newaxis, :]
                results[index] = (amplitude, int(sample_rate))
        except Exception as exc:
            self._fallback("load_audio_batch", paths[0] if paths else "<empty>", exc)
            return self._default.load_audio_batch(paths, sr=sr)

        # Non-WAV entries go through the default implementation one by one
        for index, path in enumerate(paths):
            if results[index] is None:
                results[index] = self._default.load_audio(path, sr=sr)
        return results  # type: ignore[return-value]

    # ── Output alignment ──────────────────────────────────────────────────────

    @staticmethod
    def _as_rgb_image(result) -> np.ndarray:
        """Align output dtype/shape with the default implementation: uint8, (H, W, 3)."""
        array = np.asarray(result)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)


# ─────────────────────────────────────────────────────────────────────────────
# Capability probing and singleton selector
# ─────────────────────────────────────────────────────────────────────────────

_smoke_probe_script = """
import sys

jpg_path, wav_path = sys.argv[1], sys.argv[2]

try:
    import mm
    image = mm.Image.open(jpg_path)
    if image.numpy() is None:
        raise RuntimeError("mm image decode returned None")
    audio = mm.load_audio([wav_path], sr=None)
    if not audio or len(audio) != 1:
        raise RuntimeError("mm audio load returned empty result")
except BaseException:
    # Native-layer segfaults cannot be caught (the process crashes directly); Python-layer exceptions are also treated as probe failure.
    sys.exit(1)
"""


_probe_lock = threading.Lock()
_probe_cache = None

_selection_lock = threading.Lock()
_media_backend_singleton = None
_configured_backend = None


def set_configured_media_backend(value: str) -> None:
    """Inject the GlobalConfig.media_backend configuration value (called during Pipeline initialization).

    Valid values are listed in ``VALID_MEDIA_BACKENDS``; configuration changes invalidate the cached selector.
    """
    global _configured_backend, _media_backend_singleton
    if value is None:
        value = "auto"
    if value not in VALID_MEDIA_BACKENDS:
        raise ValueError(
            f"Invalid media_backend config: {value!r}. Must be one of {VALID_MEDIA_BACKENDS}"
        )
    with _selection_lock:
        _configured_backend = value
        _media_backend_singleton = None


def _resolve_preference() -> str:
    env_value = os.environ.get(_ENV_MEDIA_BACKEND)
    if env_value:
        if env_value not in VALID_MEDIA_BACKENDS:
            raise ValueError(
                f"Invalid {_ENV_MEDIA_BACKEND} env value: {env_value!r}. "
                f"Must be one of {VALID_MEDIA_BACKENDS}"
            )
        return env_value
    if _configured_backend:
        return _configured_backend
    return "auto"


def mm_sdk_available() -> bool:
    """MultimodalSDK smoke probe (result cached in-process).

    Probe steps: platform precheck (linux + aarch64) → ``find_spec('mm')`` →
    ``import mm`` in a subprocess and actually decode a tiny JPEG + WAV.
    Subprocess probing is required: crashes from ``RTLD_GLOBAL`` symbol conflicts occur at the native layer,
    and an in-process try/except cannot catch them.
    """
    global _probe_cache
    if _probe_cache is not None:
        return _probe_cache
    with _probe_lock:
        if _probe_cache is None:
            _probe_cache = _run_probe()
        return _probe_cache


def _run_probe() -> bool:
    if sys.platform != "linux":
        return False
    if platform.machine().lower() not in ("aarch64", "arm64"):
        return False
    try:
        if importlib.util.find_spec("mm") is None:
            return False
    except Exception:
        return False
    return _smoke_test_mm()


def _smoke_test_mm() -> bool:
    """Subprocess smoke test: actually import mm and decode a tiny JPEG + WAV."""
    with tempfile.TemporaryDirectory(prefix="adf_mm_probe_") as tmp_dir:
        try:
            jpg_path, wav_path = _create_probe_fixtures(tmp_dir)
        except Exception:
            # Probe fixture generation failure (such as missing Pillow) is treated as probe failure
            return False
        try:
            completed = subprocess.run(
                [sys.executable, "-c", _smoke_probe_script, jpg_path, wav_path],
                timeout=30,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return completed.returncode == 0
        except Exception:
            return False


def _create_probe_fixtures(tmp_dir: str) -> Tuple[str, str]:
    """Generate tiny JPEG (8x8) and WAV (16 samples) probe fixtures."""
    import struct
    import wave

    from PIL import Image

    jpg_path = os.path.join(tmp_dir, "probe.jpg")
    # mm hard constraint: width must be in [10, 8192] (CheckImSize rejects smaller images); use 16x16
    Image.new("RGB", (16, 16), color=(127, 127, 127)).save(jpg_path, format="JPEG")

    wav_path = os.path.join(tmp_dir, "probe.wav")
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"".join(
            struct.pack("<h", 0) for _ in range(16)
        ))
    # The mm C++ safety policy requires the file other permission bits to be 0 (measured CheckFilePermission
    # rejects other-readable files); the fixture default 644 would cause probing to report failure in a usable aarch64 environment
    # so all are tightened to 600
    os.chmod(jpg_path, 0o600)
    os.chmod(wav_path, 0o600)
    return jpg_path, wav_path


def get_media_backend() -> MediaBackend:
    """Get the media processing backend singleton (thread-safe double-checked singleton).

    Priority: environment variable ``ADF_MEDIA_BACKEND`` > GlobalConfig.media_backend > 'auto'.

    - ``default``: always use the default implementation;
    - ``mm_sdk``: force mm; raise directly if probing fails (for users who pin acceleration);
    - ``auto``: use mm if probing passes; otherwise silently fall back to default (log only one INFO).
    """
    global _media_backend_singleton
    if _media_backend_singleton is None:
        with _selection_lock:
            if _media_backend_singleton is None:
                _media_backend_singleton = _select_media_backend()
    return _media_backend_singleton


def _select_media_backend() -> MediaBackend:
    preference = _resolve_preference()

    if preference == "default":
        return DefaultMediaBackend()

    if preference == "mm_sdk":
        if not mm_sdk_available():
            raise RuntimeError(
                "ADF media backend is pinned to 'mm_sdk' but MultimodalSDK probe "
                "failed (requires linux-aarch64, mm installed and CANN env sourced)."
            )
        logger.info("Using MultimodalSDK media backend (pinned).")
        return MMSDKMediaBackend(allow_fallback=False)

    # auto
    if mm_sdk_available():
        logger.info("Using MultimodalSDK media backend (auto).")
        return MMSDKMediaBackend(allow_fallback=True)
    logger.info("MultimodalSDK not available, using default media backend")
    return DefaultMediaBackend()


def reset_media_backend_cache() -> None:
    """Reset probe and selector caches (for tests only)."""
    global _probe_cache, _media_backend_singleton, _configured_backend
    with _probe_lock:
        _probe_cache = None
    with _selection_lock:
        _media_backend_singleton = None
        _configured_backend = None


# ── Cross-process fallback observability ─────────────────────────────────────

FALLBACK_STATS_ENV = "ADF_MM_FALLBACK_STATS_FILE"


def _record_fallback_event(operation: str, path: str, exc: Exception,
                           sink: Optional[str] = None) -> None:
    """Append each fallback event as a single-line JSON record to the aggregate file (cross-process fallback observability).

    Prefer the ``sink`` path frozen at construction time by the caller (the backend instance); it is pickled into the
    Ray worker with the instance and does not depend on environment-variable inheritance. Otherwise fall back to ``ADF_MM_FALLBACK_STATS_FILE``.
    Under O_APPEND, a small single write is atomic on Linux, so multiple processes writing to the same file complete the
    aggregation; the read side uses :func:`read_mm_fallback_stats`. Observability is a side channel: if there is no sink or the write
    fails, only log DEBUG and do not affect the main decoding path.
    """
    sink = sink or os.environ.get(FALLBACK_STATS_ENV)
    if not sink:
        return
    event = {
        "pid": os.getpid(),
        "operation": operation,
        "path": str(path),
        "error": f"{type(exc).__name__}: {exc}"[:200],
    }
    try:
        fd = os.open(sink, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except OSError as os_exc:
        logger.debug("fallback stats sink write failed (%s): %s", sink, os_exc)


def read_mm_fallback_stats(path: Optional[str] = None) -> dict:
    """Aggregate and read fallback events: ``{'total', 'by_operation', 'pids', 'events'}``.

    ``path`` defaults to the environment variable ``ADF_MM_FALLBACK_STATS_FILE``. A multi-machine Ray cluster
    requires a shared filesystem; a single machine (the current deployment shape of this framework) satisfies this naturally.
    """
    sink = path or os.environ.get(FALLBACK_STATS_ENV)
    events: list = []
    if sink and os.path.exists(sink):
        with open(sink, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    by_operation: dict = {}
    for event in events:
        key = str(event.get("operation", "?"))
        by_operation[key] = by_operation.get(key, 0) + 1
    return {
        "total": len(events),
        "by_operation": by_operation,
        "pids": sorted({int(e["pid"]) for e in events if "pid" in e}),
        "events": events,
    }


def reset_mm_fallback_stats(path: Optional[str] = None) -> None:
    """Clear the fallback event file (the parent process calls this between two pipelines to isolate statistics)."""
    sink = path or os.environ.get(FALLBACK_STATS_ENV)
    if sink and os.path.exists(sink):
        os.remove(sink)
