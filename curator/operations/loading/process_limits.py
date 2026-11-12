"""Best-effort resource limits for spawned parsing and OCR children."""
from __future__ import annotations

import math
import os
import sys
from typing import Any

_WINDOWS_JOB_HANDLE: Any = None


def _apply_windows_job_limit(max_memory_mb: int) -> None:
    """Best-effort Windows Job Object memory cap for a child process.

    Hosted environments may already place the child in a non-breakaway job;
    assignment then fails. In that case the host container/cgroup remains the
    enforcement boundary, and this helper deliberately does not claim a cap.
    """
    global _WINDOWS_JOB_HANDLE
    try:
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = 0x00000100  # PROCESS_MEMORY
        info.ProcessMemoryLimit = max_memory_mb * 1024 * 1024
        if not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            kernel32.CloseHandle(job)
            return
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return
        _WINDOWS_JOB_HANDLE = job
    except Exception:
        return


def apply_child_resource_limits(
    *,
    max_memory_mb: int,
    max_file_mb: float,
    cpu_timeout_s: float,
) -> None:
    """Apply memory, CPU and output-file limits to the current process."""
    if os.name == "posix":
        try:
            import resource

            budget = int(max_memory_mb) * 1024 * 1024
            # RLIMIT_AS also counts reserved virtual mappings from libraries
            # such as openpyxl/OpenBLAS and can reject a healthy parser before
            # it reads any input. On Linux, cap the additional mutable data
            # mappings relative to the already-imported interpreter baseline.
            # Other POSIX platforms keep the absolute RLIMIT_AS fallback.
            data_limit = getattr(resource, "RLIMIT_DATA", None)
            baseline = 0
            if sys.platform.startswith("linux") and data_limit is not None:
                try:
                    with open("/proc/self/status", "r", encoding="ascii") as handle:
                        for line in handle:
                            if line.startswith("VmData:"):
                                baseline = int(line.split()[1]) * 1024
                                break
                except (OSError, ValueError):
                    baseline = 0
            if baseline > 0:
                limit_value = baseline + budget
                resource.setrlimit(data_limit, (limit_value, limit_value))
            else:
                limit_kind = data_limit if data_limit is not None else resource.RLIMIT_AS
                resource.setrlimit(limit_kind, (budget, budget))
        except Exception:
            pass
        try:
            import resource

            cpu_seconds = max(1, int(math.ceil(cpu_timeout_s)))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except Exception:
            pass
        try:
            import resource

            max_write = int(max_file_mb * 1024 * 1024) + 64 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_write, max_write))
        except Exception:
            pass
    elif os.name == "nt":
        _apply_windows_job_limit(int(max_memory_mb))


__all__ = ["apply_child_resource_limits"]
