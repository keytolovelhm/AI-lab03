"""Stage-3 (refinement_v3) engineering constants and small IO helpers.

Read-only rules enforced by callers (see __main__.check_protected):
  * src/, results/, checkpoints/, configs/, data/, requirements.txt  -> stage-1
  * refinement/                                                     -> stage-2
Everything written by stage 3 lives under refinement_v3/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

V3 = Path(__file__).resolve().parent
ROOT = V3.parent
RUNS = V3 / "runs"
RESULTS = V3 / "results"
CONFIGS = V3 / "configs"

# Stage-1 / stage-2 trees that stage 3 must never write to.
PROTECTED_DIRS = ("src", "results", "checkpoints", "configs", "data", "refinement")
PROTECTED_FILES = ("requirements.txt",)

STAGE1_RESULTS = ROOT / "results"
STAGE1_CHECKPOINTS = ROOT / "checkpoints"
STAGE2_SUITE = ROOT / "refinement" / "runs" / "suite"
DATA_DIR = ROOT / "data"

SEEDS = (20261105, 20261106, 20261107)
# Independent random streams per seed, identical to stage 2 so same-seed initialisation lines up.
STREAM_OFFSETS = {"order": 10_000, "crop": 20_000, "flip": 30_000, "augment": 40_000, "mix": 50_000}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows, fieldnames=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty csv: {path}")
        fieldnames = list(rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def frozen_json(path: Path, value) -> None:
    """Write once; later calls must agree exactly (guards against silent re-configuration)."""
    path = Path(path)
    if path.exists():
        existing = read_json(path)
        if existing != value:
            raise ValueError(f"frozen metadata differs: {path}")
        return
    write_json(path, value)


def save_checkpoint(path: Path, state) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def peak_memory_mb() -> float:
    """Peak working set of this process (MB); 0.0 when the platform API is unavailable."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernels = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        process = kernels.GetCurrentProcess()
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return round(counters.PeakWorkingSetSize / 1e6, 1) if ok else 0.0
    except Exception:  # pragma: no cover - platform dependent
        return 0.0


def system_memory_gb():
    """(total, available) physical memory in GB, or (0, 0) when unavailable."""
    try:
        import ctypes
        from ctypes import wintypes

        class Status(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                        ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                        ("total_virt", ctypes.c_ulonglong), ("avail_virt", ctypes.c_ulonglong),
                        ("avail_ext", ctypes.c_ulonglong)]

        status = Status()
        status.length = ctypes.sizeof(Status)
        ok = ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(status))
        return (status.total / 1e9, status.avail / 1e9) if ok else (0.0, 0.0)
    except Exception:  # pragma: no cover
        return 0.0, 0.0
