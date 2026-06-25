"""Hardware profiling and disk-space validation."""
from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


class DiskSpaceError(RuntimeError):
    """Raised when a path does not have enough free disk space."""


@dataclass
class HardwareProfile:
    """Snapshot of the machine's hardware configuration."""

    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpu_model: str
    vram_gb: float
    storage_type: str
    free_disk_gb: float


def profile_hardware(shard_path: str = "") -> HardwareProfile:
    """Collect hardware specs for the current machine."""
    cpu_model = platform.processor() or "unknown"
    cpu_cores = psutil.cpu_count(logical=False) or 0
    cpu_threads = psutil.cpu_count(logical=True) or 0
    ram_gb = round(psutil.virtual_memory().total / 1e9, 1)

    gpu_model, vram_gb = _get_gpu_info()

    free_disk_gb = 0.0
    storage_type = "unknown"
    try:
        usage = shutil.disk_usage(shard_path or ".")
        free_disk_gb = round(usage.free / 1e9, 1)
        storage_type = "NVMe/SSD"
    except OSError:
        pass

    return HardwareProfile(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        gpu_model=gpu_model,
        vram_gb=vram_gb,
        storage_type=storage_type,
        free_disk_gb=free_disk_gb,
    )


def save_hardware_profile(profile: HardwareProfile, path: str) -> None:
    """Write the hardware profile as JSON to path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(profile), indent=2))


def check_disk_space(shard_path: str, required_gb: float) -> None:
    """Raise DiskSpaceError if shard_path has less than required_gb free."""
    usage = shutil.disk_usage(shard_path)
    free_gb = usage.free / 1e9
    if free_gb < required_gb:
        raise DiskSpaceError(
            f"Need {required_gb:.1f} GB free at '{shard_path}', "
            f"but only {free_gb:.1f} GB available."
        )


def _get_gpu_info() -> tuple[str, float]:
    """Return (gpu_model, vram_gb); falls back to ('none', 0.0) if no CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            return name, vram
    except Exception:
        pass
    return "none", 0.0
