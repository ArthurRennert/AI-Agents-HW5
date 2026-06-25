"""Profile this machine and write results/hardware.json (Task: Phase 1).

Run this on the actual target machine so the hardware profile is accurate — the
model-selection logic and Extension A read RAM from it.

Run from the project root:
    python experiments/profile_machine.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware.profiler import profile_hardware, save_hardware_profile

RESULTS_DIR = "results"


def main() -> None:
    profile = profile_hardware(shard_path=os.getenv("SHARD_PATH", ""))
    save_hardware_profile(profile, str(Path(RESULTS_DIR) / "hardware.json"))
    print("[profile] Wrote results/hardware.json")
    print(f"  CPU      : {profile.cpu_model} ({profile.cpu_cores}c/{profile.cpu_threads}t)")
    print(f"  RAM      : {profile.ram_gb} GB")
    print(f"  GPU      : {profile.gpu_model} ({profile.vram_gb} GB VRAM)")
    print(f"  Storage  : {profile.storage_type}, {profile.free_disk_gb} GB free")


if __name__ == "__main__":
    main()
