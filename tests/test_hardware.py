"""Tests for src/hardware/profiler.py."""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.hardware.profiler import (
    DiskSpaceError,
    HardwareProfile,
    check_disk_space,
    profile_hardware,
    save_hardware_profile,
)


def _fake_profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_model="Test CPU",
        cpu_cores=8,
        cpu_threads=16,
        ram_gb=32.0,
        gpu_model="Test GPU",
        vram_gb=24.0,
        storage_type="NVMe/SSD",
        free_disk_gb=500.0,
    )


def test_save_and_reload(tmp_path):
    profile = _fake_profile()
    out = tmp_path / "hardware.json"
    save_hardware_profile(profile, str(out))
    data = json.loads(out.read_text())
    assert data["cpu_model"] == "Test CPU"
    assert data["vram_gb"] == 24.0
    assert data["free_disk_gb"] == 500.0


def test_save_creates_parent_dirs(tmp_path):
    profile = _fake_profile()
    out = tmp_path / "nested" / "dir" / "hw.json"
    save_hardware_profile(profile, str(out))
    assert out.exists()


def test_check_disk_space_passes(tmp_path):
    with patch("src.hardware.profiler.shutil.disk_usage", return_value=MagicMock(free=int(200e9))):
        check_disk_space(str(tmp_path), required_gb=70.0)  # should not raise


def test_check_disk_space_fails(tmp_path):
    with patch("src.hardware.profiler.shutil.disk_usage", return_value=MagicMock(free=int(10e9))):
        with pytest.raises(DiskSpaceError, match="70.0 GB"):
            check_disk_space(str(tmp_path), required_gb=70.0)


def test_profile_hardware_no_gpu(tmp_path):
    with (
        patch("platform.processor", return_value="Intel Core i7"),
        patch("psutil.cpu_count", side_effect=[8, 16]),
        patch(
            "psutil.virtual_memory",
            return_value=MagicMock(total=int(32e9)),
        ),
        patch(
            "shutil.disk_usage",
            return_value=MagicMock(free=int(500e9)),
        ),
        patch("src.hardware.profiler._get_gpu_info", return_value=("none", 0.0)),
    ):
        profile = profile_hardware(str(tmp_path))
        assert profile.cpu_cores == 8
        assert profile.cpu_threads == 16
        assert profile.gpu_model == "none"
        assert profile.vram_gb == 0.0
        assert profile.free_disk_gb > 0
