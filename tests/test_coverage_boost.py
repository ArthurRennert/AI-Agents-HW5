"""Extra tests to cover GPU-adjacent and edge-case paths without real hardware."""
import time
from unittest.mock import MagicMock, patch

import pytest

# ── persistence: overwrite warning path ──────────────────────────────────────

from src.benchmark.harness import Metrics
from src.benchmark.persistence import OutputInfo, ResultRecord, ScenarioInfo, save_result


def _record(engine: str = "airllm", quant: str = "fp16") -> ResultRecord:
    return ResultRecord(
        scenario=ScenarioInfo(
            engine=engine, model="m", quant_level=quant,
            prompt_tokens=5, max_new_tokens=10, seed=42,
        ),
        metrics=Metrics(1, 2, 3, 4, 5, 6, 7),
        output=OutputInfo("hi", 10, "coherent"),
    )


def test_overwrite_warning_is_printed(tmp_path, capsys):
    save_result(_record(), str(tmp_path))
    save_result(_record(), str(tmp_path))  # second save triggers warning
    out = capsys.readouterr().out
    assert "Warning" in out or "overwriting" in out.lower()


# ── harness: power sampling success path ─────────────────────────────────────

from src.benchmark.harness import Harness


def test_power_sampling_succeeds_with_mock_nvidia_smi():
    mock_result = MagicMock()
    mock_result.stdout = "250.0\n"

    with patch("subprocess.run", return_value=mock_result):
        harness = Harness()
        harness.start()
        time.sleep(0.15)  # allow at least one power sample
        harness.record_first_token()
        metrics = harness.stop(n_output_tokens=5)

    # if a sample was collected the energy should be > 0
    assert metrics.energy_wh >= 0.0


def test_power_sampling_ignores_nvidia_smi_failure():
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        harness = Harness()
        harness.start()
        time.sleep(0.05)
        harness.record_first_token()
        metrics = harness.stop(n_output_tokens=3)
    assert metrics.energy_wh == 0.0  # no samples → no energy


# ── hardware: GPU info paths (torch imported lazily inside _get_gpu_info) ─────

from src.hardware.profiler import _get_gpu_info


def test_get_gpu_info_with_cuda_available():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 3090"
    mock_props = MagicMock()
    mock_props.total_memory = int(24e9)
    mock_torch.cuda.get_device_properties.return_value = mock_props

    # _get_gpu_info imports torch lazily inside the function; patch at sys.modules level
    with patch.dict("sys.modules", {"torch": mock_torch}):
        name, vram = _get_gpu_info()

    assert name == "NVIDIA RTX 3090"
    assert abs(vram - 24.0) < 0.5


def test_get_gpu_info_cuda_not_available():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        name, vram = _get_gpu_info()

    assert name == "none"
    assert vram == 0.0


def test_get_gpu_info_torch_raises():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA init error")

    with patch.dict("sys.modules", {"torch": mock_torch}):
        name, vram = _get_gpu_info()

    assert name == "none"
    assert vram == 0.0


# ── quantization: bnb config paths (torch + transformers imported lazily) ─────

from src.quantization.levels import get_bnb_config


def test_get_bnb_config_q8_calls_load_in_8bit():
    mock_bnb_cls = MagicMock()
    mock_transformers = MagicMock()
    mock_transformers.BitsAndBytesConfig = mock_bnb_cls
    mock_torch = MagicMock()
    mock_torch.float16 = "float16"

    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
        get_bnb_config("q8")

    mock_bnb_cls.assert_called_once_with(load_in_8bit=True)


def test_get_bnb_config_q4_uses_nf4():
    mock_bnb_cls = MagicMock()
    mock_transformers = MagicMock()
    mock_transformers.BitsAndBytesConfig = mock_bnb_cls
    mock_torch = MagicMock()
    mock_torch.float16 = "float16"

    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
        get_bnb_config("q4")

    call_kwargs = mock_bnb_cls.call_args.kwargs
    assert call_kwargs.get("load_in_4bit") is True
    assert call_kwargs.get("bnb_4bit_quant_type") == "nf4"
    assert call_kwargs.get("bnb_4bit_use_double_quant") is True


def test_get_bnb_config_q2_also_uses_4bit():
    mock_bnb_cls = MagicMock()
    mock_transformers = MagicMock()
    mock_transformers.BitsAndBytesConfig = mock_bnb_cls
    mock_torch = MagicMock()
    mock_torch.float16 = "float16"

    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
        get_bnb_config("q2")

    call_kwargs = mock_bnb_cls.call_args.kwargs
    assert call_kwargs.get("load_in_4bit") is True
