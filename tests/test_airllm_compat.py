"""Tests for the AirLLM optimum.bettertransformer compatibility shim."""
import sys

from src.runners.airllm_compat import (
    _NoOpBetterTransformer,
    install_bettertransformer_stub,
)


def test_noop_transform_returns_model_unchanged():
    model = object()
    assert _NoOpBetterTransformer.transform(model) is model
    assert _NoOpBetterTransformer.reverse(model) is model


def test_install_stub_registers_module(monkeypatch):
    # Ensure no real/previous module is present, then install the stub.
    monkeypatch.delitem(sys.modules, "optimum.bettertransformer", raising=False)

    real = type(sys)("optimum")  # minimal fake parent without the submodule
    monkeypatch.setitem(sys.modules, "optimum", real)

    installed = install_bettertransformer_stub()
    assert installed is True
    from optimum.bettertransformer import BetterTransformer

    m = object()
    assert BetterTransformer.transform(m) is m


def test_install_stub_noop_when_module_exists():
    # After the previous registration (or a real install), a second call is a no-op.
    install_bettertransformer_stub()
    assert install_bettertransformer_stub() is False
