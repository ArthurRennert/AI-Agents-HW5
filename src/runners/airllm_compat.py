"""Compatibility shim so AirLLM imports under optimum >= 2.0.

AirLLM 2.11 still does `from optimum.bettertransformer import BetterTransformer`
at import time, but optimum removed that module in v2 (it was an optional
flash-attention path, irrelevant on CPU). We register a no-op stub in
sys.modules *before* AirLLM is imported, so the import succeeds and any call to
`BetterTransformer.transform(model)` simply returns the model unchanged.

Import side effect: calling `install_bettertransformer_stub()` must happen before
`from airllm import AutoModel`.
"""
from __future__ import annotations

import sys
import types


class _NoOpBetterTransformer:
    """Drop-in replacement: every method returns the model unchanged."""

    @staticmethod
    def transform(model, *args, **kwargs):
        """Return the model untouched (no flash-attention rewrite)."""
        return model

    @staticmethod
    def reverse(model, *args, **kwargs):
        """Return the model untouched."""
        return model


def install_bettertransformer_stub() -> bool:
    """Register a stub optimum.bettertransformer if the real one is unavailable.

    Returns True if a stub was installed, False if the real module already exists.
    """
    try:
        import optimum.bettertransformer  # noqa: F401

        return False  # real module present; nothing to do
    except Exception:
        pass

    module = types.ModuleType("optimum.bettertransformer")
    module.BetterTransformer = _NoOpBetterTransformer
    sys.modules["optimum.bettertransformer"] = module
    return True
