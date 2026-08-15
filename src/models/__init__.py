"""Model adapters.

One adapter per model, shared by every reconstruction strategy.  Adapters are imported
lazily by `load_adapters` so that a JAX-only run never imports Torch and vice versa.
"""

from .base import (AdapterSpec, Conditioning, MeanFlowAdapter, ModelAdapter, ModelManager,
                   RepoSandbox, StandardFlowAdapter, ADAPTER_FACTORIES, build_initial_state,
                   register_adapter)

__all__ = ["AdapterSpec", "Conditioning", "MeanFlowAdapter", "ModelAdapter", "ModelManager",
           "RepoSandbox", "StandardFlowAdapter", "ADAPTER_FACTORIES", "build_initial_state",
           "register_adapter", "load_adapters"]


def load_adapters(models):
    """Import only the adapter modules a plan actually needs."""
    for name in models:
        if name == "jit":
            from . import jit             # noqa: F401
        elif name == "pmf":
            from . import pmf             # noqa: F401
        elif name == "sit":
            from . import sit             # noqa: F401
        elif name == "imf":
            from . import imf             # noqa: F401
        else:
            raise KeyError("No adapter module for model %r" % name)
    return ADAPTER_FACTORIES
