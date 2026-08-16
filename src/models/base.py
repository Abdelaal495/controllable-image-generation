"""The single model abstraction used by SDEdit, MPC-RHC and MPC-delta_t alike.

Above this boundary nothing is framework-specific.  Below it, each adapter keeps whatever
layout, dtype, latent normalisation, guidance rule and time direction its checkpoint needs.

    to_native_noise(noise)                  canonical NumPy N(0,I) -> the adapter's array type
    prior_sample(image_ids)                 deterministic model-native epsilon, ONE per image
    encode_pixels(pixels)                   canonical (N,256,256,3) guide -> native guide state
    initial_state(guide, t0, eps)           (1-t0) * guide + t0 * eps, natively
    to_pixels(state, differentiable=False)  native -> canonical NumPy, for display/metrics
    to_pixels(state, differentiable=True)   native -> canonical BHWC, graph intact
    velocity(state, s, cond)                standard-flow adapters -- v_theta on clock s
    transition(state, s_from, s_to, cond)   MeanFlow adapters     -- T_theta on clock s
    release()

The differentiable pixel map is the load-bearing piece: it is what lets ONE measurement
operator and ONE terminal objective serve latent and pixel models, in JAX and in PyTorch,
with no detach / NumPy round trip inside an MPC objective.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import (Backend, JaxBackend, TorchBackend, canonical_start_time, free_memory,
                     gaussian_noise, native_time, pixel_fingerprint, prior_noise_parts,
                     timed)


@dataclass
class Conditioning:
    """What the model is conditioned on, plus its own guidance settings.

    SDEdit and MPC receive exactly the same object for a given image, so class conditioning
    can never differ between the methods being compared.
    """
    labels: np.ndarray                          # (N,) int32 ImageNet class ids
    guidance: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class AdapterSpec:
    """Everything shared code may need to know about a loaded model."""
    name: str
    display_name: str
    dynamics_family: str                        # STANDARD_FLOW | MEANFLOW
    framework: str                              # "jax" | "torch"
    state_space: str                            # "latent" | "pixel"
    native_shape: Tuple[int, ...]               # without batch, in native layout
    layout: str                                 # "BHWC" | "BCHW"
    pixel_resolution: int
    prediction_kind: str
    native_time_mapping: str
    batch_size: int
    fixed_batch_shape: bool
    num_classes: int
    null_label: int
    guidance: Dict[str, Any]
    checkpoint: Dict[str, Any]
    euler_final_step_for_heun: bool = False

    def describe(self) -> str:
        return ("%-4s | %-13s | %-5s | %-6s | native=%s %s | predicts=%-13s | batch=%d%s"
                % (self.name, self.dynamics_family, self.framework, self.state_space,
                   tuple(self.native_shape), self.layout, self.prediction_kind,
                   self.batch_size, " (fixed)" if self.fixed_batch_shape else ""))


class ModelAdapter(ABC):
    """Canonical interface over every supported model."""

    spec: AdapterSpec

    def __init__(self, name: str, registry: Dict[str, Any]):
        self.name = name
        self.registry = registry
        self.forward_counter = 0

    # -- required primitives ------------------------------------------------------------
    @abstractmethod
    def to_native_noise(self, noise: np.ndarray) -> Any:
        """Canonical NumPy N(0,I) of native shape -> the adapter's array type / layout."""

    @abstractmethod
    def encode_pixels(self, pixels: np.ndarray) -> Any:
        """Canonical (N,256,256,3) float32 [-1,1] BHWC -> model-native guide state.

        Used ONLY to build a t0 < 1 initial state.  The MPC objective never encodes an
        image: it compares A(decode(x)) against y.
        """

    @abstractmethod
    def to_pixels(self, state: Any, differentiable: bool = False) -> Any:
        """Native state -> canonical (N,256,256,3) BHWC.

        differentiable=False : NumPy float32, clipped to [-1,1]; display and metrics.
        differentiable=True  : the adapter's own array type, graph intact, NOT clipped.
        """

    @abstractmethod
    def _lerp(self, guide: Any, noise: Any, keep: float, add: float) -> Any:
        """keep * guide + add * noise, in the adapter's own array library."""

    # -- shared behaviour ---------------------------------------------------------------
    def native_batch_shape(self, batch: int) -> Tuple[int, ...]:
        return (batch,) + tuple(self.spec.native_shape)

    def prior_sample(self, image_ids: Sequence[Any], replicate: int = 0) -> Any:
        """Deterministic model-native epsilon ~ N(0, I), ONE draw per image id.

        Seeding per image -- and NOT per method, solver, step count, K, lambda, optimiser or
        t0 -- guarantees that image i always receives the same epsilon.  That is exactly the
        pairing property the SDEdit-vs-MPC comparison requires.
        """
        noise = np.stack([gaussian_noise(self.spec.native_shape,
                                         *prior_noise_parts(self.spec.name, i, replicate))
                          for i in image_ids], axis=0)
        return self.to_native_noise(noise)

    def initial_state(self, guide_native: Optional[Any], t0: float, noise_native: Any) -> Any:
        """z_t0 = (1 - t0) * guide + t0 * eps, in the model's native representation.

        t0 = 1 short-circuits to the prior-noise array ITSELF, so the initial state is
        bitwise identical to `prior_sample(...)` rather than merely equal to floating-point
        tolerance -- the notebooks' exact pure-noise behaviour.
        """
        t0 = float(t0)
        if not 0.0 < t0 <= 1.0:
            raise ValueError("t0 must lie in (0, 1], got %r" % (t0,))
        if t0 >= 1.0:
            return noise_native
        if guide_native is None:
            raise ValueError("t0 = %.4g < 1 requires an initialisation guide, but none was "
                             "provided to %s." % (t0, self.spec.name))
        return self._lerp(guide_native, noise_native, 1.0 - t0, t0)

    # Alias: the SDEdit notebook called this `corrupt`, the MPC notebook `initial_state`.
    def corrupt(self, guide_native: Any, corruption_strength: float, noise_native: Any) -> Any:
        return self.initial_state(guide_native, corruption_strength, noise_native)

    def sample_native_noise(self, batch: int, *seed_parts: Any) -> Any:
        """Deterministic native noise for `batch` items, from the canonical NumPy source."""
        return self.to_native_noise(gaussian_noise(self.native_batch_shape(batch), *seed_parts))

    def native_times(self, t0: float) -> Tuple[float, float]:
        """Corruption strength -> this model's native (start, end) times."""
        s_start = canonical_start_time(t0)
        return (native_time(s_start, self.spec.native_time_mapping),
                native_time(1.0, self.spec.native_time_mapping))

    def backend(self) -> Backend:
        """The operator backend matching this adapter's framework."""
        if self.spec.framework == "torch":
            import torch
            return TorchBackend(torch, self.device, self.integration_dtype)
        if self.spec.framework == "jax":
            import jax.numpy as jnp
            return JaxBackend(jnp)
        raise ValueError("No backend for framework %r" % self.spec.framework)

    def to_numpy(self, state) -> np.ndarray:
        """Native state -> NumPy, for fingerprints and assertions (never inside a graph)."""
        if self.spec.framework == "jax":
            import jax
            return np.asarray(jax.device_get(state), np.float32)
        return np.asarray(state.detach().float().cpu().numpy(), np.float32)

    def block(self, x):
        """Force asynchronous dispatch to complete, so timings are real."""
        if self.spec.framework == "jax":
            import jax
            return jax.block_until_ready(x)
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return x

    def reset_counters(self) -> None:
        self.forward_counter = 0

    def count_forwards(self, n: int = 1) -> None:
        self.forward_counter += int(n)

    def sanity_checks(self) -> Dict[str, Any]:
        """Model-specific structural checks, so the checker needs no `if model ==` chain."""
        return {}

    def release(self) -> None:
        for attr in ("_model", "_params", "_vae", "_vae_params", "_latent_manager",
                     "_jit_step", "_jit_encode", "_transformer", "_decode_diff",
                     "_jit_decode_diff"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        free_memory()

    def __repr__(self) -> str:
        return "<%s %s>" % (type(self).__name__, self.spec.describe())


class StandardFlowAdapter(ModelAdapter):
    """Adapters whose dynamics are an instantaneous velocity field (JiT, SiT)."""

    @abstractmethod
    def velocity(self, state: Any, s: float, conditioning: Conditioning) -> Any:
        """v_theta(x, s) on the CANONICAL clock.  Differentiable w.r.t. `state`.

        The adapter maps s to its native time and applies its own published guidance rule.
        """


class MeanFlowAdapter(ModelAdapter):
    """Adapters whose dynamics are a learned finite-interval transition (pMF, iMF).

    These are NOT instantaneous velocity models and must never be approximated as such:
    `transition` is the repositories' own `sample_one_step` on the reversed MeanFlow clock.
    """

    @abstractmethod
    def transition(self, state: Any, s_from: float, s_to: float,
                   conditioning: Conditioning) -> Any:
        """T_theta(x; s_from -> s_to) on the CANONICAL clock.  Differentiable w.r.t. `state`."""


# =====================================================================================
# Adapter registry
# =====================================================================================
ADAPTER_FACTORIES: Dict[str, Any] = {}
# Download-only hooks used by `run.py --prefetch`.  They must NOT build a model: they run on
# a cluster login node, which has internet but a hard CPU-minute and memory budget.
PREFETCH_HOOKS: Dict[str, Any] = {}


def register_adapter(name: str):
    def wrap(fn):
        ADAPTER_FACTORIES[name] = fn
        return fn
    return wrap


def register_prefetch(name: str):
    def wrap(fn):
        PREFETCH_HOOKS[name] = fn
        return fn
    return wrap


def download_and_extract_zip(hf_repo: str, filename: str, cache_dir: Path) -> Path:
    """Download a checkpoint archive from the Hub and extract it once (idempotent)."""
    import zipfile
    from huggingface_hub import hf_hub_download
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(repo_id=hf_repo, filename=filename, local_dir=str(cache))
    target = cache / Path(filename).stem
    target.mkdir(exist_ok=True)
    if not any(target.iterdir()):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target)
    return target


def find_checkpoint_dir(root: Path) -> str:
    """Locate the orbax/flax checkpoint directory inside an extracted archive."""
    for base, _dirs, files in os.walk(root):
        if "checkpoint" in files or any(f.endswith(".msgpack") for f in files):
            return base
    subdirs = [p for p in Path(root).iterdir() if p.is_dir()]
    return str(subdirs[0] if subdirs else root)


class RepoSandbox:
    """Isolate repositories that share top-level package names.

    iMF and pMF both ship top-level `configs`, `utils` and `models` packages, so they cannot
    simply share sys.path.  Carried over unchanged from the notebooks.
    """

    CONFLICTING_ROOTS = ("configs", "utils", "models")
    _snapshots: Dict[str, Dict[str, Any]] = {}
    _roots: Dict[str, Tuple[str, ...]] = {}
    _active: Optional[str] = None

    def __init__(self, name: str, path: Any, extra_roots: Sequence[str] = ()):
        import sys
        self.sys = sys
        self.name = name
        # ABSOLUTE: __enter__ chdirs here, so any relative path stored by a caller
        # would afterwards resolve against the repository directory instead of the
        # original working directory.
        self.path = str(Path(path).resolve())
        self.roots = tuple(self.CONFLICTING_ROOTS) + tuple(extra_roots)
        self._prev_cwd: Optional[str] = None
        self._depth = 0

    def _owned(self) -> Dict[str, Any]:
        return {k: v for k, v in list(self.sys.modules.items())
                if k.split(".")[0] in self.roots}

    def _purge(self, roots: Sequence[str]) -> None:
        for key in [k for k in list(self.sys.modules) if k.split(".")[0] in roots]:
            self.sys.modules.pop(key, None)

    def __enter__(self) -> "RepoSandbox":
        self._depth += 1
        if self._depth > 1:
            return self
        self._prev_cwd = os.getcwd()
        prev = RepoSandbox._active
        if prev is not None and prev != self.name:
            prev_roots = RepoSandbox._roots.get(prev, self.roots)
            RepoSandbox._snapshots[prev] = {
                k: v for k, v in list(self.sys.modules.items())
                if k.split(".")[0] in prev_roots}
        if prev != self.name:
            self._purge(self.roots)
            for k, v in RepoSandbox._snapshots.get(self.name, {}).items():
                self.sys.modules[k] = v
        while self.path in self.sys.path:
            self.sys.path.remove(self.path)
        self.sys.path.insert(0, self.path)
        os.chdir(self.path)
        RepoSandbox._active = self.name
        RepoSandbox._roots[self.name] = self.roots
        return self

    def __exit__(self, *exc) -> bool:
        self._depth -= 1
        if self._depth == 0:
            RepoSandbox._snapshots[self.name] = self._owned()
            if self._prev_cwd and os.path.isdir(self._prev_cwd):
                os.chdir(self._prev_cwd)
        return False


# =====================================================================================
# Model lifecycle
# =====================================================================================
class ModelManager:
    """Lazily loads adapters, keeps at most one resident when required, releases cleanly.

    JiT is Torch and pMF is JAX, so a mixed run loads them strictly one at a time: run all
    JiT jobs, release JiT, clear GPU memory, load pMF, run, release.
    """

    def __init__(self, config: Dict[str, Any], plan, release_after_use: bool,
                 context: Optional[Dict[str, Any]] = None):
        self.config = config
        self.plan = plan
        self.release_after_use = release_after_use
        self.context = context or {}
        self._adapters: Dict[str, ModelAdapter] = {}
        self._encode_cache: Dict[Tuple[Any, ...], Any] = {}
        self.provenance: Dict[str, Dict[str, Any]] = {}
        self.load_seconds: Dict[str, float] = {}

    def acquire(self, name: str) -> ModelAdapter:
        from ..config import resolve_model_registry
        import time as _time
        if name in self._adapters:
            return self._adapters[name]
        if self.release_after_use:
            for other in [k for k in list(self._adapters) if k != name]:
                self.release(other)
        registry = resolve_model_registry(self.config, name)
        factory = ADAPTER_FACTORIES.get(name)
        if factory is None:
            raise KeyError("No adapter factory registered for %r." % name)
        print("Loading %s ..." % name.upper())
        started = _time.perf_counter()
        with timed("manager/load/%s" % name):
            adapter = factory(registry, self.context)
        # Model loading is EXCLUDED from the reported reconstruction runtime and recorded
        # separately (brief section 32).
        self.load_seconds[name] = _time.perf_counter() - started
        self._adapters[name] = adapter
        self.provenance[name] = dict(adapter.spec.checkpoint,
                                     guidance=dict(adapter.spec.guidance),
                                     load_seconds=round(self.load_seconds[name], 2))
        print("   ", adapter.spec.describe())
        print("    checkpoint:", {k: v for k, v in adapter.spec.checkpoint.items()
                                  if k in ("step", "variant", "checkpoint_source",
                                           "checkpoint_mirror", "ema", "ema_key",
                                           "parameters")})
        print("    guidance  :", adapter.spec.guidance)
        return adapter

    def release(self, name: str) -> None:
        adapter = self._adapters.pop(name, None)
        if adapter is None:
            return
        print("Releasing %s ..." % name.upper())
        self.clear_encode_cache(model=name)
        try:
            adapter.release()
        except Exception as exc:                                        # pragma: no cover
            print("   release warning:", exc)
        del adapter
        free_memory(deep=True)      # the model is gone; its compiled executables can go too

    def release_all(self) -> None:
        for name in list(self._adapters):
            self.release(name)

    # ------------------------------------------------------------------ encoded-guide cache
    def encoded_guide(self, adapter: ModelAdapter, guide_pixels: np.ndarray) -> Any:
        """Encode a guide once per (model, guide content, encode mode).

        Safe to reuse across t0, method, K, lambda, lr, n_ctrl and step counts because the
        encoding depends on none of them.  The key is a CONTENT hash, so two problems, guide
        modes or image subsets can never alias.
        """
        key = (adapter.spec.name, pixel_fingerprint(guide_pixels),
               adapter.registry.get("vae_encode_mode", "n/a"))
        if key not in self._encode_cache:
            with timed("encode_cache/%s" % adapter.spec.name):
                self._encode_cache[key] = adapter.encode_pixels(guide_pixels)
        return self._encode_cache[key]

    def clear_encode_cache(self, model: Optional[str] = None) -> None:
        if model is None:
            self._encode_cache.clear()
        else:
            for key in [k for k in list(self._encode_cache) if k[0] == model]:
                self._encode_cache.pop(key, None)
        free_memory()

    def loaded(self) -> List[str]:
        return list(self._adapters)

    def memory_report(self, prefix: str = "") -> Dict[str, float]:
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
            report = {"allocated_gib": torch.cuda.memory_allocated() / 2 ** 30,
                      "reserved_gib": torch.cuda.memory_reserved() / 2 ** 30,
                      "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
            if prefix:
                print(prefix, {k: round(v, 3) for k, v in report.items()})
            return report
        except Exception:
            return {}


# =====================================================================================
# Shared initial-state construction -- used identically by SDEdit and both MPC methods
# =====================================================================================
def build_initial_state(adapter: ModelAdapter, problem, spec, image_indices: Sequence[int],
                        manager: ModelManager) -> Any:
    """The ONE place a corrupted initial state is formed.

        t0 = 1 : the prior-noise state ITSELF (bitwise the notebooks' pure-noise path)
        t0 < 1 : (1 - t0) * encode(g(y)) + t0 * eps

    The prior noise depends only on (model, image id, replicate), so SDEdit, MPC-RHC and
    MPC-delta_t at the same t0 start from a bit-identical z_t0.
    """
    ids = [problem.image_ids[i] for i in image_indices]
    noise = adapter.prior_sample(ids, replicate=spec.replicate)
    if spec.t0 >= 1.0:
        return adapter.initial_state(None, spec.t0, noise)
    guide_chunk = np.ascontiguousarray(
        problem.initialization_guide[np.asarray(image_indices, np.int64)])
    guide_native = manager.encoded_guide(adapter, guide_chunk)
    return adapter.initial_state(guide_native, spec.t0, noise)
