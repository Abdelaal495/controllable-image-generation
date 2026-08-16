"""Shared utilities: canonical pixel contract, semantic seeding, backend shim, timing.

Extracted from the two source notebooks (MPC-Flow section 8, SDEdit "shared utilities") and
deduplicated.  Nothing here knows about models, problems or methods.

The canonical generative clock used everywhere above the model adapters is

    s = 0  ->  noise            s = 1  ->  data
    t0     ->  s_start = 1 - t0

and the canonical pixel representation is

    (N, 256, 256, 3)  float32  in [-1, 1]  BHWC
"""

from __future__ import annotations

import contextlib
import datetime
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# =====================================================================================
# Canonical time contract  (MPC notebook, section 2)
# =====================================================================================
MEANFLOW_DESCENDING = "meanflow_descending"   # native t_MF = 1 - s ; data at t_MF = 0
FLOW_ASCENDING = "flow_ascending"             # native t    = s     ; data at t    = 1

STANDARD_FLOW = "standard_flow"               # v_theta(x, s)
MEANFLOW = "meanflow"                         # T_theta(x; s -> r)


def native_time(s: float, mapping: str) -> float:
    """Canonical generation time s (0 = noise, 1 = data) -> a model's native time.

    This is the ONLY place the two opposite conventions are reconciled.  Reconstruction
    strategies never call it; adapters and the planner do.
    """
    s = float(s)
    if not -1e-6 <= s <= 1.0 + 1e-6:
        raise ValueError("canonical time s must lie in [0, 1], got %r" % (s,))
    s = min(max(s, 0.0), 1.0)
    if mapping == MEANFLOW_DESCENDING:
        return 1.0 - s
    if mapping == FLOW_ASCENDING:
        return s
    raise ValueError("Unknown native_time_mapping %r" % (mapping,))


def canonical_start_time(t0: float) -> float:
    """Corruption strength t0 -> canonical start time s_start = 1 - t0."""
    t0 = float(t0)
    if not 0.0 <= t0 <= 1.0:
        raise ValueError("t0 must lie in [0, 1], got %r" % (t0,))
    return 1.0 - t0


# =====================================================================================
# Deterministic, semantic seeding
# =====================================================================================
GLOBAL_SEED = 42


def set_global_seed(seed: int) -> None:
    global GLOBAL_SEED
    GLOBAL_SEED = int(seed)
    np.random.seed(GLOBAL_SEED)


def semantic_seed(*parts: Any) -> int:
    """Stable 63-bit integer derived from semantic fields only.

    Depends on WHAT a draw is for, never on the order loops happen to run in, so enabling
    one experiment cannot perturb another's noise and re-running a subset reproduces exactly
    the same numbers.
    """
    payload = "|".join(str(p) for p in parts)
    return int(hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest(), 16) >> 1


def derive_rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(semantic_seed(GLOBAL_SEED, *parts))


def gaussian_noise(shape: Sequence[int], *parts: Any) -> np.ndarray:
    """Canonical N(0, I) draw.

    NumPy is the single source of randomness so the same semantic seed yields the same
    numbers whether the consumer is JAX or PyTorch.
    """
    return derive_rng(*parts).standard_normal(tuple(shape)).astype(np.float32)


def prior_noise_parts(model: str, image_id: Any, replicate: int = 0) -> Tuple[Any, ...]:
    """THE generative-noise identity.

    Deliberately independent of the reconstruction method, the solver, the step count, K,
    lambda, lr, n_ctrl and t0, so one epsilon serves every paired comparison.  A replicate
    index is available for deliberate multi-trial studies.
    """
    return (model, "x0", str(image_id), "rep%d" % int(replicate))


def measurement_noise_parts(problem: str, params_key: str, image_id: Any) -> Tuple[Any, ...]:
    return ("measurement", problem, params_key, str(image_id))


def mask_parts(problem: str, params_key: str, image_id: Any) -> Tuple[Any, ...]:
    return ("mask", problem, params_key, str(image_id))


def stroke_geometry_parts(params_key: str, image_id: Any) -> Tuple[Any, ...]:
    return ("stroke_geometry", params_key, str(image_id))


SEED_RECIPES: Dict[str, str] = {
    "generative_noise": "seed(global_seed, model, 'x0', image_id, replicate)",
    "measurement": "seed(global_seed, 'measurement', problem, problem_params_key, image_id)",
    "mask": "seed(global_seed, 'mask', problem, problem_params_key, image_id)",
    "stroke_geometry": "seed(global_seed, 'stroke_geometry', problem_params_key, image_id)",
    "vae_encode": "seed(global_seed, model, 'vae_encode', guide_fingerprint)",
}


# =====================================================================================
# Canonical pixel contract
# =====================================================================================
CANONICAL_RESOLUTION = 256
PIXEL_PEAK = 2.0          # peak-to-peak of [-1, 1]; used by PSNR / SSIM


def to_float(x_uint8: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [-1,1] (exact convention of both source notebooks)."""
    return (np.asarray(x_uint8).astype(np.float32) / 127.5) - 1.0


def to_uint8(x_float: Any) -> np.ndarray:
    """float [-1,1] -> uint8 [0,255]."""
    return np.clip(127.5 * np.asarray(x_float, dtype=np.float32) + 128.0, 0, 255).astype(np.uint8)


def assert_pixel_batch(x: np.ndarray, name: str = "images", size: Optional[int] = None,
                       check_range: bool = True) -> None:
    """Enforce the canonical contract: (N,H,W,3) float32 RGB in [-1,1], BHWC."""
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError("%s must be (N,H,W,3) BHWC RGB, got %s" % (name, (x.shape,)))
    if size is not None and (x.shape[1] != size or x.shape[2] != size):
        raise ValueError("%s must be %dx%d, got %dx%d"
                         % (name, size, size, x.shape[1], x.shape[2]))
    if not np.isfinite(x).all():
        raise ValueError("%s contains non-finite values." % name)
    if check_range:
        lo, hi = float(x.min()), float(x.max())
        if lo < -1.05 or hi > 1.05:
            raise ValueError("%s outside [-1,1]: min=%.3f max=%.3f" % (name, lo, hi))


def pixel_fingerprint(x: np.ndarray) -> str:
    """Content hash of an array; used as a cache key that cannot alias across data."""
    arr = np.ascontiguousarray(np.asarray(x, np.float32))
    return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()


# =====================================================================================
# Backend shim -- the primitives an operator needs that are NOT already polymorphic
# =====================================================================================
class Backend:
    """NumPy reference backend.

    Everything else (+, -, *, strided slicing, reshape) behaves identically on NumPy
    arrays, torch tensors and JAX arrays, which is why every operator is written once.
    """
    key = "numpy"
    framework = "numpy"

    def const(self, arr):
        return np.asarray(arr, np.float32)

    def index(self, idx):
        return np.asarray(idx, np.int64)

    def take(self, x, idx, axis: int):
        return np.take(x, idx, axis=axis)

    def sum(self, x):
        return np.sum(x)

    def mean(self, x):
        return np.mean(x)

    def segment_sum(self, values, seg_index, num_segments: int):
        """values (N, P, C), seg_index (P,) int in [0, num_segments) -> (N, num_segments, C).

        Differentiable in the torch/JAX backends; the NumPy version is the reference.
        """
        n, _p, c = values.shape
        out = np.zeros((n, num_segments, c), np.float32)
        np.add.at(out, (slice(None), seg_index, slice(None)), values)
        return out


class TorchBackend(Backend):
    framework = "torch"

    def __init__(self, torch_module, device, dtype):
        self.torch = torch_module
        self.device, self.dtype = device, dtype
        self.key = "torch:%s:%s" % (device, dtype)

    def const(self, arr):
        return self.torch.from_numpy(
            np.ascontiguousarray(np.asarray(arr, np.float32))).to(device=self.device,
                                                                  dtype=self.dtype)

    def index(self, idx):
        return self.torch.from_numpy(
            np.ascontiguousarray(np.asarray(idx, np.int64))).to(self.device)

    def take(self, x, idx, axis: int):
        return self.torch.index_select(x, axis, idx)

    def sum(self, x):
        return self.torch.sum(x)

    def mean(self, x):
        return self.torch.mean(x)

    def segment_sum(self, values, seg_index, num_segments: int):
        # index_add is differentiable w.r.t. `values` (the gradient is a gather).
        n, _p, c = values.shape
        out = self.torch.zeros((n, num_segments, c), device=values.device, dtype=values.dtype)
        return out.index_add(1, seg_index, values)


class JaxBackend(Backend):
    key = "jax"
    framework = "jax"

    def __init__(self, jnp_module):
        self.jnp = jnp_module

    def const(self, arr):
        return self.jnp.asarray(np.asarray(arr, np.float32))

    def index(self, idx):
        return self.jnp.asarray(np.asarray(idx, np.int32))

    def take(self, x, idx, axis: int):
        return self.jnp.take(x, idx, axis=axis)

    def sum(self, x):
        return self.jnp.sum(x)

    def mean(self, x):
        return self.jnp.mean(x)

    def segment_sum(self, values, seg_index, num_segments: int):
        n, _p, c = values.shape
        out = self.jnp.zeros((n, num_segments, c), values.dtype)
        return out.at[:, seg_index, :].add(values)


NUMPY_BACKEND = Backend()


# =====================================================================================
# Shared signal-processing helpers used by several operators
# =====================================================================================
def reflect_indices(n: int, pad: int) -> np.ndarray:
    """Indices reproducing numpy's mode='reflect' padding, usable by every backend.

    NAMING: this is np.pad(mode='reflect'), i.e. (d c b | a b c d | c b a) with the edge
    sample NOT repeated.  scipy.ndimage calls the same thing 'mirror'.
    """
    idx = np.arange(-pad, n + pad)
    period = max(2 * n - 2, 1)
    idx = np.abs(idx) % period
    idx = np.where(idx >= n, period - idx, idx)
    return idx.astype(np.int64)


def gaussian_kernel_1d(sigma: float, size: int) -> np.ndarray:
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return (w / w.sum()).astype(np.float32)


def separable_gaussian_blur(x, kernel: np.ndarray, B: Backend, cache: Dict[Any, Any],
                            tag: str = "blur"):
    """Separable Gaussian convolution with reflect padding, in any backend.

    `x` is canonical BHWC.  Written once and used by BOTH the deblurring operator and the
    stroke renderer's final blur, so the two can never drift apart.
    """
    k = int(len(kernel))
    pad = k // 2
    h, wid = int(x.shape[1]), int(x.shape[2])

    def cached_index(name, arr):
        key = (B.key, "idx:%s:%s" % (tag, name))
        if key not in cache:
            cache[key] = B.index(arr)
        return cache[key]

    xh = B.take(x, cached_index("h%d" % h, reflect_indices(h, pad)), 1)
    out = None
    for i in range(k):
        term = float(kernel[i]) * xh[:, i:i + h, :, :]
        out = term if out is None else out + term
    xw = B.take(out, cached_index("w%d" % wid, reflect_indices(wid, pad)), 2)
    res = None
    for j in range(k):
        term = float(kernel[j]) * xw[:, :, j:j + wid, :]
        res = term if res is None else res + term
    return res


# =====================================================================================
# Small structural helpers
# =====================================================================================
def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursive dict merge used by the configuration inheritance chain."""
    import copy
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def as_sweep_list(value: Any) -> List[Any]:
    """Normalise a SWEEPABLE field to a list.  Never apply this to a non-sweepable field."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def stable_hash(*parts: Any, size: int = 8) -> str:
    return hashlib.blake2b("|".join(str(p) for p in parts).encode("utf-8"),
                           digest_size=size).hexdigest()


def canonical_params_key(params: Dict[str, Any]) -> str:
    """Order-independent, type-stable key for a problem-parameter dictionary."""
    return json.dumps({k: params[k] for k in sorted(params)}, sort_keys=True, default=str)


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def jsonable(obj):
    """Recursively coerce a structure into something json.dump accepts."""
    if isinstance(obj, dict):
        return {(k if isinstance(k, (str, int, float, bool, type(None)))
                 else "|".join(str(p) for p in k) if isinstance(k, tuple) else str(k)):
                jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def save_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(payload), fh, indent=2, default=_json_default)


def save_yaml(path, payload) -> None:
    import yaml
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(jsonable(payload), fh, sort_keys=False, default_flow_style=False)


RULE, THIN = "=" * 94, "-" * 94


# =====================================================================================
# Timing / progress / memory
# =====================================================================================
TIMING: Dict[str, List[float]] = {}
# Cheap and non-destructive: safe to run between atomic jobs.
MEMORY_HOOKS: List[Any] = []
# Destructive: discards compiled executables and other expensive-to-rebuild state, so it
# runs ONLY when a model is released.  Calling jax.clear_caches() between jobs would force
# a full recompilation of the generative model before every single job.
DEEP_MEMORY_HOOKS: List[Any] = []
_PROGRESS_ENABLED = [True]


def record_time(key: str, seconds: float) -> None:
    TIMING.setdefault(key, []).append(float(seconds))


@contextlib.contextmanager
def timed(key: str, verbose: bool = False):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        record_time(key, dt)
        if verbose:
            print("   [%s] %.2fs" % (key, dt))


def timing_summary() -> Dict[str, Dict[str, float]]:
    return {k: {"count": len(v), "total": float(np.sum(v)), "mean": float(np.mean(v))}
            for k, v in sorted(TIMING.items())}


def set_progress_enabled(flag: bool) -> None:
    _PROGRESS_ENABLED[0] = bool(flag)


def progress(iterable, desc: str = "", total: Optional[int] = None):
    if not _PROGRESS_ENABLED[0]:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=False)
    except Exception:
        return iterable


def free_memory(deep: bool = False) -> None:
    """Framework-neutral memory reclamation; each backend registers its own hook.

    deep=False  between atomic jobs -- releases tensors, KEEPS compiled executables.
    deep=True   when a model is released -- also discards compilation caches.
    """
    gc.collect()
    for hook in MEMORY_HOOKS:
        try:
            hook()
        except Exception:
            pass
    if deep:
        for hook in DEEP_MEMORY_HOOKS:
            try:
                hook()
            except Exception:
                pass


@contextlib.contextmanager
def pushd(path: Any):
    previous = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(previous)


def load_python_module(module_name: str, file_path: Any):
    """Import a repository file under a private module name (never touches sys.path)."""
    import importlib.util
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import %s from %s" % (module_name, file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def in_ipython() -> bool:
    try:
        from IPython import get_ipython           # noqa: F401
        return get_ipython() is not None
    except Exception:
        return False


def now_iso() -> str:
    return datetime.datetime.now().isoformat()


# =====================================================================================
# Accelerator detection -- pure stdlib, runs before any framework is imported
# =====================================================================================
def detect_accelerator(requested: str = "auto") -> Dict[str, Any]:
    """Detect the runtime accelerator without importing JAX or PyTorch."""
    import glob
    import shutil
    import subprocess

    info: Dict[str, Any] = {"kind": "cpu", "reason": "no accelerator device nodes found",
                            "gpu_name": None, "gpu_memory_mb": None, "requested": requested,
                            "bf16_likely": False}

    tpu_nodes = glob.glob("/dev/accel*") + glob.glob("/dev/vfio/*")
    has_tpu = (bool(tpu_nodes) or bool(os.environ.get("COLAB_TPU_ADDR"))
               or bool(os.environ.get("TPU_WORKER_ID")) or os.path.exists("/usr/share/tpu"))

    has_gpu = False
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30)
            if out.returncode == 0 and out.stdout.strip():
                parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
                info["gpu_name"] = parts[0]
                info["gpu_memory_mb"] = int(float(parts[1]))
                if len(parts) > 2:
                    try:
                        info["compute_capability"] = float(parts[2])
                        info["bf16_likely"] = float(parts[2]) >= 8.0
                    except ValueError:
                        pass
                has_gpu = True
        except Exception as exc:                                        # pragma: no cover
            print("nvidia-smi probe failed:", exc)

    if requested in ("tpu", "gpu", "cpu"):
        info.update(kind=requested, reason="forced by runtime.accelerator")
        if requested == "tpu" and not has_tpu:
            print("WARNING: accelerator='tpu' requested but no TPU device node was found.")
        if requested == "gpu" and not has_gpu:
            print("WARNING: accelerator='gpu' requested but nvidia-smi reported no GPU.")
    elif has_tpu:
        info.update(kind="tpu", reason="TPU device node present")
    elif has_gpu:
        info.update(kind="gpu", reason="nvidia-smi reports %s" % info["gpu_name"])
    return info


def get_hf_token(required: bool = False) -> Optional[str]:
    """Fetch a Hugging Face token from .env, Colab secrets or the environment.

    Never hard-coded and never written to the repository.
    """
    token = None
    try:
        from google.colab import userdata                                # type: ignore
        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and required:
        raise RuntimeError(
            "A Hugging Face token is required for the gated dataset 'ILSVRC/imagenet-1k'.\n"
            "  1. copy .env.example to .env and set HF_TOKEN=...;\n"
            "  2. Colab: key icon in the sidebar -> add a secret named HF_TOKEN;\n"
            "  3. export HF_TOKEN=... before starting;\n"
            "  4. avoid the gated dataset with data.source: local_folder.\n"
            "You must also have accepted the licence at "
            "https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    return token


def load_dotenv(path: str = ".env") -> Dict[str, str]:
    """Minimal .env loader (python-dotenv is used when available).

    Values already present in the process environment always win, so an exported HF_TOKEN
    is never overwritten by a stale file.
    """
    loaded: Dict[str, str] = {}
    try:
        from dotenv import load_dotenv as _ld                            # type: ignore
        _ld(path, override=False)
    except Exception:
        pass
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ and value:
                os.environ[key] = value
            if key:
                loaded[key] = "set" if (value or os.environ.get(key)) else "empty"
    return loaded
