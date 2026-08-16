#!/usr/bin/env python
"""SDEdit vs. MPC-Flow -- the experiment orchestrator (this repository's "notebook").

    python run.py --config configs/experiments.yaml --dry-run     # plan only, no models
    python run.py --config configs/experiments.yaml               # full run

In Colab, `%run run.py --config configs/experiments.yaml` behaves like a notebook: the
resolved plan, progress, summary tables and matplotlib figures all appear inline.

Execution is MODEL-MAJOR: each checkpoint is loaded once, every job that needs it runs, and
it is released before the next model loads -- which matters because JiT is Torch and pMF is
JAX and they should not be resident simultaneously.

Timing policy (brief section 32).  The reported reconstruction runtime covers generative
model calls, integration, MPC planning, control optimisation, backward passes and the
controlled execution.  It EXCLUDES checkpoint download, model loading, dataset download,
metrics, visualisation and image saving.  Model loading is recorded separately, and every
(model, method, shape) is warmed up untimed first so JAX compilation never lands inside a
measured run.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import checks as checks_module
from src.config import (ConfigError, MODEL_REGISTRY_DEFAULTS, METHOD_DECLARATIONS,
                        check_accelerator_compatibility, load_config, print_run_plan,
                        resolve_model_registry, resolve_run_plan, validate_config)
from src.data import DataManager
from src.metrics import configure_lpips, degraded_baseline, evaluate_reconstruction
from src.models import ModelManager, build_initial_state, load_adapters
from src.models.base import PREFETCH_HOOKS
from src.models.base import Conditioning
from src.mpc import select_reconstructor
from src.problems import ProblemStore
from src.sdedit import ReconstructionStats
from src.utils import (RULE, THIN, apply_offline_mode, detect_accelerator,
                       detect_environment, env_default, free_memory, in_ipython, jsonable,
                       load_dotenv, now_iso, pixel_fingerprint, progress, save_json,
                       save_yaml, set_global_seed, set_progress_enabled, timing_summary,
                       to_uint8)
import src.visualization as viz


# =====================================================================================
# CLI
# =====================================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiments.yaml",
                   help="experiment configuration (default: configs/experiments.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate, resolve and print the plan, then exit without loading "
                        "any model")
    p.add_argument("--check", dest="check", action="store_true", default=True,
                   help="run the structural and per-model sanity checks (default)")
    p.add_argument("--no-check", dest="check", action="store_false")
    p.add_argument("--checks-only", action="store_true",
                   help="run the structural checks against the built problems, then exit")
    p.add_argument("--max-jobs", type=int, default=None,
                   help="override runtime.max_atomic_jobs (the sweep-explosion guard)")
    p.add_argument("--output-root", default=None, help="override runtime.output_root")
    p.add_argument("--cache-root", default=None, help="override runtime.cache_root")
    p.add_argument("--models", default=None,
                   help="comma-separated subset of models to run (must appear in the config)")
    p.add_argument("--experiments", default=None,
                   help="comma-separated subset of experiments to enable")
    p.add_argument("--num-images", type=int, default=None,
                   help="override num_images for every enabled experiment (quick smoke test)")
    p.add_argument("--replicate", type=int, default=None,
                   help="stochastic replicate index; changes epsilon deliberately")
    p.add_argument("--resume", dest="resume", action="store_true", default=None,
                   help="reuse finished jobs whose resolved metadata matches (default: from "
                        "runtime.resume)")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--no-figures", action="store_true", help="skip figure generation")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the untimed warm-up (timings will then include compilation)")
    p.add_argument("--run-id", default=None, help="reuse an existing run directory name")

    cluster = p.add_argument_group(
        "cluster / offline execution",
        "Alliance (Compute Canada) compute nodes have NO internet access, so every asset "
        "must be on disk before a job starts.")
    cluster.add_argument("--prefetch", action="store_true",
                         help="download every asset the config needs (model repositories, "
                              "checkpoints, ImageNet images, LPIPS weights) and exit. Run "
                              "this ONCE on a login node; no GPU and no model build.")
    cluster.add_argument("--offline", dest="offline", action="store_true", default=None,
                         help="forbid all network access (auto-enabled under SLURM)")
    cluster.add_argument("--online", dest="offline", action="store_false",
                         help="allow network access even under SLURM")
    cluster.add_argument("--shard", default=None, metavar="K/N",
                         help="run only shard K of N (0-based) for SLURM job arrays. Shards "
                              "are contiguous over a model-major ordering, so a shard "
                              "usually loads a single checkpoint.")
    cluster.add_argument("--aggregate", action="store_true",
                         help="merge the per-job results already on disk in --run-id into "
                              "results.csv plus figures, then exit. Use after a job array.")
    return p.parse_args(argv)


def parse_shard(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    try:
        k, n = text.split("/")
        shard, total = int(k), int(n)
    except Exception:
        raise ConfigError("--shard must look like K/N, e.g. 0/4 (K is 0-based). Got %r"
                          % text)
    if total < 1 or not 0 <= shard < total:
        raise ConfigError("--shard K/N needs N >= 1 and 0 <= K < N. Got %r" % text)
    return shard, total


def select_shard(specs: Sequence, shard: int, total: int) -> List:
    """Contiguous slice of a MODEL-MAJOR ordering.

    Contiguous rather than round-robin so a shard usually needs one checkpoint: loading pMF
    costs ~2 minutes, and interleaving would make most shards pay for both models.  The
    ordering is derived from stable job ids, so every array task computes the same split
    without communicating.
    """
    from src.config import MODEL_NAMES
    order = {name: i for i, name in enumerate(MODEL_NAMES)}
    ordered = sorted(specs, key=lambda s: (order.get(s.model, 99), s.experiment, s.method,
                                           s.job_id))
    n = len(ordered)
    lo = (n * shard) // total
    hi = (n * (shard + 1)) // total
    return ordered[lo:hi]


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    rt = config.setdefault("runtime", {})
    # Job scripts set these once and every array task inherits them.
    if os.environ.get("MPCFLOW_OUTPUT_ROOT"):
        rt["output_root"] = os.environ["MPCFLOW_OUTPUT_ROOT"]
    if os.environ.get("MPCFLOW_CACHE_ROOT"):
        rt["cache_root"] = os.environ["MPCFLOW_CACHE_ROOT"]
    if args.max_jobs is not None:
        rt["max_atomic_jobs"] = int(args.max_jobs)
    if args.output_root:
        rt["output_root"] = args.output_root
    if args.cache_root:
        rt["cache_root"] = args.cache_root
    if args.replicate is not None:
        rt["replicate"] = int(args.replicate)
    if args.resume is not None:
        rt["resume"] = bool(args.resume)
    if args.experiments:
        wanted = {name.strip() for name in args.experiments.split(",") if name.strip()}
        unknown = wanted - set(config.get("experiments", {}))
        if unknown:
            raise ConfigError("--experiments names unknown experiments: %s" % sorted(unknown))
        for name, block in config["experiments"].items():
            block["enabled"] = name in wanted
    if args.models:
        wanted = {name.strip() for name in args.models.split(",") if name.strip()}
        for name, block in config["experiments"].items():
            if not block.get("enabled"):
                continue
            models = block.get("models") or {}
            kept = {k: v for k, v in models.items() if k in wanted}
            if not kept:
                block["enabled"] = False
            else:
                block["models"] = kept
    if args.num_images is not None:
        for block in config["experiments"].values():
            if block.get("enabled"):
                block["num_images"] = int(args.num_images)
    return config


# =====================================================================================
# Environment
# =====================================================================================
def ensure_repositories(plan, cache_root: Path, verbose: bool = True) -> Dict[str, Any]:
    """Clone the model repositories at their pinned revisions if they are missing.

    setup_colab.sh normally does this; doing it here too keeps a fresh checkout working
    without an extra step, and it is idempotent.
    """
    # Absolute: adapters chdir into these directories (RepoSandbox / pushd).
    repo_root = Path(cache_root).resolve() / "repos"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths, heads = {}, {}
    for model in plan.resources.models:
        spec = plan.resources.repositories[model]
        dest = repo_root / spec["dirname"]
        paths[model] = dest
        if not (dest / ".git").is_dir():
            if verbose:
                print("Cloning %s -> %s" % (spec["url"], dest))
            subprocess.run('git clone --quiet "%s" "%s"' % (spec["url"], dest), shell=True,
                           check=True)
            if spec.get("rev"):
                subprocess.run('cd "%s" && git fetch --quiet --depth 1 origin "%s" && '
                               'git checkout --quiet "%s"' % (dest, spec["rev"], spec["rev"]),
                               shell=True, check=False)
        heads[model] = subprocess.run('cd "%s" && git rev-parse HEAD' % dest, shell=True,
                                      capture_output=True, text=True).stdout.strip()
        if verbose:
            print("   %-4s %s @ %s" % (model, dest, heads[model][:12]))
    return {"repo_paths": paths, "repo_heads": heads}


def init_frameworks(plan, config: Dict[str, Any], accel: Dict[str, Any],
                    cache_root: Path, verbose: bool = True) -> Dict[str, Any]:
    """Import and configure only the frameworks this plan needs."""
    from src.utils import DEEP_MEMORY_HOOKS, MEMORY_HOOKS
    context: Dict[str, Any] = {"torch_dtypes": {}, "local_device_count": 1}
    frameworks = plan.resources.frameworks

    if "jax" in frameworks:
        if accel["kind"] == "gpu":
            # Do not grab the whole device: PyTorch may share this process.
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        jax_cache = Path(cache_root) / "jax_compilation_cache"
        jax_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(jax_cache))
        import jax
        # Persist compiled executables to disk so compilation is paid ONCE EVER rather
        # than once per session. The defaults skip small/fast entries; pMF's model step is
        # neither, but being explicit means a rerun of the same benchmark recompiles nothing.
        for option, value in (("jax_compilation_cache_dir", str(jax_cache)),
                              ("jax_persistent_cache_min_entry_size_bytes", -1),
                              ("jax_persistent_cache_min_compile_time_secs", 0.5)):
            try:
                jax.config.update(option, value)
            except Exception:
                pass                      # older JAX: the env var above still applies
        # Compatibility shims: aliases recent JAX removed but the repositories still use.
        for alias, target in (("tree_map", "tree_map"), ("tree_leaves", "tree_leaves"),
                              ("tree_flatten", "tree_flatten")):
            if not hasattr(jax, alias):
                setattr(jax, alias, getattr(jax.tree_util, target))
        jax.distributed.initialize = lambda *a, **k: None      # single-host Colab
        context["local_device_count"] = jax.local_device_count()
        # DEEP, not per-job: jax.clear_caches() discards every compiled executable. Running
        # it between jobs made pMF recompile the whole model 24 times (~470 s of compilation
        # for ~26 s of actual work). It now runs only when the model is released.
        DEEP_MEMORY_HOOKS.append(lambda: jax.clear_caches())
        if verbose:
            print("JAX             : %s | backend: %s | devices: %s"
                  % (jax.__version__, jax.default_backend(), jax.devices()))

    if "torch" in frameworks:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before cuBLAS starts
        import torch
        # AUTOGRAD STAYS ENABLED: MPC-delta_t and RHC K>1 differentiate the terminal loss
        # through the generative model.  Model PARAMETERS keep requires_grad_(False), so no
        # parameter gradients are ever allocated.  SDEdit wraps its own loop in no_grad().
        torch.set_grad_enabled(True)
        want_cuda = accel["kind"] == "gpu"
        if want_cuda and not torch.cuda.is_available():
            print("A GPU was detected but torch.cuda is unavailable; falling back to CPU.")
            want_cuda = False
        device = torch.device("cuda" if want_cuda else "cpu")
        context["torch_device"] = device

        import random as _random
        seed = int(plan.seed)
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            for flag, value in (("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False),
                                ("enable_cudnn_sdp", False), ("enable_math_sdp", True)):
                if hasattr(torch.backends.cuda, flag):
                    getattr(torch.backends.cuda, flag)(value)

        def resolve_torch_dtype(name: str):
            table = {"float32": torch.float32, "float16": torch.float16,
                     "bfloat16": torch.bfloat16}
            if name in table:
                return table[name]
            if name == "safe_auto":
                # JiT's official evaluation uses BF16 autocast.  FP16 has a much narrower
                # exponent range and destabilises pixel-space generation, so this prefers
                # BF16 and otherwise falls back to FP32 -- never FP16.
                if device.type == "cuda" and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float32
            if name != "auto":
                raise ValueError("Unsupported dtype setting: %r" % name)
            if device.type != "cuda":
                return torch.float32
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for model in plan.resources.models:
            reg = resolve_model_registry(config, model)
            if reg["framework"] == "torch":
                context["torch_dtypes"][model] = resolve_torch_dtype(reg["dtype"])
        MEMORY_HOOKS.append(lambda: torch.cuda.empty_cache()
                            if torch.cuda.is_available() else None)
        if verbose:
            print("PyTorch         : %s | device: %s | dtypes: %s"
                  % (torch.__version__, device,
                     {k: str(v) for k, v in context["torch_dtypes"].items()}))
            print("Autograd enabled: %s (required by MPC)" % torch.is_grad_enabled())
    return context


# =====================================================================================
# One atomic job
# =====================================================================================
def run_job(adapter, spec, problem, manager) -> Dict[str, Any]:
    """Execute one atomic job: one model, one problem, one method, one hyperparameter set.

    Batching is the only decision made here; every scientific setting arrived in the spec.
    Fixed-batch MeanFlow models repeat-pad a short tail chunk to their compiled shape.  Both
    terms of the MPC objective are sums over the batch and a padded row carries its own
    duplicated measurement and its own control, so padding cannot influence a real image's
    gradient or its metrics.  Padded rows are counted and reported, never treated as real.
    """
    reconstruct = select_reconstructor(spec.dynamics_family, spec.method)
    bs = max(1, int(spec.batch_size))
    indices = list(range(int(spec.num_images)))
    total = ReconstructionStats()
    outputs: List[np.ndarray] = []
    fingerprints: List[str] = []
    padded_items = 0

    for start in range(0, len(indices), bs):
        chunk = indices[start:start + bs]
        size = len(chunk)
        pad = 0
        if adapter.spec.fixed_batch_shape and size < bs:
            pad = bs - size
            chunk = chunk + [chunk[-1]] * pad
            padded_items += pad

        sub_problem = problem.subset(chunk)
        labels = (problem.labels[np.asarray(chunk, np.int64)] if problem.labels is not None
                  else np.zeros((len(chunk),), np.int32))
        # SDEdit and both MPC methods receive the SAME conditioning object for an image.
        cond = Conditioning(labels=np.asarray(labels, np.int32), guidance=dict(spec.guidance))
        x0 = build_initial_state(adapter, problem, spec, chunk, manager)
        fingerprints.append(pixel_fingerprint(adapter.to_numpy(x0)))

        x_final, stats = reconstruct(adapter, cond, x0, sub_problem, spec)
        total.merge(stats)
        pixels = adapter.to_pixels(x_final)
        outputs.append(pixels[:size] if pad else pixels)

    result = np.concatenate(outputs, axis=0)[:len(indices)]
    return {"pixels": np.asarray(result, np.float32), "stats": total,
            "padded_items": padded_items, "initial_state_fingerprints": fingerprints,
            "status": "ok" if total.finite else "non_finite_state"}


def warm_up(adapter, spec, problem, manager, verbose: bool = True) -> float:
    """One untimed reduced run per (model, method, required shape).

    pMF/iMF use JAX and pay a compilation cost on the first call for each shape.  Comparing
    a cold first run against already-compiled later jobs would make the runtime column
    meaningless, so the benchmark reports steady-state reconstruction runtime and records
    the cold/compile time separately.
    """
    # The warm-up must trace EXACTLY the trajectory the timed job will run.  Reducing
    # steps / num_mpc_steps here would change the time grid, so the real job would meet
    # uncompiled intervals and pay for their compilation inside the measured region --
    # which is precisely what "steady-state runtime" is supposed to exclude.  Only
    # n_ctrl and the image count are reduced: neither changes the traced computation.
    tiny = dataclasses.replace(
        spec,
        num_images=min(int(spec.batch_size), int(spec.num_images)),
        n_ctrl=None if spec.method == "sdedit" else 1,
        record_loss_history=False)
    started = time.perf_counter()
    run_job(adapter, tiny, problem, manager)
    seconds = time.perf_counter() - started
    if verbose:
        print("        warm-up (untimed, excluded from results): %.2fs" % seconds)
    return seconds


# =====================================================================================
# Result records
# =====================================================================================
RESULT_COLUMNS = [
    "run_id", "job_id", "task", "experiment", "problem", "problem_key", "image_id",
    "model", "model_family", "framework", "state_space", "method", "method_title",
    "t0", "canonical_start_time", "native_start_time", "native_end_time",
    "native_time_mapping", "initialization_kind", "initialization_guide_mode",
    "steps", "solver", "num_mpc_steps", "delta", "K", "lam", "n_ctrl", "lr",
    "optimizer", "warm_start", "grad_clip", "phi_normalization",
    "control_cost_normalization", "delta_t_lambda_scaling",
    "hyperparameter_source_lam", "hyperparameter_source_n_ctrl", "hyperparameter_source_lr",
    "psnr", "ssim", "lpips", "measurement_rmse", "missing_rmse", "missing_psnr",
    "missing_ssim", "observed_rmse", "observed_psnr",
    "degraded_psnr", "degraded_ssim", "degraded_lpips", "guide_psnr",
    "runtime", "runtime_per_image", "reconstruction_steps", "network_forwards",
    "model_evaluations", "planning_model_evaluations", "control_iterations",
    "backprops_through_model", "expected_model_evaluations", "expected_backprops",
    "batch_size", "num_images", "padded_items", "warmup_seconds", "model_load_seconds",
    "seed", "replicate", "generative_noise_id", "measurement_id", "guide_id",
    "initial_state_fingerprint", "conditioning_label", "cfg_scale",
    "status", "failure", "output_dir", "timestamp",
]


def build_record(spec, plan, problem, run: Optional[Dict[str, Any]],
                 metrics: Optional[Dict[str, Any]], degraded: Dict[str, Any],
                 status: str, failure: Optional[str], warmup_seconds: float,
                 load_seconds: float) -> Dict[str, Any]:
    stats = run["stats"] if run else ReconstructionStats()
    metrics = metrics or {}
    guidance = spec.guidance or {}
    record = {
        "run_id": plan.run_id, "job_id": spec.job_id, "task": spec.experiment,
        "experiment": spec.experiment, "problem": spec.problem,
        "problem_key": spec.problem_key, "image_id": "ALL",
        "model": spec.model, "model_family": spec.dynamics_family,
        "framework": spec.framework, "state_space": spec.state_space,
        "method": spec.method, "method_title": spec.method_title,
        "t0": spec.t0, "canonical_start_time": spec.canonical_start_time,
        "native_start_time": spec.native_start_time, "native_end_time": spec.native_end_time,
        "native_time_mapping": spec.native_time_mapping,
        "initialization_kind": spec.initialization_kind,
        "initialization_guide_mode": spec.initialization_guide_mode,
        "steps": spec.steps, "solver": spec.solver, "num_mpc_steps": spec.num_mpc_steps,
        "delta": spec.delta, "K": spec.K, "lam": spec.lam, "n_ctrl": spec.n_ctrl,
        "lr": spec.lr, "optimizer": spec.optimizer, "warm_start": spec.warm_start,
        "grad_clip": spec.grad_clip, "phi_normalization": spec.phi_normalization,
        "control_cost_normalization": spec.control_cost_normalization,
        "delta_t_lambda_scaling": spec.delta_t_lambda_scaling,
        "hyperparameter_source_lam": spec.hyperparameter_sources.get("lam"),
        "hyperparameter_source_n_ctrl": spec.hyperparameter_sources.get("n_ctrl"),
        "hyperparameter_source_lr": spec.hyperparameter_sources.get("lr"),
        "psnr": metrics.get("psnr"), "ssim": metrics.get("ssim"),
        "lpips": metrics.get("lpips"), "measurement_rmse": metrics.get("measurement_rmse"),
        "missing_rmse": metrics.get("missing_rmse"), "missing_psnr": metrics.get("missing_psnr"),
        "missing_ssim": metrics.get("missing_ssim"),
        "observed_rmse": metrics.get("observed_rmse"),
        "observed_psnr": metrics.get("observed_psnr"),
        "degraded_psnr": degraded.get("psnr"), "degraded_ssim": degraded.get("ssim"),
        "degraded_lpips": degraded.get("lpips"), "guide_psnr": degraded.get("guide_psnr"),
        "runtime": stats.seconds,
        "runtime_per_image": stats.seconds / max(1, int(spec.num_images)),
        "reconstruction_steps": spec.reconstruction_steps,
        "network_forwards": stats.network_forwards,
        "model_evaluations": stats.model_evals_total,
        "planning_model_evaluations": stats.model_evals_planning,
        "control_iterations": stats.control_iterations,
        "backprops_through_model": stats.backprops_through_model,
        "expected_model_evaluations": spec.expected_model_evals * spec.num_chunks,
        "expected_backprops": spec.expected_backprops * spec.num_chunks,
        "batch_size": spec.batch_size, "num_images": spec.num_images,
        "padded_items": (run or {}).get("padded_items", 0),
        "warmup_seconds": round(float(warmup_seconds), 4),
        "model_load_seconds": round(float(load_seconds), 2),
        "seed": spec.seed, "replicate": spec.replicate,
        # Identity columns: two rows that agree on these three share the same setup exactly.
        "generative_noise_id": "seed=%d|model=%s|rep=%d" % (spec.seed, spec.model,
                                                            spec.replicate),
        "measurement_id": problem.to_metadata()["measurement_fingerprint"],
        "guide_id": problem.to_metadata()["guide_fingerprint"],
        "initial_state_fingerprint": ((run or {}).get("initial_state_fingerprints") or [""])[0],
        "conditioning_label": "source_imagenet_class",
        "cfg_scale": guidance.get("scale"),
        "status": status, "failure": failure, "output_dir": None, "timestamp": now_iso(),
    }
    return record


def per_image_rows(record: Dict[str, Any], metrics: Dict[str, Any],
                   image_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    ps = metrics.get("psnr_per_image") or []
    ss = metrics.get("ssim_per_image") or []
    lp = metrics.get("lpips_per_image") or []
    for i, image_id in enumerate(image_ids):
        row = dict(record)
        row["image_id"] = image_id
        row["psnr"] = ps[i] if i < len(ps) else None
        row["ssim"] = ss[i] if i < len(ss) else None
        row["lpips"] = lp[i] if i < len(lp) else None
        rows.append(row)
    return rows


class ResultWriter:
    """Append-only, crash-safe persistence.

    Every atomic job is written the moment it finishes: a Colab crash halfway through a
    sweep never erases earlier reconstructions.
    """

    def __init__(self, run_dir: Path, shard: Optional[Tuple[int, int]] = None):
        self.run_dir = Path(run_dir)
        self.records: List[Dict[str, Any]] = []
        self.image_rows: List[Dict[str, Any]] = []
        # Parallel array tasks must never write the same file: each shard owns its own,
        # and `--aggregate` merges them from the per-job metadata afterwards.
        self.suffix = "" if shard is None else "_shard%02d" % shard[0]
        self.jsonl = self.run_dir / ("results%s.jsonl" % self.suffix)
        self.log = self.run_dir / ("experiment_log%s.jsonl" % self.suffix)

    def add(self, record: Dict[str, Any], image_rows: Sequence[Dict[str, Any]] = ()) -> None:
        self.records.append(record)
        self.image_rows.extend(image_rows)
        with open(self.jsonl, "a") as fh:
            fh.write(json.dumps(jsonable(record), default=str) + "\n")
        self.write_csv()

    def event(self, status: str, **fields: Any) -> None:
        entry = {"status": status, "time": now_iso()}
        entry.update(fields)
        with open(self.log, "a") as fh:
            fh.write(json.dumps(jsonable(entry), default=str) + "\n")

    def write_csv(self) -> Path:
        import csv
        path = self.run_dir / ("results%s.csv" % self.suffix)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in self.records:
                writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})
        if self.image_rows:
            per_image = self.run_dir / ("results_per_image%s.csv" % self.suffix)
            with open(per_image, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                for row in self.image_rows:
                    writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})
        return path


def job_dir_for(run_dir: Path, spec) -> Path:
    return (run_dir / spec.experiment / spec.model / spec.method / spec.leaf_dir)


def persist_job(run_dir: Path, spec, run: Dict[str, Any], record: Dict[str, Any],
                metrics: Dict[str, Any], problem, save_images: bool) -> Path:
    """Write one job's artefacts immediately."""
    out = job_dir_for(run_dir, spec)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["resolved_spec"] = spec.to_dict()
    payload["problem_metadata"] = problem.to_metadata()
    payload["per_image"] = {k: metrics.get(k) for k in
                            ("psnr_per_image", "ssim_per_image", "lpips_per_image")}
    save_json(out / "metadata.json", payload)
    arrays: Dict[str, Any] = {"reconstruction": to_uint8(run["pixels"])}
    if spec.record_loss_history and run["stats"].loss_history:
        arrays["loss_history"] = np.asarray(run["stats"].loss_history, np.float32)
    np.savez_compressed(out / "results.npz", **arrays)
    if save_images:
        img_dir = out / "images"
        img_dir.mkdir(exist_ok=True)
        from PIL import Image
        for i, pix in enumerate(run["pixels"]):
            Image.fromarray(to_uint8(pix)).save(img_dir / ("%s.png" % problem.image_ids[i]))
    return out


def load_finished_job(run_dir: Path, spec) -> Optional[Dict[str, Any]]:
    """Resume support: reuse a job only when its RESOLVED spec matches exactly."""
    out = job_dir_for(run_dir, spec)
    meta_path, array_path = out / "metadata.json", out / "results.npz"
    if not (meta_path.exists() and array_path.exists()):
        return None
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:
        return None
    if payload.get("status") != "ok" or payload.get("job_id") != spec.job_id:
        return None
    if payload.get("resolved_spec") != jsonable(spec.to_dict()):
        return None
    with np.load(array_path) as z:
        pixels = z["reconstruction"]
    return {"record": payload, "pixels": pixels, "output_dir": str(out)}


# =====================================================================================
# Reporting
# =====================================================================================
def print_summary_tables(records: Sequence[Dict[str, Any]]) -> None:
    """Per-configuration tables.

    Nothing is averaged across models, methods or hyperparameters: each row below is one
    atomic job, because a mean over a 4-step and a 25-step SDEdit (or over two lambdas)
    describes a run that was never executed.  The only averaging is over the images inside
    a single job, which is what `num_images` means.
    """
    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        print("No successful job to summarise.")
        return

    print("\n" + RULE)
    print("RESULTS BY RESOLVED CONFIGURATION   (one row = one atomic job, averaged over its "
          "images only)")
    print(RULE)
    print("%-18s %-5s %-26s %7s %7s %8s %10s %9s"
          % ("task", "model", "configuration", "PSNR", "SSIM", "LPIPS", "meas-RMSE",
             "s/image"))
    print(THIN)
    order = {"sdedit": 0, "mpc_rhc": 1, "mpc_delta_t": 2}
    last_task = None
    for row in sorted(ok, key=lambda r: (r["task"], r["model"], order.get(r["method"], 9),
                                         viz.config_label(r))):
        if last_task is not None and row["task"] != last_task:
            print()
        last_task = row["task"]
        degraded = row.get("degraded_psnr")
        flag = " " if (row.get("psnr") or 0) >= (degraded or -1e9) else "!"
        print("%-18s %-5s %-26s %7.2f %7.4f %8s %10.4f %9.2f %s"
              % (row["task"][:18], row["model"], viz.config_label(row)[:26], row["psnr"],
                 row["ssim"],
                 ("%.4f" % row["lpips"]) if row.get("lpips") is not None else "n/a",
                 row["measurement_rmse"], row["runtime_per_image"], flag))
    print(THIN)
    print("!  marks a reconstruction whose PSNR is BELOW the degraded observation itself,")
    print("   i.e. one that did worse than leaving the measurement untouched.")

    # ---------------------------------------------------------------- paired deltas
    print("\n" + RULE)
    print("PAIRED COMPARISON vs STEP-MATCHED SDEdit")
    print("each MPC job against the SDEdit job at the SAME task, model and t0 whose step")
    print("count is closest to its own -- never against an average of SDEdit runs")
    print(RULE)
    print("%-18s %-5s %-24s %-14s %8s %8s %9s %8s"
          % ("task", "model", "MPC configuration", "baseline", "dPSNR", "dSSIM", "dLPIPS",
             "runtime"))
    print(THIN)
    mpc = [r for r in ok if r["method"] in ("mpc_rhc", "mpc_delta_t")]
    printed = 0
    last_task = None
    for row in sorted(mpc, key=lambda r: (r["task"], r["model"], order.get(r["method"], 9),
                                          viz.config_label(r))):
        base = viz.step_matched_baseline(row, ok)
        if base is None:
            continue
        if last_task is not None and row["task"] != last_task:
            print()
        last_task = row["task"]
        printed += 1
        d_lpips = ("%+9.4f" % (row["lpips"] - base["lpips"])
                   if None not in (row.get("lpips"), base.get("lpips")) else "      n/a")
        ratio = (row["runtime_per_image"] / base["runtime_per_image"]
                 if base.get("runtime_per_image") else float("nan"))
        print("%-18s %-5s %-24s %-14s %+8.2f %+8.4f %s %7.1fx"
              % (row["task"][:18], row["model"], viz.config_label(row)[:24],
                 viz.config_label(base)[:14], row["psnr"] - base["psnr"],
                 row["ssim"] - base["ssim"], d_lpips, ratio))
    if not printed:
        print("  (no MPC job has a paired SDEdit baseline at the same t0)")
    print(THIN)
    print("dLPIPS < 0 is an improvement. 'runtime' is the multiple of the named baseline.")

    failed = [r for r in records if r["status"] != "ok"]
    if failed:
        print("\n%d job(s) did not complete:" % len(failed))
        for r in failed[:12]:
            print("  %-10s %-18s %-5s %-12s %s"
                  % (r["status"], r["task"][:18], r["model"], r["method"],
                     (r.get("failure") or "")[:70]))


def make_figures(run_dir: Path, plan, store, records, reconstructions,
                 show: bool = True) -> List[Path]:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    problems = {key: store.get(key) for key in
                dict.fromkeys(s.problem_key for s in plan.specs)}
    paths.append(viz.plot_problem_overview(problems, figures_dir / "problem_instances.png",
                                           show))
    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for spec in plan.specs:
        images = reconstructions.get(spec.job_id)
        if images is None:
            continue
        by_group.setdefault((spec.experiment, spec.model), []).append(
            {"title": spec.figure_title(), "images": images, "method": spec.method,
             "t0": spec.t0, "sort_key": spec.leaf_dir})
    for (experiment, model), entries in sorted(by_group.items()):
        spec = next(s for s in plan.specs if s.experiment == experiment and s.model == model)
        paths += viz.plot_comparison_grid(experiment, model, entries,
                                          store.get(spec.problem_key), figures_dir, show)
    for model in dict.fromkeys(s.model for s in plan.specs):
        breakdown = viz.plot_configuration_breakdown(
            records, model, figures_dir / ("configurations_%s.png" % model), show)
        if breakdown:
            paths.append(breakdown)
    deltas = viz.plot_paired_deltas(records, figures_dir / "paired_deltas.png", show)
    if deltas:
        paths.append(deltas)
    cost = viz.plot_quality_vs_cost(records, figures_dir / "quality_vs_cost.png", show)
    if cost:
        paths.append(cost)
    return paths


# =====================================================================================
# Prefetch (login node) and aggregate (after a job array)
# =====================================================================================
def prefetch_assets(plan, config: Dict[str, Any], cache_root: Path) -> Dict[str, Any]:
    """Download everything a run needs, so compute nodes never touch the network.

    Alliance/DRAC compute nodes have no route to the internet.  Anything not on disk when
    the job starts is a failure, not a slow download, so this runs once on a login node.
    Deliberately download-only: no model is built and no GPU is required.
    """
    report: Dict[str, Any] = {"cache_root": str(cache_root), "models": {}, "started": now_iso()}
    print(RULE)
    print("PREFETCH -- staging every asset for offline execution")
    print("  cache root: %s" % cache_root)
    print(RULE)

    print("\n[1/4] Model repositories")
    context = ensure_repositories(plan, cache_root)
    context["ckpt_cache"] = (cache_root / "checkpoints").resolve()
    context["ckpt_cache"].mkdir(parents=True, exist_ok=True)
    report["repositories"] = context.get("repo_heads", {})

    print("\n[2/4] Model checkpoints")
    load_adapters(plan.resources.models)
    for model in plan.resources.models:
        hook = PREFETCH_HOOKS.get(model)
        if hook is None:
            print("   %-5s no prefetch hook; it will download at first use." % model)
            report["models"][model] = {"status": "no_hook"}
            continue
        try:
            registry = resolve_model_registry(config, model)
            info = hook(registry, context)
            report["models"][model] = dict(info, status="ok")
            print("   %-5s ok  %s" % (model, info))
        except Exception as exc:
            report["models"][model] = {"status": "failed", "error": str(exc)}
            print("   %-5s FAILED: %s" % (model, exc))
            traceback.print_exc(limit=4)

    print("\n[3/4] Source images")
    try:
        data = DataManager(config, cache_root / "data")
        pool = data.pool(plan.resources.source_pool_size)
        report["data"] = {"count": len(pool), "source": pool.source,
                          "image_ids": pool.image_ids}
        print("   cached %d image(s) from %s" % (len(pool), pool.source))
    except Exception as exc:
        report["data"] = {"status": "failed", "error": str(exc)}
        print("   FAILED: %s" % exc)
        print("   (a gated-dataset failure here is almost always a missing HF_TOKEN)")

    print("\n[4/4] LPIPS weights")
    if (config.get("metrics") or {}).get("lpips", True):
        try:
            from src.metrics import lpips_per_image
            probe = np.zeros((1, 64, 64, 3), np.float32)
            ok = lpips_per_image(probe, probe) is not None
            report["lpips"] = "ok" if ok else "unavailable"
            print("   %s" % ("AlexNet weights cached" if ok
                             else "LPIPS unavailable -- results will have empty LPIPS"))
        except Exception as exc:
            report["lpips"] = "failed: %s" % exc
            print("   FAILED: %s" % exc)
    else:
        report["lpips"] = "disabled"

    report["finished"] = now_iso()
    save_json(cache_root / "prefetch_report.json", report)
    print("\n%s\nPREFETCH COMPLETE -- report: %s"
          % (RULE, cache_root / "prefetch_report.json"))
    failures = [m for m, v in report["models"].items() if v.get("status") == "failed"]
    if failures or report.get("data", {}).get("status") == "failed":
        print("SOME ASSETS FAILED (%s). Fix them here: a compute node cannot download."
              % (failures or "data"))
        return report
    print("Compute nodes can now run fully offline. Point runs at the SAME cache root:")
    print("    python run.py --config %s --cache-root %s" % ("configs/experiments.yaml",
                                                             cache_root))
    print(RULE)
    return report


def collect_finished_jobs(run_dir: Path, plan) -> Tuple[List[Dict[str, Any]],
                                                        Dict[str, np.ndarray]]:
    """Read every per-job artefact under a run directory.

    Per-job metadata.json is the source of truth, so merging the output of parallel array
    tasks needs no coordination between them and no shared append-only file.
    """
    records, reconstructions = [], {}
    by_id = {s.job_id: s for s in plan.specs}
    for meta_path in sorted(run_dir.rglob("metadata.json")):
        try:
            payload = json.loads(meta_path.read_text())
        except Exception:
            continue
        if "job_id" not in payload or payload.get("status") != "ok":
            continue
        records.append({k: payload.get(k) for k in RESULT_COLUMNS})
        array_path = meta_path.parent / "results.npz"
        if payload["job_id"] in by_id and array_path.exists():
            try:
                with np.load(array_path) as z:
                    reconstructions[payload["job_id"]] = (
                        np.asarray(z["reconstruction"], np.float32) / 127.5) - 1.0
            except Exception:
                pass
    return records, reconstructions


def aggregate_only(plan, config: Dict[str, Any], run_dir: Path, args) -> int:
    """Merge the per-job artefacts written by parallel array tasks into one result set."""
    print("\n%s\nAGGREGATE -- merging finished jobs under %s\n%s" % (RULE, run_dir, RULE))
    records, reconstructions = collect_finished_jobs(run_dir, plan)
    if not records:
        print("No finished job found. Did the array tasks write to this --run-id?")
        return 1

    done = {r["job_id"] for r in records}
    missing = [s for s in plan.specs if s.job_id not in done]
    print("  found %d finished job(s) of %d planned" % (len(records), len(plan.specs)))
    if missing:
        print("  %d job(s) are MISSING -- the merge below is partial:" % len(missing))
        for spec in missing[:10]:
            print("     %s" % spec.label)
        if len(missing) > 10:
            print("     ... and %d more" % (len(missing) - 10))

    writer = ResultWriter(run_dir)
    writer.records = records
    csv_path = writer.write_csv()
    print("  wrote %s" % csv_path)

    print_summary_tables(records)

    if not args.no_figures:
        print("\nRebuilding problem instances for the figures ...")
        cache_root = Path(plan.cache_root).resolve()
        data = DataManager(config, cache_root / "data")
        store = ProblemStore()
        store.build_all(plan.problems, data, verbose=False)
        try:
            figures = make_figures(run_dir, plan, store, records, reconstructions,
                                   show=in_ipython())
            print("  %d figure(s) written to %s" % (len(figures), run_dir / "figures"))
        except Exception as exc:
            print("  figure generation failed (%s); results.csv is unaffected." % exc)

    save_json(run_dir / "aggregate_metadata.json",
              {"merged": now_iso(), "found": len(records), "planned": len(plan.specs),
               "missing_job_ids": [s.job_id for s in missing]})
    return 0


# =====================================================================================
# Main
# =====================================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    shard = parse_shard(args.shard)
    env = load_dotenv(".env")
    if env:
        print("Loaded .env keys: %s" % ", ".join(sorted(env)))

    where = detect_environment()
    offline = where["offline"] if args.offline is None else bool(args.offline)
    if args.prefetch:
        offline = False                     # prefetch exists precisely to use the network
    apply_offline_mode(offline)
    print("Environment      : %s%s | offline=%s"
          % (where["kind"], (" [%s]" % where["cluster"]) if where["cluster"] else "", offline))
    if where["job_id"]:
        print("SLURM            : job %s%s on %s | SLURM_TMPDIR=%s"
              % (where["job_id"],
                 (" task %s" % where["array_task"]) if where["array_task"] else "",
                 where["node"], where["tmpdir"]))
    if offline and not args.prefetch:
        print("Offline mode: no download will be attempted. Anything missing from the cache")
        print("  is a hard failure -- run `python run.py --prefetch` on a login node first.")
    # Progress bars are useless in a SLURM log and produce megabytes of carriage returns.
    if not sys.stdout.isatty() and where["kind"] != "colab":
        set_progress_enabled(False)

    config = apply_cli_overrides(load_config(args.config), args)
    warnings_ = validate_config(config)
    plan = resolve_run_plan(config, warnings_, run_id=args.run_id)

    rt = config["runtime"]
    set_global_seed(plan.seed)
    set_progress_enabled(bool(rt.get("progress_bars", True)))
    configure_lpips(bool((config.get("metrics") or {}).get("lpips", True)),
                    str((config.get("metrics") or {}).get("lpips_net", "alex")))

    accel = detect_accelerator(rt.get("accelerator", "auto"))
    accel_notes = check_accelerator_compatibility(plan, accel)
    print_run_plan(plan, accel, accel_notes)

    cache_root_early = Path(rt.get("cache_root", "cache")).resolve()
    if args.prefetch:
        prefetch_assets(plan, config, cache_root_early)
        return 0

    run_dir = Path(plan.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        if not args.run_id:
            raise ConfigError("--aggregate needs --run-id naming the directory to merge.")
        return aggregate_only(plan, config, run_dir, args)

    save_yaml(run_dir / "config.yaml", plan.raw_config)
    save_yaml(run_dir / "resolved_config.yaml", plan.to_dict())
    print("\nSaved %s and %s" % (run_dir / "config.yaml", run_dir / "resolved_config.yaml"))

    if args.dry_run:
        print("\n--dry-run: the plan above was validated and resolved. No model was loaded, "
              "no checkpoint downloaded and no job executed.")
        return 0

    # ---------------------------------------------------------------- shard selection
    all_specs = plan.specs
    if shard is not None:
        selected = select_shard(all_specs, shard[0], shard[1])
        plan = dataclasses.replace(plan, specs=tuple(selected))
        models = sorted({s.model for s in selected})
        print("\n%s\nSHARD %d of %d -- %d of %d atomic jobs, model(s): %s\n%s"
              % (RULE, shard[0], shard[1], len(selected), len(all_specs),
                 ", ".join(m.upper() for m in models) or "(none)", RULE))
        if not selected:
            print("This shard is empty (more shards than jobs). Nothing to do.")
            return 0
        print("Merge the shards afterwards with:")
        print("    python run.py --config %s --run-id %s --aggregate"
              % (args.config, plan.run_id))

    # ---------------------------------------------------------------- data and problems
    cache_root = Path(plan.cache_root).resolve()
    data = DataManager(config, cache_root / "data")
    source = data.pool(plan.resources.source_pool_size)
    print("\nShared image pool (largest enabled experiment needs %d):"
          % plan.resources.source_pool_size)
    for i, image_id in enumerate(source.image_ids):
        print("   [%d] %-28s class=%d" % (i, image_id, source.labels[i]))

    print("\nBuilding inverse-problem instances (once each, shared by every model/method):")
    store = ProblemStore()
    problems = store.build_all(plan.problems, data)
    degraded_metrics = {key: degraded_baseline(p) for key, p in problems.items()}
    print("\nDegraded baseline (observation vs ground truth):")
    print("   %-17s %8s %8s %8s %10s" % ("problem", "PSNR", "SSIM", "LPIPS", "guidePSNR"))
    for key, p in problems.items():
        d = degraded_metrics[key]
        print("   %-17s %8.2f %8.4f %8s %10.2f"
              % (p.name, d["psnr"], d["ssim"],
                 ("%.4f" % d["lpips"]) if d["lpips"] is not None else "n/a",
                 d["guide_psnr"]))

    report = checks_module.CheckReport()
    if args.check or args.checks_only:
        print("\n" + RULE)
        checks_module.run_structural_checks(plan, problems, data, report)
        failures = report.failures()
        if failures:
            print("\n%d structural check(s) FAILED. Inspect them before trusting any result."
                  % len(failures))
            if not rt.get("continue_on_experiment_error", True):
                save_json(run_dir / "checks.json", report.to_dict())
                raise RuntimeError("Structural checks failed: %s"
                                   % [f.name for f in failures])
        save_json(run_dir / "checks.json", report.to_dict())

    if args.checks_only:
        print("\n--checks-only: stopping before any model is loaded.")
        return 0 if not report.failures() else 1

    # ---------------------------------------------------------------- environment
    print("\n" + RULE)
    print("ENVIRONMENT")
    print(RULE)
    context = ensure_repositories(plan, cache_root)
    context.update(init_frameworks(plan, config, accel, cache_root))
    context["ckpt_cache"] = cache_root / "checkpoints"
    context["ckpt_cache"].mkdir(parents=True, exist_ok=True)
    load_adapters(plan.resources.models)

    release = rt.get("release_model_after_use", "auto")
    if release == "auto":
        vram = accel.get("gpu_memory_mb") or 0
        release = bool(len(plan.resources.models) > 1
                       and (accel["kind"] != "gpu" or vram < 24_000
                            or len(plan.resources.frameworks) > 1))
    manager = ModelManager(config, plan, bool(release), context)

    writer = ResultWriter(run_dir, shard=shard)
    reconstructions: Dict[str, np.ndarray] = {}
    resume = bool(rt.get("resume", True))
    save_images = bool(rt.get("save_individual_images", True))
    keep_going = bool(rt.get("continue_on_experiment_error", True))
    do_warmup = not args.no_warmup

    # ---------------------------------------------------------------- model-major execution
    started = time.perf_counter()
    scheduled = plan.scheduled_models()
    print("\n" + RULE)
    print("EXECUTION -- model-major over %d atomic job(s), %d model(s)"
          % (len(plan.specs), len(scheduled)))
    print("  physical order : %s" % " -> ".join(m.upper() for m in scheduled))
    print("  logical order  : %s" % ", ".join(plan.experiment_names()))
    print("  release models : %s | resume finished jobs: %s" % (bool(release), resume))
    print(RULE)

    for position, model in enumerate(scheduled, 1):
        specs = sorted(plan.specs_for_model(model),
                       key=lambda s: (s.experiment, s.problem, s.method, s.t0,
                                      s.K or 0, s.reconstruction_steps, s.lam or 0.0))
        print("\n%s\n[%d/%d] MODEL %s -- %d job(s)\n%s"
              % (RULE, position, len(scheduled), model.upper(), len(specs), RULE))
        writer.event("model_start", model=model, jobs=len(specs))

        try:
            adapter = manager.acquire(model)
        except Exception as exc:
            message = "%s: %s" % (type(exc).__name__, exc)
            print("  MODEL LOAD FAILED: %s" % message)
            traceback.print_exc(limit=6)
            writer.event("model_load_failed", model=model, error=message)
            for spec in specs:                       # nothing is silently dropped
                writer.add(build_record(spec, plan, store.get(spec.problem_key), None, None,
                                        degraded_metrics[spec.problem_key], "skipped",
                                        "model load failed: %s" % message, 0.0, 0.0))
            if not keep_going:
                raise
            continue

        load_seconds = manager.load_seconds.get(model, 0.0)

        if args.check:
            probe = specs[0]
            checks_module.run_model_checks(adapter, store.get(probe.problem_key), probe,
                                           manager, report)
            # Fairness, measured rather than assumed: jobs that differ only by method or by
            # hyperparameters must start from a bit-identical z_t0.
            groups: Dict[Tuple, List] = {}
            for s in specs:
                groups.setdefault((s.problem_key, s.t0), []).append(s)
            for key, group in list(groups.items())[:3]:
                report.run(model, "shared_initial_state",
                           lambda g=group, k=key: checks_module.check_shared_initial_state(
                               adapter, store.get(k[0]), g, manager))
            save_json(run_dir / "checks.json", report.to_dict())

        warmed: Dict[Tuple, float] = {}
        for index, spec in enumerate(specs, 1):
            problem = store.get(spec.problem_key)
            print("\n  [%d/%d] %s" % (index, len(specs), spec.label))
            print("        job=%s guide=%s init=%s images=%d batch=%d"
                  % (spec.job_id, spec.initialization_guide_mode, spec.initialization_kind,
                     spec.num_images, spec.batch_size))
            writer.event("job_start", job_id=spec.job_id, experiment=spec.experiment,
                         model=model, method=spec.method, t0=spec.t0)

            if resume:
                finished = load_finished_job(run_dir, spec)
                if finished is not None:
                    record = finished["record"]
                    record["status"] = "ok"
                    reconstructions[spec.job_id] = (
                        np.asarray(finished["pixels"], np.float32) / 127.5) - 1.0
                    writer.add({k: record.get(k) for k in RESULT_COLUMNS})
                    print("        -> reused a finished job from %s" % finished["output_dir"])
                    continue

            # Every field below alters the time grid or the traced graph, so each
            # distinct combination needs its own untimed warm-up.
            warm_key = (spec.method, spec.batch_size, spec.problem,
                        tuple(problem.measurement.shape[1:]), spec.K, spec.t0,
                        spec.steps, spec.num_mpc_steps, spec.solver,
                        spec.phi_normalization, spec.control_cost_normalization)
            warm_seconds = warmed.get(warm_key, 0.0)
            try:
                if do_warmup and warm_key not in warmed:
                    warm_seconds = warm_up(adapter, spec, problem, manager)
                    warmed[warm_key] = warm_seconds

                run = run_job(adapter, spec, problem, manager)
                metrics = evaluate_reconstruction(run["pixels"], problem, spec)
                record = build_record(spec, plan, problem, run, metrics,
                                      degraded_metrics[spec.problem_key], run["status"],
                                      None if run["status"] == "ok" else run["status"],
                                      warm_seconds, load_seconds)
                out = persist_job(run_dir, spec, run, record, metrics, problem, save_images)
                record["output_dir"] = str(out)
                reconstructions[spec.job_id] = run["pixels"]
                writer.add(record, per_image_rows(record, metrics, problem.image_ids))
                print("        -> PSNR %.2f | SSIM %.4f | LPIPS %s | meas-RMSE %.4f"
                      % (record["psnr"], record["ssim"],
                         ("%.4f" % record["lpips"]) if record["lpips"] is not None else "n/a",
                         record["measurement_rmse"]))
                print("        -> %.1fs total, %.2fs/image | %d model eval(s) (expected %d) | "
                      "%d backprop(s)%s"
                      % (record["runtime"], record["runtime_per_image"],
                         record["model_evaluations"], record["expected_model_evaluations"],
                         record["backprops_through_model"],
                         " | padded %d" % record["padded_items"]
                         if record["padded_items"] else ""))
                writer.event("job_done", job_id=spec.job_id, psnr=record["psnr"],
                             lpips=record["lpips"], seconds=record["runtime"])
            except Exception as exc:
                message = "%s: %s" % (type(exc).__name__, exc)
                print("        FAILED: %s" % message)
                traceback.print_exc(limit=6)
                writer.add(build_record(spec, plan, problem, None, None,
                                        degraded_metrics[spec.problem_key], "failed",
                                        message, warm_seconds, load_seconds))
                writer.event("job_failed", job_id=spec.job_id, error=message)
                if not keep_going:
                    manager.release(model)
                    raise
            finally:
                free_memory()

        manager.clear_encode_cache(model=model)
        if release:
            manager.release(model)
        manager.memory_report("  memory after %s:" % model.upper())
        writer.event("model_done", model=model)

    manager.release_all()
    elapsed = time.perf_counter() - started

    # ---------------------------------------------------------------- reporting
    csv_path = writer.write_csv()
    ok = len([r for r in writer.records if r["status"] == "ok"])
    print("\n%s\nEXECUTION COMPLETE in %.1f s (%.1f min)" % (RULE, elapsed, elapsed / 60))
    print("  ok=%d  failed=%d  skipped=%d  (of %d atomic jobs)"
          % (ok, len([r for r in writer.records if r["status"] == "failed"]),
             len([r for r in writer.records if r["status"] == "skipped"]), len(plan.specs)))
    # Warm-up and loading are excluded from every reported runtime, so show them here --
    # otherwise a compilation problem hides inside the wall clock and looks like slow compute.
    print("  wall-clock breakdown (all EXCLUDED from the runtime column):")
    for model in scheduled:
        rows = [r for r in writer.records if r["model"] == model and r["status"] == "ok"]
        warm = sum(r.get("warmup_seconds") or 0.0 for r in rows)
        work = sum(r.get("runtime") or 0.0 for r in rows)
        print("    %-5s load %6.1f s | warm-up/compile %7.1f s | measured work %7.1f s%s"
              % (model.upper(), manager.load_seconds.get(model, 0.0), warm, work,
                 "   <-- compilation dominates; see README" if warm > 3 * max(work, 1e-9)
                 else ""))
    print(RULE)

    print_summary_tables(writer.records)

    figures: List[Path] = []
    if not args.no_figures:
        print("\nGenerating figures ...")
        try:
            figures = make_figures(run_dir, plan, store, writer.records, reconstructions,
                                   show=in_ipython())
        except Exception as exc:
            print("Figure generation failed (%s); results are unaffected." % exc)

    save_json(run_dir / "run_metadata.json", {
        "run_id": plan.run_id, "created": plan.created, "finished": now_iso(),
        "seed": plan.seed, "replicate": plan.replicate,
        "accelerator": accel, "elapsed_seconds": round(elapsed, 2),
        "model_provenance": manager.provenance,
        "model_load_seconds": manager.load_seconds,
        "repositories": context.get("repo_heads", {}),
        "problems": {k: p.to_metadata() for k, p in problems.items()},
        "degraded_metrics": degraded_metrics,
        "checks": report.to_dict(),
        "timing": timing_summary(),
        "figures": [str(p) for p in figures],
        "class_conditioning": (
            "Every reconstruction is conditioned on the TRUE ImageNet class of the source "
            "image, identically for SDEdit and both MPC methods. The benchmark therefore "
            "measures reconstruction given a known class, not blind restoration."),
        "warnings": list(plan.warnings) + list(accel_notes),
    })
    print("\nResults : %s" % csv_path)
    print("Run dir : %s" % run_dir)
    if figures:
        print("Figures : %d written to %s" % (len(figures), run_dir / "figures"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
