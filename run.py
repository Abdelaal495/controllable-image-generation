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
from src.models.base import Conditioning
from src.mpc import select_reconstructor
from src.problems import ProblemStore
from src.sdedit import ReconstructionStats
from src.utils import (RULE, THIN, detect_accelerator, free_memory, in_ipython, jsonable,
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
    return p.parse_args(argv)


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    rt = config.setdefault("runtime", {})
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
    repo_root = Path(cache_root) / "repos"
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
    from src.utils import MEMORY_HOOKS
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
        # Compatibility shims: aliases recent JAX removed but the repositories still use.
        for alias, target in (("tree_map", "tree_map"), ("tree_leaves", "tree_leaves"),
                              ("tree_flatten", "tree_flatten")):
            if not hasattr(jax, alias):
                setattr(jax, alias, getattr(jax.tree_util, target))
        jax.distributed.initialize = lambda *a, **k: None      # single-host Colab
        context["local_device_count"] = jax.local_device_count()
        MEMORY_HOOKS.append(lambda: jax.clear_caches())
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
    tiny = dataclasses.replace(
        spec,
        num_images=min(int(spec.batch_size), int(spec.num_images)),
        steps=1 if spec.method == "sdedit" else None,
        num_mpc_steps=None if spec.method == "sdedit" else 1,
        delta=None if spec.method == "sdedit" else spec.t0,
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

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.records: List[Dict[str, Any]] = []
        self.image_rows: List[Dict[str, Any]] = []
        self.jsonl = self.run_dir / "results.jsonl"
        self.log = self.run_dir / "experiment_log.jsonl"

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
        path = self.run_dir / "results.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in self.records:
                writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})
        if self.image_rows:
            per_image = self.run_dir / "results_per_image.csv"
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
    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        print("No successful job to summarise.")
        return

    print("\n" + RULE)
    print("RESULTS BY TASK / MODEL / METHOD   (mean over images and configurations)")
    print(RULE)
    header = ("%-18s %-5s %-12s %6s %8s %7s %8s %10s %9s"
              % ("task", "model", "method", "t0", "PSNR", "SSIM", "LPIPS", "meas-RMSE",
                 "s/image"))
    print(header)
    print(THIN)
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in ok:
        groups.setdefault((r["task"], r["model"], r["method"], r["t0"]), []).append(r)
    for key in sorted(groups):
        rows = groups[key]

        def avg(field):
            vals = [r[field] for r in rows if r.get(field) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        print("%-18s %-5s %-12s %6.2f %8.2f %7.4f %8s %10.4f %9.2f"
              % (key[0][:18], key[1], key[2][:12], key[3], avg("psnr"), avg("ssim"),
                 ("%.4f" % avg("lpips")) if not np.isnan(avg("lpips")) else "n/a",
                 avg("measurement_rmse"), avg("runtime_per_image")))

    # ---------------------------------------------------------------- paired deltas
    print("\n" + RULE)
    print("PAIRED COMPARISON vs SDEdit   (same image, y, model, t0 and epsilon)")
    print(RULE)
    print("%-18s %-5s %6s %-12s %9s %9s %10s %9s"
          % ("task", "model", "t0", "method", "dPSNR", "dSSIM", "dLPIPS", "runtime x"))
    print(THIN)
    any_pair = False
    for (task, model, t0) in sorted({(r["task"], r["model"], r["t0"]) for r in ok}):
        base = [r for r in ok if r["task"] == task and r["model"] == model
                and r["t0"] == t0 and r["method"] == "sdedit"]
        if not base:
            continue

        def mean_of(rows, field):
            vals = [r[field] for r in rows if r.get(field) is not None]
            return float(np.mean(vals)) if vals else None

        b_psnr, b_ssim = mean_of(base, "psnr"), mean_of(base, "ssim")
        b_lpips, b_rt = mean_of(base, "lpips"), mean_of(base, "runtime_per_image")
        for method in ("mpc_rhc", "mpc_delta_t"):
            rows = [r for r in ok if r["task"] == task and r["model"] == model
                    and r["t0"] == t0 and r["method"] == method]
            if not rows:
                continue
            any_pair = True
            m_psnr, m_ssim = mean_of(rows, "psnr"), mean_of(rows, "ssim")
            m_lpips, m_rt = mean_of(rows, "lpips"), mean_of(rows, "runtime_per_image")
            print("%-18s %-5s %6.2f %-12s %+9.2f %+9.4f %10s %8.1fx"
                  % (task[:18], model, t0, method[:12],
                     (m_psnr - b_psnr) if (m_psnr and b_psnr) else float("nan"),
                     (m_ssim - b_ssim) if (m_ssim and b_ssim) else float("nan"),
                     ("%+.4f" % (m_lpips - b_lpips)) if (m_lpips is not None
                                                         and b_lpips is not None) else "n/a",
                     (m_rt / b_rt) if (m_rt and b_rt) else float("nan")))
    if not any_pair:
        print("  (no method-paired group; add an SDEdit entry at the same t0 to enable this)")
    print(THIN)
    print("dLPIPS < 0 is an improvement.  'runtime x' is MPC time divided by SDEdit time at")
    print("the same t0: the compute price of whatever quality change is shown to its left.")

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
    summary = viz.plot_metric_summary(records, figures_dir / "summary_by_method.png", show)
    if summary:
        paths.append(summary)
    cost = viz.plot_quality_vs_cost(records, figures_dir / "quality_vs_cost.png", show)
    if cost:
        paths.append(cost)
    return paths


# =====================================================================================
# Main
# =====================================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    env = load_dotenv(".env")
    if env:
        print("Loaded .env keys: %s" % ", ".join(sorted(env)))

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

    run_dir = Path(plan.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "config.yaml", plan.raw_config)
    save_yaml(run_dir / "resolved_config.yaml", plan.to_dict())
    print("\nSaved %s and %s" % (run_dir / "config.yaml", run_dir / "resolved_config.yaml"))

    if args.dry_run:
        print("\n--dry-run: the plan above was validated and resolved. No model was loaded, "
              "no checkpoint downloaded and no job executed.")
        return 0

    # ---------------------------------------------------------------- data and problems
    cache_root = Path(plan.cache_root)
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

    writer = ResultWriter(run_dir)
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

            warm_key = (spec.method, spec.batch_size, spec.problem,
                        tuple(problem.measurement.shape[1:]), spec.K)
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
