"""Configuration: registries, strict validation, sweep expansion, and the run planner.

Carried over from the MPC-Flow notebook (sections 2-4) with two deliberate changes:

  * the configuration lives in `configs/experiments.yaml`, not in a notebook cell;
  * `standard_flow:` / `meanflow:` are no longer user-facing branches.  A model's dynamics
    family is a property of the model, so the user writes `models: {jit: ..., pmf: ...}`
    and the registry decides the family.

Unknown keys are errors rather than silent no-ops, and everything here runs *before* any
checkpoint is downloaded.
"""

from __future__ import annotations

import copy
import datetime
import itertools
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .utils import (FLOW_ASCENDING, MEANFLOW, MEANFLOW_DESCENDING, STANDARD_FLOW,
                    SEED_RECIPES, as_sweep_list, canonical_params_key, canonical_start_time,
                    deep_merge, native_time, stable_hash)


class ConfigError(ValueError):
    """Raised for configurations that cannot produce a meaningful run."""


# =====================================================================================
# Model registry -- technical definitions, carried over unchanged from the notebooks
# =====================================================================================
MODEL_REGISTRY_DEFAULTS: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------ iMeanFlow (JAX/latent)
    "imf": {
        "display_name": "iMF",
        "dynamics_family": MEANFLOW,
        "framework": "jax",
        "state_space": "latent",
        "native_shape": (32, 32, 4),           # H, W, C
        "layout": "BHWC",
        "pixel_resolution": 256,
        "prediction_kind": "mean_velocity",
        "native_time_mapping": MEANFLOW_DESCENDING,

        "repo_url": "https://github.com/Lyy-iiis/imeanflow.git",
        "repo_rev": "bf60cd7cb653f6628e59d48034b333c5eba445e2",
        "repo_dirname": "imeanflow",
        "hf_repo": "Lyy0725/iMF",
        "ckpt_file": "iMF-B-2.zip",
        "config_yml": "configs/eval_config.yml",
        "use_ema": True,                       # single EMA pytree (contrast with pMF)

        "vae_type": None,                      # None -> config.dataset.vae ("mse")
        "vae_encode_mode": "sample",
        "vae_decode_batch": 8,
        # Per-channel affine normalisation from utils/vae_util.LatentManager -- NOT 0.18215.
        "latent_mean": (0.86488, -0.27787343, 0.21616915, 0.3738409),
        "latent_std": (4.85503674, 5.31922414, 3.93725398, 3.9870003),

        "batch_size": 2,                       # FIXED -> one compiled shape
        "fixed_batch_shape": True,
        "num_classes": 1000,
        "null_label": 1000,
        "guidance": {"scale": 8.0, "interval": (0.40, 0.65), "mode": None},
        "packages": ("jax_stack", "diffusers", "flax", "orbax"),
        "accelerators": ("tpu", "gpu", "cpu"),
        "default_solver": None,
    },

    # ------------------------------------------------------------------ pixel MeanFlow (JAX)
    "pmf": {
        "display_name": "pMF",
        "dynamics_family": MEANFLOW,
        "framework": "jax",
        "state_space": "pixel",
        "native_shape": (256, 256, 3),
        "layout": "BHWC",
        "pixel_resolution": 256,
        "prediction_kind": "mean_velocity",
        "native_time_mapping": MEANFLOW_DESCENDING,

        "repo_url": "https://github.com/Lyy-iiis/pMF.git",
        "repo_rev": "75f6073042c21f7104686261a0c4784db4ede9d1",
        "repo_dirname": "pMF",
        "hf_repo": "Lyy0725/pMF",
        "ckpt_file": "pMF-L-16.zip",
        "config_yml": "configs/pMF_L_16_config.yml",
        "ema_key": 1000,                       # ema_params is a DICT keyed by EMA value

        "noise_scale": 1.0,
        "batch_size": 2,
        "fixed_batch_shape": True,
        "num_classes": 1000,
        "null_label": 1000,
        "guidance": {"scale": 7.0, "interval": (0.20, 0.70), "mode": None},
        "packages": ("jax_stack", "flax", "orbax"),
        "accelerators": ("tpu", "gpu", "cpu"),
        "default_solver": None,
    },

    # ------------------------------------------------------------------ SiT (Torch/latent)
    "sit": {
        "display_name": "SiT",
        "dynamics_family": STANDARD_FLOW,
        "framework": "torch",
        "state_space": "latent",
        "native_shape": (4, 32, 32),            # C, H, W
        "layout": "BCHW",
        "pixel_resolution": 256,
        "prediction_kind": "velocity",
        "native_time_mapping": FLOW_ASCENDING,

        "repo_url": "https://github.com/willisma/SiT.git",
        "repo_rev": None,
        "repo_dirname": "SiT",
        "variant": "SiT-XL/2",
        "checkpoint": "official",
        "vae_id": "stabilityai/sd-vae-ft-mse",
        "vae_encode_mode": "mean",
        "dtype": "auto",                        # bf16 where supported, else fp16, cpu -> fp32

        "batch_size": 2,
        "fixed_batch_shape": False,
        "num_classes": 1000,
        "null_label": 1000,
        "guidance": {"scale": 4.0, "interval": None, "mode": "official_first3"},
        "packages": ("torch_stack", "diffusers", "timm"),
        "accelerators": ("gpu", "cpu"),
        "default_solver": "heun",
        "euler_final_step_for_heun": False,
    },

    # ------------------------------------------------------------------ JiT (Torch/pixel)
    "jit": {
        "display_name": "JiT",
        "dynamics_family": STANDARD_FLOW,
        "framework": "torch",
        "state_space": "pixel",
        "native_shape": (3, 256, 256),
        "layout": "BCHW",
        "pixel_resolution": 256,
        "prediction_kind": "clean",             # predicts x_1; velocity is derived
        "native_time_mapping": FLOW_ASCENDING,

        "repo_url": "https://github.com/LTH14/JiT.git",
        "repo_rev": None,
        "repo_dirname": "JiT",
        "variant": "JiT-B/16",
        "checkpoint_backend": "hf_mirror",      # "hf_mirror" | "original_local"
        "hf_mirror_repo": "BiliSakura/JiT-diffusers",
        "local_checkpoint": None,
        "dtype": "safe_auto",                   # BF16 where supported, else FP32 (never FP16)
        "integration_dtype": "float32",         # solver/MPC state stays FP32 (official)
        "t_eps": 5e-2,                          # velocity denominator stabiliser near t = 1
        "noise_scale": 1.0,
        "recommended_cfg": {"JiT-B/16": 3.0, "JiT-L/16": 2.4, "JiT-H/16": 2.2},

        "batch_size": 2,
        "fixed_batch_shape": False,
        "num_classes": 1000,
        "null_label": 1000,
        "guidance": {"scale": None, "interval": (0.1, 1.0), "mode": None},  # None -> recommended
        "packages": ("torch_stack", "diffusers", "timm"),
        "accelerators": ("gpu", "cpu"),
        "default_solver": "heun",
        "euler_final_step_for_heun": True,      # official JiT sampler policy
    },
}

MODEL_NAMES: Tuple[str, ...] = tuple(MODEL_REGISTRY_DEFAULTS)


@dataclass(frozen=True)
class ModelCapabilities:
    name: str
    dynamics_family: str
    framework: str
    state_space: str
    prediction_kind: str
    supports_encoding: bool            # required for any t0 < 1 job (guide -> native space)
    supported_accelerators: Tuple[str, ...]
    fixed_batch_shape: bool
    native_time_mapping: str
    supported_solvers: Tuple[Optional[str], ...] = (None,)
    supported_methods: Tuple[str, ...] = ("sdedit", "mpc_rhc", "mpc_delta_t")

    def describe(self) -> str:
        solvers = [s for s in self.supported_solvers if s]
        return ("%-4s dynamics=%-13s fw=%-5s space=%-6s predicts=%-13s batch=%-5s%s"
                % (self.name, self.dynamics_family, self.framework, self.state_space,
                   self.prediction_kind, "fixed" if self.fixed_batch_shape else "free",
                   ("  solvers=%s" % solvers) if solvers else ""))


MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "imf": ModelCapabilities("imf", MEANFLOW, "jax", "latent", "mean_velocity", True,
                             ("tpu", "gpu", "cpu"), True, MEANFLOW_DESCENDING, (None,)),
    "pmf": ModelCapabilities("pmf", MEANFLOW, "jax", "pixel", "mean_velocity", True,
                             ("tpu", "gpu", "cpu"), True, MEANFLOW_DESCENDING, (None,)),
    "sit": ModelCapabilities("sit", STANDARD_FLOW, "torch", "latent", "velocity", True,
                             ("gpu", "cpu"), False, FLOW_ASCENDING,
                             ("euler", "heun", "rk4")),
    "jit": ModelCapabilities("jit", STANDARD_FLOW, "torch", "pixel", "clean", True,
                             ("gpu", "cpu"), False, FLOW_ASCENDING,
                             ("euler", "heun", "rk4")),
}


# =====================================================================================
# Inverse-problem declarations
# =====================================================================================
@dataclass(frozen=True)
class ProblemDeclaration:
    name: str
    title: str
    required_params: Tuple[str, ...]
    optional_params: Tuple[str, ...]
    guide_mode: str                      # ONE guide per problem; see README
    default_params: Dict[str, Any]
    description: str = ""


PROBLEM_DECLARATIONS: Dict[str, ProblemDeclaration] = {
    "denoising": ProblemDeclaration(
        "denoising", "Denoising", ("sigma",), (), "identity",
        {"sigma": 0.20},
        "A = I; y = x* + eta.  Guide g(y) = y."),
    "deblur": ProblemDeclaration(
        "deblur", "Gaussian deblurring", ("sigma", "blur_sigma"),
        ("kernel_size", "padding"), "observed",
        {"sigma": 0.05, "blur_sigma": 1.0, "kernel_size": 7, "padding": "reflect"},
        "Separable Gaussian blur.  Kernel SUPPORT is a choice here, not a paper value."),
    "super_resolution": ProblemDeclaration(
        "super_resolution", "Super-resolution x2", ("sigma", "factor"), (),
        "upsample_bicubic",
        {"sigma": 0.05, "factor": 2},
        "A keeps every second pixel; the guide bicubically lifts y back to 256x256."),
    "box_inpaint": ProblemDeclaration(
        "box_inpaint", "Box inpainting", ("sigma", "box"), (), "zero_fill",
        {"sigma": 0.05, "box": 40},
        "Central box removed.  Guide is the masked observation; no classical prefill."),
    "random_inpaint": ProblemDeclaration(
        "random_inpaint", "Random inpainting", ("sigma", "missing_fraction"), (), "zero_fill",
        {"sigma": 0.01, "missing_fraction": 0.70},
        "Pixels dropped at random.  Guide is the masked observation; no classical prefill."),
    "stroke_painting": ProblemDeclaration(
        "stroke_painting", "Stroke painting -> image", (), ("preset", "sigma", "blur_sigma"),
        "observed",
        {"preset": "medium", "sigma": 0.0, "blur_sigma": 0.8},
        "Frozen SLIC stroke geometry rendered differentiably; y = A_G(x*), no added noise."),
}

PROBLEM_NAMES: Tuple[str, ...] = tuple(PROBLEM_DECLARATIONS)
MASK_PROBLEMS: Tuple[str, ...] = ("box_inpaint", "random_inpaint")
STROKE_PRESETS: Dict[str, Dict[str, int]] = {
    "sparse": dict(n_segments=80, compactness=15, stroke_width=10),
    "medium": dict(n_segments=200, compactness=10, stroke_width=6),
    "dense": dict(n_segments=500, compactness=8, stroke_width=3),
}


# =====================================================================================
# Method declarations
# =====================================================================================
@dataclass(frozen=True)
class MethodDeclaration:
    name: str
    title: str
    is_mpc: bool
    uses_K: bool
    fields: Tuple[str, ...]
    description: str


SHARED_FIELDS: Tuple[str, ...] = ("t0",)
SDEDIT_FIELDS: Tuple[str, ...] = ("steps", "solver")
MPC_COMMON_FIELDS: Tuple[str, ...] = (
    "num_mpc_steps", "lam", "n_ctrl", "lr", "optimizer", "warm_start", "grad_clip",
    "phi_normalization", "control_cost_normalization")

METHOD_DECLARATIONS: Dict[str, MethodDeclaration] = {
    "sdedit": MethodDeclaration(
        "sdedit", "SDEdit", False, False, SHARED_FIELDS + SDEDIT_FIELDS,
        "Ordinary generation from z_t0 to the data endpoint.  No control, no measurement "
        "term: the measurement enters only through the initialisation guide."),
    "mpc_rhc": MethodDeclaration(
        "mpc_rhc", "MPC-RHC", True, True, SHARED_FIELDS + MPC_COMMON_FIELDS + ("K",),
        "Receding-horizon control over [s, 1] discretised into K steps (Algorithms 1 and 3). "
        "K = 1 does not backpropagate through the generative model."),
    "mpc_delta_t": MethodDeclaration(
        "mpc_delta_t", "MPC-delta_t", True, False,
        SHARED_FIELDS + MPC_COMMON_FIELDS + ("delta_t_lambda_scaling",),
        "Delta-t horizon control with a one-step value-function surrogate (Algorithm 2). "
        "K is meaningless here and is rejected."),
}

METHOD_NAMES: Tuple[str, ...] = tuple(METHOD_DECLARATIONS)

# The closed set of sweepable fields.  A list in ANY OTHER field is a literal value
# (e.g. guidance.interval, which is one interval and never a sweep).
SWEEPABLE_FIELDS: Tuple[str, ...] = (
    "t0", "steps", "solver", "K", "num_mpc_steps", "lam", "n_ctrl", "lr",
    "optimizer", "warm_start", "grad_clip",
    "phi_normalization", "control_cost_normalization", "delta_t_lambda_scaling",
)
MODEL_LEVEL_FIELDS: Tuple[str, ...] = ("guidance", "batch_size", "record_loss_history")

VALID_PHI_NORMALIZATIONS = ("half_sum_squared", "sum_squared", "mean_squared",
                            "gaussian_likelihood")
VALID_CONTROL_COST_NORMALIZATIONS = ("sum_squared", "mean_squared")
VALID_DELTA_T_LAMBDA_SCALINGS = ("none", "inverse_delta")
VALID_OPTIMIZERS = ("adam", "sgd")
VALID_SOLVERS = ("euler", "heun", "rk4")
SOLVER_STAGE_EVALUATIONS: Dict[str, int] = {"euler": 1, "heun": 2, "rk4": 4}
VALID_DATA_SOURCES = ("hf_imagenet_val", "local_folder")
VALID_ACCELERATORS = ("auto", "gpu", "tpu", "cpu")


# =====================================================================================
# Table E2 -- paper hyperparameters, used ONLY as fallback defaults
# =====================================================================================
# (lambda, N_ctrl, lr), MPC-Flow Appendix E.2, tuned on a CelebA 128x128 pixel-space U-Net.
# These are NOT tuned JiT/pMF values; every resolved job records where its value came from.
PAPER_E2: Dict[str, Any] = {
    "mpc_delta_t": {
        "denoising":        (15.0,  20, 0.1),
        "deblur":           (7.5,   20, 0.1),
        "super_resolution": (7.5,   20, 0.1),
        "random_inpaint":   (0.5,   20, 0.1),
        "box_inpaint":      (5.0,   20, 0.1),
    },
    # RHC is tabulated only for K = 1 and K = 3.
    "mpc_rhc": {
        1: {
            "denoising":        (0.1,   20, 0.1),
            "deblur":           (0.063, 20, 0.1),
            "super_resolution": (1.0,   20, 0.1),
            "random_inpaint":   (1.0,   20, 0.1),
            "box_inpaint":      (0.040, 20, 0.1),
        },
        3: {
            "denoising":        (0.1,   20, 0.05),
            "deblur":           (0.063, 20, 0.05),
            "super_resolution": (0.063, 20, 0.05),
            "random_inpaint":   (1.0,   20, 0.05),
            "box_inpaint":      (1.0,   20, 0.05),
        },
    },
}

# Stroke painting is NOT in Table E2 (the paper has no such task).  These are repository
# defaults, chosen conservatively, and they are labelled as such in every result row.
REPO_MPC_DEFAULTS: Dict[str, Tuple[float, int, float]] = {
    "mpc_rhc": (1.0, 20, 0.1),
    "mpc_delta_t": (5.0, 20, 0.1),
}


def resolve_mpc_hyperparameters(problem: str, method: str,
                                K: Optional[int]) -> Tuple[Dict[str, Any], str, List[str]]:
    """Return ({'lam','n_ctrl','lr'}, provenance_label, warnings) for a missing (lam,n_ctrl,lr).

    Provenance is one of
        "paper_E2"                      -- a value tabulated by MPC-Flow Appendix E.2
        "paper_E2_nearest_K(K=a->b)"    -- fallback CHOICE made here, not a published value
        "repository_default"            -- no paper entry exists for this problem
    """
    warnings_: List[str] = []
    table = PAPER_E2.get(method, {})
    if method == "mpc_delta_t":
        if problem in table:
            lam, n_ctrl, lr = table[problem]
            return {"lam": float(lam), "n_ctrl": int(n_ctrl), "lr": float(lr)}, "paper_E2", []
    elif method == "mpc_rhc":
        Kv = int(K if K is not None else 1)
        available = sorted(table)
        if available:
            if Kv in table:
                chosen, label = Kv, "paper_E2"
            else:
                chosen = min(available, key=lambda k: (abs(k - Kv), k))
                label = "paper_E2_nearest_K(K=%d->%d)" % (Kv, chosen)
                warnings_.append(
                    "K=%d is not tabulated in MPC-Flow Table E2 (only K in %s are). "
                    "lam/n_ctrl/lr fall back to K=%d; that fallback is a CHOICE made by this "
                    "repository and must not be reported as a published hyperparameter."
                    % (Kv, available, chosen))
            if problem in table[chosen]:
                lam, n_ctrl, lr = table[chosen][problem]
                return {"lam": float(lam), "n_ctrl": int(n_ctrl), "lr": float(lr)}, label, warnings_
    lam, n_ctrl, lr = REPO_MPC_DEFAULTS[method]
    warnings_.append(
        "%s/%s has no MPC-Flow Table E2 entry; using this repository's conservative default "
        "(lam=%g, n_ctrl=%d, lr=%g). Tune it before drawing conclusions."
        % (problem, method, lam, n_ctrl, lr))
    return {"lam": float(lam), "n_ctrl": int(n_ctrl), "lr": float(lr)}, "repository_default", \
        warnings_


def resolve_model_registry(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Registry defaults with the user's TECHNICAL overrides applied."""
    if name not in MODEL_REGISTRY_DEFAULTS:
        raise ConfigError("Unknown model %r (known: %s)" % (name, list(MODEL_NAMES)))
    return deep_merge(MODEL_REGISTRY_DEFAULTS[name], (config.get("models") or {}).get(name))


# =====================================================================================
# Validation
# =====================================================================================
TOP_LEVEL_KEYS = ("runtime", "data", "models", "defaults", "experiments", "metrics")
RUNTIME_KEYS = ("seed", "accelerator", "output_root", "cache_root",
                "release_model_after_use", "continue_on_experiment_error", "progress_bars",
                "save_individual_images", "verbose", "max_atomic_jobs", "resume",
                "warmup", "replicate")
DATA_KEYS = ("source", "local_folder", "image_size")
DEFAULTS_KEYS = SWEEPABLE_FIELDS + ("record_loss_history",)
EXPERIMENT_KEYS = ("enabled", "problem", "num_images", "degradation", "defaults",
                   "models", "notes")
MODEL_ENTRY_KEYS = ("methods",) + MODEL_LEVEL_FIELDS + SWEEPABLE_FIELDS
METHOD_ENTRY_KEYS = SWEEPABLE_FIELDS + ("record_loss_history",)

REMOVED_KEYS: Dict[str, str] = {
    "standard_flow": "Removed. A model's dynamics family is a property of the model, not a "
                     "user choice: write experiments.<name>.models.{jit,pmf,...} instead.",
    "meanflow": "Removed. See 'standard_flow'.",
    "enabled_models": "Removed. A model participates because it appears under an enabled "
                      "experiment.",
    "enabled_problems": "Removed. Each experiment names exactly one 'problem'.",
    "num_mpc_steps": "Not a global. Put it inside experiments.<name>.models.<model>."
                     "methods.<method>.num_mpc_steps (scalar or list).",
    "hyperparameters": "Removed. Table E2 is built in (PAPER_E2) and used as a fallback.",
    "initialization_guide_mode": "Removed. Each problem defines exactly one guide (see "
                                 "PROBLEM_DECLARATIONS); classical inpainting prefill is "
                                 "deliberately not available.",
    "preflight": "Renamed. Sanity checks are controlled by run.py --check / --no-check.",
}


def _require_mapping(value: Any, where: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("%s must be a mapping, got %s" % (where, type(value).__name__))
    return value


def _reject_unknown(block: Dict[str, Any], valid: Sequence[str], where: str,
                    removed: Optional[Dict[str, str]] = None) -> None:
    """Closed key set: unknown keys are errors, so no option can silently be a no-op."""
    for key in block:
        if removed and key in removed:
            raise ConfigError("%s.%s is no longer supported.\n    %s" % (where, key, removed[key]))
        if key not in valid:
            hint = ""
            close = [v for v in valid if isinstance(v, str)
                     and (v.startswith(str(key)[:3]) or str(key).startswith(v[:3]))]
            if close:
                hint = "\n    Did you mean: %s ?" % ", ".join(sorted(close)[:4])
            raise ConfigError("Unknown key %r in %s.\n    Valid keys: %s%s"
                              % (key, where, list(valid), hint))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    return _is_number(v) and math.isfinite(float(v))


def _check_sweep(entry: Dict[str, Any], field_name: str, where: str) -> List[Any]:
    values = as_sweep_list(entry[field_name])
    if len(values) == 0:
        raise ConfigError("%s.%s is an empty list; a sweep must contain at least one value."
                          % (where, field_name))
    return values


def _validate_sweep_values(field_name: str, values: Sequence[Any], where: str, method: str,
                           model: Optional[str], warnings_: List[str]) -> None:
    tag = "%s.%s" % (where, field_name)
    for v in values:
        if field_name == "t0":
            if not _finite(v):
                raise ConfigError("%s: t0 value %r is not a finite number." % (tag, v))
            if not 0.0 < float(v) <= 1.0:
                raise ConfigError(
                    "%s: t0=%r must lie in (0, 1]. t0 is the canonical corruption strength, "
                    "not a model-native time; t0=0 leaves nothing to generate." % (tag, v))
        elif field_name == "steps":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: steps must be an integer >= 1, got %r." % (tag, v))
        elif field_name == "solver":
            if v is not None and v not in VALID_SOLVERS:
                raise ConfigError("%s: unknown solver %r; valid: %s"
                                  % (tag, v, list(VALID_SOLVERS)))
            if model and v is not None:
                caps = MODEL_CAPABILITIES[model]
                if v not in caps.supported_solvers:
                    raise ConfigError(
                        "%s: %s does not support solver %r. %s is a MeanFlow model whose "
                        "dynamics are finite-interval transitions, not an instantaneous "
                        "velocity field, so ODE solvers do not apply."
                        % (tag, model.upper(), v, model.upper()))
        elif field_name == "K":
            if isinstance(v, bool) or not isinstance(v, int):
                raise ConfigError("%s: K must be an integer, got %r." % (tag, v))
            if v < 1:
                raise ConfigError("%s: K must be >= 1, got %d." % (tag, v))
            if v > 10:
                warnings_.append("%s: K=%d means %d nested differentiable model evaluations "
                                 "per control iteration; memory scales with K." % (tag, v, v - 1))
        elif field_name == "num_mpc_steps":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: num_mpc_steps must be an integer >= 1, got %r." % (tag, v))
        elif field_name == "lam":
            if not _finite(v) or float(v) < 0.0:
                raise ConfigError("%s: lam must be finite and non-negative, got %r." % (tag, v))
        elif field_name == "n_ctrl":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: n_ctrl must be an integer >= 1, got %r." % (tag, v))
        elif field_name == "lr":
            if not _finite(v) or float(v) <= 0.0:
                raise ConfigError("%s: lr must be a finite positive number, got %r." % (tag, v))
        elif field_name == "optimizer":
            if v not in VALID_OPTIMIZERS:
                raise ConfigError("%s: unknown optimizer %r; implemented in BOTH backends: %s"
                                  % (tag, v, list(VALID_OPTIMIZERS)))
        elif field_name == "warm_start":
            if not isinstance(v, bool):
                raise ConfigError("%s: warm_start must be a boolean, got %r." % (tag, v))
        elif field_name == "grad_clip":
            if v is not None and (not _finite(v) or float(v) <= 0.0):
                raise ConfigError("%s: grad_clip must be None or positive, got %r." % (tag, v))
        elif field_name == "phi_normalization":
            if v not in VALID_PHI_NORMALIZATIONS:
                raise ConfigError("%s: unknown phi_normalization %r; valid: %s"
                                  % (tag, v, list(VALID_PHI_NORMALIZATIONS)))
            if v != "half_sum_squared":
                warnings_.append("%s = %r changes the numerical scale of Phi, so Table E2's "
                                 "lambda values no longer mean what they meant in the paper. "
                                 "Retune lambda." % (tag, v))
        elif field_name == "control_cost_normalization":
            if v not in VALID_CONTROL_COST_NORMALIZATIONS:
                raise ConfigError("%s: unknown control_cost_normalization %r; valid: %s"
                                  % (tag, v, list(VALID_CONTROL_COST_NORMALIZATIONS)))
            if v != "sum_squared":
                warnings_.append("%s = %r rescales the control cost relative to Phi; lambda "
                                 "needs retuning." % (tag, v))
        elif field_name == "delta_t_lambda_scaling":
            if v not in VALID_DELTA_T_LAMBDA_SCALINGS:
                raise ConfigError("%s: unknown delta_t_lambda_scaling %r; valid: %s"
                                  % (tag, v, list(VALID_DELTA_T_LAMBDA_SCALINGS)))
        else:                                                       # pragma: no cover
            raise ConfigError("Sweepable field %r has no validation rule." % field_name)


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Hard-fail on impossible configurations; return the list of soft warnings."""
    warnings_: List[str] = []
    if not isinstance(config, dict):
        raise ConfigError("The configuration must be a mapping.")
    _reject_unknown(config, TOP_LEVEL_KEYS, "config", REMOVED_KEYS)
    for section in ("runtime", "data", "experiments"):
        if section not in config:
            raise ConfigError("The configuration is missing the required %r section." % section)

    # ------------------------------------------------------------------ runtime
    rt = _require_mapping(config["runtime"], "runtime")
    _reject_unknown(rt, RUNTIME_KEYS, "runtime")
    if rt.get("accelerator", "auto") not in VALID_ACCELERATORS:
        raise ConfigError("runtime.accelerator must be one of %s." % (VALID_ACCELERATORS,))
    if not isinstance(rt.get("seed", 42), int) or isinstance(rt.get("seed", 42), bool):
        raise ConfigError("runtime.seed must be an integer.")
    if int(rt.get("max_atomic_jobs", 256)) < 1:
        raise ConfigError("runtime.max_atomic_jobs must be >= 1.")
    if rt.get("release_model_after_use", "auto") not in ("auto", True, False):
        raise ConfigError("runtime.release_model_after_use must be 'auto', true or false.")
    rep = rt.get("replicate", 0)
    if isinstance(rep, bool) or not isinstance(rep, int) or rep < 0:
        raise ConfigError("runtime.replicate must be a non-negative integer (0 = the default "
                          "single trial).")

    # ------------------------------------------------------------------ data
    data = _require_mapping(config["data"], "data")
    _reject_unknown(data, DATA_KEYS, "data",
                    {"num_images": "Removed. Each experiment declares its own num_images."})
    if data.get("source", "hf_imagenet_val") not in VALID_DATA_SOURCES:
        raise ConfigError("data.source must be one of %s. Synthetic images are refused: PSNR "
                          "and LPIPS against procedural textures say nothing about an ImageNet "
                          "prior." % (VALID_DATA_SOURCES,))
    if data.get("source") == "local_folder" and not data.get("local_folder"):
        raise ConfigError("data.source='local_folder' requires data.local_folder.")
    if int(data.get("image_size", 256)) != 256:
        raise ConfigError("data.image_size must be 256: every supported checkpoint is 256x256.")

    # ------------------------------------------------------------------ models (technical only)
    models_cfg = _require_mapping(config.get("models"), "models")
    for name, block in models_cfg.items():
        if name not in MODEL_REGISTRY_DEFAULTS:
            raise ConfigError("Unknown model %r in models:; known: %s" % (name, list(MODEL_NAMES)))
        block = _require_mapping(block, "models.%s" % name)
        for key in block:
            if key in ("methods",) + SWEEPABLE_FIELDS:
                raise ConfigError(
                    "models.%s.%s is an EXPERIMENT setting, not a technical one. Put it under "
                    "experiments.<name>.models.%s." % (name, key, name))
            if key not in MODEL_REGISTRY_DEFAULTS[name]:
                raise ConfigError("Unknown technical key %r for model %r.\n    Valid keys: %s"
                                  % (key, name, sorted(MODEL_REGISTRY_DEFAULTS[name])))
    jit_cfg = resolve_model_registry(config, "jit")
    if jit_cfg["variant"] not in jit_cfg["recommended_cfg"]:
        raise ConfigError("Unknown JiT variant %r; known: %s"
                          % (jit_cfg["variant"], sorted(jit_cfg["recommended_cfg"])))
    if jit_cfg["checkpoint_backend"] not in ("hf_mirror", "original_local"):
        raise ConfigError("jit.checkpoint_backend must be 'hf_mirror' or 'original_local'.")
    if jit_cfg["checkpoint_backend"] == "original_local" and not jit_cfg.get("local_checkpoint"):
        raise ConfigError("jit.checkpoint_backend='original_local' requires jit.local_checkpoint.")
    if not (0.0 < float(jit_cfg["t_eps"]) < 1.0):
        raise ConfigError("jit.t_eps must lie in (0, 1).")
    sit_cfg = resolve_model_registry(config, "sit")
    if (sit_cfg["guidance"] or {}).get("mode") not in ("official_first3", "all_channels"):
        raise ConfigError("sit.guidance.mode must be 'official_first3' or 'all_channels'.")
    for name in MODEL_NAMES:
        reg = resolve_model_registry(config, name)
        if int(reg["batch_size"]) < 1:
            raise ConfigError("models.%s.batch_size must be >= 1." % name)
        if reg.get("vae_encode_mode") is not None and \
                reg["vae_encode_mode"] not in ("mean", "sample"):
            raise ConfigError("models.%s.vae_encode_mode must be 'mean' or 'sample'." % name)

    # ------------------------------------------------------------------ defaults
    defaults = _require_mapping(config.get("defaults"), "defaults")
    _reject_unknown(defaults, DEFAULTS_KEYS, "defaults")
    for key in defaults:
        if key in SWEEPABLE_FIELDS:
            _validate_sweep_values(key, _check_sweep(defaults, key, "defaults"), "defaults",
                                   "mpc_rhc", None, warnings_)
    if "record_loss_history" in defaults and not isinstance(defaults["record_loss_history"], bool):
        raise ConfigError("defaults.record_loss_history must be a boolean.")

    # ------------------------------------------------------------------ metrics
    metrics = _require_mapping(config.get("metrics"), "metrics")
    _reject_unknown(metrics, ("lpips", "lpips_net"), "metrics")
    if not metrics.get("lpips", True):
        warnings_.append("metrics.lpips is disabled, but LPIPS is a headline metric of this "
                         "benchmark; the results table will contain empty LPIPS columns.")

    # ------------------------------------------------------------------ experiments
    experiments = _require_mapping(config["experiments"], "experiments")
    if not experiments:
        raise ConfigError("experiments: is empty -- there is nothing to run.")
    any_enabled = False
    for exp_name, block in experiments.items():
        block = _require_mapping(block, "experiments.%s" % exp_name)
        where = "experiments.%s" % exp_name
        _reject_unknown(block, EXPERIMENT_KEYS, where, REMOVED_KEYS)
        if not block.get("enabled", False):
            continue
        any_enabled = True
        _validate_enabled_experiment(config, exp_name, block, where, warnings_)

    if not any_enabled:
        raise ConfigError("No experiment is enabled. Set `enabled: true` on at least one entry "
                          "under experiments:.")
    return warnings_


def _validate_enabled_experiment(config, exp_name, block, where, warnings_) -> None:
    problem = block.get("problem")
    if problem is None:
        raise ConfigError("%s is enabled but has no 'problem'." % where)
    if problem not in PROBLEM_DECLARATIONS:
        raise ConfigError("%s: unknown problem %r; known: %s"
                          % (where, problem, list(PROBLEM_NAMES)))
    decl = PROBLEM_DECLARATIONS[problem]

    n = block.get("num_images")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ConfigError("%s.num_images must be a positive integer, got %r." % (where, n))
    if n == 1:
        warnings_.append("%s runs on a single image; comparisons will be anecdotal." % where)

    params = _require_mapping(block.get("degradation"), "%s.degradation" % where)
    allowed = decl.required_params + decl.optional_params
    _reject_unknown(params, allowed, "%s.degradation" % where)
    merged = dict(decl.default_params)
    merged.update(params)
    for key in decl.required_params:
        if key not in merged:
            raise ConfigError("%s.degradation is missing required %r." % (where, key))
    _validate_problem_params(problem, merged, "%s.degradation" % where, warnings_)

    exp_defaults = _require_mapping(block.get("defaults"), "%s.defaults" % where)
    _reject_unknown(exp_defaults, DEFAULTS_KEYS, "%s.defaults" % where)
    for key in exp_defaults:
        if key in SWEEPABLE_FIELDS:
            _validate_sweep_values(key, _check_sweep(exp_defaults, key, "%s.defaults" % where),
                                   "%s.defaults" % where, "mpc_rhc", None, warnings_)

    models_block = _require_mapping(block.get("models"), "%s.models" % where)
    if not models_block:
        raise ConfigError("%s is enabled but no model participates. Add at least one entry "
                          "under its 'models:' block." % where)
    for model_name, entry in models_block.items():
        if model_name not in MODEL_REGISTRY_DEFAULTS:
            raise ConfigError("%s.models: unknown model %r; known: %s"
                              % (where, model_name, list(MODEL_NAMES)))
        _validate_model_entry(config, problem, decl, model_name,
                              _require_mapping(entry, "%s.models.%s" % (where, model_name)),
                              "%s.models.%s" % (where, model_name), warnings_)


def _validate_problem_params(problem: str, p: Dict[str, Any], where: str,
                             warnings_: List[str]) -> None:
    sigma = p.get("sigma", 0.0)
    if not _finite(sigma) or float(sigma) < 0.0:
        raise ConfigError("%s.sigma must be finite and non-negative, got %r." % (where, sigma))
    if float(sigma) == 0.0 and problem != "stroke_painting":
        warnings_.append("%s.sigma = 0 makes the measurement noiseless, which is not one of "
                         "the MPC-Flow paper's settings." % where)
    if problem == "super_resolution":
        factor = p.get("factor")
        if isinstance(factor, bool) or not isinstance(factor, int) or factor < 2:
            raise ConfigError("%s.factor must be an integer >= 2, got %r." % (where, factor))
        if factor != 2:
            warnings_.append("%s.factor = %d departs from the paper's x2 strided-subsampling "
                             "operator." % (where, factor))
        if 256 % int(factor) != 0:
            raise ConfigError("%s.factor must divide 256." % where)
    if problem == "deblur":
        if not _finite(p.get("blur_sigma")) or float(p["blur_sigma"]) <= 0.0:
            raise ConfigError("%s.blur_sigma must be a positive number." % where)
        ksize = p.get("kernel_size", 7)
        if isinstance(ksize, bool) or not isinstance(ksize, int) or ksize < 1 or ksize % 2 == 0:
            raise ConfigError("%s.kernel_size must be a positive ODD integer so the kernel is "
                              "centred, got %r." % (where, ksize))
        if ksize > 129:
            raise ConfigError("%s.kernel_size=%d is impractically large." % (where, ksize))
        if p.get("padding", "reflect") != "reflect":
            raise ConfigError("%s.padding: only 'reflect' padding is implemented (it is the one "
                              "the three backends compute identically by index gather)." % where)
        if ksize < 2 * int(math.ceil(3.0 * float(p["blur_sigma"]))) + 1:
            warnings_.append("%s: kernel_size=%d truncates a sigma_b=%.3g Gaussian below ~3 "
                             "sigma. The paper does not specify the kernel support, so this is "
                             "a choice made here." % (where, ksize, float(p["blur_sigma"])))
    if problem == "random_inpaint":
        frac = p.get("missing_fraction")
        if not _finite(frac) or not 0.0 < float(frac) < 1.0:
            raise ConfigError("%s.missing_fraction must lie strictly in (0, 1), got %r."
                              % (where, frac))
    if problem == "box_inpaint":
        box = p.get("box")
        if isinstance(box, bool) or not isinstance(box, int) or not 0 < box < 256:
            raise ConfigError("%s.box must be an integer in (0, 256), got %r." % (where, box))
    if problem == "stroke_painting":
        preset = p.get("preset", "medium")
        if preset not in STROKE_PRESETS:
            raise ConfigError("%s.preset must be one of %s, got %r."
                              % (where, list(STROKE_PRESETS), preset))
        if not _finite(p.get("blur_sigma", 0.8)) or float(p.get("blur_sigma", 0.8)) <= 0:
            raise ConfigError("%s.blur_sigma must be a positive number." % where)
        if float(p.get("sigma", 0.0)) > 0.0:
            warnings_.append("%s.sigma > 0 adds Gaussian noise to the stroke measurement; the "
                             "default for this task is a noiseless y = A_G(x*)." % where)


def _validate_model_entry(config, problem, decl, model_name, entry, where, warnings_) -> None:
    _reject_unknown(entry, MODEL_ENTRY_KEYS, where)
    caps = MODEL_CAPABILITIES[model_name]
    reg = resolve_model_registry(config, model_name)

    if "guidance" in entry:
        guidance = _require_mapping(entry["guidance"], "%s.guidance" % where)
        warnings_.append("%s overrides guidance with %s; the registry default (%s) is the value "
                         "published for this checkpoint, and guidance intervals are expressed "
                         "in MODEL-NATIVE time. 'interval' is one interval, never a sweep."
                         % (where, guidance, reg["guidance"]))
        if guidance.get("interval") is not None:
            iv = guidance["interval"]
            if not isinstance(iv, (list, tuple)) or len(iv) != 2:
                raise ConfigError("%s.guidance.interval must be a 2-element [lo, hi]." % where)
            lo, hi = float(iv[0]), float(iv[1])
            if not 0.0 <= lo < hi <= 1.0:
                raise ConfigError("%s.guidance.interval must satisfy 0 <= lo < hi <= 1." % where)
        if guidance.get("scale") is not None and not _finite(guidance["scale"]):
            raise ConfigError("%s.guidance.scale must be a finite number or null." % where)
    if "batch_size" in entry:
        bs = entry["batch_size"]
        if isinstance(bs, bool) or not isinstance(bs, int) or bs < 1:
            raise ConfigError("%s.batch_size must be a positive integer." % where)
        if caps.fixed_batch_shape:
            warnings_.append("%s overrides batch_size for a fixed-compiled-batch model; the "
                             "change costs exactly one recompilation." % where)
    if "record_loss_history" in entry and not isinstance(entry["record_loss_history"], bool):
        raise ConfigError("%s.record_loss_history must be a boolean." % where)

    for key in entry:
        if key in SWEEPABLE_FIELDS:
            _validate_sweep_values(key, _check_sweep(entry, key, where), where, "mpc_rhc",
                                   model_name, warnings_)

    methods = _require_mapping(entry.get("methods"), "%s.methods" % where)
    if not methods:
        raise ConfigError("%s has no 'methods'. Add at least one of %s."
                          % (where, list(METHOD_NAMES)))
    for method_name, method_entry in methods.items():
        if method_name not in METHOD_DECLARATIONS:
            raise ConfigError("%s.methods: unknown method %r; known: %s"
                              % (where, method_name, list(METHOD_NAMES)))
        if method_name not in caps.supported_methods:
            raise ConfigError("%s: %s does not support method %r; supported: %s"
                              % (where, model_name.upper(), method_name,
                                 list(caps.supported_methods)))
        _validate_method_entry(config, problem, decl, model_name, caps, method_name,
                               _require_mapping(method_entry,
                                                "%s.methods.%s" % (where, method_name)),
                               "%s.methods.%s" % (where, method_name), entry, warnings_)


def _validate_method_entry(config, problem, decl, model_name, caps, method_name, entry, where,
                           inherited_model_entry, warnings_) -> None:
    _reject_unknown(entry, METHOD_ENTRY_KEYS, where)
    method_decl = METHOD_DECLARATIONS[method_name]
    allowed = set(method_decl.fields) | {"record_loss_history"}

    for key in entry:
        if key in SWEEPABLE_FIELDS and key not in allowed:
            if key == "K":
                raise ConfigError(
                    "%s: 'K' is meaningless for %s and is rejected.\n"
                    "    %s has no planning discretisation to choose: SDEdit simply integrates, "
                    "and MPC-delta_t optimises a single control over one interval of length "
                    "delta.\n    Use method 'mpc_rhc' if you want a planning depth K."
                    % (where, method_name, method_decl.title))
            raise ConfigError(
                "%s: %r does not apply to method %r.\n    Fields for %s: %s"
                % (where, key, method_name, method_decl.title, sorted(allowed)))
    if "K" in inherited_model_entry and not method_decl.uses_K:
        warnings_.append("%s inherits a model-level 'K', which %s ignores."
                         % (where, method_name))

    for key in entry:
        if key in SWEEPABLE_FIELDS:
            _validate_sweep_values(key, _check_sweep(entry, key, where), where, method_name,
                                   model_name, warnings_)
    if "record_loss_history" in entry and not isinstance(entry["record_loss_history"], bool):
        raise ConfigError("%s.record_loss_history must be a boolean." % where)

    t0_values = [float(v) for v in as_sweep_list(
        entry.get("t0", inherited_model_entry.get("t0", 0.8)))]
    if any(t < 1.0 for t in t0_values) and not caps.supports_encoding:
        raise ConfigError("%s: t0 < 1 needs the measurement-derived guide encoded into %s's "
                          "native state space, which this adapter does not support."
                          % (where, model_name.upper()))

    # The t0 / num_mpc_steps confound (see README).
    if method_decl.is_mpc and len(set(t0_values)) > 1:
        n_steps = as_sweep_list(entry.get("num_mpc_steps",
                                          inherited_model_entry.get("num_mpc_steps", 4)))
        if len(set(n_steps)) == 1:
            warnings_.append(
                "%s sweeps t0 over %s at a FIXED num_mpc_steps=%s, so delta = t0/N varies "
                "across the sweep (%s). Equal step counts do NOT mean equal execution "
                "resolution."
                % (where, sorted(set(t0_values)), n_steps[0],
                   ", ".join("t0=%.2f->delta=%.4g" % (t, t / float(n_steps[0]))
                             for t in sorted(set(t0_values)))))


# =====================================================================================
# Resolved atomic specs
# =====================================================================================
@dataclass(frozen=True)
class JobSpec:
    """One ATOMIC job: everything the runner needs, decided here and never re-derived."""
    # -- identity ---------------------------------------------------------------------
    job_id: str
    experiment: str
    problem: str
    problem_params: Dict[str, Any]
    problem_key: str                     # -> the shared InverseProblem instance
    num_images: int
    image_ids: Tuple[str, ...]

    # -- model ------------------------------------------------------------------------
    model: str
    dynamics_family: str                 # STANDARD_FLOW | MEANFLOW
    framework: str                       # "jax" | "torch"
    state_space: str                     # "latent" | "pixel"
    batch_size: int
    fixed_batch_shape: bool
    guidance: Dict[str, Any]

    # -- method -----------------------------------------------------------------------
    method: str                          # "sdedit" | "mpc_rhc" | "mpc_delta_t"
    K: Optional[int]                     # mpc_rhc only

    # -- initialisation / time --------------------------------------------------------
    t0: float
    canonical_start_time: float          # s_start = 1 - t0
    native_start_time: float
    native_end_time: float
    native_time_mapping: str
    initialization_guide_mode: str
    initialization_kind: str             # "pure_prior_noise" | "measurement_informed"

    # -- SDEdit -----------------------------------------------------------------------
    steps: Optional[int]
    solver: Optional[str]

    # -- MPC --------------------------------------------------------------------------
    num_mpc_steps: Optional[int]
    delta: Optional[float]               # = t0 / num_mpc_steps
    lam: Optional[float]
    n_ctrl: Optional[int]
    lr: Optional[float]
    phi_normalization: Optional[str]
    control_cost_normalization: Optional[str]
    delta_t_lambda_scaling: Optional[str]
    optimizer: Optional[str]
    warm_start: Optional[bool]
    grad_clip: Optional[float]
    record_loss_history: bool

    # -- reproducibility --------------------------------------------------------------
    seed: int
    replicate: int
    seed_recipes: Dict[str, str]
    hyperparameter_sources: Dict[str, str]

    # -- per-image cost estimates -----------------------------------------------------
    expected_control_iterations: int
    expected_model_evals: int
    expected_backprops: int

    @property
    def is_mpc(self) -> bool:
        return METHOD_DECLARATIONS[self.method].is_mpc

    @property
    def num_chunks(self) -> int:
        """Batched passes covering this job's images (one network call serves a batch)."""
        return max(1, -(-int(self.num_images) // max(1, int(self.batch_size))))

    @property
    def method_title(self) -> str:
        if self.method == "mpc_rhc":
            return "MPC-RHC (K=%d)" % (self.K or 1)
        if self.method == "mpc_delta_t":
            return "MPC-delta_t"
        return "SDEdit"

    @property
    def reconstruction_steps(self) -> int:
        """Outer trajectory intervals, whatever the method calls them."""
        return int(self.steps if self.method == "sdedit" else self.num_mpc_steps)

    @property
    def leaf_dir(self) -> str:
        """Readable, collision-free directory name for one atomic job."""
        parts = ["t0=%.3f" % self.t0]
        if self.method == "sdedit":
            parts += ["steps=%d" % self.steps]
            if self.solver:
                parts.append(self.solver)
        else:
            parts += ["N=%d" % self.num_mpc_steps]
            if self.K is not None:
                parts.append("K=%d" % self.K)
            parts += ["lam=%g" % self.lam, "nctrl=%d" % self.n_ctrl, "lr=%g" % self.lr,
                      self.optimizer]
            if self.warm_start:
                parts.append("warm")
            if self.grad_clip:
                parts.append("clip=%g" % self.grad_clip)
            if self.phi_normalization != "half_sum_squared":
                parts.append("phi=%s" % self.phi_normalization)
            if self.control_cost_normalization != "sum_squared":
                parts.append("cc=%s" % self.control_cost_normalization)
            if self.delta_t_lambda_scaling not in (None, "none"):
                parts.append("lamscale=%s" % self.delta_t_lambda_scaling)
        if self.replicate:
            parts.append("rep=%d" % self.replicate)
        return "__".join(parts) + "__" + self.job_id[:8]

    @property
    def label(self) -> str:
        if self.method == "sdedit":
            return "%s | %s/%s | SDEdit | t0=%.2f steps=%d %s" % (
                self.experiment, self.model, self.problem, self.t0, self.steps,
                self.solver or "")
        return ("%s | %s/%s | %s | t0=%.2f N=%d delta=%.4g lam=%g nctrl=%d lr=%g"
                % (self.experiment, self.model, self.problem, self.method_title, self.t0,
                   self.num_mpc_steps, self.delta, self.lam, self.n_ctrl, self.lr))

    def figure_title(self) -> str:
        """Compact multi-line label for comparison grids (section 40 of the brief)."""
        lines = [MODEL_REGISTRY_DEFAULTS[self.model]["display_name"]]
        if self.method == "sdedit":
            lines.append("SDEdit")
            lines.append("t0=%.2g n=%d" % (self.t0, self.steps))
            if self.solver:
                lines.append(self.solver)
        elif self.method == "mpc_rhc":
            lines.append("MPC-RHC")
            lines.append("t0=%.2g K=%d" % (self.t0, self.K or 1))
            lines.append("N=%d lam=%g" % (self.num_mpc_steps, self.lam))
        else:
            lines.append("MPC-dt")
            lines.append("t0=%.2g N=%d" % (self.t0, self.num_mpc_steps))
            lines.append("lam=%g" % self.lam)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProblemRequest:
    """One shared inverse-problem specification, realised exactly once."""
    key: str
    problem: str
    params: Dict[str, Any]
    num_images: int
    seed: int
    guide_mode: str
    experiments: Tuple[str, ...]


@dataclass
class ResourceRequirements:
    models: Tuple[str, ...]
    frameworks: Tuple[str, ...]
    repositories: Dict[str, Dict[str, Any]]
    packages: Tuple[str, ...]
    accelerators_supported: Dict[str, Tuple[str, ...]]
    data_source: str
    source_pool_size: int
    problems: Tuple[str, ...]
    needs_lpips: bool


@dataclass
class RunPlan:
    run_id: str
    created: str
    seed: int
    replicate: int
    output_dir: str
    cache_root: str
    resources: ResourceRequirements
    problems: Tuple[ProblemRequest, ...]
    specs: Tuple[JobSpec, ...]
    warnings: Tuple[str, ...]
    workload: Dict[str, Any]
    raw_config: Dict[str, Any]

    def specs_for_model(self, model: str) -> List[JobSpec]:
        return [s for s in self.specs if s.model == model]

    def scheduled_models(self) -> List[str]:
        """Model-major PHYSICAL order, deliberately distinct from experiment order."""
        return [m for m in MODEL_NAMES if any(s.model == m for s in self.specs)]

    def experiment_names(self) -> List[str]:
        """Logical order: the order the user wrote the experiments in."""
        return [name for name in self.raw_config["experiments"]
                if any(s.experiment == name for s in self.specs)]

    def problem_request(self, key: str) -> ProblemRequest:
        for p in self.problems:
            if p.key == key:
                return p
        raise KeyError("No problem request %r in the plan." % key)

    def to_dict(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "created": self.created, "seed": self.seed,
                "replicate": self.replicate, "output_dir": self.output_dir,
                "cache_root": self.cache_root, "resources": asdict(self.resources),
                "problems": [asdict(p) for p in self.problems],
                "jobs": [s.to_dict() for s in self.specs],
                "warnings": list(self.warnings), "workload": self.workload}


# =====================================================================================
# Inheritance and sweep expansion
# =====================================================================================
BUILTIN_DEFAULTS: Dict[str, Any] = {
    "t0": 0.8,
    "steps": 50,
    "solver": None,                              # -> the model's default_solver
    "num_mpc_steps": 4,
    "phi_normalization": "half_sum_squared",
    "control_cost_normalization": "sum_squared",
    "delta_t_lambda_scaling": "none",
    "optimizer": "adam",
    "warm_start": False,
    "grad_clip": None,
    "record_loss_history": True,
}


def _inherit(field_name: str, layers: Sequence[Tuple[str, Dict[str, Any]]],
             fallback: Any, fallback_source: str):
    """Walk the precedence chain lowest -> highest; return (value, source)."""
    value, source = fallback, fallback_source
    for label, block in layers:
        if block and field_name in block:
            value, source = block[field_name], label
    return value, source


def _estimate_cost(spec_method: str, K: Optional[int], n_ctrl: int, num_steps: int,
                   dynamics_family: str, solver: Optional[str],
                   euler_final_step_for_heun: bool = False) -> Dict[str, int]:
    """Per-chunk compute estimate.  Measured counts are recorded alongside these at run time.

    SDEdit:   standard flow -> stage evaluations per step (euler 1, heun 2, rk4 4);
              MeanFlow      -> one learned transition per interval.
    MPC-RHC:  1 hoisted eval per replan + n_ctrl*(K-1) differentiable evals
              + MeanFlow only: 1 extra execution transition T(x; s -> s+delta).
    MPC-dt:   1 hoisted eval + n_ctrl look-ahead evals per replan, except the FINAL replan
              where s+delta = 1 makes the projection factor exactly zero.
    """
    if spec_method == "sdedit":
        if dynamics_family == STANDARD_FLOW:
            evals = num_steps * SOLVER_STAGE_EVALUATIONS.get(solver or "euler", 1)
            if solver == "heun" and euler_final_step_for_heun:
                evals -= 1        # the official JiT sampler takes its last step with Euler
        else:
            evals = num_steps
        return {"control_iterations": 0, "model_evals": evals, "backprops": 0}
    if spec_method == "mpc_rhc":
        k = int(K or 1)
        per_replan_planning = n_ctrl * max(0, k - 1)
        per_replan_total = 1 + per_replan_planning + (1 if dynamics_family == MEANFLOW else 0)
        return {"control_iterations": n_ctrl * num_steps,
                "model_evals": per_replan_total * num_steps,
                "backprops": per_replan_planning * num_steps}
    planning_steps = max(0, num_steps - 1)
    return {"control_iterations": n_ctrl * num_steps,
            "model_evals": num_steps + n_ctrl * planning_steps,
            "backprops": n_ctrl * planning_steps}


def resolve_run_plan(config: Dict[str, Any], warnings_: Sequence[str] = (),
                     run_id: Optional[str] = None) -> RunPlan:
    """CONFIG -> fully resolved atomic JobSpec objects.  Nothing expensive happens here."""
    warnings_ = list(warnings_)
    rt = config["runtime"]
    seed = int(rt.get("seed", 42))
    replicate = int(rt.get("replicate", 0))
    run_id = run_id or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = os.path.join(rt.get("output_root", "outputs"), run_id)
    cache_root = rt.get("cache_root", "cache")
    global_defaults = config.get("defaults") or {}

    specs: List[JobSpec] = []
    problem_entries: Dict[str, Dict[str, Any]] = {}
    pool = 0
    seen_hp_warnings = set()

    for exp_name, block in config["experiments"].items():
        if not block.get("enabled"):
            continue
        problem = block["problem"]
        decl = PROBLEM_DECLARATIONS[problem]
        num_images = int(block["num_images"])
        pool = max(pool, num_images)

        problem_params = dict(decl.default_params)
        problem_params.update(block.get("degradation") or {})
        params_key = canonical_params_key(problem_params)
        # The problem identity depends on (problem, parameters, seed) ONLY -- never on the
        # experiment name, the model or the method, so two experiments asking for the same
        # specification provably share one measurement.
        pkey = "prob_" + stable_hash(problem, params_key, seed)
        entry = problem_entries.setdefault(pkey, {
            "key": pkey, "problem": problem, "params": problem_params, "num_images": num_images,
            "seed": seed, "guide_mode": decl.guide_mode, "experiments": []})
        entry["experiments"].append(exp_name)
        # A larger request supersedes a smaller one: same seed, same per-image draws, so the
        # smaller set is a prefix of the larger.
        entry["num_images"] = max(entry["num_images"], num_images)

        exp_defaults = block.get("defaults") or {}

        for model_name, model_entry in (block.get("models") or {}).items():
            reg = resolve_model_registry(config, model_name)
            caps = MODEL_CAPABILITIES[model_name]
            model_entry = model_entry or {}

            guidance = deep_merge(dict(reg["guidance"]), model_entry.get("guidance"))
            if model_name == "jit" and guidance.get("scale") is None:
                guidance = dict(guidance, scale=reg["recommended_cfg"].get(reg["variant"]),
                                scale_source="model_size_recommendation")
            batch_size = int(model_entry.get("batch_size", reg["batch_size"]))

            for method_name, method_entry in (model_entry.get("methods") or {}).items():
                method_entry = method_entry or {}
                mdecl = METHOD_DECLARATIONS[method_name]
                layers = [("defaults", global_defaults), ("experiment", exp_defaults),
                          ("model", model_entry), ("method", method_entry)]

                axes: Dict[str, List[Any]] = {}
                axis_sources: Dict[str, str] = {}

                def axis(name: str, fallback: Any, fallback_source: str = "builtin") -> None:
                    value, source = _inherit(name, layers, fallback, fallback_source)
                    axes[name] = as_sweep_list(value)
                    axis_sources[name] = source

                axis("t0", BUILTIN_DEFAULTS["t0"])
                if method_name == "sdedit":
                    axis("steps", BUILTIN_DEFAULTS["steps"])
                    axis("solver", reg.get("default_solver"), "model_registry")
                    for name in ("K", "num_mpc_steps", "lam", "n_ctrl", "lr", "optimizer",
                                 "warm_start", "grad_clip", "phi_normalization",
                                 "control_cost_normalization", "delta_t_lambda_scaling"):
                        axes[name], axis_sources[name] = [None], "not_applicable"
                else:
                    axes["steps"], axis_sources["steps"] = [None], "not_applicable"
                    axes["solver"], axis_sources["solver"] = [None], "not_applicable"
                    axis("num_mpc_steps", BUILTIN_DEFAULTS["num_mpc_steps"])
                    for name in ("optimizer", "warm_start", "grad_clip", "phi_normalization",
                                 "control_cost_normalization"):
                        axis(name, BUILTIN_DEFAULTS[name])
                    if mdecl.uses_K:
                        axis("K", 1)
                    else:
                        axes["K"], axis_sources["K"] = [None], "not_applicable"
                    if method_name == "mpc_delta_t":
                        axis("delta_t_lambda_scaling",
                             BUILTIN_DEFAULTS["delta_t_lambda_scaling"])
                    else:
                        axes["delta_t_lambda_scaling"] = [None]
                        axis_sources["delta_t_lambda_scaling"] = "not_applicable"
                    # lam / n_ctrl / lr have no builtin fallback: their default depends on
                    # (problem, method, K) and is resolved per combination below.
                    for name in ("lam", "n_ctrl", "lr"):
                        value, source = _inherit(name, layers, "__ABSENT__", "table")
                        axes[name] = (["__ABSENT__"] if value == "__ABSENT__"
                                      else as_sweep_list(value))
                        axis_sources[name] = source

                record_loss, _ = _inherit("record_loss_history", layers,
                                          BUILTIN_DEFAULTS["record_loss_history"], "builtin")

                order = ("t0", "steps", "solver", "K", "num_mpc_steps", "lam", "n_ctrl", "lr",
                         "optimizer", "warm_start", "grad_clip", "phi_normalization",
                         "control_cost_normalization", "delta_t_lambda_scaling")
                grids = []
                for name in order:
                    seen_vals, uniq = set(), []
                    for v in axes[name]:
                        if repr(v) not in seen_vals:
                            seen_vals.add(repr(v))
                            uniq.append(v)
                    grids.append(uniq)

                for combo in itertools.product(*grids):
                    values = dict(zip(order, combo))
                    t0 = float(values["t0"])
                    s_start = canonical_start_time(t0)
                    sources = dict(axis_sources)

                    if method_name == "sdedit":
                        n_steps = None
                        delta = None
                        hp = {"lam": None, "n_ctrl": None, "lr": None}
                    else:
                        n_steps = int(values["num_mpc_steps"])
                        delta = t0 / n_steps
                        fallback, label, hp_warn = resolve_mpc_hyperparameters(
                            problem, method_name, values["K"])
                        for w in hp_warn:
                            tag = (problem, method_name, values["K"], w)
                            if tag not in seen_hp_warnings:
                                seen_hp_warnings.add(tag)
                                warnings_.append("experiments.%s.models.%s.methods.%s: %s"
                                                 % (exp_name, model_name, method_name, w))
                        hp = {}
                        for name in ("lam", "n_ctrl", "lr"):
                            if values[name] == "__ABSENT__":
                                hp[name] = fallback[name]
                                sources[name] = label
                            else:
                                hp[name] = values[name]

                    cost = _estimate_cost(method_name, values["K"],
                                          int(hp["n_ctrl"] or 0),
                                          int(values["steps"] or n_steps or 1),
                                          reg["dynamics_family"], values["solver"],
                                          bool(reg.get("euler_final_step_for_heun", False)))
                    job_id = stable_hash(run_id, exp_name, problem, params_key, model_name,
                                         method_name, values["K"], t0, values["steps"],
                                         values["solver"], n_steps, hp["lam"], hp["n_ctrl"],
                                         hp["lr"], values["optimizer"], values["warm_start"],
                                         values["grad_clip"], values["phi_normalization"],
                                         values["control_cost_normalization"],
                                         values["delta_t_lambda_scaling"], num_images,
                                         replicate, size=6)

                    specs.append(JobSpec(
                        job_id=job_id, experiment=exp_name, problem=problem,
                        problem_params=dict(problem_params), problem_key=pkey,
                        num_images=num_images, image_ids=(),
                        model=model_name, dynamics_family=reg["dynamics_family"],
                        framework=reg["framework"], state_space=reg["state_space"],
                        batch_size=batch_size, fixed_batch_shape=caps.fixed_batch_shape,
                        guidance=dict(guidance),
                        method=method_name,
                        K=(int(values["K"]) if values["K"] is not None else None),
                        t0=t0, canonical_start_time=s_start,
                        native_start_time=native_time(s_start, reg["native_time_mapping"]),
                        native_end_time=native_time(1.0, reg["native_time_mapping"]),
                        native_time_mapping=reg["native_time_mapping"],
                        initialization_guide_mode=decl.guide_mode,
                        initialization_kind=("pure_prior_noise" if t0 >= 1.0
                                             else "measurement_informed"),
                        steps=(int(values["steps"]) if values["steps"] is not None else None),
                        solver=values["solver"],
                        num_mpc_steps=n_steps, delta=delta,
                        lam=(None if hp["lam"] is None else float(hp["lam"])),
                        n_ctrl=(None if hp["n_ctrl"] is None else int(hp["n_ctrl"])),
                        lr=(None if hp["lr"] is None else float(hp["lr"])),
                        phi_normalization=values["phi_normalization"],
                        control_cost_normalization=values["control_cost_normalization"],
                        delta_t_lambda_scaling=values["delta_t_lambda_scaling"],
                        optimizer=values["optimizer"],
                        warm_start=(None if values["warm_start"] is None
                                    else bool(values["warm_start"])),
                        grad_clip=(None if values["grad_clip"] is None
                                   else float(values["grad_clip"])),
                        record_loss_history=bool(record_loss),
                        seed=seed, replicate=replicate, seed_recipes=dict(SEED_RECIPES),
                        hyperparameter_sources=sources,
                        expected_control_iterations=cost["control_iterations"],
                        expected_model_evals=cost["model_evals"],
                        expected_backprops=cost["backprops"]))

    # ---------------------------------------------------------------- resources
    required = [m for m in MODEL_NAMES if any(s.model == m for s in specs)]
    frameworks = sorted({resolve_model_registry(config, m)["framework"] for m in required})
    repositories = {m: {"url": resolve_model_registry(config, m)["repo_url"],
                        "rev": resolve_model_registry(config, m).get("repo_rev"),
                        "dirname": resolve_model_registry(config, m)["repo_dirname"]}
                    for m in required}
    packages = sorted({p for m in required
                       for p in resolve_model_registry(config, m)["packages"]})
    accel_support = {m: tuple(resolve_model_registry(config, m)["accelerators"])
                     for m in required}

    resources = ResourceRequirements(
        models=tuple(required), frameworks=tuple(frameworks), repositories=repositories,
        packages=tuple(packages), accelerators_supported=accel_support,
        data_source=config["data"].get("source", "hf_imagenet_val"),
        source_pool_size=pool, problems=tuple(dict.fromkeys(s.problem for s in specs)),
        needs_lpips=bool((config.get("metrics") or {}).get("lpips", True)))

    # ---------------------------------------------------------------- workload
    def bucket(keyfn):
        out: Dict[Any, Dict[str, int]] = {}
        for s in specs:
            e = out.setdefault(keyfn(s), {"jobs": 0, "reconstructions": 0,
                                          "control_iterations": 0, "model_evals": 0,
                                          "backprops": 0})
            e["jobs"] += 1
            e["reconstructions"] += s.num_images
            e["control_iterations"] += s.expected_control_iterations * s.num_chunks
            e["model_evals"] += s.expected_model_evals * s.num_chunks
            e["backprops"] += s.expected_backprops * s.num_chunks
        return out

    workload = {
        "enabled_experiments": len({s.experiment for s in specs}),
        "atomic_jobs": len(specs),
        "image_reconstructions": sum(s.num_images for s in specs),
        "control_iterations": sum(s.expected_control_iterations * s.num_chunks for s in specs),
        "model_evaluations": sum(s.expected_model_evals * s.num_chunks for s in specs),
        "backprops_through_model": sum(s.expected_backprops * s.num_chunks for s in specs),
        "image_model_evaluations": sum(s.expected_model_evals * s.num_images for s in specs),
        "per_model": bucket(lambda s: s.model),
        "per_method": bucket(lambda s: s.method),
        "per_experiment": bucket(lambda s: s.experiment),
        "per_experiment_model_method": bucket(
            lambda s: (s.experiment, s.model, s.method)),
    }

    # ---------------------------------------------------------------- plan-level warnings
    limit = int(rt.get("max_atomic_jobs", 256))
    if len(specs) > limit:
        raise ConfigError(
            "The resolved Cartesian product contains %d atomic jobs, above "
            "runtime.max_atomic_jobs = %d.\n    Shorten the sweep lists, or raise the limit "
            "deliberately (--max-jobs on the command line) if you really mean it."
            % (len(specs), limit))
    if len(specs) > max(32, limit // 4):
        warnings_.append("Large sweep: %d atomic jobs / %s image reconstructions."
                         % (len(specs), "{:,}".format(workload["image_reconstructions"])))
    if len(frameworks) > 1:
        warnings_.append("This run mixes JAX and PyTorch in one process. Models are loaded "
                         "sequentially and released between families, but JAX does not return "
                         "device memory eagerly; prefer runtime.release_model_after_use: true.")
    # Paired-comparison completeness: warn when SDEdit has no partner at some t0.
    for (exp, model) in {(s.experiment, s.model) for s in specs}:
        sel = [s for s in specs if s.experiment == exp and s.model == model]
        methods = {s.method for s in sel}
        if methods & {"mpc_rhc", "mpc_delta_t"} and "sdedit" not in methods:
            warnings_.append(
                "%s / %s runs MPC without a paired SDEdit baseline; the headline comparison of "
                "this repository is MPC vs SDEdit at the SAME t0." % (exp, model.upper()))
        else:
            mpc_t0 = {s.t0 for s in sel if s.is_mpc}
            sde_t0 = {s.t0 for s in sel if s.method == "sdedit"}
            missing = sorted(mpc_t0 - sde_t0)
            if missing:
                warnings_.append(
                    "%s / %s has MPC jobs at t0=%s with no SDEdit job at the same t0; those "
                    "MPC runs have no paired baseline."
                    % (exp, model.upper(), ", ".join("%.2f" % t for t in missing)))
    ids = [s.job_id for s in specs]
    if len(set(ids)) != len(ids):                                       # pragma: no cover
        raise ConfigError("Internal error: duplicate job ids were generated.")

    problem_requests = tuple(
        ProblemRequest(key=e["key"], problem=e["problem"], params=e["params"],
                       num_images=e["num_images"], seed=e["seed"], guide_mode=e["guide_mode"],
                       experiments=tuple(dict.fromkeys(e["experiments"])))
        for e in problem_entries.values())

    return RunPlan(run_id=run_id, created=datetime.datetime.now().isoformat(), seed=seed,
                   replicate=replicate, output_dir=output_dir, cache_root=cache_root,
                   resources=resources, problems=problem_requests, specs=tuple(specs),
                   warnings=tuple(warnings_), workload=workload,
                   raw_config=copy.deepcopy(config))


# =====================================================================================
# Loading and printing
# =====================================================================================
def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path) as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ConfigError("%s does not contain a YAML mapping." % path)
    return config


def check_accelerator_compatibility(plan: RunPlan, accel: Dict[str, Any]) -> List[str]:
    """Fail early for impossible model/accelerator combinations; warn about slow ones."""
    kind = accel["kind"]
    problems, notes = [], []
    for model, supported in plan.resources.accelerators_supported.items():
        if kind not in supported:
            problems.append("%s cannot run on %s (supported: %s)"
                            % (model.upper(), kind, list(supported)))
    if problems:
        raise ConfigError(
            "Unsupported accelerator for this plan:\n    " + "\n    ".join(problems)
            + "\nChange runtime.accelerator, or stop using those models.")
    if kind == "cpu":
        notes.append("Running on CPU. Viable for structural validation only: MPC needs many "
                     "forward AND backward passes through 256x256 models.")
    if kind == "tpu" and "torch" in plan.resources.frameworks:
        notes.append("A TPU was selected but this plan needs PyTorch models (SiT/JiT), which "
                     "are configured for CUDA/CPU here. They will fall back to CPU.")
    if kind == "gpu":
        vram = accel.get("gpu_memory_mb") or 0
        if vram and vram < 24_000 and len(plan.resources.models) > 1:
            notes.append("GPU has %.1f GB: models will be loaded strictly one at a time."
                         % (vram / 1024.0))
        if vram and vram < 16_000:
            notes.append("MPC-delta_t and RHC K>1 backpropagate through the generative model. "
                         "If you hit OOM, reduce models.<m>.batch_size to 1.")
        if "jit" in plan.resources.models and not accel.get("bf16_likely", False):
            notes.append("This GPU probably lacks BF16 (compute capability < 8.0). JiT will use "
                         "FP32 compute, which is correct but slower; FP16 is deliberately "
                         "avoided because it destabilises pixel-space generation.")
    max_K = max([s.K or 1 for s in plan.specs] or [1])
    if max_K > 3 and kind != "cpu":
        notes.append("The plan contains RHC with K up to %d: memory for the control graph "
                     "scales with K-1 nested differentiable model evaluations." % max_K)
    return notes


def print_run_plan(plan: RunPlan, accel: Optional[Dict[str, Any]] = None,
                   notes: Sequence[str] = (), detail: bool = True) -> None:
    """The dry-run report (brief section 28)."""
    w = plan.workload
    print(RULE_ := "=" * 94)
    print("RESOLVED RUN PLAN   |   %s" % plan.run_id)
    print(RULE_)
    if accel:
        print("Accelerator          : %s%s"
              % (accel["kind"], "  (%s)" % accel["gpu_name"] if accel.get("gpu_name") else ""))
    print("Required models      : %s" % (", ".join(
        "%s [%s/%s/%s]" % (m.upper(), MODEL_REGISTRY_DEFAULTS[m]["framework"],
                           MODEL_REGISTRY_DEFAULTS[m]["state_space"],
                           MODEL_REGISTRY_DEFAULTS[m]["dynamics_family"])
        for m in plan.resources.models) or "(none)"))
    print("Frameworks           : %s" % (", ".join(plan.resources.frameworks) or "(none)"))
    print("Data                 : source=%s, shared pool=%d image(s)"
          % (plan.resources.data_source, plan.resources.source_pool_size))
    print("Inverse problems     : %s" % ", ".join(plan.resources.problems))
    print("Global seed          : %d   (replicate %d)" % (plan.seed, plan.replicate))
    print("Output directory     : %s" % plan.output_dir)

    print("-" * 94)
    print("SHARED PROBLEM INSTANCES   built once, reused by every model and method")
    for p in plan.problems:
        print("  %-17s %-52s images=%d" % (
            p.problem, ", ".join("%s=%s" % (k, p.params[k]) for k in sorted(p.params)),
            p.num_images))
        print("  %-17s guide g(y) = %s   used by: %s"
              % ("", p.guide_mode, ", ".join(p.experiments)))

    # ---------------------------------------------------------------- job counts
    print("-" * 94)
    print("ATOMIC JOBS")
    for exp in plan.experiment_names():
        exp_specs = [s for s in plan.specs if s.experiment == exp]
        print("\n%s   (%s, %d image%s)"
              % (exp, exp_specs[0].problem, exp_specs[0].num_images,
                 "s" if exp_specs[0].num_images != 1 else ""))
        for model in MODEL_NAMES:
            model_specs = [s for s in exp_specs if s.model == model]
            if not model_specs:
                continue
            print("  %s" % MODEL_REGISTRY_DEFAULTS[model]["display_name"])
            for method in METHOD_NAMES:
                sel = [s for s in model_specs if s.method == method]
                if not sel:
                    continue
                title = METHOD_DECLARATIONS[method].title
                print("    %s %s %d jobs"
                      % (title, "." * max(2, 20 - len(title)), len(sel)))
                if detail:
                    print("      t0 %s" % _axis(sel, "t0", "%.2f"))
                    if method == "sdedit":
                        print("      steps %s | solver %s"
                              % (_axis(sel, "steps", "%s"), _axis(sel, "solver")))
                    else:
                        if method == "mpc_rhc":
                            print("      K %s" % _axis(sel, "K", "%s"))
                        print("      N %s | delta %s" % (_axis(sel, "num_mpc_steps", "%s"),
                                                         _axis(sel, "delta", "%.4g")))
                        print("      lam %s | n_ctrl %s | lr %s | source %s"
                              % (_axis(sel, "lam", "%g"), _axis(sel, "n_ctrl", "%s"),
                                 _axis(sel, "lr", "%g"),
                                 ", ".join(sorted({s.hyperparameter_sources.get("lam", "?")
                                                   for s in sel}))))

    print("-" * 94)
    print("TOTAL ATOMIC JOBS: %d" % w["atomic_jobs"])
    print("  image reconstructions       : %s" % "{:,}".format(w["image_reconstructions"]))
    print("  control iterations          : %s" % "{:,}".format(w["control_iterations"]))
    print("  generative-model evaluations: %s   (batched calls)"
          % "{:,}".format(w["model_evaluations"]))
    print("  ... as per-image work       : %s" % "{:,}".format(w["image_model_evaluations"]))
    print("  backprops through the model : %s   (SDEdit and RHC K=1 contribute ZERO)"
          % "{:,}".format(w["backprops_through_model"]))
    print("  jobs by model               : %s"
          % ", ".join("%s=%d" % (m.upper(), e["jobs"]) for m, e in w["per_model"].items()))
    print("  jobs by method              : %s"
          % ", ".join("%s=%d" % (m, e["jobs"]) for m, e in w["per_method"].items()))

    print("-" * 94)
    all_notes = list(plan.warnings) + list(notes)
    if all_notes:
        print("WARNINGS (%d)" % len(all_notes))
        for n in all_notes:
            print("  ! " + n)
    else:
        print("No warnings.")
    print("-" * 94)
    print("REMINDERS")
    print("  * delta = t0 / num_mpc_steps. Equal num_mpc_steps at different t0 does NOT mean")
    print("    equal execution resolution; the delta line above shows what each job will use.")
    print("  * MPC-Flow Table E2 hyperparameters were tuned for a CelebA 128x128 pixel-space")
    print("    U-Net. Here they are STARTING VALUES, never claims of optimality for JiT/pMF on")
    print("    ImageNet-256; every row records where its lam/n_ctrl/lr came from.")
    print("  * All methods share one epsilon per (model, image, replicate), so SDEdit and MPC")
    print("    start from a bit-identical z_t0 at equal t0.")
    print(RULE_)


def _axis(specs, attr, fmt="%s", limit=8):
    seen, out = set(), []
    for s in specs:
        v = getattr(s, attr)
        if repr(v) not in seen:
            seen.add(repr(v))
            out.append(v)
    shown = [(fmt % v) if v is not None else "-" for v in out[:limit]]
    return ", ".join(shown) + (" ... (+%d)" % (len(out) - limit) if len(out) > limit else "")
