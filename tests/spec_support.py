"""Resolved-spec access for the executable tests.

Specs come from the REAL planner: a small in-memory configuration is validated by
`config.validate_config` and resolved by `config.resolve_run_plan`, so a test that runs at
all has already proved that its method and its fields are declared, validated and swept
correctly.  That is deliberately stronger than reading `configs/experiments.yaml`, which
only contains the methods that happen to be enabled there.

Overrides are split automatically: anything the method declares as a configuration field is
written INTO the configuration (and therefore validated), and anything else is applied to
the resolved spec with `dataclasses.replace`.

If `src/config.py` cannot be imported at all, the tests fall back to the documented
stand-in below so the algorithms can still be exercised; `spec_source()` reports which path
was taken and every test script prints it.
"""

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

_SOURCE = ["unknown"]

# One experiment is enough for every method; denoising is compatible with all of them.
_BASE_CONFIG: Dict[str, Any] = {
    "runtime": {"seed": 42, "accelerator": "cpu", "output_root": "outputs",
                "cache_root": "cache", "max_atomic_jobs": 4096, "progress_bars": False},
    "data": {"source": "hf_imagenet_val", "image_size": 256},
    "metrics": {"lpips": False},
    "experiments": {},
}


def build_plan(method_entries: Dict[str, Dict[str, Any]], model: str = "pmf",
               num_images: int = 2, batch_size: int = 2, run_id: str = "test"):
    """Validate and resolve a one-experiment configuration containing `method_entries`."""
    import copy

    from src.config import resolve_run_plan, validate_config
    config = copy.deepcopy(_BASE_CONFIG)
    config["experiments"]["probe"] = {
        "enabled": True, "problem": "denoising", "num_images": int(num_images),
        "degradation": {"sigma": 0.20},
        "models": {model: {"batch_size": int(batch_size),
                           "methods": copy.deepcopy(method_entries)}},
    }
    warnings_ = validate_config(config)
    return resolve_run_plan(config, warnings_, run_id=run_id), warnings_


def make_spec(method: str, model: str = "pmf", **over):
    """A resolved JobSpec for `method`, built by the real validator and planner."""
    try:
        from src.config import METHOD_DECLARATIONS
    except ImportError:
        _SOURCE[0] = "standin"
        family = "meanflow" if model in ("pmf", "imf") else "standard_flow"
        base = StandInSpec(method=method, model=model, dynamics_family=family,
                           job_id="standin_%s_%s" % (model, method))
        return dataclasses.replace(base, **over)

    declared = set(METHOD_DECLARATIONS[method].fields) | {"record_loss_history"}
    config_over = {k: v for k, v in over.items() if k in declared}
    spec_over = {k: v for k, v in over.items() if k not in declared}
    num_images = int(spec_over.pop("num_images", 2))
    batch_size = int(spec_over.pop("batch_size", 2))
    plan, _ = build_plan({method: config_over}, model=model, num_images=num_images,
                         batch_size=batch_size)
    spec = plan.specs[0]
    _SOURCE[0] = "real"
    # Anything left is not a configuration field (a job_id for a fixture, say). It must
    # still be a real spec field, or the test is asking for something invented.
    unknown = set(spec_over) - {f.name for f in dataclasses.fields(spec)}
    if unknown:
        raise KeyError("make_spec got names that are neither config fields nor JobSpec "
                       "fields: %s" % sorted(unknown))
    return dataclasses.replace(spec, **spec_over) if spec_over else spec


def spec_source() -> str:
    return _SOURCE[0]


def source_banner() -> str:
    if _SOURCE[0] == "standin":
        return ("src/config.py could not be imported: using tests/spec_support.py's stand-in "
                "spec (the algorithms are tested, the configuration system is not)")
    return "specs resolved by src/config.py's validator and planner"


@dataclasses.dataclass
class StandInSpec:
    """Fallback double carrying only the fields `src/*.py` reads off a resolved job.

    Not a substitute for the planner: no validation, no sweep expansion, no job identity.
    """
    method: str = "sdedit"
    model: str = "pmf"
    dynamics_family: str = "meanflow"
    job_id: str = "standin"
    experiment: str = "test"

    t0: float = 0.8
    canonical_start_time: float = 0.2
    beta: float = 1.0
    replicate: int = 0
    num_images: int = 2
    batch_size: int = 2
    record_loss_history: bool = True
    phi_normalization: str = "half_mean_squared_per_measurement"
    guidance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    steps: Optional[int] = 1
    solver: Optional[str] = None

    optimizer: str = "adam"
    lr: float = 0.05
    num_opt_steps: int = 4
    grad_clip: Optional[float] = None
    warm_start: bool = False

    num_mpc_steps: Optional[int] = 2
    delta: Optional[float] = None
    delta_nominal_uniform: Optional[float] = None
    delta_min: Optional[float] = None
    delta_max: Optional[float] = None
    K: Optional[int] = 1
    lam: float = 1.0
    n_ctrl: int = 2
    control_cost_normalization: str = "sum_squared"
    delta_t_lambda_scaling: str = "none"

    num_pnp_steps: int = 4
    gamma0: float = 1.0
    alpha: float = 1.0
    noise_samples: int = 1

    num_rhso_steps: int = 2

    @property
    def is_mpc(self) -> bool:
        return self.method in ("mpc_rhc", "mpc_delta_t")


@dataclasses.dataclass
class StandInPlan:
    """The one attribute `checks.py`'s plan-level checks touch."""
    specs: tuple = ()
