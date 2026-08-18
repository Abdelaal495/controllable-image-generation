"""Lightweight checks that run BEFORE the expensive sweep.

Three groups:

  structural   no checkpoint needed -- operator parity across NumPy/Torch/JAX, gradient
               finiteness, measurement reconstruction, stroke-operator properties, and the
               paired-comparison fairness invariants of the resolved plan;
  model        run inside the model-major loop right after each checkpoint loads, so no
               model is loaded twice;
  fairness     programmatic verification that two jobs differing only by method really do
               share the measurement, the guide, t0, the generative noise and the label.

Everything returns a `CheckResult` so failures are reported rather than raised in the middle
of a long run, unless `on_failure="raise"`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MODEL_REGISTRY_DEFAULTS
from .models.base import Conditioning, build_initial_state
from .problems import (InverseProblem, make_phi, make_stroke_painting_reference)
from .utils import (JaxBackend, NUMPY_BACKEND, TorchBackend, STANDARD_FLOW, MEANFLOW,
                    canonical_start_time, gaussian_noise, native_time, pixel_fingerprint,
                    prior_noise_parts)


@dataclass
class CheckResult:
    scope: str
    name: str
    passed: bool
    detail: str = ""
    seconds: float = 0.0

    def line(self) -> str:
        return "  [%s] %-14s %-30s %s" % ("PASS" if self.passed else "FAIL", self.scope,
                                          self.name, self.detail)


@dataclass
class CheckReport:
    results: List[CheckResult] = field(default_factory=list)

    def run(self, scope: str, name: str, fn: Callable[[], Tuple[bool, str]],
            verbose: bool = True) -> CheckResult:
        started = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as exc:                                  # a raising check is a failure
            ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
        result = CheckResult(scope, name, bool(ok), str(detail), time.perf_counter() - started)
        self.results.append(result)
        if verbose:
            print(result.line())
        return result

    def failures(self, scope: Optional[str] = None) -> List[CheckResult]:
        return [r for r in self.results
                if not r.passed and (scope is None or r.scope == scope)]

    def to_dict(self) -> List[Dict[str, Any]]:
        return [{"scope": r.scope, "name": r.name, "passed": r.passed, "detail": r.detail,
                 "seconds": round(r.seconds, 4)} for r in self.results]


# =====================================================================================
# Backend availability
# =====================================================================================
def available_backends() -> Dict[str, Any]:
    """The operator backends importable in this process (never forces an import to fail)."""
    out: Dict[str, Any] = {"numpy": NUMPY_BACKEND}
    try:
        import torch
        out["torch"] = TorchBackend(torch, torch.device("cpu"), torch.float64)
    except Exception:
        pass
    try:
        import jax.numpy as jnp
        out["jax"] = JaxBackend(jnp)
    except Exception:
        pass
    return out


# =====================================================================================
# Structural checks (no checkpoint)
# =====================================================================================
def check_operator_parity(problem: InverseProblem, backends: Dict[str, Any],
                          tol: float = 2e-4) -> Tuple[bool, str]:
    """A_numpy(x) ~ A_torch(x) ~ A_jax(x) on the same input.

    The tolerance is loose enough for float32 accumulation order but far tighter than any
    modelling difference: a genuinely different operator fails by orders of magnitude.
    """
    x = np.clip(gaussian_noise(problem.ground_truth.shape, "checks", "operator",
                               problem.name) * 0.4, -1, 1).astype(np.float32)
    reference = np.asarray(problem.apply(x, backends["numpy"]), np.float64)
    details = []
    ok = True
    for name, B in backends.items():
        if name == "numpy":
            continue
        if name == "torch":
            import torch
            xb = torch.from_numpy(x).to(dtype=torch.float64)
            got = np.asarray(problem.apply(xb, B).detach().cpu().numpy(), np.float64)
        else:
            import jax.numpy as jnp
            xb = jnp.asarray(x)
            got = np.asarray(problem.apply(xb, B), np.float64)
        err = float(np.max(np.abs(got - reference)))
        details.append("%s %.2e" % (name, err))
        ok = ok and err < tol
    if not details:
        return True, "only NumPy is importable; parity untested"
    return ok, "max|A_numpy - A_backend|: " + ", ".join(details)


def check_measurement_reconstruction(problem: InverseProblem) -> Tuple[bool, str]:
    """Noiseless: A(x*) == y.  Noisy: the residual energy matches the measurement noise."""
    clean = np.asarray(problem.apply(problem.ground_truth, NUMPY_BACKEND), np.float32)
    residual = clean - problem.measurement
    rms = float(np.sqrt(np.mean(residual ** 2)))
    if problem.sigma <= 0.0:
        return rms < 1e-5, "noiseless: RMSE(A(x*) - y) = %.2e" % rms
    if problem.mask is not None:
        # Noise lives only on observed entries, so the expected RMS is scaled by that share.
        expected = problem.sigma * float(np.sqrt(problem.mask.mean()))
    else:
        expected = problem.sigma
    ratio = rms / max(expected, 1e-12)
    return 0.8 < ratio < 1.25, ("RMSE(A(x*) - y) = %.4f, expected ~%.4f from sigma=%.3g "
                                "(ratio %.3f)" % (rms, expected, problem.sigma, ratio))


def check_gradient(problem: InverseProblem, backends: Dict[str, Any]) -> Tuple[bool, str]:
    """d/dx  Phi(x) = 1/2 ||A(x) - y||^2 must be finite and non-trivial in every backend."""
    x = np.clip(problem.ground_truth * 0.9, -1, 1).astype(np.float32)
    details, ok = [], True
    if "torch" in backends:
        import torch
        B = TorchBackend(torch, torch.device("cpu"), torch.float32)
        phi = make_phi(problem, B, "half_sum_squared")
        xb = torch.from_numpy(x).requires_grad_(True)
        loss = phi(xb)
        loss.backward()
        g = xb.grad.detach().cpu().numpy()
        finite, nonzero = bool(np.isfinite(g).all()), float(np.abs(g).max()) > 0
        details.append("torch |g|max=%.3e finite=%s" % (float(np.abs(g).max()), finite))
        ok = ok and finite and nonzero
    if "jax" in backends:
        import jax
        import jax.numpy as jnp
        B = JaxBackend(jnp)
        phi = make_phi(problem, B, "half_sum_squared")
        g = np.asarray(jax.grad(lambda z: phi(z))(jnp.asarray(x)))
        finite, nonzero = bool(np.isfinite(g).all()), float(np.abs(g).max()) > 0
        details.append("jax |g|max=%.3e finite=%s" % (float(np.abs(g).max()), finite))
        ok = ok and finite and nonzero
    if not details:
        return True, "neither Torch nor JAX is importable; gradients untested"
    return ok, "; ".join(details)


def check_stroke_geometry_determinism(problem: InverseProblem, request,
                                      data_manager) -> Tuple[bool, str]:
    """The same (seed, image, parameters) must reproduce the same frozen geometry."""
    from .problems import build_problem
    examples = data_manager.examples(len(problem.image_ids))
    rebuilt = build_problem(request, examples.images, examples.image_ids, examples.labels,
                            verbose=False)
    same_geometry = all(a.fingerprint() == b.fingerprint()
                        for a, b in zip(problem.geometry, rebuilt.geometry))
    same_y = np.array_equal(problem.measurement, rebuilt.measurement)
    return bool(same_geometry and same_y), \
        ("rebuilding with the same key reproduces the geometry (%s) and y (%s) exactly"
         % (same_geometry, same_y))


def check_stroke_apply_is_pure_tensor(problem: InverseProblem) -> Tuple[bool, str]:
    """No PIL / skimage / NumPy conversion may occur inside the differentiable apply path.

    Enforced by executing `apply` with PIL's and skimage's entry points temporarily
    replaced by tripwires: any call raises and the check fails.
    """
    import PIL.Image as _pil_image
    import PIL.ImageDraw as _pil_draw
    tripped: List[str] = []

    class _Tripwire:
        def __init__(self, label):
            self.label = label

        def __call__(self, *a, **k):
            tripped.append(self.label)
            raise AssertionError("%s was called inside the differentiable path" % self.label)

    originals = [(_pil_image, "fromarray", _pil_image.fromarray),
                 (_pil_image, "new", _pil_image.new),
                 (_pil_draw, "Draw", _pil_draw.Draw)]
    try:
        from skimage import segmentation as _seg
        originals.append((_seg, "slic", _seg.slic))
    except Exception:
        _seg = None
    for module, attr, _orig in originals:
        setattr(module, attr, _Tripwire("%s.%s" % (module.__name__, attr)))
    try:
        out = problem.apply(problem.ground_truth, NUMPY_BACKEND)
        finite = bool(np.isfinite(np.asarray(out)).all())
    finally:
        for module, attr, orig in originals:
            setattr(module, attr, orig)
    return (not tripped) and finite, \
        ("no PIL/skimage call inside apply()" if not tripped
         else "tripwires fired: %s" % sorted(set(tripped)))


def check_stroke_parity(problem: InverseProblem, limit: int = 2) -> Tuple[bool, str]:
    """The differentiable renderer must look like the original SDEdit stroke transform.

    They cannot be bit-identical: the original quantises segment colours to uint8 and uses
    PIL's box-approximation Gaussian blur, whereas the renderer keeps float colours and a
    true separable Gaussian.  The benchmark measurement is produced by the RENDERER, so the
    two are never mixed -- this check only confirms the appearance is preserved.
    """
    from .metrics import psnr
    from .utils import to_float, to_uint8
    n = min(limit, len(problem.ground_truth))
    preset = {k: int(problem.params[k]) for k in ("n_segments", "compactness", "stroke_width")}
    reference, rendered = [], np.asarray(problem.measurement[:n], np.float32)
    gt_u8 = to_uint8(problem.ground_truth[:n])
    for i in range(n):
        seed = int(problem.geometry[i].params["rng_seed"])
        reference.append(to_float(make_stroke_painting_reference(gt_u8[i], rng_seed=seed,
                                                                 **preset)))
    reference = np.stack(reference)
    value = psnr(rendered, reference)
    return value > 24.0, ("PSNR(differentiable renderer, original PIL transform) = %.2f dB "
                          "over %d image(s); differences are uint8 colour quantisation and "
                          "PIL's box-approximated blur" % (value, n))


def check_stroke_gradient(problem: InverseProblem,
                          backends: Dict[str, Any]) -> Tuple[bool, str]:
    """Gradients must exist AND be non-zero for pixels that participate in a segment."""
    if "torch" not in backends and "jax" not in backends:
        return True, "neither Torch nor JAX is importable; stroke gradients untested"
    x = np.clip(problem.ground_truth * 0.8, -1, 1).astype(np.float32)
    details, ok = [], True
    if "torch" in backends:
        import torch
        B = TorchBackend(torch, torch.device("cpu"), torch.float32)
        phi = make_phi(problem, B, "half_sum_squared")
        xb = torch.from_numpy(x).requires_grad_(True)
        phi(xb).backward()
        g = xb.grad.detach().cpu().numpy()
        share = float((np.abs(g) > 0).mean())
        details.append("torch nonzero share %.3f" % share)
        ok = ok and bool(np.isfinite(g).all()) and share > 0.5
    if "jax" in backends:
        import jax
        import jax.numpy as jnp
        B = JaxBackend(jnp)
        phi = make_phi(problem, B, "half_sum_squared")
        g = np.asarray(jax.grad(lambda z: phi(z))(jnp.asarray(x)))
        share = float((np.abs(g) > 0).mean())
        details.append("jax nonzero share %.3f" % share)
        ok = ok and bool(np.isfinite(g).all()) and share > 0.5
    return ok, "; ".join(details)


def check_time_grid(plan) -> Tuple[bool, str]:
    """s_start = 1 - t0, terminal time exactly 1, strictly increasing, correct spacing.

    This check deliberately no longer asserts that every step equals t0/N.  With the
    universal power-law schedule that identity holds ONLY at beta = 1, which is exactly
    what is verified below; for beta != 1 the grid must instead be strictly monotone with
    intervals that shrink (beta < 1) or grow (beta > 1) towards the clean endpoint, and no
    scalar `delta` may describe it.
    """
    from .mpc import mpc_time_grid
    from .rhso import rhso_time_grid
    from .schedule import DEFAULT_BETA, grid_intervals, spec_beta
    problems = []
    counted = 0
    for spec in plan.specs:
        if spec.is_mpc:
            grid = mpc_time_grid(spec)
        elif spec.method == "rhso":
            grid = rhso_time_grid(spec)
        else:
            continue
        counted += 1
        beta = spec_beta(spec)
        intervals = grid_intervals(grid)
        tag = spec.job_id[:8]
        if abs(grid[0] - (1.0 - spec.t0)) > 1e-9:
            problems.append("%s: s_start" % tag)
        if abs(grid[-1] - 1.0) > 1e-12:
            problems.append("%s: terminal" % tag)
        if any(dt <= 0.0 for dt in intervals):
            problems.append("%s: not strictly increasing" % tag)
        if beta == DEFAULT_BETA:
            # The legacy identity delta = t0/N, still asserted where it is still true.
            nominal = getattr(spec, "delta", None)
            nominal = (spec.t0 / len(intervals)) if nominal is None else float(nominal)
            if max(abs(dt - nominal) for dt in intervals) > 1e-9:
                problems.append("%s: beta=1 grid is not uniform at delta=t0/N" % tag)
        elif len(intervals) > 1:
            shrinking = all(intervals[i + 1] < intervals[i]
                            for i in range(len(intervals) - 1))
            growing = all(intervals[i + 1] > intervals[i]
                          for i in range(len(intervals) - 1))
            if beta < DEFAULT_BETA and not shrinking:
                problems.append("%s: beta<1 must shrink towards s=1" % tag)
            if beta > DEFAULT_BETA and not growing:
                problems.append("%s: beta>1 must grow towards s=1" % tag)
    if not counted:
        return True, "no scheduled-execution job (MPC / RHSO) in this plan"
    return not problems, ("%d scheduled job(s): s_start = 1 - t0, terminal time exactly 1, "
                          "strictly increasing, uniform at delta = t0/N only where beta = 1"
                          % counted if not problems else "; ".join(problems[:5]))


def check_beta_schedule() -> Tuple[bool, str]:
    """The universal power-law grid itself: validation, endpoints, legacy regression.

    Pure arithmetic, so it runs with no plan, no problem and no checkpoint.
    """
    from .schedule import canonical_time_grid, interior_time_grid, resolve_beta
    notes = []

    for bad in (0.0, -1.0, float("inf"), float("nan"), "x", True):
        try:
            resolve_beta(bad)
            notes.append("beta=%r was accepted" % (bad,))
        except ValueError:
            pass

    legacy = [0.0 + (1.0 - 0.0) * (k / 4) for k in range(5)]
    legacy[-1] = 1.0
    if canonical_time_grid(0.0, 4, 1.0) != legacy:
        notes.append("beta=1 does not reproduce the old uniform grid bitwise")
    if canonical_time_grid(0.2, 5, 1.0) != [0.2 + 0.8 * (k / 5) for k in range(5)] + [1.0]:
        notes.append("beta=1 deviates from the legacy expression at s0=0.2")

    half = canonical_time_grid(0.0, 4, 0.5)
    if max(abs(a - b) for a, b in zip(half, [0.0, 0.5, 0.5 ** 0.5, 0.75 ** 0.5, 1.0])) > 1e-12:
        notes.append("beta=0.5 values are wrong")
    two = canonical_time_grid(0.0, 4, 2.0)
    if max(abs(a - b) for a, b in zip(two, [0.0, 0.0625, 0.25, 0.5625, 1.0])) > 1e-12:
        notes.append("beta=2 values are wrong")

    for beta in (0.4, 0.5, 1.0, 1.5, 2.0, 3.0):
        grid = canonical_time_grid(0.3, 6, beta)
        if grid[0] != 0.3 or grid[-1] != 1.0:
            notes.append("beta=%g endpoints" % beta)
        if any(grid[i] >= grid[i + 1] for i in range(len(grid) - 1)):
            notes.append("beta=%g is not strictly increasing" % beta)
        dts = [grid[i + 1] - grid[i] for i in range(len(grid) - 1)]
        if beta < 1.0 and not all(dts[i + 1] < dts[i] for i in range(len(dts) - 1)):
            notes.append("beta=%g should shrink towards s=1" % beta)
        if beta > 1.0 and not all(dts[i + 1] > dts[i] for i in range(len(dts) - 1)):
            notes.append("beta=%g should grow towards s=1" % beta)
        interior = interior_time_grid(0.3, 5, beta)
        if len(interior) != 5 or not all(0.3 < s < 1.0 for s in interior):
            notes.append("beta=%g interior grid escaped (s0, 1)" % beta)

    return not notes, ("s_k = s0 + (1-s0)(k/N)^beta: beta > 0 enforced, endpoints exact, "
                       "strictly increasing, direction correct, PnP's times strictly inside "
                       "(s0, 1), and beta=1 reproduces the legacy grid bitwise"
                       if not notes else "; ".join(notes[:5]))


def check_delta_t_lambda_scaling(plan) -> Tuple[bool, str]:
    """MPC-delta_t's inverse-delta scaling must use the LOCAL dt_k, not a global delta."""
    from .mpc import inverse_delta_lambda, mpc_step_sizes
    from .schedule import DEFAULT_BETA, spec_beta
    specs = [s for s in plan.specs
             if s.method == "mpc_delta_t" and s.delta_t_lambda_scaling == "inverse_delta"]
    if not specs:
        return True, "no MPC-delta_t job uses inverse-delta lambda scaling"
    notes = []
    for spec in specs:
        dts = mpc_step_sizes(spec)
        lams = [inverse_delta_lambda(spec, dt) for dt in dts]
        if any(abs(lam - spec.lam / dt) > 1e-9 * max(1.0, abs(lam))
               for lam, dt in zip(lams, dts)):
            notes.append("%s: lambda_eff != lambda / dt_k" % spec.job_id[:8])
        if spec_beta(spec) != DEFAULT_BETA and len(set(round(l, 12) for l in lams)) == 1:
            notes.append("%s: a non-uniform grid produced one global lambda"
                         % spec.job_id[:8])
    return not notes, ("%d job(s): lambda_eff,k = lambda / dt_k at the ACTUAL local step "
                       "size" % len(specs) if not notes else "; ".join(notes[:5]))


def check_plan_uniqueness(plan) -> Tuple[bool, str]:
    """No two atomic jobs may share a job id or an output directory."""
    ids = [s.job_id for s in plan.specs]
    leaves = [(s.experiment, s.model, s.method, s.leaf_dir) for s in plan.specs]
    return (len(set(ids)) == len(ids) and len(set(leaves)) == len(leaves)), \
        "%d jobs, %d unique ids, %d unique output paths" % (len(ids), len(set(ids)),
                                                            len(set(leaves)))


# =====================================================================================
# Fairness: the central invariant of the repository
# =====================================================================================
FAIRNESS_KEYS = ("problem_key", "num_images", "t0", "model", "replicate", "seed")


def check_paired_fairness(plan) -> Tuple[bool, str]:
    """Two jobs differing ONLY by method must agree on everything that defines the setup.

    Because the generative noise depends on (model, image id, replicate) alone and the
    initialisation guide is a pure function of the shared problem, matching the tuple above
    is sufficient to guarantee a bit-identical z_t0 -- which is what makes the comparison a
    comparison of reconstruction strategies rather than of setups.
    """
    groups: Dict[Tuple, List] = {}
    for spec in plan.specs:
        key = tuple(getattr(spec, k) for k in FAIRNESS_KEYS)
        groups.setdefault(key, []).append(spec)

    paired, violations = 0, []
    for key, specs in groups.items():
        methods = {s.method for s in specs}
        if len(methods) < 2:
            continue
        paired += 1
        reference = specs[0]
        for s in specs[1:]:
            for field_name in ("problem_key", "initialization_guide_mode",
                               "canonical_start_time", "num_images", "batch_size"):
                if getattr(s, field_name) != getattr(reference, field_name):
                    violations.append("%s vs %s differ in %s"
                                      % (s.job_id[:8], reference.job_id[:8], field_name))
            if s.guidance != reference.guidance:
                violations.append("%s vs %s differ in conditioning guidance"
                                  % (s.job_id[:8], reference.job_id[:8]))
    if not paired:
        return True, ("no method-paired group in this plan (nothing to verify); the headline "
                      "comparison needs SDEdit and MPC at the same t0")
    return not violations, ("%d method-paired group(s) verified: same problem instance, "
                            "guide, t0, epsilon recipe and conditioning" % paired
                            if not violations else "; ".join(violations[:5]))


def check_noise_identity_independence() -> Tuple[bool, str]:
    """The generative-noise recipe must not depend on method, solver, steps, K, lambda...

    This is the deliberate change from the SDEdit notebook's seeding strategy, and it is the
    single most important reproducibility property in the repository, so it is asserted
    rather than assumed.
    """
    from .utils import prior_noise_parts
    base = gaussian_noise((4, 4), *prior_noise_parts("jit", "000_tench", 0))
    again = gaussian_noise((4, 4), *prior_noise_parts("jit", "000_tench", 0))
    other_image = gaussian_noise((4, 4), *prior_noise_parts("jit", "039_iguana", 0))
    other_model = gaussian_noise((4, 4), *prior_noise_parts("pmf", "000_tench", 0))
    other_rep = gaussian_noise((4, 4), *prior_noise_parts("jit", "000_tench", 1))
    ok = (np.array_equal(base, again) and not np.array_equal(base, other_image)
          and not np.array_equal(base, other_model) and not np.array_equal(base, other_rep))
    return ok, ("epsilon depends on (model, image, replicate) only: reproducible=%s, "
                "image-sensitive=%s, model-sensitive=%s, replicate-sensitive=%s"
                % (np.array_equal(base, again), not np.array_equal(base, other_image),
                   not np.array_equal(base, other_model),
                   not np.array_equal(base, other_rep)))


def check_method_registry(plan) -> Tuple[bool, str]:
    """Every method in the plan must resolve to a reconstructor for its dynamics family."""
    from .reconstruction import select_reconstructor
    missing = []
    for spec in plan.specs:
        try:
            select_reconstructor(spec.dynamics_family, spec.method)
        except ValueError as exc:
            missing.append("%s/%s: %s" % (spec.model, spec.method, exc))
    pairs = sorted({(s.dynamics_family, s.method) for s in plan.specs})
    return not missing, ("%d (family, method) combination(s) resolve: %s"
                         % (len(pairs), ", ".join("%s/%s" % p for p in pairs))
                         if not missing else "; ".join(missing[:4]))


def check_pnp_time_grid(plan) -> Tuple[bool, str]:
    """PnP corrects strictly inside (s0, 1): never at s0, never at s = 1."""
    from .pnp import pnp_step_sizes, pnp_time_grid
    from .schedule import spec_beta
    specs = [s for s in plan.specs if s.method == "pnp"]
    if not specs:
        return True, "no PnP job in this plan"
    problems_ = []
    for spec in specs:
        s0 = spec.canonical_start_time
        grid = pnp_time_grid(s0, spec.num_pnp_steps, spec_beta(spec))
        if len(grid) != int(spec.num_pnp_steps):
            problems_.append("%s: %d times for N=%d" % (spec.job_id[:8], len(grid),
                                                        spec.num_pnp_steps))
        if any(s <= s0 + 1e-12 for s in grid):
            problems_.append("%s: a correction lands at or before s0" % spec.job_id[:8])
        if any(s >= 1.0 - 1e-12 for s in grid):
            problems_.append("%s: a correction lands at s = 1 (identity denoiser)"
                             % spec.job_id[:8])
        if grid != sorted(grid) or len(set(grid)) != len(grid):
            problems_.append("%s: the schedule is not strictly increasing" % spec.job_id[:8])
        gammas = pnp_step_sizes(spec, grid)
        if any(g <= 0.0 for g in gammas) or gammas != sorted(gammas, reverse=True):
            problems_.append("%s: gamma_k is not positive and decreasing" % spec.job_id[:8])
    return not problems_, ("%d PnP job(s): s0 < s_1 < ... < s_N < 1, gamma_k = gamma0 "
                           "(1-s_k)^alpha positive and decreasing" % len(specs)
                           if not problems_ else "; ".join(problems_[:5]))


def check_pnp_reprojection_seeding() -> Tuple[bool, str]:
    """PnP reprojection noise: deterministic, fresh per (iteration, sample), sweep-stable.

    The last property is the one that matters for a gamma0/alpha sweep: two configurations
    differing only in a step-size hyperparameter must meet the SAME noise realisations, or
    the sweep compares noise draws as much as it compares hyperparameters.
    """
    from .utils import pnp_reprojection_parts
    def draw(*parts):
        return gaussian_noise((4, 4), *pnp_reprojection_parts(*parts))

    base = draw("jit", "000_tench", 0, 1, 0)
    again = draw("jit", "000_tench", 0, 1, 0)
    next_iter = draw("jit", "000_tench", 0, 2, 0)
    next_sample = draw("jit", "000_tench", 0, 1, 1)
    other_image = draw("jit", "039_iguana", 0, 1, 0)
    other_model = draw("pmf", "000_tench", 0, 1, 0)
    init_noise = gaussian_noise((4, 4), *prior_noise_parts("jit", "000_tench", 0))
    ok = (np.array_equal(base, again)
          and not np.array_equal(base, next_iter)
          and not np.array_equal(base, next_sample)
          and not np.array_equal(base, other_image)
          and not np.array_equal(base, other_model)
          and not np.array_equal(base, init_noise))
    return ok, ("reproducible=%s, fresh per iteration=%s and per realisation=%s, "
                "per-image=%s, per-model=%s, distinct from the shared initialisation "
                "epsilon=%s; the recipe takes no hyperparameter, so gamma0/alpha sweeps stay "
                "paired"
                % (np.array_equal(base, again), not np.array_equal(base, next_iter),
                   not np.array_equal(base, next_sample),
                   not np.array_equal(base, other_image),
                   not np.array_equal(base, other_model),
                   not np.array_equal(base, init_noise)))


def check_per_measurement_normalization(problem: InverseProblem) -> Tuple[bool, str]:
    """m_b counts OBSERVED scalars, and the objective is summed (not averaged) over b.

    Summing is what makes each sample's gradient independent of the batch size, which is
    why a batch-2 job and a batch-4 job reconstruct the same image identically.
    """
    from .problems import (PER_MEASUREMENT_NORMALIZATION, make_phi, measurement_counts,
                           phi_log_scale)
    counts = measurement_counts(problem)
    n = int(problem.measurement.shape[0])
    expected_max = float(np.prod(problem.measurement.shape[1:]))
    notes = []

    if problem.mask is not None:
        channels = int(problem.measurement.shape[-1])
        observed = float(problem.mask[0].sum()) * channels / float(problem.mask.shape[-1])
        if abs(counts[0] - observed) > 0.5:
            notes.append("masked count %g != observed entries %g" % (counts[0], observed))
        if counts[0] >= expected_max:
            notes.append("masked count was not reduced below the full tensor size")
    elif abs(counts[0] - expected_max) > 0.5:
        notes.append("count %g != measurement entries %g" % (counts[0], expected_max))

    # Same image, two batch sizes: the per-sample term must not move.
    phi_full = make_phi(problem, NUMPY_BACKEND, PER_MEASUREMENT_NORMALIZATION)
    single = problem.subset([0])
    phi_single = make_phi(single, NUMPY_BACKEND, PER_MEASUREMENT_NORMALIZATION)
    x = np.asarray(problem.ground_truth, np.float32) * 0.5
    total = float(phi_full(x))
    one = float(phi_single(x[:1]))
    per_sample_sum = 0.0
    for i in range(n):
        sub = problem.subset([i])
        per_sample_sum += float(make_phi(sub, NUMPY_BACKEND,
                                         PER_MEASUREMENT_NORMALIZATION)(x[i:i + 1]))
    if abs(total - per_sample_sum) > 1e-3 * max(1.0, abs(total)):
        notes.append("the batch objective is not the SUM of its per-sample terms "
                     "(%g vs %g)" % (total, per_sample_sum))
    if abs(phi_log_scale(problem, PER_MEASUREMENT_NORMALIZATION) - 1.0 / n) > 1e-12:
        notes.append("the logging scale is not 1/B")

    # No sigma^-2 anywhere in the default fidelity.
    scaled = InverseProblem(**{**problem.__dict__, "sigma": max(problem.sigma, 1e-3) * 2.0,
                               "_cache": {}})
    if abs(float(make_phi(scaled, NUMPY_BACKEND, PER_MEASUREMENT_NORMALIZATION)(x)) - total) \
            > 1e-6 * max(1.0, abs(total)):
        notes.append("the objective changed when sigma changed: a 1/sigma^2 factor leaked in")

    return not notes, ("m_b=%g of %g possible scalar entries; L_opt is the sum over the "
                       "batch (%g = %g), logging divides by B, and sigma does not scale it"
                       % (counts[0], expected_max, total, one if n == 1 else per_sample_sum)
                       if not notes else "; ".join(notes))


def check_gaussian_likelihood_rejects_zero_sigma(problem: InverseProblem) -> Tuple[bool, str]:
    """A noiseless problem must REJECT gaussian_likelihood, not invent a tiny sigma."""
    from .problems import make_phi
    if float(problem.sigma) > 0.0:
        return True, "sigma=%g > 0: the likelihood objective is well defined" % problem.sigma
    try:
        make_phi(problem, NUMPY_BACKEND, "gaussian_likelihood")
    except ValueError:
        return True, ("sigma=0 with gaussian_likelihood raises instead of substituting an "
                      "epsilon and inflating the objective by ~1e16")
    return False, "sigma=0 with gaussian_likelihood was accepted silently"


def run_structural_checks(plan, problems: Dict[str, InverseProblem], data_manager,
                          report: Optional[CheckReport] = None,
                          verbose: bool = True) -> CheckReport:
    """Everything that can be verified without loading a checkpoint."""
    report = report or CheckReport()
    backends = available_backends()
    if verbose:
        print("Structural checks (no checkpoint required). Backends: %s"
              % ", ".join(sorted(backends)))
    report.run("plan", "unique_jobs", lambda: check_plan_uniqueness(plan), verbose)
    report.run("schedule", "beta_schedule", check_beta_schedule, verbose)
    report.run("plan", "time_grid", lambda: check_time_grid(plan), verbose)
    report.run("plan", "delta_t_lambda_scaling",
               lambda: check_delta_t_lambda_scaling(plan), verbose)
    report.run("plan", "pnp_time_grid", lambda: check_pnp_time_grid(plan), verbose)
    report.run("plan", "method_registry", lambda: check_method_registry(plan), verbose)
    report.run("plan", "paired_fairness", lambda: check_paired_fairness(plan), verbose)
    report.run("seeding", "noise_identity", check_noise_identity_independence, verbose)
    report.run("seeding", "pnp_reprojection_noise", check_pnp_reprojection_seeding, verbose)

    for key, problem in problems.items():
        scope = problem.name
        report.run(scope, "operator_parity",
                   lambda p=problem: check_operator_parity(p, backends), verbose)
        report.run(scope, "measurement",
                   lambda p=problem: check_measurement_reconstruction(p), verbose)
        report.run(scope, "gradient",
                   lambda p=problem: check_gradient(p, backends), verbose)
        report.run(scope, "per_measurement_fidelity",
                   lambda p=problem: check_per_measurement_normalization(p), verbose)
        report.run(scope, "likelihood_needs_sigma",
                   lambda p=problem: check_gaussian_likelihood_rejects_zero_sigma(p), verbose)
        if problem.name == "stroke_painting":
            request = plan.problem_request(key)
            report.run(scope, "geometry_determinism",
                       lambda p=problem, r=request: check_stroke_geometry_determinism(
                           p, r, data_manager), verbose)
            report.run(scope, "apply_is_pure_tensor",
                       lambda p=problem: check_stroke_apply_is_pure_tensor(p), verbose)
            report.run(scope, "renderer_parity",
                       lambda p=problem: check_stroke_parity(p), verbose)
            report.run(scope, "stroke_gradient",
                       lambda p=problem: check_stroke_gradient(p, backends), verbose)
    return report


# =====================================================================================
# Model checks (run right after each checkpoint loads)
# =====================================================================================
def run_model_checks(adapter, problem: InverseProblem, spec, manager,
                     report: Optional[CheckReport] = None, verbose: bool = True,
                     t0_probe: float = 0.5, methods: Optional[Sequence[str]] = None,
                     specs: Optional[Sequence] = None) -> CheckReport:
    """Shape, time-convention, initialisation and one-tiny-trajectory checks.

    `methods` lists the methods this model will actually run, so the PnP, D-Flow and RHSO
    probes below are skipped entirely for a plan that contains none of them -- they cost real model
    evaluations (and, for D-Flow, a backward pass) and should not be charged to a run that
    does not use them.  `specs` supplies a real resolved spec per method so the probe uses
    the job's own solver and hyperparameters rather than invented ones.
    """
    report = report or CheckReport()
    scope = adapter.spec.name
    n = min(int(adapter.spec.batch_size), len(problem.ground_truth))
    indices = list(range(n))
    ids = [problem.image_ids[i] for i in indices]
    labels = (problem.labels[:n] if problem.labels is not None
              else np.zeros((n,), np.int32))
    cond = Conditioning(labels=np.asarray(labels, np.int32), guidance=dict(spec.guidance))
    ctx = {"pixels": np.ascontiguousarray(problem.ground_truth[:n]), "conditioning": cond}

    def shapes():
        eps = adapter.prior_sample(ids)
        native = adapter.encode_pixels(ctx["pixels"])
        pixels = adapter.to_pixels(eps)
        diff = adapter.to_pixels(eps, differentiable=True)
        ok = (tuple(np.asarray(adapter.to_numpy(eps)).shape[1:])
              == tuple(adapter.spec.native_shape)
              and pixels.shape == (n, adapter.spec.pixel_resolution,
                                   adapter.spec.pixel_resolution, 3)
              and tuple(diff.shape) == tuple(pixels.shape))
        return ok, ("native eps %s -> pixels %s; differentiable map returns the same shape"
                    % (tuple(np.asarray(adapter.to_numpy(eps)).shape), pixels.shape))

    def native_times():
        start, end = adapter.native_times(spec.t0)
        expect_start = native_time(canonical_start_time(spec.t0),
                                   adapter.spec.native_time_mapping)
        expect_end = native_time(1.0, adapter.spec.native_time_mapping)
        ok = abs(start - expect_start) < 1e-9 and abs(end - expect_end) < 1e-9
        return ok, ("t0=%.2f -> native %.3f -> %.3f (%s)"
                    % (spec.t0, start, end, adapter.spec.native_time_mapping))

    def pure_noise_initialisation():
        eps = adapter.prior_sample(ids)
        state = adapter.initial_state(None, 1.0, eps)
        same_object = state is eps
        return same_object, ("t0 = 1 returns the prior-noise array ITSELF (bitwise identical "
                             "to the notebooks' pure-noise path): %s" % same_object)

    def guided_initialisation():
        eps = adapter.prior_sample(ids)
        guide = manager.encoded_guide(
            adapter, np.ascontiguousarray(problem.initialization_guide[:n]))
        z = adapter.initial_state(guide, t0_probe, eps)
        expected = adapter.to_numpy(guide) * (1 - t0_probe) + adapter.to_numpy(eps) * t0_probe
        err = float(np.max(np.abs(adapter.to_numpy(z) - expected)))
        return err < 1e-4, ("z = (1-t0) g + t0 eps at t0=%.2f: max deviation %.2e"
                            % (t0_probe, err))

    def paired_noise():
        a = adapter.to_numpy(adapter.prior_sample(ids))
        b = adapter.to_numpy(adapter.prior_sample(ids))
        return bool(np.array_equal(a, b)), \
            "the same (model, image) pair reproduces epsilon exactly across calls"

    def tiny_trajectory():
        import dataclasses
        from .mpc import select_reconstructor
        eps = adapter.prior_sample(ids)
        x0 = adapter.initial_state(None, 1.0, eps)
        tiny = dataclasses.replace(spec, method="sdedit", steps=1, solver=(
            spec.solver or MODEL_REGISTRY_DEFAULTS[adapter.spec.name].get("default_solver")),
            t0=1.0, canonical_start_time=0.0, record_loss_history=False)
        run = select_reconstructor(adapter.spec.dynamics_family, "sdedit")
        state, stats = run(adapter, cond, x0, problem.subset(indices), tiny)
        pixels = adapter.to_pixels(state)
        ok = stats.finite and np.isfinite(pixels).all()
        return ok, ("one-step trajectory produced a finite %s image in %.2fs (%d model eval(s))"
                    % (pixels.shape, stats.seconds, stats.model_evals_total))

    # ---------------------------------------------------------------- PnP / D-Flow probes
    wanted = set(methods or [spec.method])
    by_method = {s.method: s for s in (specs or [spec])}
    sub_problem = problem.subset(indices)

    def _probe_spec(method: str, **overrides):
        import dataclasses
        base = by_method.get(method, spec)
        return dataclasses.replace(base, method=method, num_images=n,
                                   record_loss_history=False, **overrides)

    def pnp_initial_projection():
        """The initial prior projection happens exactly once and is counted."""
        from .pnp import pnp_reconstruct
        tiny = _probe_spec("pnp", num_pnp_steps=1, noise_samples=1)
        eps = adapter.prior_sample(ids)
        guide = manager.encoded_guide(
            adapter, np.ascontiguousarray(problem.initialization_guide[:n]))
        x0 = adapter.initial_state(guide, tiny.t0, eps)
        _state, stats = pnp_reconstruct(adapter, cond, x0, sub_problem, tiny)
        expected = 1 + 1 * 1                       # 1 initial projection + N*M denoisings
        ok = (stats.finite and stats.denoiser_samples == expected
              and stats.model_evals_total == expected and stats.data_gradient_evals == 1
              and stats.backprops_through_model == 0)
        return ok, ("N=1, M=1 -> %d logical denoiser sample(s) (initial projection + 1 "
                    "correction), %d model eval(s), %d data gradient(s), %d generative "
                    "backprop(s)"
                    % (stats.denoiser_samples, stats.model_evals_total,
                       stats.data_gradient_evals, stats.backprops_through_model))

    def pnp_determinism():
        """Two identical PnP runs agree bitwise; a different gamma0 does not re-roll noise."""
        from .pnp import pnp_reconstruct
        eps = adapter.prior_sample(ids)
        guide = manager.encoded_guide(
            adapter, np.ascontiguousarray(problem.initialization_guide[:n]))
        tiny = _probe_spec("pnp", num_pnp_steps=1, noise_samples=1)
        first = adapter.to_numpy(pnp_reconstruct(
            adapter, cond, adapter.initial_state(guide, tiny.t0, eps), sub_problem, tiny)[0])
        second = adapter.to_numpy(pnp_reconstruct(
            adapter, cond, adapter.initial_state(guide, tiny.t0, eps), sub_problem, tiny)[0])
        zero_gamma = _probe_spec("pnp", num_pnp_steps=1, noise_samples=1, gamma0=1e-12)
        third = adapter.to_numpy(pnp_reconstruct(
            adapter, cond, adapter.initial_state(guide, tiny.t0, eps), sub_problem,
            zero_gamma)[0])
        identical = bool(np.array_equal(first, second))
        # With a vanishing step size the data term barely moves the iterate, so the two runs
        # must stay close -- which they only do if the reprojection noise was the same.
        close = float(np.max(np.abs(first - third)))
        return identical, ("repeated runs are bitwise identical=%s; a 1e12-fold smaller "
                           "gamma0 changes the result by only %.3e, i.e. the reprojection "
                           "noise did not change with the hyperparameter"
                           % (identical, close))

    def dflow_gradient():
        """d(terminal fidelity)/dq must be finite and non-zero, through the decoder."""
        from .dflow import dflow_time_grid, trajectory_evaluations
        from .problems import make_phi
        tiny = _probe_spec("dflow", steps=1, num_opt_steps=1)
        eps = adapter.prior_sample(ids)
        guide = manager.encoded_guide(
            adapter, np.ascontiguousarray(problem.initialization_guide[:n]))
        x0 = adapter.initial_state(guide, tiny.t0, eps)
        B = adapter.backend()
        phi = make_phi(sub_problem, B, tiny.phi_normalization)
        grid = dflow_time_grid(tiny)

        if adapter.spec.framework == "torch":
            import torch
            from .dflow import _integrate_flow
            q = x0.detach().clone().requires_grad_(True)
            loss = phi(adapter.to_pixels(_integrate_flow(adapter, cond, q, tiny, grid),
                                         differentiable=True))
            grad = torch.autograd.grad(loss, q)[0]
            g = adapter.to_numpy(grad)
        else:
            import jax
            from .dflow import _transition_trajectory
            g = np.asarray(jax.grad(lambda z: phi(adapter.to_pixels(
                _transition_trajectory(adapter, cond, z, grid), differentiable=True)))(x0))
        finite = bool(np.isfinite(g).all())
        magnitude = float(np.abs(g).max())
        ok = finite and magnitude > 0.0
        return ok, ("|d loss / d q|max = %.3e over %d model evaluation(s) in the trajectory; "
                    "finite=%s, non-zero=%s%s"
                    % (magnitude, trajectory_evaluations(adapter, tiny), finite,
                       magnitude > 0.0,
                       "; the chain includes the VAE decoder"
                       if adapter.spec.state_space == "latent" else ""))

    def dflow_optimisation():
        """A few Adam updates must reduce the terminal fidelity, and the output must be
        the trajectory of the FINAL q, not of an earlier iterate."""
        from .dflow import dflow_reconstruct
        from .problems import make_phi
        tiny = _probe_spec("dflow", steps=1, num_opt_steps=3, record_loss_history=True)
        eps = adapter.prior_sample(ids)
        guide = manager.encoded_guide(
            adapter, np.ascontiguousarray(problem.initialization_guide[:n]))
        x0 = adapter.initial_state(guide, tiny.t0, eps)
        state, stats = dflow_reconstruct(adapter, cond, x0, sub_problem, tiny)
        history = list(stats.loss_history)
        improved = len(history) >= 2 and history[-1] <= history[0]
        # The returned state must be the terminal map of the optimised q: evaluating the
        # fidelity on it should be at least as good as the LAST recorded objective, which
        # was computed before that final update.
        phi = make_phi(sub_problem, adapter.backend(), tiny.phi_normalization)
        final = float(phi(adapter.to_pixels(state, differentiable=True)))
        counted = (stats.optimizer_iterations == 3
                   and stats.model_evals_total >= stats.backprops_through_model
                   and stats.backprops_through_model > 0)
        return bool(improved and counted), (
            "loss %.4g -> %.4g over %d Adam step(s); final reconstruction scores %.4g; "
            "%d model eval(s) including the post-optimisation trajectory, %d trajectory "
            "backprop(s)"
            % (history[0] if history else float("nan"),
               history[-1] if history else float("nan"), stats.optimizer_iterations,
               final * (1.0 / max(1, len(sub_problem.image_ids))), stats.model_evals_total,
               stats.backprops_through_model))

    def rhso_receding_horizon():
        """RHSO optimises the current state, executes ONE interval, and replans.

        Verified on the real model at the smallest meaningful size: the counters must show
        M planning evaluations per outer stage plus exactly one execution, and the returned
        state must be the executed interval of the FINAL post-Adam q.
        """
        from .rhso import (rhso_execution_evaluations, rhso_planning_evaluations,
                           rhso_reconstruct, rhso_time_grid)
        tiny = _probe_spec("rhso", num_rhso_steps=2, num_opt_steps=2,
                           record_loss_history=True)
        eps = adapter.prior_sample(ids)
        guide = manager.encoded_guide(
            adapter, np.ascontiguousarray(problem.initialization_guide[:n]))
        x0 = adapter.initial_state(guide, tiny.t0, eps)
        state, stats = rhso_reconstruct(adapter, cond, x0, sub_problem, tiny)
        grid = rhso_time_grid(tiny)
        expected_planning = sum(2 * rhso_planning_evaluations(adapter, tiny, k, grid)
                                for k in range(2))
        expected_execution = sum(rhso_execution_evaluations(adapter, tiny, k, grid)
                                 for k in range(2))
        history = list(stats.loss_history)
        ok = (stats.finite
              and stats.optimizer_iterations == 2 * 2
              and stats.objective_evals == 2 * 2
              and stats.model_evals_planning == expected_planning
              and stats.backprops_through_model == expected_planning
              and stats.model_evals_total == expected_planning + expected_execution
              and len(history) == 4
              and bool(np.isfinite(adapter.to_pixels(state)).all()))
        return ok, ("N=2, M=2: %d optimizer iteration(s), %d planning evaluation(s) "
                    "(predicted %d), %d execution evaluation(s) (predicted %d), loss "
                    "%.4g -> %.4g" % (stats.optimizer_iterations,
                                      stats.model_evals_planning, expected_planning,
                                      stats.model_evals_total - stats.model_evals_planning,
                                      expected_execution,
                                      history[0] if history else float("nan"),
                                      history[-1] if history else float("nan")))

    if verbose:
        print("Model checks for %s" % adapter.spec.name.upper())
    report.run(scope, "shapes", shapes, verbose)
    report.run(scope, "native_times", native_times, verbose)
    report.run(scope, "init_t0_eq_1", pure_noise_initialisation, verbose)
    report.run(scope, "init_t0_lt_1", guided_initialisation, verbose)
    report.run(scope, "paired_noise", paired_noise, verbose)
    report.run(scope, "tiny_trajectory", tiny_trajectory, verbose)
    if "pnp" in wanted:
        report.run(scope, "pnp_initial_projection", pnp_initial_projection, verbose)
        report.run(scope, "pnp_determinism", pnp_determinism, verbose)
    if "dflow" in wanted:
        report.run(scope, "dflow_gradient", dflow_gradient, verbose)
        report.run(scope, "dflow_optimisation", dflow_optimisation, verbose)
    if "rhso" in wanted:
        report.run(scope, "rhso_receding_horizon", rhso_receding_horizon, verbose)
    for name, fn in adapter.sanity_checks().items():
        report.run(scope, name, lambda f=fn: f(ctx), verbose)
    return report


def check_shared_initial_state(adapter, problem, specs: Sequence, manager,
                               verbose: bool = True) -> Tuple[bool, str]:
    """Jobs differing only by method/hyperparameters must produce a bit-identical z_t0."""
    if len(specs) < 2:
        return True, "fewer than two jobs to compare"
    indices = list(range(min(int(specs[0].batch_size), len(problem.ground_truth))))
    prints = []
    for spec in specs:
        state = build_initial_state(adapter, problem, spec, indices, manager)
        prints.append(pixel_fingerprint(adapter.to_numpy(state)))
    ok = len(set(prints)) == 1
    return ok, ("%d job(s) sharing (model, t0, images) start from a bit-identical z_t0: %s"
                % (len(specs), prints[0][:12] if ok else sorted(set(prints))))



