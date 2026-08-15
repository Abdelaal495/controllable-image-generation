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
                    canonical_start_time, gaussian_noise, native_time, pixel_fingerprint)


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
    """s_start = 1 - t0, delta = t0/N, and the terminal time is exactly 1."""
    from .mpc import mpc_time_grid
    problems = []
    for spec in plan.specs:
        if not spec.is_mpc:
            continue
        grid = mpc_time_grid(spec)
        if abs(grid[0] - (1.0 - spec.t0)) > 1e-9:
            problems.append("%s: s_start" % spec.job_id[:8])
        if abs(grid[-1] - 1.0) > 1e-12:
            problems.append("%s: terminal" % spec.job_id[:8])
        if abs((grid[1] - grid[0]) - spec.delta) > 1e-9:
            problems.append("%s: delta" % spec.job_id[:8])
    return not problems, ("s_start = 1 - t0, delta = t0/N, terminal time exactly 1 for every "
                          "MPC job" if not problems else "; ".join(problems[:5]))


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
    report.run("plan", "time_grid", lambda: check_time_grid(plan), verbose)
    report.run("plan", "paired_fairness", lambda: check_paired_fairness(plan), verbose)
    report.run("seeding", "noise_identity", check_noise_identity_independence, verbose)

    for key, problem in problems.items():
        scope = problem.name
        report.run(scope, "operator_parity",
                   lambda p=problem: check_operator_parity(p, backends), verbose)
        report.run(scope, "measurement",
                   lambda p=problem: check_measurement_reconstruction(p), verbose)
        report.run(scope, "gradient",
                   lambda p=problem: check_gradient(p, backends), verbose)
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
                     t0_probe: float = 0.5) -> CheckReport:
    """Shape, time-convention, initialisation and one-tiny-trajectory checks."""
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

    if verbose:
        print("Model checks for %s" % adapter.spec.name.upper())
    report.run(scope, "shapes", shapes, verbose)
    report.run(scope, "native_times", native_times, verbose)
    report.run(scope, "init_t0_eq_1", pure_noise_initialisation, verbose)
    report.run(scope, "init_t0_lt_1", guided_initialisation, verbose)
    report.run(scope, "paired_noise", paired_noise, verbose)
    report.run(scope, "tiny_trajectory", tiny_trajectory, verbose)
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
