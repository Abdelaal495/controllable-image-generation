"""Execute the MeanFlow PnP-Flow and D-Flow code paths against a synthetic adapter.

No checkpoint is involved: the "model" is an analytic MeanFlow transition
T(x; a -> b) = x + (b - a) * (target - x) / max(1 - a, eps), which transports a point at
time a towards a fixed target at time 1 -- exactly the interface contract MeanFlowAdapter
declares.  That is enough to exercise every line of pnp.meanflow_pnp and
dflow.meanflow_dflow, their counters, their determinism and their gradients.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataclasses
import numpy as np
import jax
import jax.numpy as jnp

from src.models.base import AdapterSpec, Conditioning, MeanFlowAdapter
from src.problems import InverseProblem
from src.utils import MEANFLOW, MEANFLOW_DESCENDING

RES = 16
BATCH = 2


class ToyMeanFlow(MeanFlowAdapter):
    """Analytic learned transition; pixel space so to_pixels is the identity view."""

    def __init__(self, target):
        super().__init__("pmf", {})
        self.target = jnp.asarray(target)
        self.spec = AdapterSpec(
            name="pmf", display_name="pMF", dynamics_family=MEANFLOW, framework="jax",
            state_space="pixel", native_shape=(RES, RES, 3), layout="BHWC",
            pixel_resolution=RES, prediction_kind="mean_velocity",
            native_time_mapping=MEANFLOW_DESCENDING, batch_size=BATCH,
            fixed_batch_shape=True, num_classes=1000, null_label=1000, guidance={},
            checkpoint={})

    def to_native_noise(self, noise):
        return jnp.asarray(noise, jnp.float32)

    def encode_pixels(self, pixels):
        return jnp.asarray(pixels, jnp.float32)

    def to_pixels(self, state, differentiable=False):
        if differentiable:
            return state
        return np.asarray(jax.device_get(state), np.float32)

    def _lerp(self, guide, noise, keep, add):
        return keep * guide + add * noise

    def transition(self, state, s_from, s_to, conditioning):
        """A contraction towards the target plus a mild nonlinearity.

        Deliberately NOT a map that lands exactly on the target in one interval: such a map
        is independent of its input, which would make d(loss)/dq identically zero and the
        D-Flow gradient test vacuous.
        """
        self.count_forwards(1)
        dt = float(s_to) - float(s_from)
        return state + dt * (0.7 * (self.target - state) + 0.2 * jnp.sin(3.0 * state))


def make_problem(sigma=0.05):
    rng = np.random.default_rng(3)
    gt = rng.standard_normal((BATCH, RES, RES, 3), np.float32) * 0.3
    mask = np.ones((BATCH, RES, RES, 1), np.float32)
    mask[:, 4:10, 4:10, :] = 0.0
    y = gt * mask
    return InverseProblem(name="box_inpaint", key="k", sigma=sigma, params={},
                          ground_truth=gt, measurement=y, display_measurement=gt,
                          initialization_guide=y, guide_mode="zero_fill", mask=mask,
                          image_ids=("img0", "img1"))


def make_spec(method, **over):
    from src.config import load_config, validate_config, resolve_run_plan
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "experiments.yaml"))
    plan = resolve_run_plan(cfg, validate_config(cfg), run_id="exec_test")
    base = [s for s in plan.specs if s.method == method and s.model == "pmf"][0]
    return dataclasses.replace(base, num_images=BATCH, batch_size=BATCH,
                               record_loss_history=True, **over)


def report(name, ok, detail):
    print("  [%s] %-28s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def main():
    from src.pnp import meanflow_pnp, pnp_time_grid, pnp_step_sizes
    from src.dflow import meanflow_dflow
    from src.problems import make_phi, phi_log_scale

    problem = make_problem()
    adapter = ToyMeanFlow(problem.ground_truth)          # "prior" pulls towards the truth
    cond = Conditioning(labels=np.zeros((BATCH,), np.int32), guidance={})
    B = adapter.backend()
    results = []

    eps = adapter.prior_sample(problem.image_ids)
    guide = adapter.encode_pixels(problem.initialization_guide)

    # ---------------------------------------------------------------- PnP
    print("\nMeanFlow PnP-Flow")
    spec = make_spec("pnp", num_pnp_steps=4, noise_samples=3, gamma0=0.5, alpha=1.0)
    x0 = adapter.initial_state(guide, spec.t0, eps)
    state, stats = meanflow_pnp(adapter, cond, x0, problem, spec)

    N, M = spec.num_pnp_steps, spec.noise_samples
    results.append(report(
        "counters", stats.denoiser_samples == 1 + N * M
        and stats.model_evals_total == 1 + N * M
        and stats.network_forwards == 1 + N * M
        and stats.data_gradient_evals == N and stats.objective_evals == N
        and stats.backprops_through_model == 0 and stats.optimizer_iterations == 0,
        "1+N*M=%d denoiser samples, %d model evals, %d forwards, %d data grads, "
        "%d generative backprops"
        % (stats.denoiser_samples, stats.model_evals_total, stats.network_forwards,
           stats.data_gradient_evals, stats.backprops_through_model)))

    grid = pnp_time_grid(spec.canonical_start_time, N)
    gammas = pnp_step_sizes(spec, grid)
    results.append(report(
        "schedule", all(spec.canonical_start_time < s < 1.0 for s in grid)
        and gammas == sorted(gammas, reverse=True),
        "s in (%.3f, 1): %s ; gamma: %s"
        % (spec.canonical_start_time, ["%.4f" % s for s in grid],
           ["%.4f" % g for g in gammas])))

    again = meanflow_pnp(adapter, cond, x0, problem, spec)[0]
    results.append(report("determinism", bool(np.array_equal(np.asarray(state),
                                                             np.asarray(again))),
                          "a repeated run is bitwise identical"))

    other = dataclasses.replace(spec, gamma0=0.5001)
    shifted = meanflow_pnp(adapter, cond, x0, problem, other)[0]
    results.append(report(
        "noise independent of gamma0",
        float(np.max(np.abs(np.asarray(state) - np.asarray(shifted)))) < 1e-2,
        "a 0.02%% gamma0 change moves the result by %.2e, i.e. the reprojection noise "
        "was not re-rolled"
        % float(np.max(np.abs(np.asarray(state) - np.asarray(shifted))))))

    phi = make_phi(problem, B, spec.phi_normalization)
    scale = phi_log_scale(problem, spec.phi_normalization)
    start_loss = float(phi(adapter.to_pixels(x0, differentiable=True))) * scale
    end_loss = float(phi(state)) * scale
    results.append(report("fidelity improves", end_loss < start_loss,
                          "per-image fidelity %.5f -> %.5f over %d corrections"
                          % (start_loss, end_loss, N)))
    results.append(report("loss history", len(stats.loss_history) == N,
                          "%d recorded values, per-image scale (max %.5f)"
                          % (len(stats.loss_history), max(stats.loss_history))))
    results.append(report("finite output", stats.finite
                          and bool(np.isfinite(np.asarray(state)).all()),
                          "final state finite, shape %s" % (tuple(state.shape),)))

    # M > 1 must average DENOISED OUTPUTS, so the variance across realisations shrinks
    single = meanflow_pnp(adapter, cond, x0, problem,
                          dataclasses.replace(spec, noise_samples=1))[0]
    results.append(report("averaging over M", not np.array_equal(np.asarray(single),
                                                                 np.asarray(state)),
                          "M=1 and M=3 give different results (the average is real)"))

    # ---------------------------------------------------------------- D-Flow
    print("\nMeanFlow D-Flow")
    spec = make_spec("dflow", steps=1, num_opt_steps=6, lr=0.05)
    x0 = adapter.initial_state(guide, spec.t0, eps)
    state, stats = meanflow_dflow(adapter, cond, x0, problem, spec)
    iters = spec.num_opt_steps

    results.append(report(
        "counters", stats.optimizer_iterations == iters and stats.control_iterations == iters
        and stats.model_evals_total == 1 * (iters + 1)
        and stats.backprops_through_model == 1 * iters
        and stats.objective_evals == iters and stats.data_gradient_evals == iters,
        "%d Adam iters, %d model evals (= steps x (iters+1)), %d trajectory backprops"
        % (stats.optimizer_iterations, stats.model_evals_total,
           stats.backprops_through_model)))
    from src.config import _estimate_cost
    from src.utils import MEANFLOW as _MF
    predicted = _estimate_cost("dflow", {"steps": spec.steps, "solver": spec.solver,
                                         "num_opt_steps": spec.num_opt_steps}, _MF)
    results.append(report(
        "planner cost model == measured",
        stats.model_evals_total == predicted["model_evals"]
        and stats.backprops_through_model == predicted["backprops"]
        and stats.objective_evals + 0 == predicted["objective_evals"] - 1,
        "planner predicts %d evals / %d backprops for this resolved configuration; the run "
        "measured %d / %d"
        % (predicted["model_evals"], predicted["backprops"], stats.model_evals_total,
           stats.backprops_through_model)))

    history = stats.loss_history
    results.append(report("Adam reduces the loss", history[-1] < history[0],
                          "%.6f -> %.6f over %d iterations" % (history[0], history[-1], iters)))

    phi = make_phi(problem, B, spec.phi_normalization)
    scale = phi_log_scale(problem, spec.phi_normalization)
    final_loss = float(phi(adapter.to_pixels(state, differentiable=True))) * scale
    results.append(report(
        "output is the FINAL q", final_loss <= history[-1] + 1e-9,
        "returned state scores %.6f <= last recorded objective %.6f (it is the trajectory "
        "of the post-update q, not a stale iterate)" % (final_loss, history[-1])))

    # gradient of the terminal loss wrt q must be finite and non-zero
    from src.dflow import _transition_trajectory, dflow_time_grid
    g = np.asarray(jax.grad(lambda q: phi(adapter.to_pixels(
        _transition_trajectory(adapter, cond, q, dflow_time_grid(spec)),
        differentiable=True)))(x0))
    results.append(report("gradient wrt q", bool(np.isfinite(g).all()) and np.abs(g).max() > 0,
                          "|d loss / d q|max = %.4e, finite" % float(np.abs(g).max())))

    deep = make_spec("dflow", steps=3, num_opt_steps=2, lr=0.05)
    _s, deep_stats = meanflow_dflow(adapter, cond, x0, problem, deep)
    results.append(report(
        "multi-step trajectory", deep_stats.model_evals_total == 3 * (2 + 1)
        and deep_stats.backprops_through_model == 3 * 2,
        "steps=3 composes 3 transitions per objective: %d evals, %d backprops"
        % (deep_stats.model_evals_total, deep_stats.backprops_through_model)))

    print("\n%d/%d checks passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
