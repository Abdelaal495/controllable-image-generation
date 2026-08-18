"""Execute the framework-neutral parts of the standard-flow code paths.

PyTorch cannot be installed in this container (its CUDA dependency wheels exhaust the
disk), so `flow_pnp` and `flow_dflow` themselves -- which call torch.autograd.grad,
torch.optim.Adam and torch.no_grad -- cannot be run here.

What CAN be run is everything in those paths that is not torch-specific:

  * `dflow._integrate_flow`, the differentiable trajectory.  It touches only `+`, `*` and
    `adapter.velocity`, so a JAX-backed StandardFlowAdapter exercises exactly the same
    code, and its arithmetic is compared against an INDEPENDENT reimplementation of the
    reference integrator in sdedit.py, stage by stage.
  * `dflow.trajectory_evaluations`, including JiT's final-Euler-step policy.
  * `pnp.pnp_time_grid`, `pnp.pnp_step_sizes` and `pnp._reprojection_noise`.
  * The gradient of the terminal fidelity with respect to q through a multi-step
    trajectory, which is the quantity D-Flow optimises.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataclasses
import numpy as np
import jax
import jax.numpy as jnp

from src.models.base import AdapterSpec, Conditioning, StandardFlowAdapter
from src.utils import FLOW_ASCENDING, STANDARD_FLOW

RES, BATCH = 8, 2


class ToyFlow(StandardFlowAdapter):
    """A nonlinear, time-dependent velocity field -- enough to distinguish the solvers."""

    def __init__(self, euler_final_step_for_heun=False):
        super().__init__("jit", {})
        self.spec = AdapterSpec(
            name="jit", display_name="JiT", dynamics_family=STANDARD_FLOW,
            framework="jax",                      # so backend()/to_numpy stay usable here
            state_space="pixel", native_shape=(RES, RES, 3), layout="BHWC",
            pixel_resolution=RES, prediction_kind="velocity",
            native_time_mapping=FLOW_ASCENDING, batch_size=BATCH, fixed_batch_shape=False,
            num_classes=1000, null_label=1000, guidance={}, checkpoint={},
            euler_final_step_for_heun=euler_final_step_for_heun)

    def to_native_noise(self, noise):
        return jnp.asarray(noise, jnp.float32)

    def encode_pixels(self, pixels):
        return jnp.asarray(pixels, jnp.float32)

    def to_pixels(self, state, differentiable=False):
        return state if differentiable else np.asarray(jax.device_get(state), np.float32)

    def _lerp(self, guide, noise, keep, add):
        return keep * guide + add * noise

    def velocity(self, state, s, conditioning):
        self.count_forwards(1)
        return jnp.sin(2.0 * state) + float(s) * state * 0.5 + 0.3


def reference_integrate(adapter, cond, x, grid, solver, final_euler):
    """Independent transcription of sdedit_flow's stage arithmetic, for comparison."""
    steps = len(grid) - 1
    for k in range(steps):
        s, s_next = grid[k], grid[k + 1]
        dt = s_next - s
        step_solver = "euler" if (solver == "heun" and final_euler and k == steps - 1) \
            else solver
        v1 = adapter.velocity(x, s, cond)
        if step_solver == "euler":
            x = x + dt * v1
        elif step_solver == "heun":
            v2 = adapter.velocity(x + dt * v1, s_next, cond)
            x = x + 0.5 * dt * (v1 + v2)
        else:
            s_mid = 0.5 * (s + s_next)
            v2 = adapter.velocity(x + 0.5 * dt * v1, s_mid, cond)
            v3 = adapter.velocity(x + 0.5 * dt * v2, s_mid, cond)
            v4 = adapter.velocity(x + dt * v3, s_next, cond)
            x = x + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
    return x


def make_spec(method, **over):
    from src.config import load_config, resolve_run_plan, validate_config
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "experiments.yaml"))
    plan = resolve_run_plan(cfg, validate_config(cfg), run_id="exec_flow")
    base = [s for s in plan.specs if s.method == method and s.model == "jit"][0]
    return dataclasses.replace(base, num_images=BATCH, batch_size=BATCH, **over)


def report(name, ok, detail):
    print("  [%s] %-30s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def main():
    from src.dflow import _integrate_flow, dflow_time_grid, trajectory_evaluations
    from src.pnp import _reprojection_noise, pnp_step_sizes, pnp_time_grid

    cond = Conditioning(labels=np.zeros((BATCH,), np.int32), guidance={})
    rng = np.random.default_rng(11)
    q0 = jnp.asarray(rng.standard_normal((BATCH, RES, RES, 3), np.float32) * 0.4)
    results = []

    print("\nD-Flow differentiable trajectory (standard flow)")
    for solver in ("euler", "heun", "rk4"):
        for final_euler in (False, True):
            adapter = ToyFlow(final_euler)
            spec = make_spec("dflow", steps=3, solver=solver)
            grid = dflow_time_grid(spec)
            adapter.reset_counters()
            mine = _integrate_flow(adapter, cond, q0, spec, grid)
            measured = adapter.forward_counter
            adapter.reset_counters()
            ref = reference_integrate(adapter, cond, q0, grid, solver, final_euler)
            predicted = trajectory_evaluations(adapter, spec)
            err = float(np.max(np.abs(np.asarray(mine) - np.asarray(ref))))
            results.append(report(
                "%s%s" % (solver, " + final-Euler" if final_euler else ""),
                err == 0.0 and measured == predicted,
                "matches the reference integrator bitwise (max diff %g); %d model "
                "evaluation(s), predicted %d" % (err, measured, predicted)))

    adapter = ToyFlow(True)
    spec = make_spec("dflow", steps=3, solver="heun")
    results.append(report(
        "JiT final-Euler saves a stage",
        trajectory_evaluations(adapter, spec) == 3 * 2 - 1,
        "heun over 3 steps costs %d evaluations, not 6, because the official sampler's "
        "last step is Euler" % trajectory_evaluations(adapter, spec)))

    print("\nD-Flow terminal gradient through a multi-step trajectory")
    from src.problems import InverseProblem, make_phi
    gt = np.asarray(rng.standard_normal((BATCH, RES, RES, 3), np.float32) * 0.3)
    problem = InverseProblem(name="denoising", key="k", sigma=0.05, params={},
                             ground_truth=gt, measurement=gt, display_measurement=gt,
                             initialization_guide=gt, guide_mode="identity",
                             image_ids=("a", "b"))
    adapter = ToyFlow(False)
    spec = make_spec("dflow", steps=2, solver="heun")
    grid = dflow_time_grid(spec)
    phi = make_phi(problem, adapter.backend(), spec.phi_normalization)
    loss_fn = lambda q: phi(adapter.to_pixels(_integrate_flow(adapter, cond, q, spec, grid),
                                              differentiable=True))
    value, grad = jax.value_and_grad(loss_fn)(q0)
    g = np.asarray(grad)
    results.append(report("gradient is finite and non-zero",
                          bool(np.isfinite(g).all()) and np.abs(g).max() > 0,
                          "loss %.5f, |d loss / d q|max = %.4e through %d chained model "
                          "evaluations" % (float(value), float(np.abs(g).max()),
                                           trajectory_evaluations(adapter, spec))))
    stepped = q0 - 0.05 * grad
    results.append(report("a descent step reduces the loss",
                          float(loss_fn(stepped)) < float(value),
                          "%.6f -> %.6f after one manual gradient step"
                          % (float(value), float(loss_fn(stepped)))))

    print("\nPnP schedule and reprojection noise (framework-neutral)")
    spec = make_spec("pnp", num_pnp_steps=6, gamma0=2.0, alpha=0.6)
    grid = pnp_time_grid(spec.canonical_start_time, spec.num_pnp_steps)
    gammas = pnp_step_sizes(spec, grid)
    s0 = spec.canonical_start_time
    expected = [s0 + (k / 7.0) * (1 - s0) for k in range(1, 7)]
    results.append(report(
        "s_k = s0 + k/(N+1)*(1-s0)",
        max(abs(a - b) for a, b in zip(grid, expected)) < 1e-12
        and all(s0 < s < 1 for s in grid),
        "%s, strictly inside (%.2f, 1)" % (["%.4f" % s for s in grid], s0)))
    results.append(report(
        "gamma_k = gamma0 (1-s_k)^alpha",
        max(abs(g - 2.0 * (1 - s) ** 0.6) for g, s in zip(gammas, grid)) < 1e-12,
        "%s" % ["%.4f" % g for g in gammas]))

    a = np.asarray(_reprojection_noise(adapter, spec, ["a", "b"], 1, 0))
    b = np.asarray(_reprojection_noise(adapter, spec, ["a", "b"], 1, 0))
    c = np.asarray(_reprojection_noise(adapter, spec, ["a", "b"], 2, 0))
    per_image = np.asarray(_reprojection_noise(adapter, spec, ["b", "a"], 1, 0))
    swapped_ok = np.array_equal(a[0], per_image[1]) and np.array_equal(a[1], per_image[0])
    results.append(report(
        "noise is per image and per iteration",
        np.array_equal(a, b) and not np.array_equal(a, c) and swapped_ok
        and a.shape == (2, RES, RES, 3),
        "shape %s; reproducible, fresh per iteration, and follows the IMAGE rather than "
        "the batch position (a reordered batch gives each image its own draw back)"
        % (a.shape,)))

    print("\n%d/%d checks passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
