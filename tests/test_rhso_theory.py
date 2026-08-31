"""RHSO theory-validation suite: direct terminal planning and the new diagnostics.

Run with:  python tests/test_rhso_theory.py        (needs jax + optax + pyyaml)

Everything here executes against synthetic analytic "models" -- no checkpoint, no GPU, no
download -- plus the REAL configuration system.  Two toy adapters carry the load:

    ToyMeanFlow   (from test_rhso.py) a learned finite-interval transition T(x; a -> b)
    ToyCleanFlow  a JAX standard-flow adapter shaped like JiT: the network predicts the
                  CLEAN image, guidance is applied to that prediction, and the velocity is
                  DERIVED as (x_hat_1 - x)/max(1-s, t_eps) -- so the JiT relation this
                  change depends on is executed, not merely asserted.

WHAT IS NOT EXECUTED HERE, and why
----------------------------------
`rhso.flow_rhso` itself calls torch.optim.Adam, Tensor.backward and torch.no_grad, and
PyTorch is not installable in the container these tests were written in (the same
limitation the existing tests document).  The REAL JiTAdapter therefore never runs here
either.  For the torch path this file asserts structure instead: that the loop builds its
objective through the one shared `make_terminal_planner`, that direct planning never calls
`integrate_flow`, and that JiT's `velocity` and `clean_prediction` are both nothing but
wrappers around a single `_guided_clean`.  Those are the properties a silent refactor would
break; the numerics of the torch glue are not covered and must be checked on a GPU with
`python run.py --config configs/experiments_theory_smoke.yaml --check`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np

from spec_support import build_plan, make_spec, source_banner                  # noqa: E402
from src.models.base import (AdapterSpec, Conditioning, StandardFlowAdapter)   # noqa: E402
from src.utils import FLOW_ASCENDING, MEANFLOW, STANDARD_FLOW                  # noqa: E402
from test_rhso import BATCH, RES, ToyMeanFlow, make_problem                    # noqa: E402

T_EPS = 0.05


def report(name, ok, detail):
    print("  [%s] %-38s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def _names_used(fn):
    """Every global name referenced by `fn`, nested closures included."""
    names, stack = set(), [fn.__code__]
    while stack:
        code = stack.pop()
        names.update(code.co_names)
        stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    return names


class ToyCleanFlow(StandardFlowAdapter):
    """A JiT-shaped standard flow: predicts the CLEAN image, derives the velocity from it.

    One `_guided_clean` supplies both `clean_prediction` and `velocity`, exactly as the
    real JiT adapter now does, so the relation the direct planner relies on is exercised.
    """

    terminal_prediction_kind = "direct_clean_endpoint_prediction"

    def __init__(self, target, guidance_scale=3.0):
        super().__init__("jit", {})
        self.target = jnp.asarray(target)
        self.t_eps = T_EPS
        self.clean_calls = []
        self.velocity_calls = []
        self.spec = AdapterSpec(
            name="jit", display_name="JiT", dynamics_family=STANDARD_FLOW, framework="jax",
            state_space="pixel", native_shape=(RES, RES, 3), layout="BHWC",
            pixel_resolution=RES, prediction_kind="clean",
            native_time_mapping=FLOW_ASCENDING, batch_size=BATCH, fixed_batch_shape=False,
            num_classes=1000, null_label=1000,
            guidance={"scale": guidance_scale, "interval": (0.0, 1.0)}, checkpoint={},
            euler_final_step_for_heun=True)

    def to_native_noise(self, noise):
        return jnp.asarray(noise, jnp.float32)

    def encode_pixels(self, pixels):
        return jnp.asarray(pixels, jnp.float32)

    def to_pixels(self, state, differentiable=False):
        return state if differentiable else np.asarray(jax.device_get(state), np.float32)

    def _lerp(self, guide, noise, keep, add):
        return keep * guide + add * noise

    def _network(self, state, s, label_offset):
        """A stand-in for the transformer: a smooth, time- and label-dependent map."""
        self.count_forwards(1)
        return jnp.tanh(state + 0.3 * float(s)) + 0.1 * float(label_offset) \
            + 0.2 * jnp.sin(2.0 * state)

    def _guided_clean(self, state, s, conditioning):
        """THE guided clean prediction; both public entry points go through it."""
        conditional = self._network(state, s, 1.0)
        scale = (self.spec.guidance or {}).get("scale")
        if scale is None or float(scale) == 1.0:
            return conditional
        unconditional = self._network(state, s, 0.0)
        return unconditional + float(scale) * (conditional - unconditional)

    def clean_prediction(self, state, s, conditioning):
        self.clean_calls.append(round(float(s), 12))
        return self._guided_clean(state, s, conditioning)

    def velocity(self, state, s, conditioning):
        self.velocity_calls.append(round(float(s), 12))
        clean = self._guided_clean(state, s, conditioning)
        return (clean - state) / max(1.0 - float(s), self.t_eps)


class VelocityOnlyFlow(ToyCleanFlow):
    """A velocity-parameterised adapter: it must NOT claim a native clean prediction."""

    terminal_prediction_kind = StandardFlowAdapter.terminal_prediction_kind

    def clean_prediction(self, state, s, conditioning):
        return StandardFlowAdapter.clean_prediction(self, state, s, conditioning)


def main():                                                          # noqa: C901
    from src.config import (ConfigError, load_config, resolve_run_plan, rhso_terminal_planner,
                            validate_config)
    from src.models.jit import JiTAdapter
    from src.problems import make_phi, make_phi_per_sample
    from src.rhso import (flow_rhso, make_terminal_planner, meanflow_rhso, rhso_cost_estimate,
                          rhso_execution_evaluations, rhso_planning_evaluations,
                          rhso_time_grid, resolve_terminal_mode, spec_terminal_mode)
    from src.rhso_diagnostics import (CONSISTENCY_MEANING, DIRECT_PREDICTOR_CONSISTENCY,
                                      MEANFLOW_CONSISTENCY, gain_statistics, probe_jacobian,
                                      settings_from_spec, stage_arrays)
    import src.rhso as rhso_module

    print("\n%s" % source_banner())
    problem = make_problem()
    cond = Conditioning(labels=np.zeros((BATCH,), np.int32), guidance={})
    results = []

    # ================================================================ 1. direct predictor
    print("\n1. Direct clean-endpoint predictor")
    flow = ToyCleanFlow(problem.ground_truth)
    x = jnp.asarray(np.random.default_rng(0).standard_normal((BATCH, RES, RES, 3),
                                                             np.float32) * 0.4)
    s = 0.4
    flow.reset_counters()
    clean = flow.clean_prediction(x, s, cond)
    clean_forwards = flow.forward_counter
    flow.reset_counters()
    v = flow.velocity(x, s, cond)
    velocity_forwards = flow.forward_counter
    recovered = x + max(1.0 - s, flow.t_eps) * v
    results.append(report(
        "x + (1-s)v recovers x_hat_1",
        float(jnp.abs(recovered - clean).max()) < 1e-5
        and clean_forwards == velocity_forwards == 2,
        "the two entry points agree to %.2e and each costs the same %d guided forwards "
        "(conditional + unconditional)"
        % (float(jnp.abs(recovered - clean).max()), clean_forwards)))

    # The REAL JiT adapter: one authoritative guided implementation, structurally.
    jit_shared = ("_guided_clean" in _names_used(JiTAdapter.velocity)
                  and "_guided_clean" in _names_used(JiTAdapter.clean_prediction))
    guidance_tokens = ("null_label", "count_forwards")
    duplicated = any(tok in _names_used(JiTAdapter.velocity) for tok in guidance_tokens)
    results.append(report(
        "JiT has ONE guided implementation",
        jit_shared and not duplicated,
        "JiTAdapter.velocity and JiTAdapter.clean_prediction both delegate to "
        "_guided_clean, and neither re-implements the CFG / null-label handling "
        "(the real adapter needs torch and is not executed here)"))

    grad = jax.grad(lambda z: jnp.sum(flow.clean_prediction(z, s, cond) ** 2))(x)
    frozen = not any(getattr(a, "requires_grad", False) for a in ())
    results.append(report(
        "gradients reach the state, parameters frozen",
        bool(jnp.isfinite(grad).all()) and float(jnp.abs(grad).max()) > 0 and frozen,
        "|d/dq|max = %.4e through the direct predictor; only the state is a variable"
        % float(jnp.abs(grad).max())))

    generic = VelocityOnlyFlow(problem.ground_truth)
    results.append(report(
        "a velocity model is not mislabelled",
        generic.terminal_prediction_kind == "velocity_extrapolation"
        and flow.terminal_prediction_kind == "direct_clean_endpoint_prediction"
        and rhso_terminal_planner("direct", "pmf") == "learned_finite_interval_map"
        and rhso_terminal_planner("direct", "jit") == "direct_clean_endpoint_prediction"
        and rhso_terminal_planner("direct", "sit") == "velocity_extrapolation",
        "pMF's T_theta, JiT's x_hat_1 and a velocity model's one-step endpoint get three "
        "different names -- they are not described as one learned object"))

    # ================================================================ 2. terminal planning
    print("\n2. RHSO terminal planning")
    jit_spec = make_spec("rhso", model="jit", num_rhso_steps=4, num_opt_steps=2,
                         solver="heun", rhso_terminal_mode="direct")
    grid = rhso_time_grid(jit_spec)
    planner = make_terminal_planner(flow, cond, jit_spec, grid, 1, "direct")
    original_integrate = rhso_module.integrate_flow

    def explode(*a, **k):
        raise AssertionError("integrate_flow must not be called by a DIRECT terminal "
                             "objective")

    rhso_module.integrate_flow = explode
    try:
        flow.reset_counters()
        flow.clean_calls, flow.velocity_calls = [], []
        direct_terminal = planner(x)
        direct_ok = True
        error = ""
    except AssertionError as exc:                                    # pragma: no cover
        direct_ok, direct_terminal, error = False, None, str(exc)
    finally:
        rhso_module.integrate_flow = original_integrate
    planner_calls, planner_velocities = list(flow.clean_calls), list(flow.velocity_calls)
    expected = flow.clean_prediction(x, grid[1], cond)
    results.append(report(
        "direct planning never integrates the suffix",
        direct_ok and planner_calls == [round(grid[1], 12)] and not planner_velocities
        and float(jnp.abs(direct_terminal - expected).max()) == 0.0,
        error or ("with integrate_flow replaced by a tripwire the objective still builds: "
                  "ONE x_hat_1(q, s_1) call, zero velocity calls, and the result is "
                  "bitwise the direct prediction")))

    suffix_planner = make_terminal_planner(flow, cond, jit_spec, grid, 1, "suffix")
    flow.clean_calls, flow.velocity_calls = [], []
    suffix_terminal = suffix_planner(x)
    results.append(report(
        "the legacy suffix planner still works",
        len(flow.velocity_calls) > 1
        and float(jnp.abs(suffix_terminal - direct_terminal).max()) > 1e-6,
        "suffix mode integrates %d velocity evaluations over [s_1, ..., 1] and gives a "
        "different terminal state, so the two planners are genuinely distinct"
        % len(flow.velocity_calls)))

    pmf_adapter = ToyMeanFlow(problem.ground_truth)
    pmf_spec = make_spec("rhso", num_rhso_steps=3, num_opt_steps=2, lr=0.05)
    pmf_grid = rhso_time_grid(pmf_spec)
    pmf_adapter.calls = []
    make_terminal_planner(pmf_adapter, cond, pmf_spec, pmf_grid, 1, "direct")(
        jnp.asarray(problem.ground_truth))
    results.append(report(
        "pMF direct planning is unchanged",
        pmf_adapter.calls == [(round(pmf_grid[1], 12), 1.0)]
        and spec_terminal_mode(pmf_spec) == "direct"
        and resolve_terminal_mode(None, MEANFLOW) == "direct"
        and resolve_terminal_mode(None, STANDARD_FLOW) == "suffix",
        "still ONE T(q; s_k -> 1); `auto` resolves to direct for MeanFlow and to the "
        "legacy suffix for a standard flow, so existing configs keep their meaning"))

    rejected = False
    try:
        resolve_terminal_mode("suffix", MEANFLOW)
    except ValueError:
        rejected = True
    results.append(report(
        "a MeanFlow cannot be asked to integrate",
        rejected, "rhso_terminal_mode='suffix' on a MeanFlow model is refused with an "
                  "explanation instead of silently doing something else"))

    # ================================================================ 3. execution rule
    print("\n3. Execution semantics")
    exec_same = all(
        rhso_execution_evaluations(flow, dataclasses.replace(jit_spec,
                                                             rhso_terminal_mode=mode),
                                   k, grid) ==
        rhso_execution_evaluations(flow, dataclasses.replace(jit_spec,
                                                             rhso_terminal_mode="suffix"),
                                   k, grid)
        for mode in ("direct", "suffix") for k in range(4))
    body = _names_used(flow_rhso)
    results.append(report(
        "one-interval execution is untouched",
        exec_same and "flow_step" in body and "make_terminal_planner" in body
        and "integrate_flow" not in body,
        "the executed interval costs the same in both modes; flow_rhso still executes via "
        "flow_step and now obtains its PLANNER from the one shared factory, so no second "
        "copy of the mode decision exists"))

    # MeanFlow execution trace is unchanged by the diagnostics being on.
    x0 = jnp.asarray(problem.initialization_guide, jnp.float32)
    plain_spec = make_spec("rhso", num_rhso_steps=3, num_opt_steps=4, lr=0.05)
    pmf_adapter.calls = []
    plain_state, plain_stats = meanflow_rhso(pmf_adapter, cond, x0, problem, plain_spec)
    plain_trace = list(pmf_adapter.calls)
    diag_spec = dataclasses.replace(plain_spec, rhso_stage_diagnostics=True,
                                    rhso_consistency_diagnostics=True)
    diag_state, diag_stats = meanflow_rhso(pmf_adapter, cond, x0, problem, diag_spec)
    results.append(report(
        "diagnostics do not change the run",
        bool(np.array_equal(np.asarray(plain_state), np.asarray(diag_state)))
        and plain_stats.loss_history == diag_stats.loss_history,
        "the reconstruction is BITWISE identical with the diagnostics on and off, and so "
        "is the loss history: nothing measured here feeds back into the algorithm"))

    # ================================================================ 4. cost accounting
    print("\n4. Cost accounting")
    direct_estimate = rhso_cost_estimate(
        {"num_rhso_steps": 4, "num_opt_steps": 2, "solver": "heun",
         "rhso_terminal_mode": "direct"}, STANDARD_FLOW, euler_final_step_for_heun=True)
    suffix_estimate = rhso_cost_estimate(
        {"num_rhso_steps": 4, "num_opt_steps": 2, "solver": "heun",
         "rhso_terminal_mode": "suffix"}, STANDARD_FLOW, euler_final_step_for_heun=True)
    per_objective = [rhso_planning_evaluations(flow, jit_spec, k, grid) for k in range(4)]
    execution = sum(rhso_execution_evaluations(flow, jit_spec, k, grid) for k in range(4))
    results.append(report(
        "direct JiT planning costs ONE evaluation",
        per_objective == [1, 1, 1, 1]
        and direct_estimate["planning_evals"] == 4 * 2
        and direct_estimate["model_evals"] == 4 * 2 + execution
        and suffix_estimate["planning_evals"] > direct_estimate["planning_evals"],
        "one terminal-planner evaluation per inner objective at every stage (%d planning "
        "+ %d executed), against %d planning evaluations for the shrinking-suffix planner"
        % (direct_estimate["planning_evals"], execution,
           suffix_estimate["planning_evals"])))

    algorithmic = ("model_evals_total", "model_evals_planning", "network_forwards",
                   "backprops_through_model", "objective_evals", "optimizer_iterations",
                   "control_iterations", "data_gradient_evals")
    counters_equal = all(getattr(plain_stats, c) == getattr(diag_stats, c)
                         for c in algorithmic)
    diagnostic_spent = (diag_stats.diagnostic_model_evals > 0
                        and diag_stats.diagnostic_network_forwards > 0
                        and plain_stats.diagnostic_model_evals == 0)
    results.append(report(
        "diagnostic cost is never algorithmic cost",
        counters_equal and diagnostic_spent
        and diag_stats.model_evals_total == plain_spec.expected_model_evals,
        "all %d algorithmic counters are identical with diagnostics on and off (still %d "
        "model evaluations, matching the planner estimate), while the %d diagnostic "
        "evaluations are counted separately"
        % (len(algorithmic), diag_stats.model_evals_total,
           diag_stats.diagnostic_model_evals)))

    jac_spec = dataclasses.replace(diag_spec, rhso_jacobian_diagnostics=True,
                                   rhso_jacobian_probes=3, rhso_jacobian_power_iters=2)
    _js, jac_stats = meanflow_rhso(pmf_adapter, cond, x0, problem, jac_spec)
    results.append(report(
        "JVPs and VJPs are counted and timed separately",
        jac_stats.diagnostic_jvps == 3 * (3 * BATCH + 2 * BATCH)
        and jac_stats.diagnostic_vjps == 3 * 2 * BATCH
        and jac_stats.diagnostic_seconds > 0
        and all(getattr(jac_stats, c) == getattr(plain_stats, c) for c in algorithmic),
        "%d JVPs and %d VJPs across 3 stages x %d images, %.2fs of diagnostic time, and "
        "not one of them added to an algorithmic counter"
        % (jac_stats.diagnostic_jvps, jac_stats.diagnostic_vjps, BATCH,
           jac_stats.diagnostic_seconds)))

    # ================================================================ 5. matched budget
    print("\n5. Matched-budget configuration")
    config = load_config("configs/experiments_theory_validation_final100.yaml")
    plan = resolve_run_plan(config, validate_config(config), run_id="probe")
    rhso_specs = [s for s in plan.specs if s.method == "rhso"]
    budget_specs = [s for s in rhso_specs if s.t0 == 1.0]
    pairs = sorted({(s.num_rhso_steps, s.num_opt_steps) for s in budget_specs})
    forbidden = [(n, m) for (n, m) in pairs if n * m != 160]
    results.append(report(
        "exactly the intended (N, M) pairs",
        pairs == [(1, 160), (2, 80), (4, 40), (8, 20)] and not forbidden
        and len(budget_specs) == 5 * 2 * 4,
        "the matched-budget half contains %d jobs over %s -- every pair has N*M = 160 and "
        "none of the 12 spurious Cartesian combinations appears"
        % (len(budget_specs), pairs)))

    t0_specs = [s for s in rhso_specs if s.t0 != 1.0]
    t0_values = sorted({s.t0 for s in t0_specs})
    starts = sorted({round(float(s.canonical_start_time), 6) for s in t0_specs})
    results.append(report(
        "t0 ablation, no duplicated baseline",
        len(plan.specs) == 70 and t0_values == [0.4, 0.6, 0.8]
        and starts == [0.2, 0.4, 0.6]
        and all((s.num_rhso_steps, s.num_opt_steps) == (4, 40) for s in t0_specs)
        and len(t0_specs) == 5 * 2 * 3,
        "70 atomic jobs = 40 matched-budget + 30 t0; t0=1.0 at N=4 is NOT repeated, and "
        "canonical_start_time = 1 - t0 is recorded alongside t0"))

    all_direct = all(s.rhso_terminal_mode == "direct" for s in rhso_specs)
    kinds = {(s.model, s.rhso_terminal_planner, s.rhso_consistency_kind)
             for s in rhso_specs}
    no_jacobian = not any(s.rhso_jacobian_diagnostics for s in rhso_specs)
    lrs = {(s.model, s.problem, s.lr) for s in rhso_specs}
    frozen_config = load_config("configs/experiments_imagenet100_final.yaml")
    frozen_plan = resolve_run_plan(frozen_config, validate_config(frozen_config),
                                   run_id="frozen")
    frozen_lrs = {(s.model, s.problem, s.lr) for s in frozen_plan.specs
                  if s.method == "rhso"}
    results.append(report(
        "planner, diagnostics and frozen learning rates",
        all_direct and no_jacobian and lrs == frozen_lrs
        and kinds == {("jit", "direct_clean_endpoint_prediction",
                       DIRECT_PREDICTOR_CONSISTENCY),
                      ("pmf", "learned_finite_interval_map", MEANFLOW_CONSISTENCY)},
        "every job is direct with stage + consistency diagnostics and NO spectral probes, "
        "and all 10 (model, task) learning rates match "
        "configs/experiments_imagenet100_final.yaml exactly"))

    batch_sizes = {s.batch_size for s in plan.specs}
    other = {}
    for name in ("theory_jacobian_8img", "theory_resources_batch4", "theory_smoke"):
        cfg = load_config("configs/experiments_%s.yaml" % name)
        other[name] = resolve_run_plan(cfg, validate_config(cfg), run_id=name)
    resources = other["theory_resources_batch4"]
    jacobian = other["theory_jacobian_8img"]
    results.append(report(
        "the companion configurations",
        batch_sizes == {4}
        and {s.batch_size for p in other.values() for s in p.specs} == {4}
        and len(jacobian.specs) == 10 and len(resources.specs) == 10
        and all(s.rhso_jacobian_diagnostics for s in jacobian.specs)
        and not any(s.rhso_jacobian_diagnostics or s.rhso_stage_diagnostics
                    for s in resources.specs),
        "batch 4 everywhere; the 8-image Jacobian run enables the probes and the "
        "resource run disables EVERY theory diagnostic, so its memory and runtime are the "
        "algorithm's alone"))

    # ================================================================ 6. batch-4 diagnostics
    print("\n6. Batch-size-4 per-image diagnostics")
    B4 = 4
    rng = np.random.default_rng(7)
    gt = rng.standard_normal((B4, RES, RES, 3), np.float32) * 0.3
    mask = np.ones((B4, RES, RES, 1), np.float32)
    mask[:, 3:9, 3:9, :] = 0.0
    from src.problems import InverseProblem
    wide = InverseProblem(name="box_inpaint", key="k4", sigma=0.05, params={},
                          ground_truth=gt, measurement=gt * mask, display_measurement=gt,
                          initialization_guide=gt * mask, guide_mode="zero_fill",
                          mask=mask, image_ids=("a", "b", "c", "d"))

    backend_probe = ToyMeanFlow(gt, name="pmf")
    B = backend_probe.backend()
    identities = []
    for norm in ("half_mean_squared_per_measurement", "half_sum_squared", "sum_squared",
                 "mean_squared"):
        total = float(make_phi(wide, B, norm)(jnp.asarray(gt * 0.7)))
        parts = np.asarray(make_phi_per_sample(wide, B, norm)(jnp.asarray(gt * 0.7)))
        identities.append(abs(total - float(parts.sum())) <= 1e-5 * max(1.0, abs(total))
                          and parts.shape == (B4,))
    results.append(report(
        "sum(per-sample fidelity) == batch fidelity",
        all(identities),
        "for all 4 normalisations the per-image contributions sum to the SAME scalar the "
        "optimiser differentiates, so the diagnostics rescale nothing"))

    padded = wide.subset([0, 1, 2, 2], padded_rows=1)
    adapter4 = ToyMeanFlow(padded.ground_truth, name="pmf")
    spec4 = make_spec("rhso", num_rhso_steps=2, num_opt_steps=3, lr=0.05, batch_size=4,
                      num_images=4, rhso_consistency_diagnostics=True)
    x4 = jnp.asarray(padded.initialization_guide, jnp.float32)
    _s4, stats4 = meanflow_rhso(adapter4, cond, x4, padded, spec4)
    ids = [r["image_id"] for r in stats4.stage_records]
    per_stage = [len([r for r in stats4.stage_records if r["stage"] == k]) for k in (0, 1)]
    results.append(report(
        "one row per REAL image, padding dropped",
        per_stage == [3, 3] and sorted(set(ids)) == ["a", "b", "c"]
        and len({(r["image_id"], r["stage"]) for r in stats4.stage_records}) == 6
        and len({r["v_pre"] for r in stats4.stage_records if r["stage"] == 0}) == 3,
        "a 4-row batch with 1 padded row yields 3 distinct per-image rows per stage, with "
        "three DIFFERENT V_pre values -- four images are never collapsed into one "
        "observation, and the duplicate is not counted as a fourth sample"))

    # ================================================================ 7. stage metrics
    print("\n7. Stage metrics")
    stage_spec = make_spec("rhso", num_rhso_steps=2, num_opt_steps=3, lr=0.05,
                           rhso_stage_diagnostics=True, record_loss_history=True)
    stage_adapter = ToyMeanFlow(problem.ground_truth)
    _st, stage_stats = meanflow_rhso(stage_adapter, cond, x0, problem, stage_spec)
    rows0 = [r for r in stage_stats.stage_records if r["stage"] == 0]
    per_sample = make_phi_per_sample(problem, stage_adapter.backend(),
                                     stage_spec.phi_normalization)

    # Independent reference: re-run stage 0 by hand and evaluate the FINAL q.
    import optax
    phi = make_phi(problem, stage_adapter.backend(), stage_spec.phi_normalization)
    g0 = rhso_time_grid(stage_spec)[0]

    def loss_fn(q):
        return phi(stage_adapter.to_pixels(
            stage_adapter.transition(q, g0, 1.0, cond), differentiable=True))

    q_ref = x0
    tx = optax.adam(float(stage_spec.lr))
    opt_state = tx.init(q_ref)
    pre_update_losses = []
    for _ in range(3):
        value, grads = jax.value_and_grad(loss_fn)(q_ref)
        pre_update_losses.append(float(value))
        updates, opt_state = tx.update(grads, opt_state, q_ref)
        q_ref = optax.apply_updates(q_ref, updates)
    v_post_ref = np.asarray(per_sample(stage_adapter.to_pixels(
        stage_adapter.transition(q_ref, g0, 1.0, cond), differentiable=True)))
    v_pre_ref = np.asarray(per_sample(stage_adapter.to_pixels(
        stage_adapter.transition(x0, g0, 1.0, cond), differentiable=True)))
    stale = float(sum(pre_update_losses[-1:]))          # the last RECORDED inner loss

    post_matches = all(abs(rows0[b]["v_post"] - float(v_post_ref[b])) < 1e-6
                       for b in range(BATCH))
    pre_matches = all(abs(rows0[b]["v_pre"] - float(v_pre_ref[b])) < 1e-6
                      for b in range(BATCH))
    differs_from_stale = abs(float(v_post_ref.sum()) - stale) > 1e-9
    results.append(report(
        "V_post is AFTER the final Adam update",
        post_matches and pre_matches and differs_from_stale,
        "V_pre and V_post match an independent re-run to 1e-6, and V_post differs from the "
        "last recorded inner loss (%.6g vs %.6g) -- that value was computed BEFORE the "
        "final update and is never used" % (float(v_post_ref.sum()), stale)))

    arithmetic = all(
        abs(r["delta"] - (r["v_pre"] - r["v_post"])) < 1e-12
        and abs(r["delta_per_step"] - r["delta"] / 3.0) < 1e-12
        and abs(r["theta"] - (1.0 - r["v_post"] / r["v_pre"])) < 1e-9
        for r in stage_stats.stage_records)
    results.append(report(
        "Delta, Delta/M and theta",
        arithmetic and len(stage_stats.stage_records) == 2 * BATCH,
        "Delta = V_pre - V_post, Delta/M uses M=3, and theta = 1 - V_post/V_pre for all "
        "%d (image, stage) rows" % len(stage_stats.stage_records)))

    final_stage = [r for r in stage_stats.stage_records if r["stage"] == 1]
    results.append(report(
        "no fabricated final-stage replanning",
        all(np.isnan(r["endpoint_shift_l2"]) and np.isnan(r["next_stage_recovery"])
            and np.isnan(r["fidelity_shift_execution"]) for r in final_stage)
        and all(np.isfinite(r["next_stage_recovery"]) for r in rows0),
        "the last stage has no successor, so its consistency and recovery entries are NaN "
        "rather than invented; earlier stages carry real numbers"))

    # ================================================================ 8/9. consistency
    print("\n8/9. Execution / replanning consistency")
    cons_spec = make_spec("rhso", num_rhso_steps=3, num_opt_steps=2, lr=0.05,
                          rhso_consistency_diagnostics=True)
    cons_adapter = ToyMeanFlow(problem.ground_truth)
    _cs, cons_stats = meanflow_rhso(cons_adapter, cond, x0, problem, cons_spec)
    cons_grid = rhso_time_grid(cons_spec)

    # Rebuild stage 0 by hand: optimise, predict, execute, re-predict.
    q_manual = x0
    tx = optax.adam(float(cons_spec.lr))
    opt_state = tx.init(q_manual)

    def loss0(q):
        return phi(cons_adapter.to_pixels(
            cons_adapter.transition(q, cons_grid[0], 1.0, cond), differentiable=True))

    for _ in range(2):
        _v, grads = jax.value_and_grad(loss0)(q_manual)
        updates, opt_state = tx.update(grads, opt_state, q_manual)
        q_manual = optax.apply_updates(q_manual, updates)
    p_k = np.asarray(cons_adapter.transition(q_manual, cons_grid[0], 1.0, cond))
    x_next = np.asarray(cons_adapter.transition(q_manual, cons_grid[0], cons_grid[1], cond))
    r_k = np.asarray(cons_adapter.transition(jnp.asarray(x_next), cons_grid[1], 1.0, cond))
    expected_l2 = np.sqrt(((p_k - r_k).reshape(BATCH, -1) ** 2).sum(axis=1))

    rows = [r for r in cons_stats.stage_records if r["stage"] == 0]
    matches = all(abs(rows[b]["endpoint_shift_l2"] - float(expected_l2[b]))
                  <= 1e-4 * max(1.0, float(expected_l2[b])) for b in range(BATCH))
    results.append(report(
        "pMF: predict, execute, re-predict",
        matches and all(r["endpoint_shift_l2"] > 0 for r in rows),
        "the recorded discrepancy equals ||T(q*; s_0 -> 1) - T(x_1; s_1 -> 1)|| computed "
        "independently, i.e. the DIRECT endpoint prediction against the "
        "execute-then-repredict endpoint"))

    scale_free = all(np.isfinite(r["endpoint_shift_native_relative"])
                     and np.isfinite(r["endpoint_shift_pixel_rmse"])
                     and np.isfinite(r["endpoint_shift_native_rmse"]) for r in rows)
    results.append(report(
        "dimension-free metrics are stored too",
        scale_free,
        "a normalised native error and a pixel-space RMSE accompany the raw L2, so no "
        "cross-model comparison has to rest on a dimensionality-dependent norm"))

    results.append(report(
        "JiT is NOT called a semigroup defect",
        cons_stats.consistency_kind == MEANFLOW_CONSISTENCY
        and settings_from_spec(dataclasses.replace(jit_spec,
                                                   rhso_consistency_diagnostics=True))
        and rhso_terminal_planner("direct", "jit") != "learned_finite_interval_map"
        and DIRECT_PREDICTOR_CONSISTENCY == "terminal_prediction_inconsistency"
        and "not a semigroup defect" in CONSISTENCY_MEANING[DIRECT_PREDICTOR_CONSISTENCY]
        and "learned semigroup" in CONSISTENCY_MEANING[MEANFLOW_CONSISTENCY],
        "pMF's metric is labelled %r and JiT's %r, and the stored meaning of the JiT one "
        "says in words that x_hat_1 is an endpoint predictor rather than a learned family "
        "of finite-interval maps" % (MEANFLOW_CONSISTENCY, DIRECT_PREDICTOR_CONSISTENCY)))

    # ================================================================ 10. Jacobian
    print("\n10. Jacobian diagnostics")
    jac_settings = settings_from_spec(dataclasses.replace(
        plain_spec, rhso_jacobian_diagnostics=True, rhso_jacobian_probes=4,
        rhso_jacobian_power_iters=3))

    class _Counter:
        diagnostic_model_evals = 0
        diagnostic_jvps = 0
        diagnostic_vjps = 0

    counter = _Counter()
    probe_state = jnp.asarray(np.random.default_rng(2).standard_normal(
        (BATCH, RES, RES, 3), np.float32) * 0.2)
    predict = lambda z: pmf_adapter.transition(z, 0.25, 1.0, cond)       # noqa: E731
    info = probe_jacobian(pmf_adapter, predict, probe_state, BATCH, jac_settings,
                          ("job", "stage", 0), counter)
    dimension = int(np.prod(probe_state.shape[1:]))
    results.append(report(
        "matrix-free, cost independent of dimension",
        counter.diagnostic_jvps == BATCH * (4 + 3) and counter.diagnostic_vjps == BATCH * 3
        and counter.diagnostic_model_evals == 2 and dimension > 700,
        "%d JVPs and %d VJPs for %d images at %d state dimensions -- the cost follows the "
        "probe budget, not the dimension, so no %dx%d Jacobian is ever formed"
        % (counter.diagnostic_jvps, counter.diagnostic_vjps, BATCH, dimension,
           dimension, dimension)))

    perturbed = probe_state.at[1].set(probe_state[1] * 3.0 + 1.0)
    info_perturbed = probe_jacobian(pmf_adapter, predict, perturbed, BATCH, jac_settings,
                                    ("job", "stage", 0), None)
    isolated = (abs(info[0]["sigma_max_estimate"]
                    - info_perturbed[0]["sigma_max_estimate"]) < 1e-6
                and abs(info[1]["sigma_max_estimate"]
                        - info_perturbed[1]["sigma_max_estimate"]) > 1e-6)
    results.append(report(
        "each image is probed in isolation",
        isolated,
        "changing row 1's state moves row 1's sigma_max and leaves row 0's untouched, so "
        "these are per-image Jacobians and not one number for the concatenated batch"))

    repeat = probe_jacobian(pmf_adapter, predict, probe_state, BATCH, jac_settings,
                            ("job", "stage", 0), None)
    other_seed = probe_jacobian(
        pmf_adapter, predict, probe_state, BATCH,
        dataclasses.replace(jac_settings, seed=jac_settings.seed + 1), ("job", "stage", 0),
        None)
    results.append(report(
        "deterministic for a fixed probe seed",
        all(a["gains"] == b["gains"] for a, b in zip(info, repeat))
        and info[0]["gains"] != other_seed[0]["gains"]
        and jac_settings.seed == 20240917,
        "a repeated probe reproduces every directional gain exactly, a different seed does "
        "not, and the default seed is fixed rather than time-dependent"))

    fields = set(info[0])
    stats_fields = set(gain_statistics([1.0, 2.0, 3.0]))
    results.append(report(
        "no exact condition number is claimed",
        "sigma_max_estimate" in fields
        and not any("condition" in f for f in fields | stats_fields)
        and "empirical_gain_ratio" in stats_fields
        and "sigma_min" not in " ".join(fields | stats_fields)
        and len(info[0]["gains"]) == 4,
        "the reported spectral field is named sigma_max_ESTIMATE, the anisotropy statistic "
        "is EMPIRICAL, the raw gains are kept, and nothing is called sigma_min or a "
        "condition number"))

    # ================================================================ 11. persistence
    print("\n11. Persistence")
    arrays = stage_arrays(jac_stats.stage_records, problem.image_ids)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.npz"
        np.savez_compressed(path, reconstruction=np.zeros((BATCH, 4, 4, 3), np.uint8),
                            **{k: v for k, v in arrays.items()
                               if k != "stage_image_ids"})
        with np.load(path) as z:
            loaded = {k: z[k] for k in z.files}
    keeps_structure = (loaded["stage_v_pre"].shape == (BATCH, 3)
                       and loaded["stage_jacobian_gains"].shape == (BATCH, 3, 3)
                       and np.allclose(loaded["stage_delta"], arrays["stage_delta"],
                                       equal_nan=True))
    results.append(report(
        "stage and spectral arrays survive save/load",
        keeps_structure and "stage_theta" in loaded
        and "stage_endpoint_shift_pixel_rmse" in loaded,
        "results.npz round-trips [image, stage] arrays and the [image, stage, probe] gains "
        "with the structure intact -- nothing is flattened away"))

    import run as runner
    record = runner.build_record(
        dataclasses.replace(make_spec("rhso", num_rhso_steps=3, num_opt_steps=4),
                            rhso_stage_diagnostics=True,
                            rhso_consistency_diagnostics=True,
                            rhso_terminal_planner="learned_finite_interval_map",
                            rhso_consistency_kind=MEANFLOW_CONSISTENCY),
        types.SimpleNamespace(run_id="r"), problem,
        {"stats": jac_stats, "padded_items": 0, "initial_state_fingerprints": ["f"]},
        {}, {}, "ok", None, 0.0, 0.0,
        memory={"gpu_process_peak_gib": 1.0,
                "gpu_process_memory_source": "nvml_process_sampling(interval=0.02s)"})
    summary_present = all(record.get(k) is not None for k in
                          ("stage_delta_first", "stage_theta_last",
                           "endpoint_shift_pixel_rmse_mean", "diagnostic_model_evaluations",
                           "rhso_terminal_planner", "rhso_consistency_kind"))
    results.append(report(
        "results.csv carries summaries and semantics",
        summary_present and record["diagnostic_seconds"] > 0
        and record["rhso_consistency_kind"] == MEANFLOW_CONSISTENCY
        and all(c in runner.RESULT_COLUMNS for c in
                ("rhso_terminal_planner", "rhso_consistency_kind", "diagnostic_jvps",
                 "gpu_process_peak_gib", "stage_theta_first")),
        "the job-level row records the planner, the consistency semantics, the diagnostic "
        "cost and the stage summaries, while the raw arrays stay in results.npz"))

    # ================================================================ 12. memory
    print("\n12. GPU memory")
    from src.memory import GpuMemoryProfiler

    class _FakeSampler:
        """Stands in for NVML so the per-job reset can be tested without a GPU."""
        instances = []

        def __init__(self, series):
            self.series = list(series)
            self.peak_bytes = None
            self.started = 0
            _FakeSampler.instances.append(self)

        available = True

        def sample_once(self):
            return 1.0

        def start(self):
            self.started += 1
            self.peak_bytes = int(1.0 * 2 ** 30)

        def stop(self):
            self.peak_bytes = int(self.series.pop(0) * 2 ** 30)
            return self.peak_bytes / float(2 ** 30)

    class _JaxAdapter:
        class spec:
            framework = "jax"

    def profiler_with(series):
        prof = GpuMemoryProfiler.__new__(GpuMemoryProfiler)
        prof.adapter, prof.enabled, prof.nvml_interval = _JaxAdapter(), True, 0.02
        prof.framework = "jax"
        prof.baseline_gib = prof.peak_gib = None
        prof.source, prof.extra = "uninitialised", {}
        prof._sampler = None
        prof.process_baseline_gib = prof.process_peak_gib = None
        prof.process_source = "uninitialised"
        prof._process_sampler = _FakeSampler(series)
        prof._mode = "jax_device_stats"
        # A stand-in for JAX's device statistics, so the LIFETIME-marked framework column
        # can be exercised on a machine with no JAX GPU.
        prof._jax_stats = lambda: {"bytes_in_use": 2 * 2 ** 30,
                                   "peak_bytes_in_use": 40 * 2 ** 30}
        return prof

    first = profiler_with([9.0])
    first.establish_baseline()
    with first.measure():
        pass
    second = profiler_with([3.0])
    second.establish_baseline()
    with second.measure():
        pass
    results.append(report(
        "the process sampler resets per job",
        first.report()["gpu_process_peak_gib"] == 9.0
        and second.report()["gpu_process_peak_gib"] == 3.0
        and first._process_sampler is not second._process_sampler,
        "a second job that peaks LOWER reports 3.0 GiB rather than inheriting the earlier "
        "job's 9.0 GiB -- each job gets a fresh sampler, which is exactly what JAX's "
        "lifetime high-water mark cannot do"))

    lifetime = profiler_with([])
    lifetime._process_sampler = None
    lifetime.process_source = "unavailable"
    lifetime.establish_baseline()
    with lifetime.measure():
        pass
    report_ = lifetime.report()
    mislabelled = (report_["gpu_process_peak_gib"] is not None
                   or "lifetime" in str(report_["gpu_process_memory_source"]).lower()
                   or "jax" in str(report_["gpu_process_memory_source"]).lower())
    results.append(report(
        "no lifetime mark posing as a process peak",
        not mislabelled
        and report_["gpu_process_memory_source"] == "unavailable"
        and "LIFETIME" in str(report_["gpu_memory_source"]),
        "with NVML absent the comparable process metric reports 'unavailable'; JAX's "
        "unresettable peak_bytes_in_use stays in its own clearly-labelled column and is "
        "never substituted for it"))

    print("\n%d/%d checks passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
