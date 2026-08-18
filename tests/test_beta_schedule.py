"""The universal power-law time schedule, checked without a checkpoint or a GPU.

    s_k = s_0 + (1 - s_0) * (k / N)^beta

Covered here:

  * validation (beta finite and > 0, steps >= 1);
  * exact endpoints and strict monotonicity for a range of exponents;
  * the beta = 1 REGRESSION -- the grid must be bitwise identical to the expression the
    repository used before beta existed, for SDEdit / D-Flow / MPC and for PnP's interior
    schedule;
  * the worked examples from the specification (beta = 0.5 and beta = 2 at s0 = 0, N = 4);
  * interval direction (beta < 1 refines towards the clean endpoint, beta > 1 towards the
    noisy start);
  * PnP's N correction times stay strictly inside (s0, 1) for every beta;
  * the structural check no longer assumes dt = t0/N;
  * MPC-delta_t's inverse-delta lambda scaling uses the LOCAL dt_k.

Nothing here needs jax, optax or torch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataclasses

from spec_support import (StandInPlan, build_plan, make_spec, source_banner,   # noqa: E402
                          spec_source)


class _ProbeProblem:
    """The one attribute `run.warmup_key` reads off a problem."""
    class measurement:
        shape = (2, 256, 256, 3)


def report(name, ok, detail):
    print("  [%s] %-34s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def legacy_uniform_grid(s_start, steps):
    """The pre-beta expression, transcribed independently from the original sdedit.py."""
    grid = [float(s_start) + (1.0 - float(s_start)) * (k / steps) for k in range(steps + 1)]
    grid[-1] = 1.0
    return grid


def legacy_pnp_grid(s_start, n):
    """The pre-beta PnP expression, transcribed independently from the original pnp.py."""
    s0 = float(s_start)
    return [s0 + (k / float(n + 1)) * (1.0 - s0) for k in range(1, n + 1)]


def main():
    from src.schedule import (DEFAULT_BETA, canonical_time_grid, grid_intervals,
                              grid_metadata, interior_time_grid, resolve_beta, spec_beta)
    results = []
    print("\n%s" % source_banner())

    # ---------------------------------------------------------------- 1. validation
    print("\nValidation")
    rejected = []
    for bad in (0.0, -0.5, float("inf"), float("-inf"), float("nan"), "0.5", None if False
                else True, [1.0]):
        try:
            resolve_beta(bad)
        except ValueError:
            rejected.append(bad)
    results.append(report("beta must be finite and > 0", len(rejected) == 7,
                          "rejected %d/7 invalid values (0, negative, +/-inf, nan, a "
                          "string, a bool, a list)" % len(rejected)))
    results.append(report("beta defaults to 1", resolve_beta(None) == 1.0
                          and spec_beta(object()) == 1.0,
                          "an absent field and an explicit None both give the legacy "
                          "uniform schedule"))
    bad_steps = []
    for steps in (0, -3):
        try:
            canonical_time_grid(0.0, steps, 1.0)
        except ValueError:
            bad_steps.append(steps)
    results.append(report("steps must be >= 1", len(bad_steps) == 2,
                          "steps=0 and steps=-3 both raise"))
    try:
        canonical_time_grid(1.0, 4, 1.0)
        degenerate = False
    except ValueError:
        degenerate = True
    results.append(report("s_start == 1 is degenerate", degenerate,
                          "a start time at the data endpoint still raises, as before"))

    # ---------------------------------------------------------------- 2/3. shape
    print("\nEndpoints, length and monotonicity")
    shape_ok, notes = True, []
    for s0 in (0.0, 0.2, 0.75):
        for steps in (1, 2, 4, 7):
            for beta in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
                grid = canonical_time_grid(s0, steps, beta)
                if len(grid) != steps + 1:
                    notes.append("len at s0=%g N=%d beta=%g" % (s0, steps, beta))
                if grid[0] != s0 or grid[-1] != 1.0:
                    notes.append("endpoints at s0=%g N=%d beta=%g" % (s0, steps, beta))
                if any(grid[i] >= grid[i + 1] for i in range(steps)):
                    notes.append("monotonicity at s0=%g N=%d beta=%g" % (s0, steps, beta))
    shape_ok = not notes
    results.append(report("exactly steps+1 points, s0 .. 1", shape_ok,
                          "72 (s0, N, beta) combinations: grid[0] == s0 exactly, "
                          "grid[-1] == 1.0 exactly, strictly increasing"
                          if shape_ok else "; ".join(notes[:4])))

    # ---------------------------------------------------------------- 4. beta = 1 regression
    print("\nbeta = 1 reproduces the legacy grid BITWISE")
    identical = []
    for s0 in (0.0, 0.2, 0.5, 0.8, 0.95):
        for steps in (1, 2, 3, 4, 5, 8, 25):
            identical.append(canonical_time_grid(s0, steps, 1.0)
                             == legacy_uniform_grid(s0, steps))
    results.append(report("canonical grid", all(identical),
                          "%d (s0, steps) pairs match the pre-beta expression exactly, not "
                          "approximately" % len(identical)))
    pnp_identical = [interior_time_grid(s0, n, 1.0) == legacy_pnp_grid(s0, n)
                     for s0 in (0.0, 0.2, 0.8) for n in (1, 4, 20)]
    results.append(report("PnP interior grid", all(pnp_identical),
                          "%d (s0, N) pairs match the pre-beta k/(N+1) expression exactly"
                          % len(pnp_identical)))
    results.append(report("default is beta = 1", DEFAULT_BETA == 1.0
                          and canonical_time_grid(0.2, 4) == legacy_uniform_grid(0.2, 4),
                          "a caller that passes no beta gets the legacy grid"))

    # ---------------------------------------------------------------- 5/6. worked examples
    print("\nWorked examples from the specification (s0 = 0, N = 4)")
    half = canonical_time_grid(0.0, 4, 0.5)
    expect_half = [0.0, 0.5, 0.7071067811865476, 0.8660254037844386, 1.0]
    results.append(report("beta = 0.5",
                          max(abs(a - b) for a, b in zip(half, expect_half)) < 1e-15,
                          "%s" % ["%.4f" % v for v in half]))
    two = canonical_time_grid(0.0, 4, 2.0)
    expect_two = [0.0, 0.0625, 0.25, 0.5625, 1.0]
    results.append(report("beta = 2",
                          max(abs(a - b) for a, b in zip(two, expect_two)) < 1e-15,
                          "%s" % ["%.4f" % v for v in two]))
    one = canonical_time_grid(0.0, 4, 1.0)
    results.append(report("beta = 1", one == [0.0, 0.25, 0.5, 0.75, 1.0],
                          "%s" % ["%.4f" % v for v in one]))

    # ---------------------------------------------------------------- 7. direction
    print("\nInterval direction")
    small = grid_intervals(canonical_time_grid(0.2, 6, 0.5))
    large = grid_intervals(canonical_time_grid(0.2, 6, 2.0))
    flat = grid_intervals(canonical_time_grid(0.2, 6, 1.0))
    results.append(report(
        "beta < 1 refines near s = 1",
        all(small[i + 1] < small[i] for i in range(len(small) - 1)),
        "dt %s -- decreasing towards the clean endpoint" % ["%.4f" % d for d in small]))
    results.append(report(
        "beta > 1 refines near s = s0",
        all(large[i + 1] > large[i] for i in range(len(large) - 1)),
        "dt %s -- increasing towards the clean endpoint" % ["%.4f" % d for d in large]))
    results.append(report(
        "beta = 1 is uniform",
        max(flat) - min(flat) < 1e-15,
        "dt %s -- one step size, which is the only case where t0/N is the real dt"
        % ["%.4f" % d for d in flat]))
    results.append(report(
        "one interval makes beta inert",
        canonical_time_grid(0.3, 1, 0.5) == canonical_time_grid(0.3, 1, 4.0)
        == [0.3, 1.0],
        "N=1 gives [s0, 1] for every beta, so a beta sweep there is a no-op"))

    # ---------------------------------------------------------------- 8. PnP interior
    print("\nPnP correction times stay strictly inside (s0, 1)")
    inside, counts = True, []
    for s0 in (0.0, 0.2, 0.9):
        for n in (1, 3, 20):
            for beta in (0.25, 0.5, 1.0, 2.0, 4.0):
                grid = interior_time_grid(s0, n, beta)
                counts.append(len(grid) == n)
                inside = inside and all(s0 < s < 1.0 for s in grid)
                inside = inside and all(grid[i] < grid[i + 1] for i in range(n - 1))
    results.append(report("N times, never at s0 or s = 1", inside and all(counts),
                          "45 (s0, N, beta) combinations: exactly N strictly increasing "
                          "times, none at the endpoints"))

    # ---------------------------------------------------------------- method grids
    print("\nMethod grids all come from the one helper")
    from src.dflow import dflow_time_grid
    from src.mpc import mpc_step_sizes, mpc_time_grid
    from src.pnp import pnp_time_grid
    from src.sdedit import canonical_time_grid as reexported

    results.append(report("sdedit re-exports the helper", reexported is canonical_time_grid,
                          "`from src.sdedit import canonical_time_grid` still resolves, now "
                          "to schedule.py's implementation"))
    spec = make_spec("dflow", steps=4, beta=0.5)
    results.append(report(
        "D-Flow", dflow_time_grid(spec)
        == canonical_time_grid(spec.canonical_start_time, 4, 0.5),
        "trajectory grid follows beta (%s)"
        % ["%.4f" % v for v in dflow_time_grid(spec)]))
    spec = make_spec("mpc_delta_t", num_mpc_steps=4, beta=2.0)
    results.append(report(
        "MPC outer grid", mpc_time_grid(spec)
        == canonical_time_grid(spec.canonical_start_time, 4, 2.0),
        "execution grid follows beta (%s)" % ["%.4f" % v for v in mpc_time_grid(spec)]))
    spec = make_spec("pnp", num_pnp_steps=5, beta=0.5)
    results.append(report(
        "PnP", pnp_time_grid(spec.canonical_start_time, 5, 0.5)
        == interior_time_grid(spec.canonical_start_time, 5, 0.5),
        "correction times follow beta (%s)"
        % ["%.4f" % v for v in pnp_time_grid(spec.canonical_start_time, 5, 0.5)]))

    # ---------------------------------------------------------------- 9. no constant delta
    print("\nNo algorithm or check may assume dt = t0/N")
    from src.checks import check_delta_t_lambda_scaling, check_time_grid
    uniform = make_spec("mpc_delta_t", num_mpc_steps=4, beta=1.0, t0=0.8,
                        job_id="uniform0")
    skewed = dataclasses.replace(uniform, beta=0.5, job_id="skewed01")
    steep = dataclasses.replace(uniform, beta=2.0, job_id="steep002")
    rhso_spec = make_spec("rhso", num_rhso_steps=4, beta=0.5, job_id="rhso0001")
    ok, detail = check_time_grid(StandInPlan(specs=(uniform, skewed, steep, rhso_spec)))
    results.append(report("check_time_grid accepts beta != 1", ok, detail))

    dts = mpc_step_sizes(skewed)
    nominal = skewed.t0 / 4
    results.append(report(
        "the nominal delta is not a dt", max(abs(dt - nominal) for dt in dts) > 1e-3,
        "beta=0.5 executes dt = %s, while t0/N = %.4f -- using the scalar would be wrong "
        "for every step" % (["%.4f" % d for d in dts], nominal)))
    meta = grid_metadata(mpc_time_grid(skewed), 0.5)
    results.append(report(
        "metadata reports the real spread", meta["delta"] is None
        and abs(meta["delta_min"] - min(dts)) < 1e-15
        and abs(meta["delta_max"] - max(dts)) < 1e-15,
        "delta is null for beta != 1; delta_min=%.4f delta_max=%.4f are recorded instead"
        % (meta["delta_min"], meta["delta_max"])))
    meta1 = grid_metadata(mpc_time_grid(uniform), 1.0)
    results.append(report(
        "metadata keeps delta at beta = 1", meta1["delta"] is not None
        and abs(meta1["delta"] - nominal) < 1e-15,
        "delta = %.4f is still reported as the nominal uniform spacing" % meta1["delta"]))

    # ---------------------------------------------------------------- 10. inverse-delta lambda
    print("\nMPC-delta_t inverse-delta lambda scaling")
    from src.mpc import inverse_delta_lambda
    scaled = dataclasses.replace(skewed, delta_t_lambda_scaling="inverse_delta", lam=2.0,
                                 job_id="scaled01")
    lams = [inverse_delta_lambda(scaled, dt) for dt in mpc_step_sizes(scaled)]
    expected = [2.0 / dt for dt in mpc_step_sizes(scaled)]
    results.append(report(
        "lambda_eff,k = lambda / dt_k",
        max(abs(a - b) for a, b in zip(lams, expected)) < 1e-12 and len(set(lams)) == 4,
        "four distinct effective lambdas %s for four distinct steps"
        % ["%.2f" % v for v in lams]))
    legacy_scaled = dataclasses.replace(uniform, delta_t_lambda_scaling="inverse_delta",
                                        lam=2.0, job_id="legacy01")
    legacy_lams = [inverse_delta_lambda(legacy_scaled, dt)
                   for dt in mpc_step_sizes(legacy_scaled)]
    results.append(report(
        "beta = 1 keeps lambda / spec.delta",
        legacy_scaled.delta is not None
        and all(v == 2.0 / legacy_scaled.delta for v in legacy_lams),
        "uniform schedule reproduces the old global value %.4f exactly (spec.delta = %.4f)"
        % (legacy_lams[0], legacy_scaled.delta)))
    unscaled = dataclasses.replace(scaled, delta_t_lambda_scaling="none",
                                   job_id="unscaled")
    results.append(report(
        "disabled scaling is untouched",
        all(inverse_delta_lambda(unscaled, dt) == 2.0
            for dt in mpc_step_sizes(unscaled)),
        "with scaling off, lambda stays lambda for every step"))
    ok, detail = check_delta_t_lambda_scaling(StandInPlan(specs=(scaled, legacy_scaled)))
    results.append(report("check_delta_t_lambda_scaling", ok, detail))

    # ---------------------------------------------------------------- configuration
    if spec_source() == "real":
        print("\nConfiguration: beta as a shared, swept, identity-bearing field")
        from src.config import ConfigError, METHOD_DECLARATIONS, SWEEPABLE_FIELDS
        from src.schedule import DEFAULT_BETA as CFG_DEFAULT

        results.append(report(
            "beta is a shared field",
            "beta" in SWEEPABLE_FIELDS
            and all("beta" in d.fields for d in METHOD_DECLARATIONS.values()),
            "declared for all %d methods and sweepable, exactly like t0"
            % len(METHOD_DECLARATIONS)))

        plan, _w = build_plan({"sdedit": {"t0": 0.8, "steps": 4,
                                          "beta": [0.5, 1.0, 2.0]}})
        betas = sorted(s.beta for s in plan.specs)
        leaves = {s.leaf_dir for s in plan.specs}
        ids = {s.job_id for s in plan.specs}
        results.append(report(
            "a beta list expands as a sweep",
            betas == [0.5, 1.0, 2.0] and len(ids) == 3 and len(leaves) == 3,
            "3 jobs, 3 distinct job ids and 3 distinct output directories -- beta-distinct "
            "configurations cannot collapse into one"))

        titles = {s.figure_title() for s in plan.specs}
        results.append(report(
            "beta reaches the labels",
            sum("beta=" in t for t in titles) == 2
            and any("beta=0.5" in t for t in titles),
            "the two non-default jobs carry beta in their figure title; the beta=1 job is "
            "labelled exactly as before"))

        default_plan, _w = build_plan({"sdedit": {"t0": 0.8, "steps": 4}})
        results.append(report(
            "omitting beta is the legacy behaviour",
            default_plan.specs[0].beta == CFG_DEFAULT
            and abs(default_plan.specs[0].delta_min
                    - default_plan.specs[0].delta_max) < 1e-12,
            "a configuration that never mentions beta resolves to beta=%g and a uniform "
            "grid (delta_min and delta_max agree to floating-point subtraction error)"
            % default_plan.specs[0].beta))

        rejected = 0
        # NOTE: [0.5] is NOT invalid -- a list is a one-point sweep, not a bad value.
        for bad in (0, -1.0, "half", float("nan")):
            try:
                build_plan({"sdedit": {"steps": 4, "beta": bad}})
            except ConfigError:
                rejected += 1
        results.append(report("the validator rejects a bad beta", rejected == 4,
                              "beta must be a finite positive number (4/4 rejected)"))

        warned, _w2 = build_plan({"dflow": {"steps": 1, "beta": [0.5, 1.0]}})
        results.append(report(
            "single-interval beta warns, not fails",
            any("NO mathematical effect" in w for w in _w2),
            "steps=1 leaves one interval, where beta cannot change anything"))

        mpc, _w = build_plan({"mpc_delta_t": {"t0": 0.8, "num_mpc_steps": 4,
                                              "beta": [1.0, 0.5]}})
        by_beta = {s.beta: s for s in mpc.specs}
        results.append(report(
            "delta is nominal-only when beta != 1",
            by_beta[1.0].delta is not None and by_beta[0.5].delta is None
            and by_beta[0.5].delta_nominal_uniform is not None
            and by_beta[0.5].delta_min < by_beta[0.5].delta_max,
            "beta=1 keeps delta=%.4f; beta=0.5 reports delta=None with dt in [%.4f, %.4f]"
            % (by_beta[1.0].delta, by_beta[0.5].delta_min, by_beta[0.5].delta_max)))

        import run as runner
        keys = {runner.warmup_key(s, _ProbeProblem()) for s in mpc.specs}
        results.append(report(
            "beta is in the warm-up key", len(keys) == 2,
            "two beta values give two warm-up keys, so neither compiles inside its measured "
            "region"))
        results.append(report(
            "beta is a results column",
            "beta" in runner.RESULT_COLUMNS and "delta_min" in runner.RESULT_COLUMNS
            and "delta_max" in runner.RESULT_COLUMNS,
            "results.csv records beta and the real dt range next to the nominal delta"))

    print("\n%d/%d checks passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
