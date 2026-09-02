"""Theory-validation diagnostics for RHSO -- measurements ABOUT the method, never part of it.

Everything in this module is an EXTRA measurement.  None of it enters the optimised
objective, none of it changes a gradient, none of it changes the executed trajectory, and
none of it is counted as algorithmic compute:

    * the optimiser still differentiates exactly `rhso.rhso_total_objective`, built from the
      batch-summed `problems.make_phi`.  The per-image numbers recorded here come from
      `problems.make_phi_per_sample`, which is the same fidelity written per batch element
      (`sum_b per_sample == make_phi`, asserted by a test) and is never differentiated;
    * every model call made here increments the DIAGNOSTIC counters
      (`diagnostic_model_evals`, `diagnostic_network_forwards`, `diagnostic_jvps`,
      `diagnostic_vjps`) and the diagnostic clock, and the RHSO loops subtract the
      diagnostic seconds from the reported reconstruction runtime;
    * diagnostics can be switched off entirely, in which case the loops behave exactly as
      they did before this module existed.

Three families of diagnostic
----------------------------
1. STAGE OPTIMISATION EFFICIENCY  (`rhso_stage_diagnostics`)
   Per real image and per outer stage k:

       V_pre    the terminal fidelity at the state ENTERING the stage, before any Adam
                update -- i.e. Phi_b( P(x_{s_k}, s_k) )
       V_post   the terminal fidelity at the FINAL q, AFTER all M Adam updates.  It is
                evaluated explicitly, because the loop's last recorded inner loss was
                computed BEFORE the last update and therefore belongs to a state that no
                longer exists
       Delta_k  = V_pre - V_post,  and Delta_k / M
       theta_k  = 1 - V_post / V_pre   (NaN when V_pre is not positive and finite)

2. EXECUTION / REPLANNING CONSISTENCY  (`rhso_consistency_diagnostics`)
   At the optimised state q_k*, the terminal planner predicts an endpoint

       p_k = P(q_k*, s_k)

   The algorithm then executes ONE interval to x_{k+1} and, at the next stage, predicts

       r_k = P(x_{k+1}, s_{k+1})

   `r_k` is exactly the next stage's V_pre prediction, so the pair costs no extra model
   call beyond the two per stage that (1) already needs.  What ||p_k - r_k|| MEANS differs
   by family and is labelled accordingly by `consistency_kind`:

       pMF / iMF   `meanflow_semigroup_defect`: T_theta is a learned FAMILY of
                   finite-interval transport maps, so p_k and r_k are two members of that
                   family applied to the same trajectory, and their disagreement is a
                   defect of the learned semigroup T(.; s_k -> 1) vs
                   T(.; s_{k+1} -> 1) o T(.; s_k -> s_{k+1}).
       JiT-direct  `terminal_prediction_inconsistency`: x_hat_1 is an endpoint PREDICTOR,
                   not a learned family of finite-interval maps.  The shift is what the
                   predictor does after the state moves; calling it a semigroup defect
                   would assert a structure JiT does not have.

3. TERMINAL-MAP ANISOTROPY  (`rhso_jacobian_diagnostics`)
   A matrix-free, per-image probe of the terminal planner's Jacobian at the state entering
   the stage.  See `probe_jacobian` for exactly what is and is not claimed.

4. GRADIENT-ALIGNED ENDPOINT AUTHORITY  (`rhso_gradient_authority_diagnostics`)
   The same Jacobian, but contracted against the direction the TASK actually asks for
   rather than against random directions:

       A_k = ||J_k^T grad_{x_1} Phi|| / ||grad_{x_1} Phi||

   (3) asks how anisotropic the terminal map is; (4) asks how much authority the state at
   s_k has over the measurement objective specifically.  They answer different questions
   and are switched on independently, so a run can carry both and compare them on exactly
   the same images and stages.  See `probe_gradient_authority`.

Batch handling
--------------
Every diagnostic is PER IMAGE.  A chunk of four images produces four rows per stage, not
one.  Rows that are repeat padding added to fill a fixed compiled batch shape
(`InverseProblem.padded_rows`) are dropped: they are duplicates of the last real image and
are not independent samples.  Padding still contributes nothing to any gradient, exactly as
before, because the objective is summed over the batch.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .utils import MEANFLOW, STANDARD_FLOW, derive_rng

# `problems` imports `config`, and `config` imports the diagnostic defaults declared below,
# so the per-sample fidelity is imported where it is used rather than at module scope.

# What ||p_k - r_k|| means, per family.  Stored next to the numbers so no downstream table
# can mislabel a JiT endpoint shift as a learned-semigroup defect.
MEANFLOW_CONSISTENCY = "meanflow_semigroup_defect"
DIRECT_PREDICTOR_CONSISTENCY = "terminal_prediction_inconsistency"
SUFFIX_CONSISTENCY = "suffix_integration_endpoint_shift"

CONSISTENCY_MEANING: Dict[str, str] = {
    MEANFLOW_CONSISTENCY: (
        "||T(q*; s_k -> 1) - T(x_{k+1}; s_{k+1} -> 1)||: the disagreement between two "
        "members of the LEARNED FAMILY of finite-interval transport maps applied to the "
        "same trajectory, i.e. a defect of the learned semigroup."),
    DIRECT_PREDICTOR_CONSISTENCY: (
        "||x_hat_1(q*, s_k) - x_hat_1(x_{k+1}, s_{k+1})||: how much the model's DIRECT "
        "clean-endpoint PREDICTION moves once the optimised state is actually executed "
        "one interval. x_hat_1 is an endpoint predictor, NOT a learned family of "
        "finite-interval transport maps, so this is not a semigroup defect."),
    SUFFIX_CONSISTENCY: (
        "||G(q*; s_k -> 1) - G(x_{k+1}; s_{k+1} -> 1)||: the endpoint shift of the "
        "integrated remaining suffix, which mixes model error with solver discretisation "
        "error and is reported only for the legacy suffix planner."),
}

# Deliberately fixed and documented, so a diagnostic run is reproducible without the user
# having to remember to set a seed.
DEFAULT_JACOBIAN_SEED = 20240917
DEFAULT_JACOBIAN_PROBES = 8
DEFAULT_JACOBIAN_POWER_ITERATIONS = 8

# Below this endpoint-gradient norm the authority RATIO is undefined and is reported as
# NaN.  Dividing by an arbitrary epsilon instead would manufacture a finite-looking
# authority out of a measurement that carries no direction at all, so it is deliberately
# not done; both norms are still recorded, so the reader can see WHY the ratio is missing.
GRADIENT_AUTHORITY_MIN_ENDPOINT_NORM = 1e-12


# =====================================================================================
# Settings
# =====================================================================================
@dataclass(frozen=True)
class DiagnosticSettings:
    """Which diagnostics run, and with what probe budget."""
    stage: bool = False
    consistency: bool = False
    jacobian: bool = False
    probes: int = DEFAULT_JACOBIAN_PROBES
    power_iterations: int = DEFAULT_JACOBIAN_POWER_ITERATIONS
    seed: int = DEFAULT_JACOBIAN_SEED
    # Appended AFTER `seed` on purpose: any existing positional construction of the six
    # fields above keeps its meaning.
    gradient_authority: bool = False

    @property
    def any_enabled(self) -> bool:
        return bool(self.stage or self.consistency or self.jacobian
                    or self.gradient_authority)

    def to_metadata(self) -> Dict[str, Any]:
        return {"stage_diagnostics": bool(self.stage),
                "consistency_diagnostics": bool(self.consistency),
                "jacobian_diagnostics": bool(self.jacobian),
                "jacobian_probes": int(self.probes),
                "jacobian_power_iterations": int(self.power_iterations),
                "jacobian_seed": int(self.seed),
                "gradient_authority_diagnostics": bool(self.gradient_authority),
                "gradient_authority_min_endpoint_norm":
                    float(GRADIENT_AUTHORITY_MIN_ENDPOINT_NORM)}


def settings_from_spec(spec) -> DiagnosticSettings:
    """Resolve a job's diagnostic settings, defaulting to EVERYTHING OFF.

    Read through `getattr` so a spec produced before these fields existed -- or a
    lightweight stand-in built by a test -- resolves to the pre-change behaviour.
    """
    def flag(name: str) -> bool:
        return bool(getattr(spec, name, False) or False)

    def number(name: str, fallback: int) -> int:
        value = getattr(spec, name, None)
        return int(fallback if value is None else value)

    consistency = flag("rhso_consistency_diagnostics")
    authority = flag("rhso_gradient_authority_diagnostics")
    return DiagnosticSettings(
        # A consistency measurement is a comparison of two stage-terminal predictions, so
        # it implies the stage bookkeeping; asking for it alone is not an error.  The
        # gradient-authority probe implies it for the same reason: its numbers live on the
        # per-(image, stage) rows the stage bookkeeping creates.
        stage=flag("rhso_stage_diagnostics") or consistency or authority,
        consistency=consistency,
        gradient_authority=authority,
        jacobian=flag("rhso_jacobian_diagnostics"),
        probes=number("rhso_jacobian_probes", DEFAULT_JACOBIAN_PROBES),
        power_iterations=number("rhso_jacobian_power_iters",
                                DEFAULT_JACOBIAN_POWER_ITERATIONS),
        seed=number("rhso_jacobian_seed", DEFAULT_JACOBIAN_SEED))


def consistency_kind(dynamics_family: str, terminal_mode: str) -> str:
    """The label for what ||p_k - r_k|| measures, for this family and planner."""
    if dynamics_family == MEANFLOW:
        return MEANFLOW_CONSISTENCY
    if terminal_mode == "suffix":
        return SUFFIX_CONSISTENCY
    return DIRECT_PREDICTOR_CONSISTENCY


# =====================================================================================
# Small framework-neutral helpers
# =====================================================================================
def _to_numpy(adapter, state) -> np.ndarray:
    """Native state -> float64 NumPy, OUTSIDE any autograd graph."""
    return np.asarray(adapter.to_numpy(state), np.float64)


def _pixels_numpy(adapter, state) -> np.ndarray:
    """Canonical BHWC pixels as NumPy, UNCLIPPED.

    `to_pixels(differentiable=False)` clips to [-1, 1] for display and metrics; clipping
    would hide part of an endpoint discrepancy, so the differentiable path is used and the
    graph is left behind by converting immediately.
    """
    pixels = adapter.to_pixels(state, differentiable=True)
    if adapter.spec.framework == "jax":
        import jax
        return np.asarray(jax.device_get(jax.lax.stop_gradient(pixels)), np.float64)
    return np.asarray(pixels.detach().float().cpu().numpy(), np.float64)


def _rows_l2(a: np.ndarray) -> np.ndarray:
    """Per-sample Euclidean norm of a batched array."""
    flat = a.reshape(int(a.shape[0]), -1)
    return np.sqrt((flat * flat).sum(axis=1))


def _rows_rmse(a: np.ndarray) -> np.ndarray:
    flat = a.reshape(int(a.shape[0]), -1)
    return np.sqrt((flat * flat).mean(axis=1))


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, np.float64)
    good = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0)
    out[good] = numerator[good] / denominator[good]
    return out


def gain_statistics(gains: Sequence[float]) -> Dict[str, float]:
    """Robust summaries of a set of directional gains g_i = ||J r_i|| / ||r_i||.

    `empirical_gain_ratio` is p95/p05 OF THE SAMPLED DIRECTIONS.  It is a lower bound on
    the true condition number and is deliberately NOT called one: random directions in high
    dimension concentrate, so the smallest sampled gain is almost never near sigma_min.
    """
    arr = np.asarray([g for g in gains], np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {k: float("nan") for k in
                ("gain_min", "gain_max", "gain_mean", "gain_std", "gain_p05", "gain_p50",
                 "gain_p95", "empirical_gain_ratio", "empirical_log_anisotropy")}
    p05, p50, p95 = (float(np.percentile(finite, q)) for q in (5.0, 50.0, 95.0))
    ratio = p95 / p05 if p05 > 0.0 else float("nan")
    return {
        "gain_min": float(finite.min()), "gain_max": float(finite.max()),
        "gain_mean": float(finite.mean()), "gain_std": float(finite.std()),
        "gain_p05": p05, "gain_p50": p50, "gain_p95": p95,
        "empirical_gain_ratio": float(ratio),
        "empirical_log_anisotropy": float(math.log(ratio)) if ratio and ratio > 0
        else float("nan"),
    }


# =====================================================================================
# Matrix-free Jacobian probing
# =====================================================================================
def probe_jacobian(adapter, predict, x, real_rows: int, settings: DiagnosticSettings,
                   seed_parts: Sequence[Any], counters: Optional[Any] = None
                   ) -> List[Dict[str, Any]]:
    """Matrix-free anisotropy probe of the terminal planner's Jacobian, PER IMAGE.

        pMF / iMF   J_k = D_x T_theta(x_{s_k}; s_k -> 1)
        JiT-direct  J_k = D_x x_hat_1(x_{s_k}, s_k)

    The full Jacobian is NEVER formed: with a 256x256x3 state it has ~4.3e10 entries.  Only
    JVPs, VJPs and their composition J^T J are used, and both frameworks reuse ONE
    linearisation for every probe of a stage.

    Batch isolation.  The theory is about the Jacobian of ONE image.  The models act
    independently across batch elements, so a tangent supported on row b produces an output
    supported on row b, and every probe here uses such a masked tangent/cotangent and reads
    only row b.  A condition number of the concatenated four-image batch is never computed.
    Padded rows are skipped.

    What is reported, and what is NOT claimed
    -----------------------------------------
        sigma_max_estimate        power iteration on J^T J -- the DOMINANT singular scale,
                                  and only that.  Power iteration says nothing about
                                  sigma_min, so no condition number is derived from it.
        gains                     the raw directional gains ||J r_i|| / ||r_i|| for
                                  `probes` deterministic random unit directions
        empirical_gain_ratio      p95/p05 of those gains: an EMPIRICAL, sample-based
                                  anisotropy statistic and a lower bound on the true
                                  condition number, never an exact one.

    Returns one dict per REAL image, in batch order.
    """
    framework = adapter.spec.framework
    probes = max(1, int(settings.probes))
    iterations = max(1, int(settings.power_iterations))
    try:
        if framework == "jax":
            return _probe_jax(predict, x, real_rows, probes, iterations, settings.seed,
                              seed_parts, counters)
        return _probe_torch(predict, x, real_rows, probes, iterations, settings.seed,
                            seed_parts, counters)
    except Exception as exc:                                             # pragma: no cover
        # A diagnostic must never take a reconstruction down with it.
        return [{"sigma_max_estimate": float("nan"), "gains": [float("nan")] * probes,
                 "jacobian_error": "%s: %s" % (type(exc).__name__, exc),
                 **gain_statistics([])} for _ in range(max(0, int(real_rows)))]


def _unit_directions(seed: int, seed_parts: Sequence[Any], row: int, count: int,
                     shape: Sequence[int]) -> List[np.ndarray]:
    """`count` deterministic unit vectors in one sample's state space.

    Seeded by (diagnostic seed, job/stage identity, ROW) so the directions a given image
    receives do not depend on which other images share its batch.
    """
    rng = derive_rng(seed, *seed_parts, "row", int(row))
    out = []
    for _ in range(int(count)):
        r = rng.standard_normal(tuple(int(d) for d in shape)).astype(np.float32)
        norm = float(np.sqrt((r.astype(np.float64) ** 2).sum()))
        out.append(r / norm if norm > 0 else r)
    return out


def _bump(counters, name: str, amount: int = 1) -> None:
    if counters is not None:
        setattr(counters, name, getattr(counters, name, 0) + int(amount))


def _probe_jax(predict, x, real_rows, probes, iterations, seed, seed_parts, counters):
    import jax
    import jax.numpy as jnp

    x = jax.lax.stop_gradient(x)
    fn = lambda z: predict(z)                                            # noqa: E731

    # ONE linearisation and ONE vjp for the whole stage: every probe below is then a linear
    # application, not another forward pass through the network.
    y, jvp_fn = jax.linearize(fn, x)
    _y2, vjp_fn = jax.vjp(fn, x)
    _bump(counters, "diagnostic_model_evals", 2)

    sample_shape = tuple(int(d) for d in x.shape[1:])
    results = []
    for b in range(int(real_rows)):
        directions = _unit_directions(seed, seed_parts, b, probes + 1, sample_shape)
        gains = []
        for r in directions[:probes]:
            tangent = jnp.zeros_like(x).at[b].set(jnp.asarray(r))
            out = jvp_fn(tangent)
            _bump(counters, "diagnostic_jvps")
            gains.append(float(jnp.linalg.norm(jnp.asarray(out)[b].reshape(-1))))

        v = jnp.asarray(directions[probes])
        rayleigh = float("nan")
        for _ in range(iterations):
            tangent = jnp.zeros_like(x).at[b].set(v)
            jv = jvp_fn(tangent)
            _bump(counters, "diagnostic_jvps")
            cotangent = jnp.zeros_like(jnp.asarray(jv)).at[b].set(jnp.asarray(jv)[b])
            (jtjv,) = vjp_fn(cotangent)
            _bump(counters, "diagnostic_vjps")
            w = jnp.asarray(jtjv)[b]
            rayleigh = float(jnp.vdot(v, w).real / jnp.vdot(v, v).real)
            norm = float(jnp.linalg.norm(w.reshape(-1)))
            if not np.isfinite(norm) or norm == 0.0:
                break
            v = w / norm
        sigma_max = float(np.sqrt(rayleigh)) if np.isfinite(rayleigh) and rayleigh >= 0 \
            else float("nan")
        results.append({"sigma_max_estimate": sigma_max, "gains": [float(g) for g in gains],
                        **gain_statistics(gains)})
    return results


def _probe_torch(predict, x, real_rows, probes, iterations, seed, seed_parts, counters):
    import torch

    base = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        y = predict(base)
        _bump(counters, "diagnostic_model_evals", 1)
        # The double-backward trick gives a REUSABLE jvp and vjp from a single forward pass
        # and works on every torch version (no dependency on torch.func):
        #     g(u) = J^T u        -> d g / d u applied to v  =  J v
        u = torch.zeros_like(y, requires_grad=True)
        (g,) = torch.autograd.grad(y, base, grad_outputs=u, create_graph=True,
                                   retain_graph=True)

        def jvp(v):
            _bump(counters, "diagnostic_jvps")
            return torch.autograd.grad(g, u, grad_outputs=v, retain_graph=True)[0]

        def vjp(w):
            _bump(counters, "diagnostic_vjps")
            return torch.autograd.grad(y, base, grad_outputs=w, retain_graph=True)[0]

        sample_shape = tuple(int(d) for d in base.shape[1:])
        results = []
        for b in range(int(real_rows)):
            directions = _unit_directions(seed, seed_parts, b, probes + 1, sample_shape)
            gains = []
            for r in directions[:probes]:
                tangent = torch.zeros_like(base)
                tangent[b] = torch.as_tensor(r, dtype=base.dtype, device=base.device)
                out = jvp(tangent)
                gains.append(float(out[b].reshape(-1).norm()))

            v = torch.as_tensor(directions[probes], dtype=base.dtype, device=base.device)
            rayleigh = float("nan")
            for _ in range(iterations):
                tangent = torch.zeros_like(base)
                tangent[b] = v
                jv = jvp(tangent)
                cotangent = torch.zeros_like(jv)
                cotangent[b] = jv[b]
                jtjv = vjp(cotangent)
                w = jtjv[b]
                denom = float((v * v).sum())
                rayleigh = float((v * w).sum()) / denom if denom > 0 else float("nan")
                norm = float(w.reshape(-1).norm())
                if not np.isfinite(norm) or norm == 0.0:
                    break
                v = w / norm
            sigma_max = float(np.sqrt(rayleigh)) if np.isfinite(rayleigh) and rayleigh >= 0 \
                else float("nan")
            results.append({"sigma_max_estimate": sigma_max,
                            "gains": [float(g_) for g_ in gains],
                            **gain_statistics(gains)})
    return results


# =====================================================================================
# Gradient-aligned endpoint authority
# =====================================================================================
def probe_gradient_authority(adapter, predict, x, real_rows: int, fidelity_of_state,
                             counters: Optional[Any] = None) -> List[Dict[str, Any]]:
    """TASK-ALIGNED authority of the terminal predictor at the state entering a stage.

    Where `probe_jacobian` asks how the terminal map stretches RANDOM directions, this asks
    how much of the direction the measurement actually cares about survives the pull-back:

        g_end_k = grad_{x_1} Phi(x_1; y)   at  x_1 = P_k(x_{t_k})
        J_k     = D_{x_{t_k}} P_k(x_{t_k})
        A_k     = ||J_k^T g_end_k|| / ||g_end_k||

    By the chain rule J_k^T g_end_k is exactly grad_{x_{t_k}} Phi(P_k(x_{t_k}); y), so the
    Jacobian is NEVER materialised: one forward pass through the terminal planner, one
    gradient of the fidelity at its output, and one VJP back through the planner with that
    gradient as the cotangent.

    Three things this deliberately does NOT do:

      * it does not include the state-anchor `mu` penalty.  That penalty is a property of
        the optimisation problem, not of the model's authority over the measurement, and
        its gradient (q - anchor)/d is exactly zero at the state entering a stage anyway,
        so including it would be both wrong and invisible;
      * it does not touch the optimisation.  `x` is detached first and nothing computed
        here is returned to the caller's graph;
      * it does not divide by an epsilon.  See GRADIENT_AUTHORITY_MIN_ENDPOINT_NORM.

    `P_k` is whatever the current planner is -- the caller passes the SAME
    `make_terminal_planner` closure the inner objective differentiates -- so a pMF job
    measures its learned finite-interval map and a JiT-direct job measures x_hat_1.

    Batch isolation is structural rather than probed: the models act independently across
    batch elements and the fidelity is a sum over them, so row b of either gradient is the
    gradient of image b's own fidelity and nothing else.  Padded rows are never read.

    Returns one dict per REAL image, in batch order.
    """
    try:
        if adapter.spec.framework == "jax":
            g_end, g_state = _authority_jax(predict, fidelity_of_state, x, counters)
        else:
            g_end, g_state = _authority_torch(predict, fidelity_of_state, x, counters)
    except Exception as exc:                                             # pragma: no cover
        # A diagnostic must never take a reconstruction down with it.
        return [{"endpoint_fidelity_grad_norm": float("nan"),
                 "state_fidelity_grad_norm": float("nan"),
                 "gradient_aligned_authority": float("nan"),
                 "log_gradient_aligned_authority": float("nan"),
                 "gradient_authority_error": "%s: %s" % (type(exc).__name__, exc)}
                for _ in range(max(0, int(real_rows)))]

    endpoint_norms = _rows_l2(g_end)
    state_norms = _rows_l2(g_state)
    results: List[Dict[str, Any]] = []
    for b in range(max(0, int(real_rows))):
        endpoint = float(endpoint_norms[b])
        state = float(state_norms[b])
        if (np.isfinite(endpoint) and np.isfinite(state)
                and endpoint >= GRADIENT_AUTHORITY_MIN_ENDPOINT_NORM):
            authority = state / endpoint
            log_authority = (float(math.log(authority)) if authority > 0.0
                             else float("nan"))
        else:
            # Documented, not silent: the ratio is undefined, so it is NaN and the two
            # norms that produced it stay on the row.
            authority = float("nan")
            log_authority = float("nan")
        results.append({"endpoint_fidelity_grad_norm": endpoint,
                        "state_fidelity_grad_norm": state,
                        "gradient_aligned_authority": float(authority),
                        "log_gradient_aligned_authority": float(log_authority)})
    return results


def _authority_torch(predict, fidelity_of_state, x, counters):
    import torch

    base = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        terminal = predict(base)                       # ONE terminal-planner evaluation
        _bump(counters, "diagnostic_model_evals", 1)
        # The endpoint gradient is taken with respect to a LEAF copy of the prediction, so
        # it is grad_{x_1} Phi and carries nothing of the planner in it.
        endpoint = terminal.detach().clone().requires_grad_(True)
        (g_end,) = torch.autograd.grad(fidelity_of_state(endpoint), endpoint)
        (g_state,) = torch.autograd.grad(terminal, base, grad_outputs=g_end,
                                         retain_graph=False, allow_unused=True)
        _bump(counters, "diagnostic_vjps", 1)
        if g_state is None:                                              # pragma: no cover
            g_state = torch.zeros_like(base)
    return (np.asarray(g_end.detach().float().cpu().numpy(), np.float64),
            np.asarray(g_state.detach().float().cpu().numpy(), np.float64))


def _authority_jax(predict, fidelity_of_state, x, counters):
    import jax

    x = jax.lax.stop_gradient(x)
    terminal, vjp_fn = jax.vjp(predict, x)             # ONE terminal-planner evaluation
    _bump(counters, "diagnostic_model_evals", 1)
    g_end = jax.grad(fidelity_of_state)(jax.lax.stop_gradient(terminal))
    (g_state,) = vjp_fn(g_end)
    _bump(counters, "diagnostic_vjps", 1)
    return (np.asarray(jax.device_get(g_end), np.float64),
            np.asarray(jax.device_get(g_state), np.float64))


# =====================================================================================
# The per-stage recorder
# =====================================================================================
# Every scalar field of one [image, stage] diagnostic row, and the dtype it is stored with.
STAGE_FLOAT_FIELDS = (
    "s_from", "s_to", "v_pre", "v_post", "delta", "delta_per_step", "theta",
    "anchor_penalty_post", "stage_seconds",
    "endpoint_shift_l2", "endpoint_shift_native_rmse", "endpoint_shift_native_relative",
    "endpoint_shift_pixel_rmse", "fidelity_shift_execution", "next_stage_recovery",
    "next_stage_theta",
    "jacobian_sigma_max", "jacobian_gain_min", "jacobian_gain_max", "jacobian_gain_mean",
    "jacobian_gain_std", "jacobian_gain_p05", "jacobian_gain_p50", "jacobian_gain_p95",
    "jacobian_empirical_gain_ratio", "jacobian_empirical_log_anisotropy",
    # Task-aligned endpoint authority.  APPENDED, never inserted: an .npz written before
    # these existed simply lacks the three arrays, and every reader that asks for a field
    # by name is unaffected.
    "endpoint_fidelity_grad_norm", "state_fidelity_grad_norm",
    "gradient_aligned_authority", "log_gradient_aligned_authority",
)
STAGE_INT_FIELDS = ("stage", "image_row")


class RhsoStageRecorder:
    """Collects [image, stage] diagnostics for ONE batch chunk of one RHSO run.

    The RHSO loops call, in order:

        begin_stage(k, s_k, s_next, x_in)     before the inner optimisation
        end_stage(k, q_star, penalty)         after the FINAL Adam update
        finish()                              after the last stage

    `begin_stage` is what produces V_pre and the endpoint prediction r_{k-1} that closes the
    previous stage's consistency measurement, so a stage's consistency row is filled in only
    once the NEXT stage begins.  The final stage therefore has no consistency measurement
    at all, and its fields stay NaN: there is no subsequent replanning to observe, and
    inventing one would be a fabricated number.
    """

    def __init__(self, adapter, cond, spec, problem, grid, settings: DiagnosticSettings,
                 terminal_mode: str, predict: Callable[[Any, float], Any],
                 counters: Optional[Any] = None, job_id: str = ""):
        self.adapter = adapter
        self.cond = cond
        self.spec = spec
        self.problem = problem
        self.grid = list(grid)
        self.settings = settings
        self.terminal_mode = terminal_mode
        self.predict = predict
        self.counters = counters
        self.job_id = job_id
        self.consistency_kind = consistency_kind(adapter.spec.dynamics_family, terminal_mode)
        self.seconds = 0.0

        batch = int(problem.measurement.shape[0])
        self.real_rows = max(0, batch - int(getattr(problem, "padded_rows", 0) or 0))
        self.image_ids = list(problem.image_ids[:self.real_rows])
        from .problems import make_phi_per_sample                  # local: import cycle
        self._phi_per_sample = make_phi_per_sample(problem, adapter.backend(),
                                                   spec.phi_normalization)
        self._backend = adapter.backend()
        self._rows: Dict[int, List[Dict[str, Any]]] = {}
        self._predicted_endpoint: Optional[np.ndarray] = None       # p_k, native, NumPy
        self._predicted_pixels: Optional[np.ndarray] = None
        self._predicted_stage: Optional[int] = None

    # ---------------------------------------------------------------- plumbing
    def _clock(self):
        """Time and count everything inside as DIAGNOSTIC work, never as algorithm work.

        The adapter's forward counter is zeroed on entry and harvested on exit into
        `diagnostic_network_forwards`, so a diagnostic CFG pair can never be added to the
        job's algorithmic `network_forwards`.  The RHSO loops only ever read that counter
        immediately after their own model calls, so borrowing it here is safe.
        """
        recorder = self

        class _Timer:
            def __enter__(self):
                self.started = time.perf_counter()
                recorder.adapter.reset_counters()
                return self

            def __exit__(self, *exc):
                recorder.seconds += time.perf_counter() - self.started
                _bump(recorder.counters, "diagnostic_network_forwards",
                      int(getattr(recorder.adapter, "forward_counter", 0)))
                recorder.adapter.reset_counters()
                return False

        return _Timer()

    def _fidelity_per_image(self, prediction) -> np.ndarray:
        """Per-image terminal fidelity of an already-computed endpoint prediction.

        `make_phi_per_sample` is the SAME fidelity the optimiser sums; this is never
        differentiated, so no gradient anywhere changes.
        """
        values = self._phi_per_sample(self.adapter.to_pixels(prediction,
                                                             differentiable=True))
        if self.adapter.spec.framework == "jax":
            import jax
            values = np.asarray(jax.device_get(jax.lax.stop_gradient(values)), np.float64)
        else:
            values = np.asarray(values.detach().float().cpu().numpy(), np.float64)
        return values

    def _fidelity_of_state(self, terminal_state):
        """Phi as a differentiable function of the NATIVE terminal state.

        The batch SUM of the per-sample fidelity, which is the same scalar the optimiser
        differentiates (`sum_b make_phi_per_sample == make_phi`, asserted by a test) and
        whose gradient therefore has, in row b, exactly image b's own endpoint gradient.
        The adapter's differentiable `to_pixels` is inside Phi, not inside the terminal
        map, so the Jacobian in the authority ratio is the terminal PREDICTOR's alone.
        """
        pixels = self.adapter.to_pixels(terminal_state, differentiable=True)
        return self._backend.sum(self._phi_per_sample(pixels))

    def _predict_and_score(self, state, s: float):
        with self._clock():
            prediction = self._detached(self.predict(state, float(s)))
            _bump(self.counters, "diagnostic_model_evals", 1)
            fidelity = self._fidelity_per_image(prediction)
            native = _to_numpy(self.adapter, prediction)
            pixels = _pixels_numpy(self.adapter, prediction)
        return fidelity, native, pixels

    def _detached(self, state):
        if self.adapter.spec.framework == "jax":
            import jax
            return jax.lax.stop_gradient(state)
        return state.detach()

    def _row(self, stage: int, row: int) -> Dict[str, Any]:
        rows = self._rows.setdefault(int(stage), [])
        while len(rows) <= row:
            entry = {name: float("nan") for name in STAGE_FLOAT_FIELDS}
            entry.update({"stage": int(stage), "image_row": len(rows),
                          "image_id": self.image_ids[len(rows)]
                          if len(rows) < len(self.image_ids) else "",
                          "gains": []})
            rows.append(entry)
        return rows[row]

    # ---------------------------------------------------------------- the two hooks
    def begin_stage(self, stage: int, s_from: float, s_to: float, x_in) -> None:
        """V_pre at the incoming state, plus the optional Jacobian probe at that state."""
        if not self.settings.stage or self.real_rows <= 0:
            return
        fidelity, native, pixels = self._predict_and_score(x_in, s_from)
        for b in range(self.real_rows):
            row = self._row(stage, b)
            row["s_from"] = float(s_from)
            row["s_to"] = float(s_to)
            row["v_pre"] = float(fidelity[b])

        # Close the PREVIOUS stage's consistency measurement: this prediction is r_{k-1}.
        if (self.settings.consistency and self._predicted_endpoint is not None
                and self._predicted_stage == stage - 1):
            self._record_consistency(stage - 1, native, pixels, fidelity)

        if self.settings.jacobian:
            with self._clock():
                probes = probe_jacobian(
                    self.adapter, lambda z, _s=float(s_from): self.predict(z, _s),
                    x_in, self.real_rows, self.settings,
                    (self.job_id, "stage", int(stage)), self.counters)
            for b, info in enumerate(probes[:self.real_rows]):
                row = self._row(stage, b)
                row["jacobian_sigma_max"] = float(info.get("sigma_max_estimate",
                                                           float("nan")))
                row["gains"] = list(info.get("gains", []))
                for key in ("gain_min", "gain_max", "gain_mean", "gain_std", "gain_p05",
                            "gain_p50", "gain_p95", "empirical_gain_ratio",
                            "empirical_log_anisotropy"):
                    row["jacobian_" + key] = float(info.get(key, float("nan")))
                if info.get("jacobian_error"):
                    row["jacobian_error"] = info["jacobian_error"]

        if self.settings.gradient_authority:
            # Measured at the state ENTERING the stage -- the same place and the same
            # linearisation point as the Jacobian probe above -- because what it reports is
            # the authority AVAILABLE when the stage begins.
            with self._clock():
                authority = probe_gradient_authority(
                    self.adapter, lambda z, _s=float(s_from): self.predict(z, _s),
                    x_in, self.real_rows, self._fidelity_of_state, self.counters)
            for b, info in enumerate(authority[:self.real_rows]):
                row = self._row(stage, b)
                for key in ("endpoint_fidelity_grad_norm", "state_fidelity_grad_norm",
                            "gradient_aligned_authority",
                            "log_gradient_aligned_authority"):
                    row[key] = float(info.get(key, float("nan")))
                if info.get("gradient_authority_error"):
                    row["gradient_authority_error"] = info["gradient_authority_error"]

    def end_stage(self, stage: int, q_star, anchor_penalty: Optional[float] = None) -> None:
        """V_post at the FINAL post-update q, and the endpoint p_k it predicts."""
        if not self.settings.stage or self.real_rows <= 0:
            return
        fidelity, native, pixels = self._predict_and_score(q_star, self.grid[stage])
        m = max(1, int(getattr(self.spec, "num_opt_steps", 1) or 1))
        for b in range(self.real_rows):
            row = self._row(stage, b)
            v_pre = float(row["v_pre"])
            v_post = float(fidelity[b])
            row["v_post"] = v_post
            row["delta"] = v_pre - v_post
            row["delta_per_step"] = (v_pre - v_post) / float(m)
            row["theta"] = (1.0 - v_post / v_pre) if (np.isfinite(v_pre) and v_pre > 0.0) \
                else float("nan")
            if anchor_penalty is not None:
                row["anchor_penalty_post"] = float(anchor_penalty)
        self._predicted_endpoint = native
        self._predicted_pixels = pixels
        self._predicted_stage = int(stage)

    def record_stage_seconds(self, stage: int, seconds: float) -> None:
        """ALGORITHM seconds of one stage (the caller has already removed diagnostic time)."""
        if not self.settings.stage or self.real_rows <= 0:
            return
        for b in range(self.real_rows):
            self._row(stage, b)["stage_seconds"] = float(seconds)

    def _record_consistency(self, stage: int, r_native: np.ndarray, r_pixels: np.ndarray,
                            r_fidelity: np.ndarray) -> None:
        p_native, p_pixels = self._predicted_endpoint, self._predicted_pixels
        diff = p_native[:self.real_rows] - r_native[:self.real_rows]
        pixel_diff = p_pixels[:self.real_rows] - r_pixels[:self.real_rows]
        l2 = _rows_l2(diff)
        native_rmse = _rows_rmse(diff)
        relative = _safe_ratio(l2, _rows_l2(p_native[:self.real_rows]))
        pixel_rmse = _rows_rmse(pixel_diff)
        for b in range(self.real_rows):
            row = self._row(stage, b)
            row["endpoint_shift_l2"] = float(l2[b])
            row["endpoint_shift_native_rmse"] = float(native_rmse[b])
            row["endpoint_shift_native_relative"] = float(relative[b])
            row["endpoint_shift_pixel_rmse"] = float(pixel_rmse[b])
            # Executing one interval and replanning changes the measurement fidelity of the
            # PREDICTED endpoint by this much (positive = the prediction got worse).
            row["fidelity_shift_execution"] = float(r_fidelity[b]) - float(row["v_post"])

    def finish(self) -> List[Dict[str, Any]]:
        """Fill in the next-stage recovery terms and return the finished rows.

        A stage's `next_stage_recovery` is the improvement the NEXT stage's optimisation
        achieved from the state actually reached, i.e. Delta_{k+1}. The final stage has no
        successor, so it keeps NaN rather than a fabricated value -- the same convention as
        its consistency fields.
        """
        stages = sorted(self._rows)
        for stage in stages:
            nxt = self._rows.get(stage + 1)
            if nxt is None:
                continue
            for b, row in enumerate(self._rows[stage]):
                if b < len(nxt):
                    row["next_stage_recovery"] = float(nxt[b]["delta"])
                    row["next_stage_theta"] = float(nxt[b]["theta"])
        out: List[Dict[str, Any]] = []
        for stage in stages:
            out.extend(self._rows[stage])
        return out


# =====================================================================================
# Persistence
# =====================================================================================
def stage_arrays(records: Sequence[Dict[str, Any]], image_ids: Sequence[str]
                 ) -> Dict[str, np.ndarray]:
    """[image, stage] arrays from the flat per-(image, stage) records.

    The structure is deliberately NOT flattened: the whole point of these diagnostics is to
    compare stage k against stage k+1 for the same image.  Missing entries are NaN.
    Directional-gain probes become a third axis, [image, stage, probe].
    """
    if not records:
        return {}
    order = {str(image_id): i for i, image_id in enumerate(image_ids)}
    rows = [r for r in records if str(r.get("image_id", "")) in order]
    if not rows:
        return {}
    n_images = len(image_ids)
    n_stages = max(int(r["stage"]) for r in rows) + 1
    out: Dict[str, np.ndarray] = {}
    for name in STAGE_FLOAT_FIELDS:
        grid = np.full((n_images, n_stages), np.nan, np.float64)
        for r in rows:
            grid[order[str(r["image_id"])], int(r["stage"])] = float(r.get(name, np.nan))
        out["stage_" + name] = grid.astype(np.float32)
    probes = max((len(r.get("gains") or []) for r in rows), default=0)
    if probes:
        gains = np.full((n_images, n_stages, probes), np.nan, np.float32)
        for r in rows:
            values = list(r.get("gains") or [])
            if values:
                gains[order[str(r["image_id"])], int(r["stage"]), :len(values)] = values
        out["stage_jacobian_gains"] = gains
    # A fixed-width unicode array, NOT dtype=object: an object array in an .npz can only
    # be read back with allow_pickle=True, which neither the resume path nor `--aggregate`
    # enables (nor should they).
    out["stage_image_ids"] = np.asarray([str(i) for i in image_ids], dtype=np.str_)
    out["stage_index"] = np.arange(n_stages, dtype=np.int32)
    return out


def stage_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """A few job-level scalars for results.csv.  The raw arrays stay in results.npz."""
    if not records:
        return {}
    by_stage: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        by_stage.setdefault(int(r["stage"]), []).append(r)
    stages = sorted(by_stage)

    def mean(stage: int, field_name: str) -> Optional[float]:
        values = [float(r.get(field_name, np.nan)) for r in by_stage[stage]]
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else None

    def overall(field_name: str) -> Optional[float]:
        values = [float(r.get(field_name, np.nan)) for r in records]
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else None

    first, last = stages[0], stages[-1]
    return {
        "stage_count": len(stages),
        "stage_delta_first": mean(first, "delta"),
        "stage_delta_last": mean(last, "delta"),
        "stage_delta_per_step_first": mean(first, "delta_per_step"),
        "stage_delta_per_step_last": mean(last, "delta_per_step"),
        "stage_theta_first": mean(first, "theta"),
        "stage_theta_last": mean(last, "theta"),
        "stage_v_pre_first": mean(first, "v_pre"),
        "stage_v_post_last": mean(last, "v_post"),
        "endpoint_shift_native_relative_mean": overall("endpoint_shift_native_relative"),
        "endpoint_shift_pixel_rmse_mean": overall("endpoint_shift_pixel_rmse"),
        "fidelity_shift_execution_mean": overall("fidelity_shift_execution"),
        "next_stage_recovery_mean": overall("next_stage_recovery"),
        "jacobian_sigma_max_first": mean(first, "jacobian_sigma_max"),
        "jacobian_sigma_max_last": mean(last, "jacobian_sigma_max"),
        "jacobian_log_anisotropy_first": mean(first, "jacobian_empirical_log_anisotropy"),
        "jacobian_log_anisotropy_last": mean(last, "jacobian_empirical_log_anisotropy"),
        # `mean` / `overall` already drop non-finite entries, so a stage whose ratio is the
        # documented NaN is EXCLUDED from these averages rather than poisoning them; a
        # summary over no finite value at all stays None.
        "gradient_aligned_authority_first": mean(first, "gradient_aligned_authority"),
        "gradient_aligned_authority_last": mean(last, "gradient_aligned_authority"),
        "gradient_aligned_authority_mean": overall("gradient_aligned_authority"),
        "endpoint_fidelity_grad_norm_mean": overall("endpoint_fidelity_grad_norm"),
        "state_fidelity_grad_norm_mean": overall("state_fidelity_grad_norm"),
    }
