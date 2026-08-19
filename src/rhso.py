"""RHSO: Receding-Horizon State Optimization.

A reconstruction strategy of this repository (not a published method).  At every scheduled
time it optimises THE CURRENT GENERATIVE STATE against its own terminal consequence, then
executes a single scheduled interval and starts over:

    at s_k, from the current state x_k:

        anchor = stop_gradient(x_k)               <- FIXED for the whole stage
        q^(0)  = x_k
        for j = 0 .. M-1:
            fidelity = Phi( to_pixels( G_{s_k -> 1}( q^(j) ), differentiable=True ) )
            R        = sum_b 1/(2 d_b) || q^(j)_b - anchor_b ||^2
            loss     = fidelity + mu * R
            q^(j+1)  = AdamUpdate( q^(j), d loss / d q )
        q* = q^(M)

        x_{k+1} = G_{s_k -> s_{k+1}}( q* )        <- only ONE interval is executed

    then repeat at s_{k+1}, from a FRESH optimisation problem AND a FRESH anchor taken
    from the state actually reached.

The outer times are the repository's universal power-law schedule with N = num_rhso_steps:

    s_k = s_0 + (1 - s_0) * (k / N)^beta

What RHSO is NOT
----------------
* NOT D-Flow.  D-Flow optimises ONE state (the trajectory's starting point) against the
  terminal reconstruction and then executes the WHOLE planned trajectory.  RHSO throws the
  optimisation problem away after committing a single interval and re-optimises from the
  state it actually reached.  The closed-loop replanning is the method.
* NOT MPC.  MPC optimises an explicit control variable u that is added to the dynamics, and
  pays for it with a control penalty ||u||^2 and a lambda trade-off.  RHSO has no u, no
  control cost, no lambda and no K: the decision variable IS the state.  `mu` below is NOT
  MPC's lambda -- it weights a displacement of the STATE from where the trajectory actually
  was, not the magnitude of an added control signal, and the two are neither numerically
  nor conceptually interchangeable.

State-anchor regularisation (optional; mu = 0 by default)
---------------------------------------------------------
Pure terminal fidelity lets q drift arbitrarily far from x_k, and an over-optimised q can
score well on the measurement while leaving the generative manifold -- the observed failure
mode is measurement consistency continuing to improve while PSNR / SSIM / LPIPS degrade.
`mu > 0` adds a trust-region-like penalty on that displacement:

    R(q, x_k) = sum_b  1/(2 d_b)  || q_b - x_{k,b} ||_2^2

    (squared difference -> MEAN over the non-batch state dimensions -> times 1/2
     -> SUMMED over the batch)

The batch sum matches the repository's per-measurement fidelity convention, so one sample's
gradient never depends on the batch size, and the 1/d normalisation means a useful mu does
not scale with the state dimensionality (a pixel model and a latent model are comparable).

The anchor is the state at the START of the stage, detached, and it does NOT move as q
moves.  It is not the trajectory's origin x_0, and it is not a moving average.  Nothing
about the schedule, the planner, the execution step, the Adam reset or the cost accounting
changes: the penalty contains no model evaluation.  mu is an EXPERIMENTAL extension
motivated by observed over-optimisation, not a theoretical requirement.

Family differences (both are implemented; neither is a special case of the other)
--------------------------------------------------------------------------------
STANDARD FLOW (JiT, SiT)
    The terminal planner has to integrate: G_{s_k -> 1} is a differentiable fixed-step
    solve over the REMAINING SUFFIX of the same outer grid, [s_k, s_{k+1}, ..., 1], with
    the repository's existing solver semantics (Euler / Heun / RK4 and the adapter's own
    final-step policy).  No second planning-resolution hyperparameter is introduced, and
    the planning cost therefore shrinks as k grows.

MEANFLOW (pMF, iMF)
    The terminal planner is ONE direct learned finite-interval transition
    T_theta(q ; s_k -> 1).  No ODE is constructed, no instantaneous velocity is inferred,
    and the remaining outer intervals are NOT composed for the objective.  One forward and
    one backward network evaluation per inner iteration, at any k -- which is what makes
    MeanFlow models computationally well suited to a receding-horizon state optimiser.

Differentiability contract
--------------------------
The inner objective path

    q -> terminal planner -> native terminal state -> to_pixels(differentiable=True)
      -> A(x) -> fidelity

contains no NumPy conversion, no detach, no PIL and no stop-gradient.  For latent models
(iMF, SiT) the gradient continues through the VAE decoder; there is no separate pixel/latent
mathematics, only the adapter abstraction.  Model parameters stay frozen -- only q moves.

Execution is inference only: no graph is carried into the next outer stage, and the Adam
state is rebuilt from scratch at every stage because the optimisation problem itself
changes when the terminal transport map changes.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

from .dflow import flow_step, integrate_flow, solver_evaluations
from .models.base import Conditioning, MeanFlowAdapter, ModelAdapter, StandardFlowAdapter
from .problems import make_phi, phi_log_scale
from .schedule import canonical_time_grid, spec_beta
from .sdedit import ReconstructionStats, _is_finite
from .utils import MEANFLOW, STANDARD_FLOW


def rhso_time_grid(spec) -> List[float]:
    """The N + 1 outer times s_0 ... 1 of one RHSO run (N = num_rhso_steps)."""
    return canonical_time_grid(spec.canonical_start_time, int(_field(spec, "num_rhso_steps")),
                               spec_beta(spec))


def rhso_planning_grid(grid: List[float], stage: int) -> List[float]:
    """The times ONE standard-flow inner objective integrates over at outer stage k.

    The REMAINING SUFFIX of the same outer grid, [s_k, s_{k+1}, ..., 1] -- deliberately not
    a second, independently-resolved planning discretisation.  Exposed as a function so the
    property can be asserted without running a solver.
    """
    stage = int(stage)
    if not 0 <= stage <= len(grid) - 2:
        raise ValueError("stage %d is outside the RHSO grid (%d intervals)"
                         % (stage, len(grid) - 1))
    return list(grid[stage:])


def _field(spec, name: str):
    """Read a required RHSO field, with a message that names the missing wiring."""
    try:
        value = getattr(spec, name)
    except AttributeError:
        raise AttributeError(
            "RHSO needs the resolved job field %r. Declare it in src/config.py "
            "(METHOD_DECLARATIONS['rhso'], SWEEPABLE_FIELDS, BUILTIN_DEFAULTS and JobSpec) "
            "-- see docs/schedule_and_rhso.md." % name)
    if value is None:
        raise ValueError("RHSO field %r is None; it has no meaningful default." % name)
    return value


def rhso_mu(spec) -> float:
    """The state-anchor regularisation weight of one resolved job.

    Absent or None -> 0.0, i.e. the vanilla objective, so a spec predating `mu` (and any
    stand-in used by a test) behaves exactly as it did before the field existed.
    """
    mu = getattr(spec, "mu", None)
    if mu is None:
        return 0.0
    mu = float(mu)
    if not math.isfinite(mu) or mu < 0.0:
        raise ValueError(
            "RHSO mu must be a finite, non-negative state-regularisation weight, got %r. "
            "mu = 0 disables the penalty." % (mu,))
    return mu


def state_anchor_penalty(B, q, anchor):
    """R(q, x_k) = sum_b (1 / 2 d_b) * ||q_b - x_{k,b}||^2 -- one implementation, both families.

    Written against the `Backend` abstraction and using only `-`, `*`, `reshape` and a
    global `sum`, all of which behave identically on NumPy arrays, torch tensors and JAX
    arrays.  Torch RHSO and MeanFlow RHSO therefore share this exact normalisation by
    construction rather than by two parallel transcriptions of a formula.

        squared difference -> MEAN over the non-batch dimensions (the 1/d factor)
                           -> times 1/2
                           -> SUM over the batch

    Summing over the batch (rather than averaging) keeps each sample's gradient independent
    of the batch size, exactly as `problems.make_phi` does for the per-measurement
    fidelity.  Dividing by d keeps a useful mu independent of the state dimensionality.
    """
    batch = int(q.shape[0])
    diff = (q - anchor).reshape(batch, -1)
    dims = int(diff.shape[1])
    return 0.5 * B.sum(diff * diff) / float(dims)


def rhso_total_objective(B, fidelity, q, anchor, mu: float):
    """(total, R) for one inner iteration.  `total` is the scalar Adam differentiates.

    At mu = 0 the returned total is the fidelity object ITSELF, so the differentiated graph
    is identical to the one this method used before the penalty existed -- backward
    compatibility is structural, not a matter of a zero multiplier cancelling later.  R is
    still computed and reported, because "how far did q drift" is worth logging even when
    nothing penalises it.
    """
    penalty = state_anchor_penalty(B, q, anchor)
    if mu == 0.0:
        return fidelity, penalty
    return fidelity + mu * penalty, penalty


def _require_adam(spec) -> None:
    """Adam is the only optimiser RHSO advertises, and the only one it implements."""
    if (spec.optimizer or "adam") != "adam":
        raise ValueError(
            "RHSO supports only Adam in this iteration, got optimizer=%r. The state "
            "optimisation is a fresh problem at every outer stage, so a stateful "
            "alternative would need its own reset semantics before it could be honest."
            % (spec.optimizer,))


def rhso_planning_evaluations(adapter: ModelAdapter, spec, stage: int,
                              grid: List[float]) -> int:
    """Model evaluations in ONE inner objective at outer stage `stage`.

    MeanFlow: exactly one direct transition s_k -> 1, at every stage.
    Standard flow: the solver's stage count over the remaining suffix [s_k, ..., 1], which
    shrinks by one interval per stage; the adapter's final-Euler policy applies because the
    suffix does land on the data endpoint.
    """
    if adapter.spec.dynamics_family != STANDARD_FLOW:
        return 1
    return solver_evaluations(adapter, spec.solver or "euler",
                              len(rhso_planning_grid(grid, stage)) - 1,
                              reaches_data_endpoint=True)


def rhso_execution_evaluations(adapter: ModelAdapter, spec, stage: int,
                               grid: List[float]) -> int:
    """Model evaluations to execute the single interval s_k -> s_{k+1}."""
    if adapter.spec.dynamics_family != STANDARD_FLOW:
        return 1
    last = int(stage) == len(grid) - 2
    return solver_evaluations(adapter, spec.solver or "euler", 1,
                              reaches_data_endpoint=last)


def rhso_cost_estimate(values: Dict[str, Any], dynamics_family: str,
                       euler_final_step_for_heun: bool = False) -> Dict[str, int]:
    """Exact planner cost model for one RHSO job -- for `--dry-run`.

    Kept here, next to the loop it describes, so the two cannot drift apart; `config.py`'s
    `_estimate_cost` is expected to call it (see docs/schedule_and_rhso.md).  It takes the
    resolved values dict rather than a spec because the planner has no adapter, and takes
    the final-step policy as a plain flag for the same reason.

        MeanFlow        N*M planning evaluations, N*M backprops, N*(M+1) evaluations total
        Standard flow   sum over stages of the solver's stage count on the remaining
                        suffix, times M, plus one executed interval per stage
    """
    from .config import SOLVER_STAGE_EVALUATIONS                     # local: avoid a cycle

    n = int(values.get("num_rhso_steps") or 0)
    m = int(values.get("num_opt_steps") or 0)
    if n < 1 or m < 1:
        return {"model_evals": 0, "planning_evals": 0, "backprops": 0,
                "objective_evals": 0, "optimizer_iterations": 0}

    if dynamics_family != STANDARD_FLOW:
        planning = n * m
        return {"model_evals": n * (m + 1), "planning_evals": planning,
                "backprops": planning, "objective_evals": planning,
                "optimizer_iterations": planning}

    solver = values.get("solver") or "euler"
    stages = SOLVER_STAGE_EVALUATIONS[solver]
    drops_a_stage = solver == "heun" and bool(euler_final_step_for_heun)
    planning = execution = 0
    for k in range(n):
        suffix = n - k                                    # intervals from s_k to 1
        planning += m * (suffix * stages - (1 if drops_a_stage else 0))
        execution += 1 * stages - (1 if (drops_a_stage and k == n - 1) else 0)
    return {"model_evals": planning + execution, "planning_evals": planning,
            "backprops": planning, "objective_evals": n * m,
            "optimizer_iterations": n * m}


# =====================================================================================
# Standard flow (JiT, SiT): differentiable integration over the remaining suffix
# =====================================================================================
def flow_rhso(adapter: StandardFlowAdapter, cond: Conditioning, x0, problem,
              spec) -> Tuple[Any, ReconstructionStats]:
    """RHSO for an instantaneous-velocity model, with Adam on the current state."""
    import torch

    _require_adam(spec)
    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    log_scale = phi_log_scale(problem, spec.phi_normalization)
    grid = rhso_time_grid(spec)
    N = int(_field(spec, "num_rhso_steps"))
    M = int(_field(spec, "num_opt_steps"))
    mu = rhso_mu(spec)

    stats = ReconstructionStats()
    started = time.perf_counter()
    x = x0.detach().clone()

    for k in range(N):
        s_k, s_next = grid[k], grid[k + 1]
        suffix = rhso_planning_grid(grid, k)     # [s_k, s_{k+1}, ..., 1]
        per_objective = rhso_planning_evaluations(adapter, spec, k, grid)

        # A FRESH optimisation variable AND a fresh Adam state: the terminal transport map
        # has changed, so moments accumulated at s_{k-1} describe a different problem.
        # The anchor is a SEPARATE, detached copy of the same state: it is what q is
        # measured against for the whole stage and it never receives a gradient or an
        # update, so `q` moving cannot drag it along.
        x_anchor = x.detach().clone()
        q = x_anchor.clone().requires_grad_(True)
        opt = torch.optim.Adam([q], lr=float(spec.lr))

        for iteration in range(M):
            opt.zero_grad(set_to_none=True)
            adapter.reset_counters()
            terminal = integrate_flow(adapter, cond, q, spec, suffix,
                                      reaches_data_endpoint=True)
            fidelity = phi(adapter.to_pixels(terminal, differentiable=True))
            loss, penalty = rhso_total_objective(B, fidelity, q, x_anchor, mu)
            loss.backward()
            opt.step()

            stats.control_iterations += 1
            stats.optimizer_iterations += 1
            stats.objective_evals += 1
            stats.data_gradient_evals += 1
            stats.model_evals_planning += per_objective
            stats.model_evals_total += per_objective
            stats.backprops_through_model += per_objective
            stats.network_forwards += adapter.forward_counter
            if spec.record_loss_history:
                # Same per-image scaling for all three, so the recorded numbers still
                # satisfy total = fidelity + mu * R after scaling.
                stats.loss_history.append(float(loss.detach()) * log_scale)
                stats.fidelity_history.append(float(fidelity.detach()) * log_scale)
                stats.state_penalty_history.append(float(penalty.detach()) * log_scale)
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(
                    "RHSO objective became non-finite at outer stage %d/%d (s=%.6f), "
                    "optimizer iteration %d/%d." % (k + 1, N, s_k, iteration + 1, M))

        # ---- execute ONE interval, with the FINAL post-update q; inference only --------
        adapter.reset_counters()
        with torch.no_grad():
            x = flow_step(adapter, cond, q.detach(), s_k, s_next, spec.solver or "euler",
                          apply_final_euler_policy=(bool(adapter.spec.euler_final_step_for_heun)
                                                    and k == N - 1)).detach()
        stats.model_evals_total += rhso_execution_evaluations(adapter, spec, k, grid)
        stats.network_forwards += adapter.forward_counter
        if not _is_finite(adapter, x):
            raise FloatingPointError(
                "%s produced a non-finite state executing RHSO interval %d/%d "
                "(s=%.6f -> %.6f)." % (adapter.spec.name, k + 1, N, s_k, s_next))

    adapter.block(x)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x)
    return x, stats


# =====================================================================================
# MeanFlow (pMF, iMF): ONE direct learned transition to the clean endpoint
# =====================================================================================
def meanflow_rhso(adapter: MeanFlowAdapter, cond: Conditioning, x0, problem,
                  spec) -> Tuple[Any, ReconstructionStats]:
    """RHSO on the learned finite-interval transport.

    The inner objective is a SINGLE transition T(q ; s_k -> 1) -- not a composition of the
    remaining outer intervals and not an ODE -- which is the capability that makes this
    method cheap for a MeanFlow model: the planning cost does not depend on how many
    intervals are left.
    """
    import jax
    import jax.numpy as jnp
    import optax

    _require_adam(spec)
    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    log_scale = phi_log_scale(problem, spec.phi_normalization)
    grid = rhso_time_grid(spec)
    N = int(_field(spec, "num_rhso_steps"))
    M = int(_field(spec, "num_opt_steps"))
    mu = rhso_mu(spec)

    stats = ReconstructionStats()
    started = time.perf_counter()
    x = x0

    for k in range(N):
        s_k, s_next = grid[k], grid[k + 1]

        # The anchor is closed over as a CONSTANT: stop_gradient here means the penalty's
        # gradient flows to q only, and nothing rebinds it during the M inner iterations.
        x_anchor = jax.lax.stop_gradient(x)

        def loss_fn(q, _s=s_k, _anchor=x_anchor):
            terminal = adapter.transition(q, _s, 1.0, cond)      # ONE direct transition
            fidelity = phi(adapter.to_pixels(terminal, differentiable=True))
            total, penalty = rhso_total_objective(B, fidelity, q, _anchor, mu)
            return total, (fidelity, penalty)

        # has_aux keeps the diagnostics out of the differentiated scalar: the gradient is
        # of `total` alone, so mu = 0 reproduces the pre-penalty gradient exactly.
        value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

        # Fresh variable, fresh Adam moments -- see the standard-flow twin.
        q = x_anchor
        tx = optax.adam(float(spec.lr))
        opt_state = tx.init(q)

        for iteration in range(M):
            adapter.reset_counters()
            (loss, (fidelity, penalty)), grads = value_and_grad(q)
            updates, opt_state = tx.update(grads, opt_state, q)
            q = optax.apply_updates(q, updates)

            stats.control_iterations += 1
            stats.optimizer_iterations += 1
            stats.objective_evals += 1
            stats.data_gradient_evals += 1
            stats.model_evals_planning += 1
            stats.model_evals_total += 1
            stats.backprops_through_model += 1
            stats.network_forwards += adapter.forward_counter
            if spec.record_loss_history:
                stats.loss_history.append(float(loss) * log_scale)
                stats.fidelity_history.append(float(fidelity) * log_scale)
                stats.state_penalty_history.append(float(penalty) * log_scale)
            if not bool(jnp.isfinite(loss)):
                raise FloatingPointError(
                    "RHSO objective became non-finite at outer stage %d/%d (s=%.6f), "
                    "optimizer iteration %d/%d." % (k + 1, N, s_k, iteration + 1, M))

        # ---- execute ONE interval with the FINAL q -------------------------------------
        # The last inner objective was evaluated BEFORE the last Adam update, so its
        # terminal prediction belongs to a state that no longer exists; it is never reused.
        adapter.reset_counters()
        x = jax.lax.stop_gradient(adapter.transition(q, s_k, s_next, cond))
        stats.model_evals_total += 1
        stats.network_forwards += adapter.forward_counter
        if not _is_finite(adapter, x):
            raise FloatingPointError(
                "%s produced a non-finite state executing RHSO transition %d/%d "
                "(s=%.6f -> %.6f)." % (adapter.spec.name, k + 1, N, s_k, s_next))

    x = adapter.block(x)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x)
    return x, stats


def rhso_reconstruct(adapter: ModelAdapter, cond: Conditioning, x0, problem,
                     spec) -> Tuple[Any, ReconstructionStats]:
    """Family dispatch, for callers that hold an adapter but not a registry."""
    if spec.dynamics_family == STANDARD_FLOW:
        return flow_rhso(adapter, cond, x0, problem, spec)
    if spec.dynamics_family == MEANFLOW:
        return meanflow_rhso(adapter, cond, x0, problem, spec)
    raise ValueError("No RHSO strategy for dynamics family %r" % spec.dynamics_family)
