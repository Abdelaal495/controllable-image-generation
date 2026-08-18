"""D-Flow reconstruction: optimise the starting generative state through the flow.

Based on Ben-Hamu, Puny, Gat, Karrer, Singer and Lipman, "D-Flow: Differentiating through
Flows for Controlled Generation" (ICML 2024).  The principle kept here is exactly theirs:

    min_q  L( G(q) ),     G = the generative map,  q = the state the generation starts from

with the gradient taken THROUGH the generative trajectory.  That is the defining
difference from PnP, which never differentiates the network, and it is why this method is
the expensive end of the benchmark.

What is published, and what this file adapts
--------------------------------------------
PUBLISHED (the principle):
  * optimising the starting state by differentiating a terminal measurement loss through a
    differentiable solve of the generative ODE.

ADAPTED for this repository (documented, never presented as the paper's algorithm):
  * INITIALISATION.  q^(0) is the repository's SHARED z_t0.  The paper initialises either
    from the source distribution or from its variance-preserving "blend" of noise with the
    backward solve of the observation; neither is used here, because every method in this
    benchmark must start from the same state.
  * INTERMEDIATE START (t0 < 1).  At t0 = 1 (s0 = 0) this is ordinary source-point
    optimisation.  At t0 < 1 the optimised variable is an INTERMEDIATE flow state, which is
    a research extension of this repository -- an interesting one, since it is what makes
    D-Flow directly comparable with SDEdit and MPC at the same t0, but not the published
    setup.  `initialization_kind` and `canonical_start_time` are recorded for every job.
  * OPTIMISER.  Fixed-budget Adam.  The paper uses LBFGS with line search and stops at a
    task-dependent target PSNR; a ground-truth-based stopping rule has no place in a
    benchmark, and LBFGS is deliberately out of scope for this iteration.
  * NO REGULARISATION.  The paper's chi^d source-norm regulariser, source NLL and gradient
    clipping are not implemented, so this is the R == 0 ("implicit regularisation") case.
  * FIDELITY.  The per-measurement normalisation of section 3 (see problems.make_phi), not
    negative PSNR and not 1/(2 sigma^2) ||Hx - y||^2.
  * MEANFLOW (research extension).  A MeanFlow model has no ODE to solve, so the trajectory
    is a composition of learned finite-interval transitions.  With steps = 1 it is the
    single map T_theta(q ; s0 -> 1), giving a differentiable D-Flow trajectory that costs
    ONE network evaluation forward and one backward.  A one-step Euler integration of an
    ordinary Flow-Matching velocity field is NOT the same object: it is a coarse
    approximation of a trajectory, whereas the MeanFlow transition is trained to be the
    finite-interval map.

Trajectory code lives HERE, not in sdedit.py: SDEdit stays a pure-inference sampler with
no graph-retention switch, and this module owns its own differentiable integration.
"""

from __future__ import annotations

import time
from typing import Any, List, Tuple

from .config import SOLVER_STAGE_EVALUATIONS
from .models.base import Conditioning, MeanFlowAdapter, ModelAdapter, StandardFlowAdapter
from .problems import make_phi, phi_log_scale
from .sdedit import ReconstructionStats, _is_finite, canonical_time_grid
from .utils import MEANFLOW, STANDARD_FLOW


def dflow_time_grid(spec) -> List[float]:
    """The steps + 1 canonical times s0 ... 1 the trajectory is discretised over."""
    return canonical_time_grid(spec.canonical_start_time, int(spec.steps))


def trajectory_evaluations(adapter: ModelAdapter, spec) -> int:
    """Model evaluations in ONE forward trajectory -- not the same as `steps`.

    A standard-flow step costs the solver's stage count (euler 1, heun 2, rk4 4), minus the
    one stage JiT's official sampler drops on its final Heun step.  A MeanFlow step costs
    exactly one learned transition.
    """
    steps = int(spec.steps)
    if adapter.spec.dynamics_family != STANDARD_FLOW:
        return steps
    solver = spec.solver or "euler"
    evals = steps * SOLVER_STAGE_EVALUATIONS[solver]
    if solver == "heun" and adapter.spec.euler_final_step_for_heun:
        evals -= 1
    return evals


def _require_adam(spec) -> None:
    if (spec.optimizer or "adam") != "adam":
        raise ValueError(
            "D-Flow supports only Adam in this iteration, got optimizer=%r. The published "
            "method uses LBFGS with line search, which is deliberately not implemented "
            "here." % (spec.optimizer,))


# =====================================================================================
# Standard flow (JiT, SiT): differentiable fixed-step integration
# =====================================================================================
def _integrate_flow(adapter: StandardFlowAdapter, cond: Conditioning, q, spec,
                    grid: List[float]):
    """G(q ; s0 -> 1) by fixed-step integration, kept INSIDE the autograd graph.

    No torch.no_grad, no detach, no NumPy or PIL conversion anywhere on this path: the
    whole chain q -> trajectory -> to_pixels -> A(x) -> fidelity has to stay differentiable
    with respect to q.  Model parameters remain frozen; only q is optimised.
    """
    solver = spec.solver or "euler"
    if solver not in SOLVER_STAGE_EVALUATIONS:
        raise ValueError("Unknown solver %r" % solver)
    final_euler = bool(adapter.spec.euler_final_step_for_heun)
    steps = len(grid) - 1

    x = q
    for k in range(steps):
        s, s_next = grid[k], grid[k + 1]
        dt = s_next - s
        step_solver = solver
        if solver == "heun" and final_euler and k == steps - 1:
            step_solver = "euler"

        v1 = adapter.velocity(x, s, cond)
        if step_solver == "euler":
            x = x + dt * v1
        elif step_solver == "heun":
            v2 = adapter.velocity(x + dt * v1, s_next, cond)
            x = x + 0.5 * dt * (v1 + v2)
        else:                                                            # rk4
            s_mid = 0.5 * (s + s_next)
            v2 = adapter.velocity(x + 0.5 * dt * v1, s_mid, cond)
            v3 = adapter.velocity(x + 0.5 * dt * v2, s_mid, cond)
            v4 = adapter.velocity(x + dt * v3, s_next, cond)
            x = x + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
    return x


def flow_dflow(adapter: StandardFlowAdapter, cond: Conditioning, x0, problem,
               spec) -> Tuple[Any, ReconstructionStats]:
    """D-Flow for an instantaneous-velocity model, with Adam on q."""
    import torch

    _require_adam(spec)
    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    log_scale = phi_log_scale(problem, spec.phi_normalization)
    grid = dflow_time_grid(spec)
    per_objective = trajectory_evaluations(adapter, spec)

    stats = ReconstructionStats()
    started = time.perf_counter()

    # The optimised variable IS the shared initial state; nothing else is trainable.
    q = x0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=float(spec.lr))

    for iteration in range(int(spec.num_opt_steps)):
        opt.zero_grad(set_to_none=True)
        adapter.reset_counters()
        x_terminal = _integrate_flow(adapter, cond, q, spec, grid)
        loss = phi(adapter.to_pixels(x_terminal, differentiable=True))
        loss.backward()
        opt.step()

        stats.control_iterations += 1
        stats.optimizer_iterations += 1
        stats.objective_evals += 1
        stats.data_gradient_evals += 1
        stats.model_evals_total += per_objective
        stats.model_evals_planning += per_objective
        stats.network_forwards += adapter.forward_counter
        # Every model evaluation on the trajectory is traversed by the backward pass.
        stats.backprops_through_model += per_objective
        if spec.record_loss_history:
            stats.loss_history.append(float(loss.detach()) * log_scale)
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(
                "D-Flow objective became non-finite at optimizer iteration %d/%d."
                % (iteration + 1, int(spec.num_opt_steps)))

    # The reconstruction must correspond to the FINAL optimised q, so one more trajectory
    # is evaluated after the last Adam update.  It is counted like any other.
    adapter.reset_counters()
    with torch.no_grad():
        x_final = _integrate_flow(adapter, cond, q.detach(), spec, grid)
    stats.model_evals_total += per_objective
    stats.network_forwards += adapter.forward_counter
    stats.objective_evals += 0            # a reconstruction, not an objective evaluation

    adapter.block(x_final)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x_final)
    return x_final, stats


# =====================================================================================
# MeanFlow (pMF, iMF): composition of learned transitions -- RESEARCH EXTENSION
# =====================================================================================
def _transition_trajectory(adapter: MeanFlowAdapter, cond: Conditioning, q,
                           grid: List[float]):
    """G(q ; s0 -> 1) as a composition of learned finite-interval transitions.

    With a single interval this is exactly T_theta(q ; s0 -> 1).  No ODE is constructed and
    no instantaneous velocity is inferred: a MeanFlow model does not have one.
    """
    x = q
    for i in range(len(grid) - 1):
        x = adapter.transition(x, grid[i], grid[i + 1], cond)
    return x


def meanflow_dflow(adapter: MeanFlowAdapter, cond: Conditioning, x0, problem,
                   spec) -> Tuple[Any, ReconstructionStats]:
    """D-Flow through learned MeanFlow transitions.  steps = 1 is the one-step experiment."""
    import jax
    import jax.numpy as jnp
    import optax

    _require_adam(spec)
    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    log_scale = phi_log_scale(problem, spec.phi_normalization)
    grid = dflow_time_grid(spec)
    per_objective = trajectory_evaluations(adapter, spec)

    def terminal(q):
        return _transition_trajectory(adapter, cond, q, grid)

    def loss_fn(q):
        return phi(adapter.to_pixels(terminal(q), differentiable=True))

    value_and_grad = jax.value_and_grad(loss_fn)

    stats = ReconstructionStats()
    started = time.perf_counter()

    q = x0
    tx = optax.adam(float(spec.lr))
    opt_state = tx.init(q)

    for iteration in range(int(spec.num_opt_steps)):
        adapter.reset_counters()
        loss, grads = value_and_grad(q)
        updates, opt_state = tx.update(grads, opt_state, q)
        q = optax.apply_updates(q, updates)

        stats.control_iterations += 1
        stats.optimizer_iterations += 1
        stats.objective_evals += 1
        stats.data_gradient_evals += 1
        stats.model_evals_total += per_objective
        stats.model_evals_planning += per_objective
        stats.network_forwards += adapter.forward_counter
        stats.backprops_through_model += per_objective
        if spec.record_loss_history:
            stats.loss_history.append(float(loss) * log_scale)
        if not bool(jnp.isfinite(loss)):
            raise FloatingPointError(
                "D-Flow objective became non-finite at optimizer iteration %d/%d."
                % (iteration + 1, int(spec.num_opt_steps)))

    adapter.reset_counters()
    x_final = jax.lax.stop_gradient(terminal(q))
    stats.model_evals_total += per_objective
    stats.network_forwards += adapter.forward_counter

    x_final = adapter.block(x_final)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x_final)
    return x_final, stats


def dflow_reconstruct(adapter: ModelAdapter, cond: Conditioning, x0, problem,
                      spec) -> Tuple[Any, ReconstructionStats]:
    """Family dispatch, for callers that hold an adapter but not a registry."""
    if spec.dynamics_family == STANDARD_FLOW:
        return flow_dflow(adapter, cond, x0, problem, spec)
    if spec.dynamics_family == MEANFLOW:
        return meanflow_dflow(adapter, cond, x0, problem, spec)
    raise ValueError("No D-Flow strategy for dynamics family %r" % spec.dynamics_family)
