"""MPC-Flow reconstruction: receding-horizon control and delta-t control.

Carried over from the MPC-Flow notebook (section 17) as faithfully as possible, for both
dynamics families.  Everything runs on the canonical clock s in [0,1] (s = 0 noise,
s = 1 data).  A job starts at

    s_start = 1 - t0,      delta = (1 - s_start)/N = t0/N,

and performs N receding-horizon replans over [s_start, 1].  With t0 = 1 this is
s_start = 0, delta = 1/N -- the paper's setting.

Standard flows (JiT, SiT), explicit Euler as in the paper
    RHC (Alg. 1/3)   x_{k+1} = x_k + h [ v(x_k, s_k) + u_k ],   h = (1-s)/K
                     J(U)    = sum_k h ||u_k||^2 + lambda Phi(x_K)
                     execute x <- x + delta [ v(x, s) + u_0* ]
    MPC-dt (Alg. 2)  x_{s+dt}(u) = x_s + delta [ v(x_s, s) + u ]
                     V_hat(s, x)  = Phi( x + (1-s) v(x, s) )              <- eq. (9)
                     J(u)         = ||u||^2 + lambda V_hat(s+delta, x_{s+dt}(u))

MeanFlows (pMF, iMF), on the LEARNED finite-interval transport -- not an Euler step
    RHC              x_{k+1} = T(x_k; s_k -> s_{k+1}) + h_k u_k
    K = 1            xbar_1 = T(x_s; s -> 1) is independent of u, so no backprop
    MPC-dt           V_hat(s+dt, x) = Phi( T(x; s+delta -> 1) )

The MeanFlow MPC-delta_t value-to-go is a RESEARCH EXTENSION: the MPC-Flow paper does not
propose or evaluate it.  It is preserved exactly as the notebook implemented it.

The differentiable path

    control -> trajectory -> native terminal state -> to_pixels(differentiable=True)
            -> A(x) -> measurement loss

contains no NumPy conversion, no detach and no PIL.  For latent models the gradient
continues through the VAE decoder.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import numpy as np

from .models.base import Conditioning, MeanFlowAdapter, ModelAdapter, StandardFlowAdapter
from .problems import make_control_cost, make_phi
from .sdedit import ReconstructionStats, canonical_time_grid
from .utils import MEANFLOW, STANDARD_FLOW


def mpc_time_grid(spec) -> list:
    """The N+1 canonical execution times s_start, s_start+delta, ..., 1."""
    return canonical_time_grid(spec.canonical_start_time, int(spec.num_mpc_steps))


def _record_loss(stats: ReconstructionStats, loss, spec) -> None:
    if spec.record_loss_history:
        stats.loss_history.append(float(loss))


# =====================================================================================
# Standard-flow MPC (JiT, SiT)
# =====================================================================================
def _torch_optimizer(params, spec):
    """Both advertised optimisers are really implemented; the validator rejects any other."""
    import torch
    if spec.optimizer == "adam":
        return torch.optim.Adam(list(params), lr=float(spec.lr))
    if spec.optimizer == "sgd":
        return torch.optim.SGD(list(params), lr=float(spec.lr))
    raise ValueError("Unsupported optimizer %r" % spec.optimizer)


def flow_mpc_rhc(adapter: StandardFlowAdapter, cond: Conditioning, x0, problem,
                 spec) -> Tuple[Any, ReconstructionStats]:
    """Receding-horizon control for an ordinary flow model.  K = 1 is Algorithm 3."""
    import torch
    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    ctrl_cost = make_control_cost(B, spec.control_cost_normalization)
    K = int(spec.K or 1)
    grid = mpc_time_grid(spec)

    stats = ReconstructionStats()
    t_start = time.perf_counter()
    x = x0.detach().clone()
    warm = None

    for step in range(int(spec.num_mpc_steps)):
        s, s_next = grid[step], grid[step + 1]
        remaining = 1.0 - s                       # shortened horizon when t0 < 1
        h = remaining / K

        # v_theta at the CURRENT state does not depend on U (it is the k = 0 evaluation and
        # also the one Algorithm 1 line 7 executes with).  Computing it once, outside the
        # optimisation loop and outside the graph, is exact algebra -- not an approximation.
        adapter.reset_counters()
        with torch.no_grad():
            v0 = adapter.velocity(x, s, cond).detach()
        stats.model_evals_total += 1
        stats.network_forwards += adapter.forward_counter

        if spec.warm_start and warm is not None:
            U = [w.detach().clone().requires_grad_(True) for w in warm]
        else:
            U = [torch.zeros_like(x, requires_grad=True) for _ in range(K)]
        opt = _torch_optimizer(U, spec)

        for _it in range(int(spec.n_ctrl)):
            opt.zero_grad(set_to_none=True)
            adapter.reset_counters()
            if K == 1:
                # x' = x + (1-s)(v0 + u); v0 is constant -> no model in the graph at all.
                x_terminal = x + remaining * (v0 + U[0])
                control_term = remaining * ctrl_cost(U[0])
            else:
                xk = x + h * (v0 + U[0])
                for k in range(1, K):
                    vk = adapter.velocity(xk, s + k * h, cond)     # differentiable on purpose
                    xk = xk + h * (vk + U[k])
                x_terminal = xk
                control_term = h * sum(ctrl_cost(u) for u in U)
            loss = control_term + spec.lam * phi(
                adapter.to_pixels(x_terminal, differentiable=True))
            loss.backward()
            if spec.grad_clip:
                torch.nn.utils.clip_grad_norm_(U, float(spec.grad_clip))
            opt.step()
            stats.control_iterations += 1
            stats.model_evals_planning += max(0, K - 1)
            stats.model_evals_total += max(0, K - 1)
            stats.network_forwards += adapter.forward_counter
            stats.backprops_through_model += max(0, K - 1)
            _record_loss(stats, loss.detach(), spec)

        with torch.no_grad():
            x = (x + (s_next - s) * (v0 + U[0].detach())).detach()
        warm = [u.detach() for u in U]

    adapter.block(x)
    stats.seconds = time.perf_counter() - t_start
    stats.finite = bool(torch.isfinite(x).all())
    return x, stats


def flow_mpc_delta_t(adapter: StandardFlowAdapter, cond: Conditioning, x0, problem,
                     spec) -> Tuple[Any, ReconstructionStats]:
    """Delta-t horizon control with the paper's one-step Euler value approximation (eq. 9)."""
    import torch
    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    ctrl_cost = make_control_cost(B, spec.control_cost_normalization)
    grid = mpc_time_grid(spec)
    lam_eff = (spec.lam / spec.delta if spec.delta_t_lambda_scaling == "inverse_delta"
               else spec.lam)

    stats = ReconstructionStats()
    t_start = time.perf_counter()
    x = x0.detach().clone()
    warm = None

    for step in range(int(spec.num_mpc_steps)):
        s, s_next = grid[step], grid[step + 1]
        dt = s_next - s

        adapter.reset_counters()
        with torch.no_grad():
            v0 = adapter.velocity(x, s, cond).detach()          # constant w.r.t. u
        stats.model_evals_total += 1
        stats.network_forwards += adapter.forward_counter

        remaining = 1.0 - s_next          # the (1 - t) factor of the value approximation

        u = (warm.detach().clone().requires_grad_(True)
             if (spec.warm_start and warm is not None)
             else torch.zeros_like(x, requires_grad=True))
        opt = _torch_optimizer([u], spec)

        for _it in range(int(spec.n_ctrl)):
            opt.zero_grad(set_to_none=True)
            adapter.reset_counters()
            x_next = x + dt * (v0 + u)
            if remaining > 1e-9:
                v_next = adapter.velocity(x_next, s_next, cond)  # differentiable on purpose
                x_projected = x_next + remaining * v_next
                used_model = 1
            else:
                # At s_next = 1 the projection factor is exactly zero, so V_hat = Phi(x_next).
                # Skipping the call is algebraically identical and saves a forward+backward.
                x_projected = x_next
                used_model = 0
            # Algorithm 2 carries NO delta factor on ||u||^2 (unlike RHC, which carries h).
            loss = ctrl_cost(u) + lam_eff * phi(
                adapter.to_pixels(x_projected, differentiable=True))
            loss.backward()
            if spec.grad_clip:
                torch.nn.utils.clip_grad_norm_([u], float(spec.grad_clip))
            opt.step()
            stats.control_iterations += 1
            stats.model_evals_planning += used_model
            stats.model_evals_total += used_model
            stats.network_forwards += adapter.forward_counter
            stats.backprops_through_model += used_model
            _record_loss(stats, loss.detach(), spec)

        with torch.no_grad():
            x = (x + dt * (v0 + u.detach())).detach()
        warm = u.detach()

    adapter.block(x)
    stats.seconds = time.perf_counter() - t_start
    stats.finite = bool(torch.isfinite(x).all())
    return x, stats


# =====================================================================================
# MeanFlow MPC (pMF, iMF)
# =====================================================================================
def _jax_optimizer(spec):
    """optax counterpart of _torch_optimizer; the same two optimisers, really implemented."""
    import optax
    if spec.optimizer == "adam":
        tx = optax.adam(float(spec.lr))
    elif spec.optimizer == "sgd":
        tx = optax.sgd(float(spec.lr))
    else:
        raise ValueError("Unsupported optimizer %r" % spec.optimizer)
    if spec.grad_clip:
        tx = optax.chain(optax.clip_by_global_norm(float(spec.grad_clip)), tx)
    return tx


def _optimize_jax(loss_fn, init, spec, stats: ReconstructionStats):
    """Gradient descent on the control, in JAX/Optax."""
    import jax
    import optax
    tx = _jax_optimizer(spec)
    state = tx.init(init)
    params = init
    value_and_grad = jax.value_and_grad(loss_fn)
    for _it in range(int(spec.n_ctrl)):
        loss, grads = value_and_grad(params)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        stats.control_iterations += 1
        _record_loss(stats, loss, spec)
    return params


def meanflow_mpc_rhc(adapter: MeanFlowAdapter, cond: Conditioning, x0, problem,
                     spec) -> Tuple[Any, ReconstructionStats]:
    """Receding-horizon control on the learned finite-interval transition."""
    import jax
    import jax.numpy as jnp
    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    ctrl_cost = make_control_cost(B, spec.control_cost_normalization)
    K = int(spec.K or 1)
    grid = mpc_time_grid(spec)
    lam = spec.lam

    stats = ReconstructionStats()
    t_start = time.perf_counter()
    x = x0
    warm = None

    for step in range(int(spec.num_mpc_steps)):
        s, s_next = grid[step], grid[step + 1]
        remaining = 1.0 - s
        h = remaining / K
        s_plan = [s + k * h for k in range(K + 1)]
        s_plan[-1] = 1.0                  # exactly the terminal time, free of rounding

        adapter.reset_counters()
        if K == 1:
            # xbar_1 = T(x_s; s -> 1) is independent of u, so the network never enters the
            # graph -- the MeanFlow analogue of the paper's K = 1 memory claim.
            nominal = jax.lax.stop_gradient(adapter.transition(x, s, 1.0, cond))
            init = warm if (spec.warm_start and warm is not None) else jnp.zeros_like(x)

            def loss_fn(u, _nominal=nominal, _rem=remaining):
                x_terminal = _nominal + _rem * u
                return (_rem * ctrl_cost(u)
                        + lam * phi(adapter.to_pixels(x_terminal, differentiable=True)))
        else:
            # The first transition also leaves the current state and is independent of U.
            first = jax.lax.stop_gradient(adapter.transition(x, s_plan[0], s_plan[1], cond))
            init = (warm if (spec.warm_start and warm is not None)
                    else jnp.zeros((K,) + x.shape, jnp.float32))

            def loss_fn(U, _first=first, _grid=tuple(s_plan), _h=h):
                xk = _first + _h * U[0]
                for k in range(1, K):
                    xk = adapter.transition(xk, _grid[k], _grid[k + 1], cond) + _h * U[k]
                control_term = _h * sum(ctrl_cost(U[k]) for k in range(K))
                return control_term + lam * phi(adapter.to_pixels(xk, differentiable=True))

        stats.model_evals_total += 1
        stats.network_forwards += adapter.forward_counter

        adapter.reset_counters()
        params = _optimize_jax(loss_fn, init, spec, stats)
        planning_evals = adapter.forward_counter
        stats.model_evals_planning += planning_evals
        stats.model_evals_total += planning_evals
        stats.network_forwards += planning_evals
        stats.backprops_through_model += int(spec.n_ctrl) * max(0, K - 1)

        u0 = params if K == 1 else params[0]
        warm = params

        # Execution uses T(x; s -> s+delta), which is NOT one of the planning intervals, so
        # MeanFlow RHC pays one extra transition per replan relative to the standard-flow case.
        adapter.reset_counters()
        x = adapter.transition(x, s, s_next, cond) + (s_next - s) * u0
        stats.model_evals_total += 1
        stats.network_forwards += adapter.forward_counter

    x = adapter.block(x)
    stats.seconds = time.perf_counter() - t_start
    stats.finite = bool(jnp.isfinite(x).all())
    return x, stats


def meanflow_mpc_delta_t(adapter: MeanFlowAdapter, cond: Conditioning, x0, problem,
                         spec) -> Tuple[Any, ReconstructionStats]:
    """Delta-t horizon control whose value-to-go is the model's own transport to s = 1.

    RESEARCH EXTENSION: the MeanFlow analogue of eq. (9).  The MPC-Flow paper does not
    propose or evaluate it.
    """
    import jax
    import jax.numpy as jnp
    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    ctrl_cost = make_control_cost(B, spec.control_cost_normalization)
    grid = mpc_time_grid(spec)
    lam_eff = (spec.lam / spec.delta if spec.delta_t_lambda_scaling == "inverse_delta"
               else spec.lam)

    stats = ReconstructionStats()
    t_start = time.perf_counter()
    x = x0
    warm = None

    for step in range(int(spec.num_mpc_steps)):
        s, s_next = grid[step], grid[step + 1]
        dt = s_next - s

        # The short nominal transition is independent of u AND is exactly what the execution
        # step applies, so it is computed once and reused for both.
        adapter.reset_counters()
        nominal_next = jax.lax.stop_gradient(adapter.transition(x, s, s_next, cond))
        stats.model_evals_total += 1
        stats.network_forwards += adapter.forward_counter

        at_terminal = s_next >= 1.0 - 1e-9
        init = warm if (spec.warm_start and warm is not None) else jnp.zeros_like(x)

        def loss_fn(u, _nom=nominal_next, _sn=s_next, _terminal=at_terminal, _dt=dt):
            x_next = _nom + _dt * u
            # T(x; 1 -> 1) is the identity by construction (t - r = 0), so it is skipped.
            x_terminal = x_next if _terminal else adapter.transition(x_next, _sn, 1.0, cond)
            return ctrl_cost(u) + lam_eff * phi(
                adapter.to_pixels(x_terminal, differentiable=True))

        adapter.reset_counters()
        u_star = _optimize_jax(loss_fn, init, spec, stats)
        planning_evals = adapter.forward_counter
        stats.model_evals_planning += planning_evals
        stats.model_evals_total += planning_evals
        stats.network_forwards += planning_evals
        stats.backprops_through_model += 0 if at_terminal else int(spec.n_ctrl)

        x = nominal_next + dt * u_star     # execution reuses the cached transition
        warm = u_star

    x = adapter.block(x)
    stats.seconds = time.perf_counter() - t_start
    stats.finite = bool(jnp.isfinite(x).all())
    return x, stats


# =====================================================================================
# This module's entries in the reconstruction registry -- MPC only.
#
# The GLOBAL dispatcher used to live here, which made mpc.py the owner of unrelated
# methods.  It now lives in `reconstruction.py`; this table declares only what MPC itself
# implements, and `select_reconstructor` below is a thin alias kept so that older imports
# (`from src.mpc import select_reconstructor`) keep working.
# =====================================================================================
SOLVERS = {
    (STANDARD_FLOW, "mpc_rhc"): flow_mpc_rhc,
    (STANDARD_FLOW, "mpc_delta_t"): flow_mpc_delta_t,
    (MEANFLOW, "mpc_rhc"): meanflow_mpc_rhc,
    (MEANFLOW, "mpc_delta_t"): meanflow_mpc_delta_t,
}


def select_reconstructor(dynamics_family: str, method: str):
    """Deprecated alias for `reconstruction.select_reconstructor`."""
    from .reconstruction import select_reconstructor as _select
    return _select(dynamics_family, method)
