"""PnP-Flow reconstruction: Plug-and-Play restoration with a Flow-Matching denoiser.

Based on Martin, Gagneux, Hagemann and Steidl, "PnP-Flow: Plug-and-Play Image Restoration
with Flow Matching" (ICLR 2025).  The published algorithm cycles

    gradient step on the data fidelity  ->  interpolation back onto the flow path
                                        ->  denoising with D_t = Id + (1 - t) v_theta

and, crucially, never backpropagates through the network.  That property is preserved here
exactly; it is what makes PnP the cheap member of this benchmark.

What is published, and what this file adapts
--------------------------------------------
PUBLISHED (close to the paper):
  * the three-step cycle above, and the time-dependent denoiser D_s = Id + (1 - s) v_theta;
  * the step-size schedule gamma_k = gamma0 (1 - s_k)^alpha;
  * averaging the denoised output over several noise realisations (Remark 3).

ADAPTED for this repository (documented, never presented as the paper's algorithm):
  * INITIALISATION.  The paper starts from an arbitrary x^(0) at t = 0 and is explicitly
    insensitive to it.  Here PnP must start from the SAME shared z_t0 as SDEdit, MPC and
    D-Flow, because that is the fairness invariant this benchmark is built on.  z_t0 lives
    ON the flow path at s0 = 1 - t0, so it is first mapped to a clean-ish iterate by ONE
    prior projection, and the correction schedule then runs on (s0, 1) rather than [0, 1).
  * FIDELITY SCALE.  The data term is the per-measurement normalisation of section 3 of
    the brief (see problems.make_phi), not 1/(2 sigma^2) ||Hx - y||^2.
  * M defaults to 1, where the paper averages over 5 realisations.
  * MEANFLOW (research extension).  A MeanFlow model has no instantaneous velocity, so the
    denoiser becomes the learned finite-interval transition to the clean endpoint,
    D_s(q) = T_theta(q ; s -> 1).  For a standard Flow-Matching model D_s is the
    conditional-mean estimate E[X_1 | X_s = q]; that interpretation does NOT automatically
    transfer to an arbitrary learned MeanFlow transition, and no such claim is made here.

Compute semantics
-----------------
The initial prior projection is part of the method: it is timed, counted and included in
the peak-memory measurement.  It is NOT one of `num_pnp_steps`.  With N corrections and M
noise realisations the logical prior applications are 1 + N*M.
"""

from __future__ import annotations

import time
from typing import Any, List, Tuple

import numpy as np

from .models.base import Conditioning, MeanFlowAdapter, ModelAdapter, StandardFlowAdapter
from .problems import make_phi, phi_log_scale
from .sdedit import ReconstructionStats, _is_finite
from .utils import MEANFLOW, STANDARD_FLOW, gaussian_noise, pnp_reprojection_parts


def pnp_time_grid(s_start: float, num_steps: int) -> List[float]:
    """The N correction times, strictly inside (s_start, 1).

        s_k = s0 + [k / (N + 1)] * (1 - s0),      k = 1 .. N

    The offset by one interval is deliberate.  Correcting AT s0 would repeat the state the
    initial prior projection already consumed, and correcting AT s = 1 would apply a
    denoiser with a factor (1 - s) = 0, i.e. an identity step that costs a full network
    evaluation and changes nothing.
    """
    n = int(num_steps)
    if n < 1:
        raise ValueError("num_pnp_steps must be >= 1, got %r" % (num_steps,))
    s0 = float(s_start)
    if not 0.0 <= s0 < 1.0:
        raise ValueError("PnP needs a start time in [0, 1), got %r" % (s_start,))
    grid = [s0 + (k / float(n + 1)) * (1.0 - s0) for k in range(1, n + 1)]
    assert all(s0 < s < 1.0 for s in grid)
    return grid


def pnp_step_sizes(spec, grid: List[float]) -> List[float]:
    """gamma_k = gamma0 * (1 - s_k)^alpha  -- no 1/sigma^2, by design (see the brief)."""
    gamma0, alpha = float(spec.gamma0), float(spec.alpha)
    return [gamma0 * (1.0 - s) ** alpha for s in grid]


def _reprojection_noise(adapter: ModelAdapter, spec, image_ids, iteration: int, sample: int):
    """One fresh-but-deterministic native noise draw per (image, iteration, realisation).

    Drawn per IMAGE ID rather than per batch position, so a batch of 2 and a batch of 4
    give image i the same realisations, and never from the shared initialisation epsilon:
    reusing that would couple every reprojection to the starting state.
    """
    noise = np.stack([gaussian_noise(adapter.spec.native_shape,
                                     *pnp_reprojection_parts(adapter.spec.name, image_id,
                                                             spec.replicate, iteration, sample))
                      for image_id in image_ids], axis=0)
    return adapter.to_native_noise(noise)


# =====================================================================================
# Standard flow (JiT, SiT) -- the published algorithm's denoiser
# =====================================================================================
def flow_pnp(adapter: StandardFlowAdapter, cond: Conditioning, x0, problem,
             spec) -> Tuple[Any, ReconstructionStats]:
    """PnP-Flow for an instantaneous-velocity model.

    Performs ZERO backpropagation through the generative model.  The only gradient taken is
    of the data-fidelity term with respect to the current iterate; for a latent model that
    gradient flows through the VAE decoder and the measurement operator, which is a
    measurement-side gradient and is counted as `data_gradient_evals`, never as a
    generative-trajectory backprop.
    """
    import torch

    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    log_scale = phi_log_scale(problem, spec.phi_normalization)
    s0 = float(spec.canonical_start_time)
    grid = pnp_time_grid(s0, int(spec.num_pnp_steps))
    gammas = pnp_step_sizes(spec, grid)
    M = int(spec.noise_samples or 1)
    image_ids = list(problem.image_ids)

    stats = ReconstructionStats()
    started = time.perf_counter()

    # ---------------------------------------------------------------- initial projection
    # z_t0 lies on the flow path at s0, so one denoiser evaluation turns it into the clean
    # iterate the PnP loop expects.  Counted, timed and measured -- but not a PnP step.
    adapter.reset_counters()
    with torch.no_grad():
        x = x0 + (1.0 - s0) * adapter.velocity(x0, s0, cond)
        x = x.detach()
    stats.model_evals_total += 1
    stats.denoiser_samples += 1
    stats.network_forwards += adapter.forward_counter
    if not _is_finite(adapter, x):
        raise FloatingPointError(
            "%s produced a non-finite state in the PnP initial prior projection at s=%.6f."
            % (adapter.spec.name, s0))

    # ---------------------------------------------------------------- correction cycles
    for k, (s_k, gamma_k) in enumerate(zip(grid, gammas), start=1):
        # -- 1. data-consistency gradient step ---------------------------------------
        # A FRESH leaf every iteration: PnP must not accumulate a graph across cycles.
        z = x.detach().clone().requires_grad_(True)
        loss = phi(adapter.to_pixels(z, differentiable=True))
        grad = torch.autograd.grad(loss, z)[0]
        with torch.no_grad():
            z_data = (z - gamma_k * grad).detach()
        stats.data_gradient_evals += 1
        stats.objective_evals += 1
        if spec.record_loss_history:
            stats.loss_history.append(float(loss.detach()) * log_scale)
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError(
                "PnP data-fidelity gradient is non-finite at iteration %d (s=%.6f). gamma0 "
                "is probably too large for this problem's fidelity scale." % (k, s_k))

        # -- 2. stochastic reprojection + 3. prior denoising --------------------------
        adapter.reset_counters()
        accumulated = None
        with torch.no_grad():
            for i in range(M):
                eps = _reprojection_noise(adapter, spec, image_ids, k, i)
                # q = s_k * z_data + (1 - s_k) * eps, in the adapter's own representation.
                q = adapter.corrupt(z_data, 1.0 - s_k, eps)
                denoised = q + (1.0 - s_k) * adapter.velocity(q, s_k, cond)
                accumulated = denoised if accumulated is None else accumulated + denoised
            # AVERAGE THE DENOISED OUTPUTS, not the noises: averaging eps first would
            # simply shrink the noise towards zero and defeat the reprojection.
            x = (accumulated / float(M)).detach()
        stats.model_evals_total += M
        stats.denoiser_samples += M
        stats.network_forwards += adapter.forward_counter

        if not _is_finite(adapter, x):
            raise FloatingPointError(
                "%s produced a non-finite state at PnP iteration %d/%d (s=%.6f)."
                % (adapter.spec.name, k, len(grid), s_k))

    adapter.block(x)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x)
    return x, stats


# =====================================================================================
# MeanFlow (pMF, iMF) -- RESEARCH EXTENSION
# =====================================================================================
def meanflow_pnp(adapter: MeanFlowAdapter, cond: Conditioning, x0, problem,
                 spec) -> Tuple[Any, ReconstructionStats]:
    """PnP with the learned finite-interval transition as the denoiser.

    RESEARCH EXTENSION.  The PnP-Flow paper defines its denoiser from an instantaneous
    velocity field; a MeanFlow model does not have one.  D_s(q) = T_theta(q ; s -> 1) is
    the natural analogue -- the model's own learned transport of a point at time s to the
    clean endpoint -- but it inherits none of the paper's conditional-mean guarantees.

    Like the standard-flow version, this never differentiates the generative model: each
    transition output is stop-gradient'ed so no MeanFlow transition can leak into the next
    iteration's data gradient.
    """
    import jax
    import jax.numpy as jnp

    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    log_scale = phi_log_scale(problem, spec.phi_normalization)
    s0 = float(spec.canonical_start_time)
    grid = pnp_time_grid(s0, int(spec.num_pnp_steps))
    gammas = pnp_step_sizes(spec, grid)
    M = int(spec.noise_samples or 1)
    image_ids = list(problem.image_ids)

    def fidelity(state):
        return phi(adapter.to_pixels(state, differentiable=True))

    value_and_grad = jax.value_and_grad(fidelity)

    stats = ReconstructionStats()
    started = time.perf_counter()

    # ---------------------------------------------------------------- initial projection
    adapter.reset_counters()
    x = jax.lax.stop_gradient(adapter.transition(x0, s0, 1.0, cond))
    stats.model_evals_total += 1
    stats.denoiser_samples += 1
    stats.network_forwards += adapter.forward_counter
    if not _is_finite(adapter, x):
        raise FloatingPointError(
            "%s produced a non-finite state in the PnP initial transition T(x; %.6f -> 1)."
            % (adapter.spec.name, s0))

    # ---------------------------------------------------------------- correction cycles
    for k, (s_k, gamma_k) in enumerate(zip(grid, gammas), start=1):
        loss, grad = value_and_grad(x)
        z_data = jax.lax.stop_gradient(x - gamma_k * grad)
        stats.data_gradient_evals += 1
        stats.objective_evals += 1
        if spec.record_loss_history:
            stats.loss_history.append(float(loss) * log_scale)
        if not bool(jnp.isfinite(grad).all()):
            raise FloatingPointError(
                "PnP data-fidelity gradient is non-finite at iteration %d (s=%.6f). gamma0 "
                "is probably too large for this problem's fidelity scale." % (k, s_k))

        adapter.reset_counters()
        accumulated = None
        for i in range(M):
            eps = _reprojection_noise(adapter, spec, image_ids, k, i)
            q = adapter.corrupt(z_data, 1.0 - s_k, eps)
            denoised = adapter.transition(q, s_k, 1.0, cond)
            accumulated = denoised if accumulated is None else accumulated + denoised
        x = jax.lax.stop_gradient(accumulated / float(M))
        stats.model_evals_total += M
        stats.denoiser_samples += M
        stats.network_forwards += adapter.forward_counter

        if not _is_finite(adapter, x):
            raise FloatingPointError(
                "%s produced a non-finite state at PnP iteration %d/%d (s=%.6f)."
                % (adapter.spec.name, k, len(grid), s_k))

    x = adapter.block(x)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x)
    return x, stats


def pnp_reconstruct(adapter: ModelAdapter, cond: Conditioning, x0, problem,
                    spec) -> Tuple[Any, ReconstructionStats]:
    """Family dispatch, for callers that hold an adapter but not a registry."""
    if spec.dynamics_family == STANDARD_FLOW:
        return flow_pnp(adapter, cond, x0, problem, spec)
    if spec.dynamics_family == MEANFLOW:
        return meanflow_pnp(adapter, cond, x0, problem, spec)
    raise ValueError("No PnP strategy for dynamics family %r" % spec.dynamics_family)
