"""Ordinary SDEdit reconstruction -- the paired baseline.

This module contains reconstruction ONLY.  Degradation lives in `problems.py`, and the
corrupted initial state is built by `models.base.build_initial_state`, exactly as it is for
both MPC methods:

        z_t0 = (1 - t0) * g(y) + t0 * eps

SDEdit then runs ordinary generation from s_start = 1 - t0 to the data endpoint s = 1.  The
measurement enters ONLY through the guide; there is no data-fidelity term and no control.
That is the point: it is what MPC's trajectory control is being compared against.

Standard flows (JiT, SiT)
    fixed-step integration of the instantaneous velocity field, with the Euler / Heun / RK4
    solvers from the SDEdit notebook.  Heun's final step falls back to Euler when the
    adapter advertises that policy (JiT's official sampler does).

MeanFlows (pMF, iMF)
    successive learned interval transitions along the canonical grid.  pMF is NOT treated as
    an ordinary velocity model; `steps` is the number of MeanFlow intervals and is always
    recorded explicitly in the results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SOLVER_STAGE_EVALUATIONS
from .models.base import Conditioning, MeanFlowAdapter, ModelAdapter, StandardFlowAdapter
from .utils import MEANFLOW, STANDARD_FLOW


@dataclass
class ReconstructionStats:
    """Measured compute for one reconstruction (per chunk), never an estimate."""
    control_iterations: int = 0
    model_evals_planning: int = 0        # v_theta / T_theta calls inside a control loop
    model_evals_total: int = 0           # every dynamics evaluation
    network_forwards: int = 0            # actual forward passes (CFG counted separately)
    backprops_through_model: int = 0
    loss_history: List[float] = field(default_factory=list)
    seconds: float = 0.0
    finite: bool = True

    def merge(self, other: "ReconstructionStats") -> None:
        self.control_iterations += other.control_iterations
        self.model_evals_planning += other.model_evals_planning
        self.model_evals_total += other.model_evals_total
        self.network_forwards += other.network_forwards
        self.backprops_through_model += other.backprops_through_model
        self.loss_history.extend(other.loss_history)
        self.seconds += other.seconds
        self.finite = self.finite and other.finite


def canonical_time_grid(s_start: float, steps: int) -> List[float]:
    """The `steps` + 1 canonical execution times s_start ... 1 (exact at both ends).

    Both families use this grid; the adapters convert each s to their own native time, so
    no model-specific clock inversion appears in this file.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1, got %r" % (steps,))
    if abs(1.0 - float(s_start)) < 1e-12:
        raise ValueError("Degenerate time grid: s_start == 1 leaves nothing to generate. "
                         "Use t0 > 0.")
    grid = [float(s_start) + (1.0 - float(s_start)) * (k / steps) for k in range(steps + 1)]
    grid[-1] = 1.0                      # kill accumulated rounding at the terminal time
    return grid


def _is_finite(adapter: ModelAdapter, x) -> bool:
    if adapter.spec.framework == "torch":
        import torch
        return bool(torch.isfinite(x).all())
    import jax.numpy as jnp
    return bool(jnp.isfinite(x).all())


# =====================================================================================
# Standard flow: fixed-step ODE integration
# =====================================================================================
def sdedit_flow(adapter: StandardFlowAdapter, cond: Conditioning, x0, spec) -> Tuple[Any,
                                                                                     ReconstructionStats]:
    """Euler / Heun / RK4 integration from s_start to s = 1 (SDEdit notebook, section 14)."""
    import torch
    solver = spec.solver or "euler"
    if solver not in SOLVER_STAGE_EVALUATIONS:
        raise ValueError("Unknown solver %r" % solver)
    grid = canonical_time_grid(spec.canonical_start_time, int(spec.steps))
    # The final-step policy is a MODEL property, read from the adapter -- not a name check.
    final_euler = bool(adapter.spec.euler_final_step_for_heun)

    stats = ReconstructionStats()
    started = time.perf_counter()
    # Sampling is pure inference: no graph is needed, unlike MPC.
    with torch.no_grad():
        x = x0.detach().clone()
        for k in range(int(spec.steps)):
            s, s_next = grid[k], grid[k + 1]
            dt = s_next - s
            adapter.reset_counters()

            v1 = adapter.velocity(x, s, cond)
            step_solver = solver
            if solver == "heun" and final_euler and k == int(spec.steps) - 1:
                step_solver = "euler"

            if step_solver == "euler":
                x_next = x + dt * v1
                evals = 1
            elif step_solver == "heun":
                v2 = adapter.velocity(x + dt * v1, s_next, cond)
                x_next = x + 0.5 * dt * (v1 + v2)
                evals = 2
            else:                                                        # rk4
                s_mid = 0.5 * (s + s_next)
                v2 = adapter.velocity(x + 0.5 * dt * v1, s_mid, cond)
                v3 = adapter.velocity(x + 0.5 * dt * v2, s_mid, cond)
                v4 = adapter.velocity(x + dt * v3, s_next, cond)
                x_next = x + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
                evals = 4

            stats.model_evals_total += evals
            stats.network_forwards += adapter.forward_counter
            if not _is_finite(adapter, x_next):
                raise FloatingPointError(
                    "%s produced a non-finite state at step %d/%d (s=%.6f -> %.6f) using %s."
                    % (adapter.spec.name, k, int(spec.steps) - 1, s, s_next, step_solver))
            x = x_next

    adapter.block(x)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x)
    return x, stats


# =====================================================================================
# MeanFlow: successive learned interval transitions
# =====================================================================================
def sdedit_meanflow(adapter: MeanFlowAdapter, cond: Conditioning, x0,
                    spec) -> Tuple[Any, ReconstructionStats]:
    """Successive transitions T(x; s_k -> s_{k+1}) along the canonical grid.

    This is the notebooks' ordinary MeanFlow sampling, unchanged.  With `steps: 1` it is the
    single-interval transition straight to the data endpoint, which is what pMF is designed
    for; larger values subdivide the same interval.
    """
    grid = canonical_time_grid(spec.canonical_start_time, int(spec.steps))
    stats = ReconstructionStats()
    started = time.perf_counter()
    x = x0

    for k in range(int(spec.steps)):
        adapter.reset_counters()
        x = adapter.transition(x, grid[k], grid[k + 1], cond)
        stats.model_evals_total += 1
        stats.network_forwards += adapter.forward_counter
        if not _is_finite(adapter, x):
            raise FloatingPointError(
                "%s produced a non-finite state at MeanFlow transition %d/%d (s=%.4f -> %.4f)."
                % (adapter.spec.name, k, int(spec.steps) - 1, grid[k], grid[k + 1]))

    x = adapter.block(x)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x)
    return x, stats


def sdedit_reconstruct(adapter: ModelAdapter, cond: Conditioning, x0, problem,
                       spec) -> Tuple[Any, ReconstructionStats]:
    """Family dispatch.  `problem` is accepted for interface symmetry with the MPC solvers
    and is deliberately unused: SDEdit never sees the measurement."""
    if spec.dynamics_family == STANDARD_FLOW:
        return sdedit_flow(adapter, cond, x0, spec)
    if spec.dynamics_family == MEANFLOW:
        return sdedit_meanflow(adapter, cond, x0, spec)
    raise ValueError("No SDEdit strategy for dynamics family %r" % spec.dynamics_family)
