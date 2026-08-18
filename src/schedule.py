"""The ONE canonical time schedule, shared by every reconstruction strategy.

Every method in this repository discretises the canonical interval [s_0, 1] (s = 0 noise,
s = 1 data).  Until now that discretisation was uniform and the helper lived in
`sdedit.py`, which made an algorithm module the owner of a quantity SDEdit, MPC, PnP and
D-Flow all depend on.  It now lives here, in a module that knows nothing about any method,
and it is generalised by ONE new hyperparameter:

    s_k = s_0 + (1 - s_0) * (k / N)^beta,        k = 0, ..., N

    beta < 1   large steps early, progressively smaller steps near the clean endpoint
    beta = 1   EXACTLY the repository's legacy uniform grid
    beta > 1   small steps early, progressively larger steps near the clean endpoint

For s_0 = 0, N = 4:

    beta = 0.5   0, 0.5,    0.7071, 0.8660, 1
    beta = 1     0, 0.25,   0.5,    0.75,   1
    beta = 2     0, 0.0625, 0.25,   0.5625, 1

`beta` controls TIME DISCRETISATION ONLY.  It is not PnP's `alpha` (which scales the
data-consistency step size), it is not a step count, and it never changes how many model
evaluations a method performs.

Backward compatibility is exact, not approximate: `beta == 1.0` takes the ORIGINAL linear
code path verbatim, so a re-run of an old configuration reproduces the same floating-point
grid bit for bit rather than to within a rounding error.

A power-law family of this shape was studied in the Flower paper's time-discretisation
ablation, where an exponent equivalent to beta = 0.5 -- more resolution near the clean
endpoint -- behaved favourably at low step counts.  Applying it uniformly to SDEdit, MPC,
PnP, D-Flow and RHSO is this repository's generalisation, not a claim about that paper.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

# The legacy schedule, and the value every configuration that says nothing gets.
DEFAULT_BETA = 1.0


def resolve_beta(beta: Any = None) -> float:
    """Validate a beta value and return it as a float.  `None` means the default.

    Accepts anything numeric; rejects non-finite values and beta <= 0, which do not
    describe a monotone schedule.
    """
    if beta is None:
        return DEFAULT_BETA
    if isinstance(beta, bool):                     # bool is an int; a boolean beta is a typo
        raise ValueError("beta must be a positive finite number, got %r" % (beta,))
    try:
        value = float(beta)
    except (TypeError, ValueError):
        raise ValueError("beta must be a positive finite number, got %r" % (beta,))
    if not math.isfinite(value):
        raise ValueError("beta must be finite, got %r" % (beta,))
    if value <= 0.0:
        raise ValueError("beta must be > 0, got %r" % (beta,))
    return value


def spec_beta(spec: Any) -> float:
    """The beta of a resolved job spec, defaulting to the legacy uniform schedule.

    Read through `getattr` on purpose: a spec produced before `beta` existed -- or by a
    caller that constructs its own lightweight spec -- keeps working and gets 1.0.
    """
    return resolve_beta(getattr(spec, "beta", None))


def is_uniform(beta: Any = None) -> bool:
    """True when the schedule is the legacy uniform one."""
    return resolve_beta(beta) == DEFAULT_BETA


def canonical_time_grid(s_start: float, steps: int,
                        beta: Any = DEFAULT_BETA) -> List[float]:
    """The `steps` + 1 canonical execution times s_start ... 1 (exact at both ends).

        s_k = s_start + (1 - s_start) * (k / steps)^beta

    Both dynamics families use this grid; the adapters convert each s to their own native
    time, so no model-specific clock inversion appears in any method module.

    `beta == 1.0` follows the ORIGINAL linear expression so old runs reproduce exactly.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be >= 1, got %r" % (steps,))
    s0 = float(s_start)
    if abs(1.0 - s0) < 1e-12:
        raise ValueError("Degenerate time grid: s_start == 1 leaves nothing to generate. "
                         "Use t0 > 0.")
    exponent = resolve_beta(beta)

    if exponent == DEFAULT_BETA:
        # ---- legacy path, character for character, so beta=1 is a bitwise regression ----
        grid = [s0 + (1.0 - s0) * (k / steps) for k in range(steps + 1)]
    else:
        grid = [s0 + (1.0 - s0) * (k / steps) ** exponent for k in range(steps + 1)]

    grid[0] = s0                        # exact start (k = 0 already gives this)
    grid[-1] = 1.0                      # kill accumulated rounding at the terminal time
    _assert_strictly_increasing(grid, exponent, steps)
    return grid


def interior_time_grid(s_start: float, num_points: int,
                       beta: Any = DEFAULT_BETA) -> List[float]:
    """`num_points` times strictly INSIDE (s_start, 1) -- PnP's correction schedule.

        s_k = s_start + (1 - s_start) * (k / (N + 1))^beta,      k = 1 .. N

    Equivalently: the canonical grid with N + 1 intervals, with both endpoints removed.
    Correcting AT s_start would repeat the state the initial prior projection already
    consumed, and correcting AT s = 1 would apply a denoiser with a factor (1 - s) = 0.
    Neither is allowed for any beta.
    """
    n = int(num_points)
    if n < 1:
        raise ValueError("num_points must be >= 1, got %r" % (num_points,))
    s0 = float(s_start)
    if not 0.0 <= s0 < 1.0:
        raise ValueError("interior_time_grid needs a start time in [0, 1), got %r"
                         % (s_start,))
    exponent = resolve_beta(beta)

    if exponent == DEFAULT_BETA:
        # ---- legacy path, character for character (see canonical_time_grid) ----
        grid = [s0 + (k / float(n + 1)) * (1.0 - s0) for k in range(1, n + 1)]
    else:
        grid = [s0 + ((k / float(n + 1)) ** exponent) * (1.0 - s0) for k in range(1, n + 1)]

    _assert_strictly_increasing([s0] + grid + [1.0], exponent, n + 1)
    return grid


def _assert_strictly_increasing(grid: Sequence[float], beta: float, steps: int) -> None:
    """A schedule that is not strictly increasing is not a schedule.

    Only reachable for extreme exponents, where (k/N)^beta underflows to zero for the
    first few k and two consecutive times collapse onto s_start.
    """
    for i in range(len(grid) - 1):
        if not grid[i] < grid[i + 1]:
            raise ValueError(
                "beta = %r collapses the time grid: s_%d = %r is not strictly before "
                "s_%d = %r at N = %d. Use a beta closer to 1, or fewer intervals."
                % (beta, i, grid[i], i + 1, grid[i + 1], steps))


# =====================================================================================
# Derived quantities -- for metadata, validation and cost models
# =====================================================================================
def grid_intervals(grid: Sequence[float]) -> List[float]:
    """dt_k = s_{k+1} - s_k.  THE per-step step size: no algorithm may assume t0/N."""
    return [float(grid[k + 1]) - float(grid[k]) for k in range(len(grid) - 1)]


def nominal_uniform_delta(s_start: float, steps: int) -> float:
    """(1 - s_start) / steps -- the spacing a UNIFORM grid would have.

    Reported as metadata only.  It is the actual step size when beta == 1 and it is NOT
    the actual step size otherwise, which is why nothing downstream may use it as a dt.
    """
    return (1.0 - float(s_start)) / int(steps)


def grid_metadata(grid: Sequence[float], beta: Any = DEFAULT_BETA) -> Dict[str, Any]:
    """Schedule facts worth recording next to a result row.

    `delta` is the nominal uniform spacing and is None whenever beta != 1, because a
    non-uniform trajectory has no single physical step size.  `delta_min` / `delta_max`
    always describe the real thing.
    """
    exponent = resolve_beta(beta)
    intervals = grid_intervals(grid)
    uniform = exponent == DEFAULT_BETA
    return {
        "beta": exponent,
        "uniform_schedule": uniform,
        "s_start": float(grid[0]),
        "num_intervals": len(intervals),
        "delta": (nominal_uniform_delta(grid[0], len(intervals)) if uniform else None),
        "delta_nominal_uniform": nominal_uniform_delta(grid[0], len(intervals)),
        "delta_min": min(intervals) if intervals else None,
        "delta_max": max(intervals) if intervals else None,
        "times": [float(s) for s in grid],
    }


def describe_schedule(beta: Any = DEFAULT_BETA) -> str:
    """One line for a log or a figure caption."""
    exponent = resolve_beta(beta)
    if exponent == DEFAULT_BETA:
        return "uniform (beta = 1)"
    if exponent < DEFAULT_BETA:
        return "power law beta = %g: finer near the clean endpoint" % exponent
    return "power law beta = %g: finer near the noisy start" % exponent
