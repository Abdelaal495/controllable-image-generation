# The `beta` time schedule, and RHSO

Two additions, documented together because RHSO uses the schedule and the schedule is
available to every method including RHSO.

---

## A. The universal power-law time schedule

Every strategy discretises the canonical interval `[s0, 1]`, where `s0 = 1 - t0`
(`s = 0` noise, `s = 1` data). That discretisation used to be uniform. It is now

```
s_k = s0 + (1 - s0) · (k / N)^beta,        k = 0 … N
```

with one new shared hyperparameter:

| `beta` | effect |
|---|---|
| `< 1` | large steps early, progressively **smaller** steps near clean space — more resolution where the image is formed |
| `= 1` | **exactly** the legacy uniform schedule |
| `> 1` | small steps early, progressively **larger** steps near clean space — more resolution near the noisy start |

Worked examples at `s0 = 0`, `N = 4`:

```
beta = 0.5    0, 0.5,    0.7071, 0.8660, 1
beta = 1      0, 0.25,   0.5,    0.75,   1
beta = 2      0, 0.0625, 0.25,   0.5625, 1
```

`beta` controls **time discretisation only**. It is not a step count, it never changes how
many model evaluations a method performs, and it is **not** PnP's `alpha` — that scales the
data-consistency step size `γ_k = γ0·(1 − s_k)^α` and keeps its meaning unchanged. The two
are deliberately never coupled: `beta` moves *where* the corrections happen, `alpha` shapes
*how large* they are at whatever times result.

A power-law family of this shape was studied in the Flower paper's time-discretisation
ablation, where an exponent equivalent to `beta = 0.5` behaved favourably on inverse
problems at low step counts. Applying it uniformly to SDEdit, MPC, PnP, D-Flow and RHSO is
this repository's generalisation — Flower proposes neither that generalisation nor RHSO.

### One implementation

`src/schedule.py` owns the schedule and nothing else:

```python
canonical_time_grid(s_start, steps, beta)   # steps + 1 points, s_start … 1
interior_time_grid(s_start, n, beta)        # n points strictly inside (s_start, 1)  [PnP]
grid_intervals(grid)                        # the ACTUAL dt_k = s_{k+1} - s_k
grid_metadata(grid, beta)                   # what a result row should record
spec_beta(spec)                             # spec.beta, or 1.0 if the field is absent
```

`(k/N)^beta` appears nowhere else. `canonical_time_grid` validates `steps ≥ 1` and
`beta` finite and `> 0`, returns exactly `steps + 1` points, starts exactly at `s_start`,
ends exactly at `1.0`, and is strictly increasing (an exponent so extreme that two times
collapse is an error, not a silently degenerate grid).

**`beta == 1.0` takes the original linear code path verbatim**, so old experiments
reproduce the same floating-point grid *bitwise*, not merely to within a rounding error.
`sdedit.canonical_time_grid` remains importable and now resolves to this implementation.

### Per method

| method | what `beta` shapes | what it does **not** shape |
|---|---|---|
| `sdedit` | the `steps` execution intervals; each solver step uses its own `dt_k` | solver arithmetic (Euler / Heun / RK4 and the adapter's final-step policy) |
| `dflow` | the differentiable trajectory grid; gradients still flow through all of it | anything else; at `steps = 1` `beta` is inert |
| `mpc_rhc` | **only** the outer receding-horizon execution grid (`N = num_mpc_steps`) | the internal `K`-step planning discretisation, still uniform over the remaining horizon `1 − s_k` |
| `mpc_delta_t` | the outer execution grid; every `dt` is the local `s_{k+1} − s_k` | — |
| `pnp` | the `N` correction times, still strictly inside `(s0, 1)` | `alpha`, `gamma0`, and the reprojection noise seeding |
| `rhso` | the outer stage grid, and hence the standard-flow planning suffix | — |

`beta` and `K` are not confounded on purpose: **`beta` says how often the controller
replans and how much it commits; `K` says how finely it plans.**

### The end of the constant `delta`

A non-uniform trajectory has no single physical step size. Concretely:

* nothing may use `spec.delta` (nominal `t0/N`) as an actual `dt` when `beta ≠ 1`;
* `MPC-Δt`'s `delta_t_lambda_scaling: inverse_delta` is now **step-dependent**,
  `λ_eff,k = λ / dt_k` (`src.mpc.inverse_delta_lambda`). With `beta = 1` and a planner-
  supplied `spec.delta`, that scalar is used verbatim, so legacy runs are bitwise
  unchanged. With scaling disabled, `λ` stays `λ`;
* `checks.check_time_grid` no longer asserts `dt = t0/N`. It asserts the endpoints,
  strict monotonicity, uniformity **where `beta = 1`**, and the correct interval direction
  otherwise;
* `schedule.grid_metadata` reports `delta: null` for `beta ≠ 1` and adds
  `delta_nominal_uniform`, `delta_min`, `delta_max`.

The job-defining quantities are now **`t0`, `N`, `beta`**.

If `beta` is swept where the relevant trajectory has a single interval (`steps: 1`,
`num_mpc_steps: 1`, `num_rhso_steps: 1`), it has no mathematical effect — a warning, not an
error.

---

## B. RHSO — Receding-Horizon State Optimization

Method key `rhso`, implemented for **both** dynamics families in `src/rhso.py`.

```
at s_k, holding the current state x_k:

    anchor = stop_gradient(x_k)              ← FIXED for the whole stage
    q⁽⁰⁾   = x_k
    for j = 0 … M-1:
        fidelity = Φ( to_pixels( G_{s_k → 1}( q⁽ʲ⁾ ), differentiable=True ) )
        R        = Σ_b  ‖q⁽ʲ⁾_b − anchor_b‖²  /  (2 d_b)
        loss     = fidelity + μ · R
        q⁽ʲ⁺¹⁾  = AdamUpdate( q⁽ʲ⁾, ∂loss/∂q )
    q* = q⁽ᴹ⁾

    x_{k+1} = G_{s_k → s_{k+1}}( q* )        ← only ONE interval is executed

then discard the problem and start again at s_{k+1}, with a FRESH anchor = x_{k+1}
```

with `N = num_rhso_steps`, `M = num_opt_steps`, `μ = mu` and the outer times from the
`beta` schedule. In one line: **optimise the current state using its terminal prediction →
execute one scheduled interval → replan.**

The variable being optimised is the **current generative state itself**. There is no
control `u`, no control penalty, no `λ`, no `K`, no `τ` and no adaptive step size.
`μ = 0` (the default) is the plain terminal-fidelity objective and is exactly what RHSO did
before the penalty existed.

### State-anchor regularisation (`mu`) — optional, off by default

Pure terminal fidelity places no cost on *where* `q` goes: it may drift far from the state
the generative trajectory actually reached and still score well on the measurement. The
failure mode this addresses is empirical — pushing the inner optimisation harder keeps
improving measurement consistency while PSNR / SSIM / LPIPS start to degrade. `mu > 0` adds
a trust-region-like penalty on that displacement:

```
R(q, x_k) = Σ_b  1/(2 d_b) · ‖q_b − x_{k,b}‖²

    squared difference  →  MEAN over the non-batch state dimensions  →  × ½  →  SUM over the batch

L_k(q) = Φ( G_{s_k → 1}(q) )  +  μ · R(q, x_k)
```

* **The anchor is `x_k`**: the state at the *start of that outer stage*, detached. It is
  **not** the trajectory origin `x_0`, and it is **not** a moving average — it does not
  follow `q`. Stage `k+1` takes a fresh anchor from the state actually executed.
* **Batch sum, not batch mean**, matching the repository's per-measurement fidelity: one
  sample's gradient never depends on the batch size.
* **Divided by `d_b`**, the number of scalar state dimensions per sample, so a useful `μ`
  does not scale with image or latent dimensionality — a pixel model and a latent model are
  on comparable footing. An unnormalised `‖q − x_k‖²` would not be.
* **No `σ²`, `dt`, `beta` or time dependence.** `μ` is the only weight in this iteration.
* The penalty contains **no generative-model evaluation**, so every compute counter and
  every dry-run cost estimate is unchanged (verified by a check, not asserted).

`μ` is *not* MPC's `λ`. MPC's `λ` weights the magnitude of an **added control signal** `u`
in a dynamics that RHSO does not have; `μ` weights a **displacement of the state** from
where the trajectory actually was. They are different quantities on different objects, with
different units, and are not interchangeable — which is why the field is separately named
and rejected outside RHSO.

This is an **experimental extension** motivated by observed over-optimisation. It is not
claimed to be theoretically required, and no value of `μ` is claimed to be optimal; `μ = 0`
is a first-class point of the sweep and remains the default.

### The two families differ in the planner, not in the method

**Standard flow (JiT, SiT).** `G_{s_k → 1}` has to be integrated: a differentiable
fixed-step solve over the **remaining suffix of the same outer grid**,
`[s_k, s_{k+1}, …, 1]`, using the repository's existing solver semantics — Euler / Heun /
RK4 and the adapter's own final-step policy, via the shared
`dflow.integrate_flow` / `dflow.flow_step`. No second planning-resolution hyperparameter is
introduced, so planning cost shrinks as `k` grows. Execution then applies **one**
`flow_step` over `[s_k, s_{k+1}]`, under `no_grad`; the Heun→Euler fallback applies only to
the interval that genuinely lands on `s = 1`.

**MeanFlow (pMF, iMF).** The planner is **one direct learned finite-interval transition**
`T_θ(q; s_k → 1)`. The remaining outer intervals are *not* composed for the objective, no
ODE is constructed and no instantaneous velocity is inferred. Execution is exactly one
transition `T_θ(q*; s_k → s_{k+1})`. One forward and one backward per inner iteration at
every `k`, which is what makes MeanFlow models computationally well suited to a
receding-horizon state optimiser — RHSO is available to standard flows, it is simply
dearer there.

Both paths share the same objective, the same fidelity `Φ`, the same
`to_pixels(differentiable=True)` and the same adapter abstraction. Latent models (iMF, SiT)
differentiate through the VAE decoder; there is **no separate pixel/latent RHSO
mathematics** and no model-name special-casing.

### Contracts

* The inner objective path `q → planner → native terminal state → to_pixels(differentiable=True) → A(x) → Φ`
  contains no NumPy conversion, no `detach`, no PIL and no stop-gradient. Model parameters
  stay frozen; only `q` moves.
* **Adam state is rebuilt at every outer stage.** Moments are never carried from `s_k` to
  `s_{k+1}`: the optimisation problem changes when the terminal transport map changes.
* Execution is inference only — no graph is carried into the next outer stage.
* The executed interval always uses the **final post-update `q*`**. The last inner
  objective was evaluated *before* the last Adam update, so its terminal prediction belongs
  to a state that no longer exists and is never reused.
* The anchor is a **stop-gradient constant for the whole stage**: `x_anchor = x.detach()`
  in torch, `jax.lax.stop_gradient(x)` closed over by `loss_fn` in JAX. It never receives a
  gradient or an update. Both families call the same
  `rhso.state_anchor_penalty(B, q, anchor)`, written against the `Backend` abstraction, so
  the normalisation is shared by construction rather than transcribed twice.

### RHSO vs D-Flow vs MPC

| | what is optimised | what is executed after optimising |
|---|---|---|
| **D-Flow** | one starting/intermediate state, globally | the **whole** planned trajectory |
| **MPC** | explicit control variables `u` added to the dynamics, with a control penalty `λ‖u‖²` | one interval, then replan |
| **RHSO** | the **current generative state**, optionally with `μ·R` anchoring it to the state the trajectory reached | one interval, then re-optimise from the state actually reached |

RHSO is *terminal planning + state optimisation + partial execution + replanning*. It is
not implemented by calling D-Flow in a loop; it has its own reconstruction implementation
and reuses only low-level numerical helpers.

### Configuration

```yaml
rhso: {t0: 0.8, beta: [0.5, 1.0, 2.0], num_rhso_steps: 4, num_opt_steps: 10, lr: 0.01}
```

Sweeping the state-regularisation weight (the values below are **syntax, not
recommendations** — nothing here is tuned):

```yaml
rhso:
  t0: 1.0
  beta: 0.5
  num_rhso_steps: 4
  num_opt_steps: 20
  lr: 0.02
  mu: [0.0, 0.01, 0.1, 1.0]        # 0.0 = the unpenalised objective
  optimizer: adam
  phi_normalization: half_mean_squared_per_measurement
```

| scope | fields |
|---|---|
| shared | `t0`, `beta` |
| `rhso` | `num_rhso_steps`, `num_opt_steps`, `lr`, `mu`, `optimizer`, `phi_normalization` |
| `rhso` on a standard flow only | `solver` (same allowed values and semantics as the other standard-flow methods) |

`mu` must be finite and `>= 0`; it defaults to `0.0`, takes part in the job id, and appears
in the output directory name only when it is non-zero (so paths written before the field
existed are unchanged). It is rejected for every other method — including with MPC's `lam`
present in the same file — because no other method here optimises a state in place.

`optimizer: adam` is the only supported value in this iteration. `solver` is meaningless for
MeanFlow RHSO and must be rejected there, exactly as it is for the other MeanFlow methods.
Deliberately **absent**: `lam`, `control_cost_normalization`, `K`, `warm_start`, `tau`,
`alpha_min`, adaptive horizons, adaptive or time-dependent `mu`, and any new noise
hyperparameter.

### Cost accounting

Counted honestly per the repository's existing distinctions — one outer stage is **not**
one model evaluation.

**MeanFlow**, with `N` stages and `M` inner iterations:

```
planning model evaluations = N · M
generative backprops       = N · M
optimizer iterations       = N · M
objective evaluations      = N · M
model evaluations total    = N · (M + 1)
```

The `+1` per stage is the execution transition for `q*`.

**Standard flow**: planning cost shrinks with `k`, so it is computed from the solver's real
stage counts over each remaining suffix, plus one executed interval per stage:

```
planning = Σ_k  M · ( (N − k) · stages(solver) − [heun && euler_final_step_for_heun] )
execution = Σ_k ( stages(solver) − [heun && euler_final_step_for_heun && k = N−1] )
```

`rhso.rhso_cost_estimate(values, dynamics_family, euler_final_step_for_heun)` returns
exactly these numbers and is the function `config._estimate_cost` calls, so the dry-run plan
and the measured counters cannot drift apart.

**`mu` does not appear anywhere above, and that is the point.** The state penalty is a
handful of elementwise tensor operations on `q` and a constant; it evaluates no generative
model, so planning evaluations, total model evaluations, generative backprops, optimizer
iterations, objective evaluations and the dry-run estimate are all identical at every `mu`.
A regression check runs the same job at `mu = 0` and `mu > 0` and compares every counter.

### Diagnostics

The scalar Adam differentiates is the **total** objective `Φ + μ·R`, and `loss_history`
records exactly that — so at `μ = 0` the history is what it always was. Two parallel lists
in `ReconstructionStats` split it, on the same per-image scale:

| field | meaning |
|---|---|
| `loss_history` | the optimised total `Φ + μ·R` |
| `fidelity_history` | the terminal measurement fidelity `Φ` alone |
| `state_penalty_history` | the **unweighted** displacement `R` (multiply by `mu` from the result row for the weighted contribution) |

`loss_history[i] == fidelity_history[i] + mu · state_penalty_history[i]` for every recorded
iteration. `R` is stored unweighted so a `mu` sweep can be re-weighted after the fact. All
three land in the per-job `results.npz` when `record_loss_history` is on, and are empty for
every method that has no such split. `mu` itself, and where it came from, are columns in
`results.csv`.

---

## C. How it is wired into `config.py` and `run.py`

**`src/config.py`**

* `beta` joins `t0` in `SHARED_FIELDS`, so it is declared for **every** method; it is in
  `SWEEPABLE_FIELDS` and `BUILTIN_DEFAULTS` (`1.0`), and it is a `JobSpec` field.
* Validated through `schedule.resolve_beta` (finite, `> 0`); a `beta` sweep on a
  single-interval trajectory **warns** rather than fails.
* `beta` and `num_rhso_steps` take part in `job_id`, and `beta` appears in `leaf_dir`
  whenever it is not 1 — so two runs differing only in `beta` can never share an output
  directory or a plotting group.
* `METHOD_DECLARATIONS["rhso"]` declares
  `t0, beta, num_rhso_steps, num_opt_steps, lr, mu, optimizer, phi_normalization, solver`.
  `mu` is sweepable, defaults to `0.0` with provenance `builtin` (the feature *off*, not an
  untuned guess), takes part in `job_id`, appears in `leaf_dir` and the figure label only
  when non-zero, and is rejected for every other method.
  `lam`, `control_cost_normalization` and `K` are rejected with an explanation;
  `optimizer` is restricted to `adam` by `VALID_RHSO_OPTIMIZERS`; `solver` is accepted only
  where the model's capabilities allow one, so a MeanFlow RHSO job refuses it exactly as
  the other MeanFlow methods do. Every entry of `MODEL_CAPABILITIES` supports `rhso`, and
  `rhso` is in `COMPARED_METHODS`.
* `_estimate_cost("rhso", …)` delegates to `rhso.rhso_cost_estimate`, so the dry-run plan
  and the measured counters cannot drift apart.
* `delta` is now **nominal uniform spacing only, and `None` whenever `beta != 1`**. Three
  derived fields are recorded for every method instead:
  `delta_nominal_uniform`, `delta_min`, `delta_max`.
* The dry-run report prints `beta` on every method, `executed dt` as real min–max pairs,
  and reminders that `delta = t0/N` holds only at `beta = 1`.

**`run.py`**

* `warmup_key` includes `beta`, `num_rhso_steps` and `mu`: the first two change the resolved
  time grid and the third changes the objective that is traced and differentiated, so
  two such jobs never share a warm-up and neither compiles inside its measured region.
* `warm_up` reduces RHSO's inner Adam budget to one iteration and **never** reduces
  `num_rhso_steps` or `beta` — every outer stage traces a different computation.
* `RESULT_COLUMNS` gains `beta`, `num_rhso_steps`, `mu`, `hyperparameter_source_mu`,
  `delta_nominal_uniform`, `delta_min` and `delta_max`; `persist_job` writes
  `fidelity_history` and `state_penalty_history` beside `loss_history` in `results.npz`; `METHOD_ORDER` places RHSO after D-Flow; `run_metadata.json` records the
  time-schedule policy.

Nothing else in either file changed, and a configuration that mentions neither `beta` nor
`rhso` resolves exactly as it did before.
