# PnP-Flow and D-Flow in this repository

This benchmark compares five reconstruction strategies against one shared corrupted state.
Two of them — **PnP-Flow** and **D-Flow** — were added after the SDEdit / MPC core, and both
had to be adapted before they could sit in the same comparison. This document says exactly
what comes from the papers, what was adapted to make the comparison fair, and what is a
research extension of this repository with no published backing at all.

Read it before quoting any number from these two methods.

---

## Why adaptation was necessary at all

The repository's central invariant is:

```
same problem + same model + same t0 + same epsilon
        ==>  the ONLY thing that varies is the reconstruction strategy
```

Both papers initialise their algorithms in ways that violate it. PnP-Flow starts from an
arbitrary iterate at `t = 0` and is explicitly insensitive to it; D-Flow either starts from
the source distribution or from a variance-preserving blend of noise with a backward solve
of the observation. Either choice would give one method a different starting point from
SDEdit and MPC, and the resulting table would compare *initialisations* as much as
strategies.

So both are initialised from the shared

```
z_t0 = (1 − t0) · g(y) + t0 · ε ,        s0 = 1 − t0
```

and the algorithms are adjusted around that constraint. This is a deliberate trade: the
comparison becomes clean, and the correspondence with each published algorithm becomes
approximate in a way that has to be stated rather than buried.

---

## PnP-Flow

> Martin, Gagneux, Hagemann, Steidl. *PnP-Flow: Plug-and-Play Image Restoration with Flow
> Matching.* ICLR 2025.

### Published, and kept

* The three-step cycle: a gradient step on the data fidelity, interpolation back onto the
  flow path, then denoising.
* The time-dependent denoiser built from the velocity field, `D_s(q) = q + (1 − s)·v_θ(q, s)`.
* The step-size schedule `γ_k = γ0 · (1 − s_k)^α`.
* Averaging the denoised output over several noise realisations (the paper's Remark 3).
* **No backpropagation through the generative model.** This is the property that makes PnP
  the cheap member of the benchmark, and it is preserved exactly. `backprops_through_model`
  is zero for every PnP job, by construction and by check.

### Adapted

| | Paper | Here |
|---|---|---|
| Initialisation | arbitrary `x⁽⁰⁾`, insensitive | the shared `z_t0`, followed by **one** prior projection |
| Time range | the full `[0, 1)` | `(s0, 1)`, with `s0 = 1 − t0` |
| Fidelity | `1/(2σ²)‖Hx − y‖²` | `F_b = 1/(2 m_b) · Σ (A(x)_j − y_j)²`, summed over the batch |
| Noise realisations | 5 | `noise_samples`, default **1** |
| `γ0`, `α` | tuned per dataset and task on a validation split | repository defaults, recorded as `repository_default_untuned` |

The **initial prior projection** deserves emphasis. `z_t0` lies *on* the flow path at `s0`,
which is not what the PnP loop expects to consume, so it is first mapped to a clean-ish
iterate by a single denoiser application. That projection is part of the method: it is
timed, counted, and inside the measured memory region. It is **not** one of
`num_pnp_steps`. With `N` corrections and `M` realisations the logical prior applications
are `1 + N·M`, and that is what `denoiser_samples` reports.

The **correction schedule** is

```
s_k = s0 + [k / (N + 1)] · (1 − s0),      k = 1 … N
```

strictly inside `(s0, 1)`. Correcting *at* `s0` would repeat the state the initial
projection already consumed; correcting *at* `s = 1` would apply a denoiser with factor
`(1 − s) = 0` — an identity step costing a full network evaluation. The structural check
`pnp_time_grid` asserts both.

### Research extension: PnP on a MeanFlow model

A MeanFlow model has no instantaneous velocity field, so `D_s` cannot be formed as the
paper forms it. The denoiser here is the model's own learned transport to the clean
endpoint:

```
D_s(q) = T_θ(q ; s → 1)
```

**This is not the paper's algorithm and carries none of its guarantees.** For a standard
Flow-Matching model, `D_s` is the conditional-mean estimate `E[X_1 | X_s = q]`, and the
plug-and-play interpretation of the denoising step rests on that. An arbitrary learned
MeanFlow transition is *not* known to be a conditional-mean estimator, and this repository
makes no such claim. What the MeanFlow PnP results show is how the method behaves when the
prior step is replaced by a learned finite-interval map — an empirical question, not a
transfer of theory.

---

## D-Flow

> Ben-Hamu, Puny, Gat, Karrer, Singer, Lipman. *D-Flow: Differentiating through Flows for
> Controlled Generation.* ICML 2024.

### Published, and kept

* The principle: `min_q L(G(q))`, optimising the state the generation starts from, with the
  gradient taken **through** the generative trajectory. This is the defining contrast with
  PnP and the reason D-Flow is the expensive end of the benchmark.

### Adapted

| | Paper | Here |
|---|---|---|
| Initialisation | source distribution, or a blend of noise with the backward solve of `y` | the shared `z_t0` |
| Optimiser | LBFGS with line search | fixed-budget **Adam** (`num_opt_steps`, `lr`) |
| Stopping | at a task-dependent target PSNR | never — the budget is fixed |
| Regularisation | χ^d source-norm penalty, source NLL, gradient clipping | **none** (the `R = 0`, implicit-regularisation case) |
| Loss | negative PSNR / `1/(2σ²)‖Hx − y‖²` | the per-measurement fidelity below |

A ground-truth-dependent stopping rule cannot appear in a benchmark: it would let one
method peek at the answer. LBFGS and the regularisers are simply out of scope for this
iteration, and the validator **rejects** `optimizer: sgd` or `optimizer: lbfgs` for D-Flow
rather than silently running something else under the same name.

The reconstruction always corresponds to the **final** optimised `q`: after the last Adam
update, one more trajectory is evaluated, and that extra evaluation is counted in
`model_evaluations`. Returning the last iterate seen during optimisation would report a
reconstruction one gradient step stale.

### Research extension: intermediate-state D-Flow (`t0 < 1`)

At `t0 = 1` (`s0 = 0`) this is ordinary source-point optimisation. At `t0 < 1` the optimised
variable is an **intermediate flow state** — the measurement-informed `z_t0`. This is what
makes D-Flow directly comparable with SDEdit, MPC and PnP at the same `t0`, and it is *not*
the published setup. Every such job emits a warning at validation time, and
`initialization_kind` / `canonical_start_time` are recorded in every result row. Call it
"truncated" or "intermediate-state" D-Flow when reporting; do not call it D-Flow without
qualification.

### Research extension: D-Flow on a MeanFlow model

There is no ODE to differentiate through. The trajectory is a composition of learned
finite-interval transitions, and with `steps = 1` it is the single map `T_θ(q ; s0 → 1)` —
one network evaluation forward, one backward, per objective.

**A one-step Euler integration of a velocity field is not the same object as a one-step
MeanFlow map.** The first is a coarse approximation of a trajectory whose error shrinks as
steps increase; the second is trained to *be* the finite-interval map. A table row showing
"D-Flow, steps = 1" for JiT and for pMF is therefore comparing two different things at
equal step count, which is why `model_evaluations`, runtime and memory are reported
alongside — those are comparable, `steps` is not.

---

## The shared fidelity scale

Both methods default to `phi_normalization: half_mean_squared_per_measurement`:

```
F_b(x_b) = (1 / 2 m_b) · Σ_{j ∈ Ω_b} (A(x_b)_j − y_b,j)²
L_opt    = Σ_b F_b                       (SUMMED over the batch, never averaged)
L_log    = (1 / B) · Σ_b F_b             (what the loss history reports)
```

Three decisions inside that:

* **`m_b` counts actual observed scalars.** Super-resolution uses the true low-resolution
  `y`, not the 256×256 bicubic guide. Inpainting counts observed entries × channels, per
  image — masks are drawn per image, so `m_b` genuinely differs across a batch (a verified
  example: `[11088, 10413, 9588]`). Stroke painting counts the rendered measurement's
  entries with no invented "effective degrees of freedom" correction.
* **Summed, not averaged, over the batch.** Reconstruction variables are independent per
  image, so summing keeps each sample's gradient invariant to batch size and to repeat
  padding. Averaging would make a batch-2 job and a batch-4 job optimise different problems.
* **No `1/σ²` factor.** A σ of 0.05 would inflate the objective 200-fold and move the useful
  `γ0` and `lr` by the same factor without changing anything scientific. `γ0` and `lr` carry
  that calibration, and they are swept. `gaussian_likelihood` remains available as an
  explicit choice — but for a noiseless problem it is **rejected**, at both validation and
  run time, instead of substituting a tiny ε and scaling the objective by ~1e16.

MPC's defaults are untouched by all of this: it still defaults to `half_sum_squared`, so
every Table E2 λ keeps the meaning it had. This was regression-tested.

---

## Cost accounting

The compute columns distinguish things that used to be one number:

| Column | Meaning |
|---|---|
| `model_evaluations` | every generative evaluation (velocity or transition) |
| `backprops_through_model` | backward passes through *generative* evaluations — **0 for PnP**, `trajectory × iterations` for D-Flow |
| `data_gradient_evaluations` | gradients of the data-fidelity term w.r.t. the state. For a latent model this differentiates the VAE decoder; that is a measurement-side gradient, **not** a generative backprop |
| `objective_evaluations` | evaluations of the fidelity scalar |
| `optimizer_iterations` | Adam updates. One iteration is **not** one network call — a D-Flow iteration traverses a whole trajectory |
| `denoiser_samples` | logical prior applications, including PnP's initial projection and each of its `M` realisations |

Expected values are computed at plan time (`--dry-run` shows them) and the measured values
sit next to them in `results.csv`, so a mismatch is visible rather than assumed away.

---

## GPU memory

Per atomic job: `gpu_baseline_gib` (steady state after load and warm-up, immediately before
the reconstruction), `gpu_peak_gib` (during it), `gpu_incremental_peak_gib` (the difference —
what the method itself added on top of a resident model), and `gpu_memory_source`.

* The measured region is **only** the reconstruction. Model loading, dataset preparation,
  metrics, visualisation and the untimed warm-up are outside it. PnP's initial projection
  and D-Flow's backward pass are inside it — they are part of the method.
* Numbers are **per job at its configured batch size** and are never divided by the batch.
  Activation memory does not decompose into an honest per-image figure. Run the final
  memory sweep at `batch_size: 1` if you want the single-image number.
* `gpu_memory_source` matters. A Torch allocator high-water mark and a **sampled NVML
  process peak** are not the same kind of measurement, and a sampled peak can miss a brief
  spike. The JAX path uses NVML sampling because `peak_bytes_in_use` is a *lifetime*
  high-water mark with no reset API — it cannot answer "what did this job peak at" once an
  earlier job peaked higher. Nothing in the profiler changes JAX's preallocation setting,
  allocator, compilation-cache policy or device lifecycle to obtain a number.
* If no source is available, the columns are empty and `gpu_memory_source` says
  `unavailable`. They are never filled with a guess.

---

## What to be careful claiming

1. PnP's MeanFlow denoiser has **no MMSE / conditional-mean guarantee**. Do not describe it
   as a denoiser in the Bayesian sense.
2. D-Flow at `t0 < 1` is **not** the published method. Say "intermediate-state" or
   "truncated".
3. `steps = 1` means different things for a velocity model and a MeanFlow model. Compare
   `model_evaluations`, runtime and memory, not `steps`.
4. `γ0`, `α`, `lr` and `num_opt_steps` are **untuned repository defaults**. Their provenance
   column literally says `repository_default_untuned`. A single-configuration comparison
   between a tuned method and an untuned one measures the tuning, not the method.
5. The step-matched SDEdit baseline pairs jobs by *closest step count*, which is a weak
   notion of matched cost across methods. That is why the runtime ratio and memory columns
   sit next to every delta.
