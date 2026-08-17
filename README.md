# SDEdit vs. MPC-Flow on ImageNet-256

A small research repository that asks one question:

> Given the **same** degraded observation, the **same** generative model, the **same**
> corruption strength `t0` and the **same** sampled generative noise, how much do
> **MPC-RHC** and **MPC-Δt** improve reconstruction relative to ordinary **SDEdit**, and
> what computational cost do those improvements require?

It is a merge of two Jupyter notebooks — an MPC-Flow inverse-problems notebook and a
unified SDEdit-style editing notebook — into one coherent codebase with a **single model
abstraction** and a **single inverse-problem abstraction** shared by every method.

The central invariant:

```
same problem + same model + same t0 + same epsilon
        ==>  the ONLY thing that varies is the reconstruction strategy
```

---

## 1. Supported models

| model | dynamics family | framework | state space | default |
|---|---|---|---|---|
| **JiT** (JiT-B/16) | standard flow (predicts x₁, velocity derived) | PyTorch | pixel `3×256×256` | **on** |
| **pMF** (pMF-L-16) | MeanFlow (finite-interval transition) | JAX | pixel `256×256×3` | **on** |
| SiT (SiT-XL/2) | standard flow (predicts velocity) | PyTorch | SD-VAE latent `4×32×32` | off |
| iMF (iMF-B-2) | MeanFlow | JAX | SD-VAE latent `32×32×4` | off |

SiT and iMF are **fully implemented and usable** — they are simply not part of the default
workload. `configs/experiments.yaml` shows how to switch them on; nothing else changes,
because the registry supplies the dynamics family.

pMF and iMF are never approximated as ordinary velocity models. Their dynamics are the
learned finite-interval transition `T_θ(x; s→r)`, and the MeanFlow-specific MPC logic is
preserved.

## 2. Six tasks

Each builds an inverse-problem instance from a clean ImageNet image `x*`, generally
`y = A(x*) + η`.

| task | `A` | measurement | guide `g(y)` |
|---|---|---|---|
| `denoising` | identity | `256×256` | `y` |
| `deblur` | separable Gaussian blur, reflect padding | `256×256` | `y` |
| `super_resolution` | `x[:, ::2, ::2, :]` — a **real** low-resolution operator | `128×128` | bicubic lift to `256×256` |
| `box_inpaint` | `M ⊙ x`, central box | `256×256` | `y` (zero-filled) |
| `random_inpaint` | `M ⊙ x`, random pixels dropped | `256×256` | `y` (zero-filled) |
| `stroke_painting` | `A_G(x)` — frozen SLIC stroke geometry, differentiable renderer | `256×256` | `y` |

Every operator is written **once** against a three-primitive backend shim and executed
identically by NumPy (to build `y`), PyTorch (JiT/SiT objectives) and JAX (pMF/iMF
objectives). Reflect padding is an index gather rather than each framework's own `pad`, so
the backends agree to ~1e-8 rather than merely approximately.

For super-resolution, `y` and `g(y)` are kept strictly separate: MPC's data-fidelity term
uses `A(x) − y` at `128×128`, while the bicubic `256×256` lift exists only to initialise.
**The bicubic upsample never replaces `A` inside the loss.**

## 3. Three reconstruction methods

All three start from the *same* SDEdit-style corrupted state

```
z_t0 = (1 − t0) · g(y) + t0 · ε
```

formed by one function (`models.base.build_initial_state`). At `t0 = 1` the implementation
returns the prior-noise array **itself**, so the pure-noise path is bitwise identical to the
notebooks' rather than merely numerically close.

- **`sdedit`** — ordinary generation from `s_start = 1 − t0` to the data endpoint. Standard
  flows use the Euler / Heun / RK4 integrator from the SDEdit notebook (Heun's final step
  falls back to Euler when the adapter advertises that policy, as JiT's official sampler
  does). MeanFlows use successive learned interval transitions; `steps` is the number of
  intervals and is always recorded explicitly. There is no measurement term — the
  measurement enters only through the guide.
- **`mpc_rhc`** — receding-horizon control (Algorithms 1 and 3). `K = 1` hoists the
  dynamics evaluation out of the graph, so it does **zero** backprop through the generative
  model.
- **`mpc_delta_t`** — Δt-horizon control with the one-step value surrogate (Algorithm 2).
  `K` is meaningless here and is **rejected** by the validator.

The MPC terminal objective is always evaluated in canonical pixel space,

```
Φ(x) = data-fidelity(A(x), y)
```

and the differentiable path

```
control → trajectory → native terminal state → to_pixels(differentiable=True) → A(x) → loss
```

contains no NumPy conversion, no `detach` and no PIL. For latent models the gradient
continues through the VAE decoder.

## 4. Installation (Colab)

```bash
!git clone <this-repo>
%cd <repo>
!bash setup_colab.sh
```

`setup_colab.sh` detects the accelerator, installs the JAX and/or PyTorch stacks in the
order that is known to work on Colab, repairs Pillow, and clones the model repositories at
their pinned revisions. It is idempotent.

> **One restart.** Installing JAX replaces shared libraries an already-running Python
> process has imported. If the JAX stack is (re)installed the script says so; restart the
> Colab runtime **once**, then continue. Re-running the script afterwards is a no-op.

Then:

```python
%run run.py --config configs/experiments.yaml
```

`%run` gives notebook-like behaviour: the resolved plan, progress, summary tables and
matplotlib figures all appear inline. There are no setup cells to execute by hand.

Non-Colab users can `pip install -r requirements.txt` after installing a PyTorch and/or
JAX build matching their hardware.

## 5. `.env` setup

ImageNet-1k is a **gated** Hugging Face dataset.

```bash
cp .env.example .env      # then set HF_TOKEN=hf_...
```

You must also accept the licence at
<https://huggingface.co/datasets/ILSVRC/imagenet-1k>. A Colab secret named `HF_TOKEN`, or an
exported `HF_TOKEN`, works too and always takes precedence over the file. `.env` is
gitignored; `.env.example` is committed; no real token is ever stored in the repository.

No token is needed if you use `data.source: local_folder`.

## 6. Configuration

Everything lives in `configs/experiments.yaml`. One entry per experiment:

```yaml
experiments:
  denoising:
    enabled: true
    problem: denoising
    num_images: 8
    degradation:
      sigma: 0.20
    models:
      jit:
        methods:
          sdedit:      {t0: [0.6, 0.8], steps: [25, 50], solver: heun}
          mpc_rhc:     {t0: [0.6, 0.8], num_mpc_steps: [4, 8], K: [1, 3],
                        lam: [0.05, 0.1], n_ctrl: 20, lr: 0.1}
          mpc_delta_t: {t0: [0.6, 0.8], num_mpc_steps: [4, 8],
                        lam: [5, 10], n_ctrl: 20, lr: 0.1}
      pmf:
        methods:
          sdedit:      {t0: [0.6, 0.8], steps: [1, 2]}
          mpc_rhc:     {t0: [0.6, 0.8], num_mpc_steps: [2, 4], K: [1, 3]}
          mpc_delta_t: {t0: [0.6, 0.8], num_mpc_steps: [2, 4]}
```

You never write `standard_flow:` or `meanflow:`. **You specify the model; the
implementation determines its family.**

### 7. Scalar / list sweeps

Every sweepable field accepts a scalar or a list; lists expand as a Cartesian product.

| scope | fields |
|---|---|
| shared | `t0` |
| `sdedit` | `steps`, `solver` |
| MPC | `num_mpc_steps`, `lam`, `n_ctrl`, `lr`, `optimizer`, `warm_start`, `grad_clip`, `phi_normalization`, `control_cost_normalization` |
| `mpc_rhc` only | `K` |
| `mpc_delta_t` only | `delta_t_lambda_scaling` |

The sweepable set is **closed**: any other list (e.g. `guidance.interval`) is a literal
value, never a sweep.

Validation is strict — unknown keys are errors, not silent no-ops:

```
n_ctrls: 20     ->  Unknown key 'n_ctrls' ... Did you mean: n_ctrl ?
K: 3 (sdedit)   ->  'K' is meaningless for sdedit and is rejected.
solver: heun (pMF) -> pMF is a MeanFlow model; ODE solvers do not apply.
```

`K` on `mpc_delta_t`, unknown models/methods/degradation parameters, `t0 ∉ (0,1]`,
non-positive step counts, mask fractions outside `(0,1)` and even blur kernels are all
rejected **before** any checkpoint is downloaded.

### 8. Dry run

```bash
python run.py --config configs/experiments.yaml --dry-run
```

validates, resolves defaults, expands sweeps, determines model resources, computes atomic
jobs, prints the plan and the warnings, and exits without loading anything:

```
denoising   (denoising, 2 images)
  JiT
    SDEdit .............. 2 jobs
    MPC-RHC ............. 1 jobs
    MPC-delta_t ......... 1 jobs
...
TOTAL ATOMIC JOBS: 48
```

`runtime.max_atomic_jobs` (default 256) is a guard against a sweep that accidentally
explodes; exceeding it is an error until you raise it deliberately (`--max-jobs`).

### 9. Normal run

```bash
python run.py --config configs/experiments.yaml
```

Useful flags: `--models jit`, `--experiments denoising,deblurring`, `--num-images 1`,
`--checks-only`, `--no-check`, `--replicate 1`, `--no-resume`, `--no-figures`,
`--no-warmup`, `--run-id <existing>`.

Execution is **model-major**: each checkpoint is loaded once, every job that needs it runs,
and it is released before the next model loads — which matters because JiT is Torch and pMF
is JAX and they should not be resident simultaneously.

## 10. Outputs

```
outputs/<run_id>/
├── config.yaml              the configuration as given
├── resolved_config.yaml     every resolved atomic job, with provenance
├── results.csv              one row per atomic job
├── results_per_image.csv    one row per (job, image)
├── results.jsonl            appended as each job finishes
├── run_metadata.json        provenance, checks, timing, warnings
├── checks.json
├── experiment_log.jsonl
├── denoising/<model>/<method>/<resolved-hyperparameters>/
│   ├── metadata.json        the full resolved spec + metrics + problem metadata
│   ├── results.npz          reconstructions (+ MPC loss history)
│   └── images/*.png
├── deblurring/ ... super_resolution/ ... box_inpainting/ ...
├── random_inpainting/ ... stroke_painting/
└── figures/
    ├── problem_instances.png
    ├── <task>_<model>_page_01.png   (paginated; nothing is silently omitted)
    ├── summary_by_method.png
    └── quality_vs_cost.png
```

Every atomic job is written **the moment it finishes**, so a Colab crash halfway through a
sweep never erases earlier reconstructions. With `runtime.resume: true` a rerun reuses a
finished job only when its *resolved spec* matches exactly.

### Figures

`problem_instances.png` sizes its grid to the number of images actually in the run.

`configurations_<model>.png` shows **one bar per resolved configuration**, per model, with
**no averaging across models, methods or hyperparameters** — a mean over a 4-step and a
25-step SDEdit, or over two lambdas, describes a run that was never executed. The only
averaging is over the images inside a single job, which is what `num_images` means. A dashed
line marks the degraded observation itself, so it is immediately visible whether a
reconstruction beat doing nothing. Bars group on the structural configuration (method,
steps/`N`/`K`); lambda is task-dependent via Table E2 and is annotated per bar.

`paired_deltas.png` shows each MPC job minus its **step-matched SDEdit baseline** — the
SDEdit job at the same task, model and t0 whose step count is closest — annotated with the
runtime multiple. Never against an average of SDEdit runs.

### Metrics

Every reconstruction reports **PSNR, SSIM, LPIPS and runtime per image**, per-image and
averaged. LPIPS is a headline metric and is not optional in a real run.

Secondary diagnostics: measurement consistency `RMSE(A(x̂) − y)` — important because MPC
explicitly optimises it while SDEdit does not — and, for the two inpainting tasks,
missing-region and observed-region errors separately, since full-image PSNR is otherwise
dominated by the already-observed pixels.

### Compute cost

Recorded per job: runtime, runtime/image, network forwards, model evaluations, planning
evaluations, control iterations, backprops through the model, reconstruction steps, batch
size and padding. The summary prints a paired table so you can say
*"MPC improved LPIPS by X but required Y× the runtime"*.

**Runtime definition.** The reported reconstruction runtime **includes** generative model
calls, integration, MPC planning, control optimisation, backward passes and controlled
execution. It **excludes** checkpoint download, model loading, dataset download, metrics,
visualisation and image saving — model loading is recorded separately. Before any timed
measurement, every `model × method × required shape` gets an **untimed warm-up**, so JAX
compilation never lands inside a measured run; the reported figure is steady-state runtime
and the warm-up cost is stored in its own column.

## 11. Scientific comparison assumptions

For every comparison the following are held constant, and only the reconstruction strategy
changes: source image, source class label, degradation operator and parameters, measurement
noise, inpainting mask, stroke geometry, initialisation guide, `t0`, generative noise `ε`,
model, and model conditioning.

**Shared randomness policy.** The generative-noise identity is

```
seed(global_seed, model, "x0", image_id, replicate)
```

and deliberately **does not** depend on the method, the solver, the number of steps, the MPC
hyperparameters, `K`, `λ`, the optimiser or the number of control iterations. This is a
deliberate change from the SDEdit notebook's seeding strategy, which folded solver and step
count into the noise identity and would therefore have made SDEdit-vs-MPC comparisons
unpaired. One `ε` per `(model, image, replicate)` serves every sweep point, so changing `t0`
moves along the *same* corruption direction:

```
z_0.6 = 0.4·g(y) + 0.6·ε        z_0.8 = 0.2·g(y) + 0.8·ε
```

`runtime.replicate` is the one deliberate way to draw a different `ε`.

Measurements are generated **once per problem instance** and cached across the whole sweep:
one noise draw, one mask, one low-resolution measurement, one stroke geometry — reused by
every model and method. Measurement noise for the inpainting tasks is masked, so unobserved
entries stay exactly zero in `y` and never become observations.

`src/checks.py` verifies these invariants programmatically rather than relying on developer
discipline: three-backend operator parity, gradient finiteness, `A(x*) ≈ y`, the noise
identity's independence, paired-fairness across method groups, and — measured on the loaded
model — that jobs differing only by method start from a **bit-identical** `z_t0`.

### The `t0` / MPC-step confound

If the trajectory from `t0` is divided into `N` intervals, `δ = t0/N` changes when `t0`
changes at fixed `num_mpc_steps`. The planner warns about this, and every result row records
`t0`, `num_mpc_steps` **and** `delta`. Equal step counts do not mean equal execution
resolution.

### Trajectory-discretisation matching

Where the algorithms discretise comparably, the default config matches SDEdit's `steps` to
the MPC methods' `num_mpc_steps` so the comparison is also step-matched. Where they do not
define "steps" the same way (pMF's learned intervals vs. an Euler grid), the difference is
recorded precisely rather than forced.

## 12. Class-conditioning assumption

JiT and pMF are ImageNet class-conditional, and every reconstruction is conditioned on the
**true ImageNet class of the source image** — identically for SDEdit and both MPC methods.
The benchmark should therefore be described as *reconstruction conditioned on the known
source class*, not as blind restoration. This is recorded in `run_metadata.json`.

## 13. Fixed-geometry stroke operator — caveat

The stroke task needs a differentiable forward operator, and the original SDEdit stroke
transform is not one: it uses SLIC segmentation, `np.where`, eigenvector selection,
percentiles, `uint8` conversion and PIL drawing.

So geometry and rendering are separated:

1. **Geometry `G` is extracted once**, with the original algorithm — same SLIC call, same
   `RandomState` draw order, same 80th-percentile half-length, same clipping, same PIL line
   rasterisation. Instead of the segment's mean colour, the rasteriser draws the segment's
   **index**, giving an exact owner map. This step is non-differentiable and does not need to
   be: it happens outside the MPC graph.
2. **`G` is then frozen**, and `A_G(x)` is a pure tensor function: a differentiable segment
   mean `c_j(x) = Σ M_j x / Σ M_j`, one gather that reproduces the base canvas *and* the
   stroke overwrite in the original drawing order, and a fixed separable Gaussian
   convolution replacing PIL's blur.

**The benchmark measurement is produced by the renderer**, `y = A_G(x*)` — not by the old
PIL transform — so `y` and the MPC objective use the same mathematical operator. The parity
check reports ≈51 dB agreement between the renderer and the original transform (the
remaining difference is `uint8` colour quantisation and PIL's box-approximated blur).
Additional tests confirm deterministic geometry for a fixed seed, identical geometry across
all methods, NumPy/Torch/JAX parity, non-zero gradients, and **no PIL/skimage call inside
the differentiable `apply()` path** (enforced with tripwires, not by inspection).

No measurement noise is added by default; the objective is `Φ(x) = ½‖A_G(x) − y‖²`.

> **Scientifically important:** unlike a random inpainting mask, which is independent of
> image content, the stroke geometry is **derived from the source image** and then frozen.
> It encodes `x*`'s own superpixel structure, so the stroke measurement carries more
> information about `x*` than its visual sparsity suggests. This is recorded in the
> problem metadata.

## 14. Deliberate changes from the source notebooks

Everything else was preserved; these are the changes, with reasons:

| change | why |
|---|---|
| Generative-noise identity no longer includes solver / steps / method / MPC hyperparameters | The SDEdit notebook's recipe would have given SDEdit and MPC different `ε`, making the headline comparison unpaired. |
| `standard_flow:` / `meanflow:` config branches removed | A model's dynamics family is a property of the model, not a user choice. The registry supplies it. |
| Classical inpainting prefill (Telea, Navier-Stokes) removed | Prefilling changes what the generative model is asked to do and would confound the SDEdit-vs-MPC comparison. The guide is the zero-filled observation. Attempting to configure it produces an explicit migration error. |
| One guide mode per problem, chosen by the problem | Removes a degree of freedom that could silently differ between methods. |
| `y` for stroke painting comes from the differentiable renderer, not the PIL transform | Otherwise the measurement and the MPC objective would be different operators. |
| Autograd stays enabled globally; SDEdit uses `torch.no_grad()` locally | The SDEdit notebook disabled autograd globally and used `inference_mode`; inference-mode tensors can never enter an autograd graph, and MPC needs one. Model parameters still keep `requires_grad_(False)`. |
| `diffusers` installed once at `>=0.36,<0.40` | The two notebooks installed it twice with different bounds; the intersection is what both stacks actually need. |
| Untimed warm-up before every timed run | Comparing a cold JAX-compiled first run against later compiled ones would have made the runtime column meaningless. |

## 15. Repository layout

```
configs/experiments.yaml   the only file you normally edit
src/
  config.py                registries, validation, sweeps, planner, Table E2 provenance
  data.py                  the shared ImageNet pool, stable image ids
  problems.py              ALL six degradations, guides, stroke geometry + renderer, Φ
  models/
    base.py                the one adapter interface, registry, ModelManager
    jit.py  pmf.py  sit.py  imf.py
  sdedit.py                ordinary reconstruction only (no degradation logic)
  mpc.py                   RHC and Δt, both families, plus the dispatcher
  metrics.py               PSNR, SSIM, LPIPS, measurement consistency, masked metrics
  visualization.py         paginated comparison grids, summary plots
  checks.py                parity, gradient, stroke and fairness tests
  utils.py                 canonical clock, seeding, backend shim, timing
run.py                     the notebook replacement / orchestrator
setup_colab.sh  requirements.txt  .env.example  .gitignore
```

## 16. Hyperparameter provenance

`λ`, `N_ctrl` and `lr` default to MPC-Flow **Appendix E.2**. Those values were tuned for a
**CelebA 128×128 pixel-space U-Net** with a particular loss normalisation — they are
*starting values* here, **not** tuned JiT/pMF hyperparameters, and `λ`'s balance further
depends on the dimension of each model's control space.

Every resolved job records where each value came from, in the
`hyperparameter_source_{lam,n_ctrl,lr}` columns:

- `paper_E2` — a value genuinely tabulated by the paper;
- `paper_E2_nearest_K(K=a->b)` — the paper tabulates RHC only for `K ∈ {1, 3}`; any other
  `K` falls back to the nearest tabulated one, with a prominent warning. **This is a
  fallback choice made by this repository, not a published hyperparameter for that `K`.**
- `repository_default` — no paper entry exists (stroke painting has no MPC-Flow analogue);
- `defaults` / `experiment` / `model` / `method` — you set it explicitly.

The objective normalisation is likewise explicit and selectable
(`half_sum_squared` — the default and the form whose gradient the paper writes down —
`sum_squared`, `mean_squared`, `gaussian_likelihood`), with control-cost normalisation
`sum_squared` or `mean_squared`. Changing either rescales `Φ` and therefore invalidates
Table E2's `λ`; the validator warns when you do.

## 17. Running on Alliance (Compute Canada) clusters

The repository runs unchanged on Colab **and** on Nibi, Narval and Rorqual. One cluster fact
shapes everything: **compute nodes have no internet access.** Every checkpoint, dataset and
git repository must be on disk before a job starts.

```bash
# LOGIN NODE (has internet, but a ~10 CPU-minute budget)
bash setup_cluster.sh                    # venv from Alliance wheels + stage all assets

# SUBMIT (also from the login node; sbatch returns immediately)
bash submit.sh                           # one GPU, whole config
bash submit.sh --array 8                 # 8 GPUs in parallel
python run.py --run-id run_<ID> --aggregate    # merge the shards
```

**No tracked file is ever edited per cluster.** `#SBATCH` lines are plain comments and
cannot read variables, but sbatch command-line flags override them -- so `submit.sh` passes
the account, GPU type and shard count from `cluster.env`, which is gitignored and written
for you by `setup_cluster.sh`. `git pull` therefore never conflicts with your local
settings. The GPU type is auto-detected (`narval` -> `a100:1`, `nibi`/`rorqual` ->
`h100:1`), and the account from `~/projects` when you have exactly one. Override anything
for a single submission:

```bash
bash submit.sh --time 6:00:00 --mem 96G --gpu h100:2
bash submit.sh --dry-run                 # print the sbatch command without submitting
```

The array script reads its shard count from `SLURM_ARRAY_TASK_COUNT` rather than a constant,
so `--array` and the shard count cannot drift out of sync and silently drop jobs.

What the code does for you:

- **Environment detection.** `detect_environment()` distinguishes Colab, a SLURM compute
  node, a cluster login node and a local machine. Under SLURM it enables offline mode
  automatically (`HF_HUB_OFFLINE` etc.), so a missing asset is an immediate, legible error
  instead of a multi-minute socket timeout buried in a job log. Override with
  `--online` / `--offline`. Progress bars switch off when stdout is not a TTY.
- **`--prefetch`.** Downloads model repositories, checkpoints, ImageNet images and LPIPS
  weights, then exits. Download-only: it builds no model and needs no GPU, so it respects
  the login-node budget. Writes `cache/prefetch_report.json`.
- **`--shard K/N`.** Splits the resolved plan across a SLURM job array. Shards are
  contiguous over a **model-major** ordering, so a task normally loads one checkpoint rather
  than paying pMF's ~2-minute load twice. Every task derives the same split from stable job
  ids, with no coordination. Tasks never write a shared file: each owns
  `results_shardKK.csv` and its own per-job directories.
- **`--aggregate`.** Merges the per-job `metadata.json` files into one `results.csv` plus
  figures. Reports any planned job that is missing rather than silently producing a partial
  table.
- **Path overrides.** `MPCFLOW_CACHE_ROOT` and `MPCFLOW_OUTPUT_ROOT` (or `--cache-root` /
  `--output-root`) keep the multi-gigabyte cache and outputs off the small, backed-up
  `/home` and on `$SCRATCH`. The job scripts stage the read-only cache to `$SLURM_TMPDIR`
  (node-local NVMe) for repeated checkpoint reads.

`setup_cluster.sh` installs from the Alliance wheelhouse (`pip install --no-index`) so torch
and jax link against the cluster's own CUDA/cuDNN, and falls back to PyPI only for what has
no wheel. It writes `activate_cluster.sh`, which the job scripts source. Anaconda is not
used — the Alliance does not permit it.

Cluster GPU specifiers: `--gpus-per-node=h100:1` on Nibi and Rorqual, `a100:1` on Narval.

## 18. JAX compilation and runtime hygiene

pMF and iMF run on JAX, which compiles per array shape. Three mechanisms keep compilation
out of the measured runtime:

1. **Untimed warm-up.** Before any timed job, the *exact* resolved trajectory is executed
   once with `n_ctrl = 1`. Reducing `steps`/`num_mpc_steps` in the warm-up would change the
   time grid, so the real job would meet uncompiled intervals and pay for them inside the
   measured region. The warm-up cache is keyed on everything that alters the traced graph
   (method, batch, problem, measurement shape, `K`, `t0`, `steps`, `num_mpc_steps`, solver,
   both normalisations).
2. **Compiled executables survive between jobs.** `jax.clear_caches()` is a *deep* hook that
   runs only when a model is released, never in the per-job cleanup path. Calling it between
   jobs discards every compiled executable and forces a full recompilation before each job —
   a measured 470 s of compilation for 26 s of work on a six-task run.
3. **Persistent on-disk cache.** Compiled executables are written under
   `cache/jax_compilation_cache/`, so rerunning the same benchmark recompiles nothing. Delete
   that directory if you change JAX or the model code.

The completion summary prints a per-model breakdown of load / warm-up / measured work, and
flags any model where compilation dominates. `warmup_seconds` and `model_load_seconds` are
columns in `results.csv`; neither is inside `runtime`.

`diagnose_pmf_timing.py` times a repeated interval against a new one and reports the
`jax.jit` cache size, if you need to confirm what a given machine is doing.

## 19. Known limitations

- The primary baseline is **paired SDEdit at the same `t0`**. A compute-matched SDEdit
  baseline (giving SDEdit as many network evaluations as MPC uses) would be interesting and
  is not implemented; the recorded evaluation counts make it straightforward to add.
- MeanFlow MPC-Δt uses the model's own transport to `s = 1` as the value-to-go. That is a
  **research extension**: the MPC-Flow paper does not propose or evaluate it. MeanFlow RHC
  likewise adapts the paper's algorithm to a learned interval transition.
- `t0 < 1` is a measurement-informed initialisation and an extension of MPC-Flow, which
  evaluates the pure-noise (`t0 = 1`) setting. Both are available; the initialisation kind
  is recorded in every row.
- The curated ImageNet list holds 32 classes. Use `data.source: local_folder` for a larger
  pool (with `labels.json`, or files named `<classid>_<name>.png`).
