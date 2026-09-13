# Mean Flows as Controllable Priors for Solving Inverse Problems

Training-free inverse-problem solvers on pretrained flow priors, compared under one invariant:

```
same measurement + same model + same start time t0 + same generative noise
        ==>  the only thing that varies is the reconstruction strategy
```

MeanFlow models expose a learned finite-interval transport `T(x; t -> s)`, so the clean
sample reached from any intermediate state is one network evaluation away.  This repository
uses that interface for existing solvers and for **Receding-Horizon State Optimization
(RHSO)**, which optimises the current state through its predicted clean sample, advances one
interval, and replans.

## What is compared

| strategy | reference | backprop through the prior |
|---|---|---|
| `sdedit` | SDEdit-style generation from the corrupted state | no |
| `pnp` | PnP-Flow (ICLR 2025) | no |
| `dflow` | D-Flow (ICML 2024) | whole trajectory |
| `mpc_rhc`, `mpc_delta_t` | MPC-Flow, receding-horizon and delta-t variants | planning horizon |
| `rhso` | this paper | one terminal prediction per stage |

| model | family | framework | state space |
|---|---|---|---|
| `jit` (JiT-B/16) | standard flow | PyTorch | pixel |
| `pmf` (pMF-L/16) | MeanFlow | JAX | pixel |
| `sit` (SiT-XL/2) | standard flow | PyTorch | SD-VAE latent |
| `imf` (iMF-B-2) | MeanFlow | JAX | SD-VAE latent |

Inverse problems: `denoising`, `deblur`, `super_resolution` (true 2x subsampling),
`random_inpaint`, `box_inpaint`, `stroke_painting`.  Every operator is written once against a
backend shim and evaluated identically in NumPy, PyTorch and JAX.  Metrics: PSNR, SSIM, LPIPS,
measurement RMSE, runtime and evaluation counts per job.

## Setup

```bash
git clone <this repository> && cd controllable-image-generation
bash setup_colab.sh            # installs torch / jax for the detected accelerator and the model repos
pip install -r requirements.txt
```

On Alliance clusters (Narval, Nibi, Rorqual) use `setup_cluster.sh` and `submit.sh` instead;
see [`docs/quickstart_cluster.md`](docs/quickstart_cluster.md).  A Colab walkthrough is in
[`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb).

**Data.**  Small runs read ImageNet-1k validation images from Hugging Face; the dataset is
gated, so put a token in `.env` (`cp .env.example .env`).  Larger runs use a local pool:

```bash
python scripts/build_local_imagenet_pool.py --frozen-manifest benchmarks/imagenet100_c42_i43/manifest.csv
```

rebuilds the paper's frozen 100-image benchmark (`benchmarks/imagenet100_c42_i43/` holds the
manifest and checksums) from an ungated mirror into `cache/data/imagenet100_c42_i43_mirror`.

## Running

```bash
python run.py --config configs/experiments.yaml --dry-run   # validate and print the job plan
python run.py --config configs/experiments.yaml             # run it
```

A config lists experiments as `problem x model x method` with hyperparameters; any list value
is swept as a Cartesian product.  Useful flags: `--models jit,pmf`, `--experiments denoising`,
`--num-images 4`, `--no-figures`, `--run-id NAME`, `--shard K/N` with `--aggregate` for job
arrays.  Every finished job is written immediately under `outputs/<run_id>/` (`results.csv`,
`results_per_image.csv`, per-job `metadata.json`, reconstructions and figures), and reruns
resume from finished jobs.

## Reproducing the paper

The frozen-100 benchmark (Table 1) is a three-stage protocol at `t0 = 1`:

```bash
# Stage 1: screening grid, 4 images, one process per model
python run.py --config configs/experiments_hpo_full_pmf.yaml --run-id hpo_pmf --no-figures
# Stage 2: top three per cell + manuscript values + grid-edge extensions, 8 images
python scripts/hpo_stage2.py --run outputs/hpo_pmf --only-model pmf --out configs/experiments_hpo_stage2_pmf.yaml
python run.py --config configs/experiments_hpo_stage2_pmf.yaml --run-id stage2_pmf --no-figures
python scripts/hpo_stage2_round.py --model pmf --stage1 outputs/hpo_pmf --stage2 outputs/stage2_pmf --out configs/experiments_hpo_stage2_pmf_r2.yaml
# Final: the Stage-2 winner inside the Stage-1 grid, 100 frozen images
python scripts/make_final_frozen100_configs.py --run outputs/stage2_pmf --model pmf --grid-only outputs/hpo_pmf --out configs/experiments_final_frozen100_pmf.yaml
python run.py --config configs/experiments_final_frozen100_pmf.yaml --run-id final_pmf --no-figures
# Tables and paired tests
python scripts/make_final_tables.py --run outputs/final_pmf ... --out results/final/tables.md
python scripts/make_results_tex.py  --run outputs/final_pmf ... --out results/final/results.tex
python scripts/paired_tests.py      --run outputs/final_pmf ... --metric lpips --ref rhso
```

The final configs for all four models are committed, and the CSVs of every stage are under
`results/` so the selection can be audited without rerunning:
[`docs/best_hyperparameters.md`](docs/best_hyperparameters.md) lists the winner of every cell
and how the manuscript's values rank.  The RHSO ablations of the appendix (stage count,
noise strength, state anchor, schedule, initialisation time) are
`configs/experiments_theory_*.yaml`, documented in
[`docs/theory_validation_changelog.md`](docs/theory_validation_changelog.md).  Qualitative
figures come from `scripts/make_paper_figures.py`, the 2-D illustration from
[`toy/`](toy/README.md).

## Repository layout

```
run.py                    entry point: validate, plan, run, aggregate
src/                      problems, schedule, strategies (sdedit, pnp, dflow, mpc, rhso), metrics, checks
src/models/               adapters for JiT, SiT (PyTorch) and pMF, iMF (JAX)
configs/                  experiments.yaml (default), Stage-1 grids, final benchmark, theory ablations
scripts/                  pool builder, HPO stage generators, table / LaTeX / figure scripts
results/                  Stage-1, Stage-2 and final CSVs, tables.md, results.tex, paired tests
benchmarks/               manifest and checksums of the frozen ImageNet-100 benchmark
docs/                     method notes, cluster guides, troubleshooting, extension guide
tests/                    executable checks against analytic models (no GPU, no checkpoints)
toy/                      RHSO on a two-moons prior in 2-D
slurm/, submit.sh         job scripts for Slurm clusters
```

## Documentation

* [`docs/schedule_and_rhso.md`](docs/schedule_and_rhso.md): the shared power-law time
  schedule, RHSO, and its diagnostics
* [`docs/methods_pnp_dflow.md`](docs/methods_pnp_dflow.md): what was adapted in PnP-Flow and
  D-Flow to fit the shared-state comparison
* [`docs/extending.md`](docs/extending.md): adding a strategy, a model or a task
* [`docs/troubleshooting.md`](docs/troubleshooting.md),
  [`docs/clusters_narval_rorqual.md`](docs/clusters_narval_rorqual.md)

## Tests

```bash
python tests/test_beta_schedule.py          # pure Python
python tests/test_rhso.py                   # jax + optax
python tests/test_meanflow_pnp_dflow.py     # jax + optax
```

Each script prints `N/N checks passed`; see [`tests/README.md`](tests/README.md) for the full list.
