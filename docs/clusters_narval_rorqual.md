# Narval and Rorqual: one repository, two clusters

This repository runs on more than one Digital Research Alliance of Canada cluster from a
**single shared Git checkout**. Nothing cluster-specific is committed. This document is the
reference for how that separation works and how to reproduce either cluster from scratch.

`docs/quickstart_cluster.md` is the step-by-step walkthrough; this file is the "why", plus
the exact Narval/Rorqual differences.

---

## 1. The separation

```text
shared tracked repository
    |
    +-- generic scientific code            src/, run.py, configs/
    +-- generic experiment configs         configs/experiments*.yaml
    +-- generic submit/setup machinery     setup_cluster.sh, submit.sh, slurm/
    |                                      scripts/cluster_modules.sh
    |
    +-- Narval behaviour                   GPU: A100
    |     the Alliance DEFAULT CUDA/cuDNN modules, unchanged
    |
    +-- Rorqual behaviour                  GPU: H100
          CUDA: cuda/12.9   cuDNN: cudnn/9.13.1.26

machine-specific LOCAL files (gitignored, never committed)
    cluster.env
    activate_cluster.sh
```

Everything that differs between clusters is selected **conditionally at setup time** by
`scripts/cluster_modules.sh`, or lives in a **local generated file**. There are no
`submit_narval.sh` / `submit_rorqual.sh` copies, and no tracked file needs editing per
cluster — which is what stops `git pull` from conflicting with a working installation.

---

## 2. Local, generated, gitignored files

Two files are written for you by `setup_cluster.sh` and are listed in `.gitignore`:

| File | What it is | Regenerate with |
|---|---|---|
| `activate_cluster.sh` | modules + venv activation + environment variables | `bash setup_cluster.sh --refresh` |
| `cluster.env` | your SLURM defaults: account, GPU, walltime, shards | edit by hand; only created if absent |

**Never commit either one.** They encode one machine's paths, one cluster's module names and
*your* allocation. Committing `cluster.env` would put an account name in the public history
and would break every other user's setup on the next pull.

`cluster.env` is where your local submission defaults live:

```bash
MPCFLOW_ACCOUNT=def-yourpi_gpu     # YOUR allocation. Never in a tracked file.
MPCFLOW_GPU=h100:1                 # a100:1 on Narval, h100:1 on Rorqual/Nibi
MPCFLOW_TIME=03:00:00
MPCFLOW_MEM=64G
MPCFLOW_CPUS=8
MPCFLOW_SHARDS=4
MPCFLOW_CONFIG=configs/experiments.yaml
```

`submit.sh` reads it and passes the values as `sbatch` command-line flags, which override
the `#SBATCH` defaults in `slurm/*.sh`. The job scripts themselves are never edited.

### Why Narval and Rorqual can share one repository

They have **different `cluster.env` files on different filesystems**. Your Narval home
directory holds a `cluster.env` saying `a100:1`; your Rorqual home directory holds one
saying `h100:1`. Neither is in Git, so neither can overwrite the other, and `git pull` on
either cluster touches neither. `cluster.env.example` is the tracked template; it contains
no account and no cluster-specific value.

The same applies to `activate_cluster.sh`: each cluster generates its own, with its own
module lines, and Git never sees either.

---

## 3. Narval — unchanged

| | |
|---|---|
| GPU family | **A100** |
| CUDA / cuDNN | the Alliance **default (unversioned)** `module load cuda cudnn` |
| `MPCFLOW_GPU` | `a100:1` |

Narval is **deliberately not listed** in `scripts/cluster_modules.sh`. Any cluster that is
not named there resolves to exactly the line the repository has always used:

```bash
module load cuda cudnn >/dev/null 2>&1 || module load cuda >/dev/null 2>&1 || true
```

This is asserted by a test (`tests/test_cluster_and_aggregation.py`), which compares the
resolved Narval line against that string **byte for byte** and separately asserts that
`12.9` never appears in it. If someone later pins a CUDA version globally, that test fails.

Narval therefore needs no special instructions: follow the shared workflow in §5.

Your account belongs in `cluster.env`, not in any tracked file.

---

## 4. Rorqual — cluster-specific

| | |
|---|---|
| GPU family | **H100 80 GB** |
| CUDA module | **`cuda/12.9`** |
| cuDNN module | **`cudnn/9.13.1.26`** |
| `MPCFLOW_GPU` | `h100:1` |

### Why the versions are pinned

The unversioned default on Rorqual resolves to **CUDA 12.6.2** (`ptxas V12.6.77`). During a
pMF/JAX GPU run, JAX warns that CUDA compilers **up to and including 12.6.2** can miscompile
certain clamping edge cases, and recommends 12.6.3 or newer. Rather than accept a compiler
JAX itself flags, the setup pins `cuda/12.9` (`ptxas V12.9.86`).

`cudnn/9.13.1.26` is the cuDNN that pairs with it: `module spider` reports it requires
`StdEnv/2023` + `cudacore/.12.9.1`. The newer `cudnn/9.21.1.3` requires CUDA 13.2 and is
**not** interchangeable here.

### How it is applied

`scripts/cluster_modules.sh` is the single source of truth, and `setup_cluster.sh` uses it
in **both** places that must agree:

1. the module stack loaded while the virtualenv is **built and verified**;
2. the module stack written into the generated **`activate_cluster.sh`**.

They cannot drift, because both call the same function. A generated Rorqual
`activate_cluster.sh` therefore loads CUDA 12.9 and cuDNN 9.13.1.26 rather than reverting to
the old unversioned default — which is precisely the failure the first smoke run hit.

The pinned form still falls back to the unversioned modules if the pinned versions ever
disappear from the module tree, so a module-tree change degrades to the previous behaviour
instead of failing setup outright.

To pin a stack on some other cluster without editing the file:

```bash
MPCFLOW_CUDA_MODULE=cuda/13.2 MPCFLOW_CUDNN_MODULE=cudnn/9.21.1.3 bash setup_cluster.sh
```

Your Rorqual account belongs in `cluster.env`. It is not committed anywhere.

---

## 5. Shared Alliance workflow

Identical on both clusters. **Steps 1–5 run on a LOGIN node** (compute nodes have no
internet); step 7 runs on a compute node.

```bash
# 1. clone (login node)
mkdir -p "$SCRATCH/controllable-image-generation" && cd "$_"
git clone <repo-url> repo && cd repo

# 2. token for the gated ImageNet dataset (skip for data.source: local_folder)
cp .env.example .env && nano .env          # HF_TOKEN=hf_...

# 3. build the environment AND stage every asset. LOGIN NODE.
bash setup_cluster.sh --venv "$HOME/controllable-image-generation-env"

# 4. activate
source activate_cluster.sh

# 5. prefetch anything a new config added (compute nodes are OFFLINE)
python run.py --config configs/experiments_theory_smoke.yaml --prefetch

# 6. check the resolved plan before spending GPU time
python run.py --config configs/experiments_theory_smoke.yaml --dry-run

# 7. submit
bash submit.sh --dry-run                   # print the sbatch command only
bash submit.sh --array 2 --config configs/experiments_theory_smoke.yaml

# 8. monitor
squeue -u "$USER"
sacct -j <ARRAY_JOB_ID> --format=JobID,State,ExitCode,Elapsed
tail -f logs/*<ARRAY_JOB_ID>*.out

# 9. aggregate the shards (login node, seconds, no GPU) -- REQUIRED
python run.py --config configs/experiments_theory_smoke.yaml \
              --run-id run_<ARRAY_JOB_ID> --aggregate
```

**Local benchmark folder.** With `data.source: local_folder`, the images are expected under
the configured benchmark directory. This repository ships two frozen selections:
`benchmarks/imagenet100_c42_i43/` (the paper's **100-image** benchmark, with manifest and
checksums) and `benchmarks/imagenet1000_c42_i43/` (the **1000-image** benchmark, one
validation image per class). Build each into its own folder on a login node before
submitting, e.g.
`python scripts/build_local_imagenet_pool.py --frozen-manifest benchmarks/imagenet1000_c42_i43/manifest.csv --out cache/data/imagenet1000_c42_i43_mirror`.
No HF token is needed in that mode, and step 5 stages only checkpoints and LPIPS weights.
The 1000-image configs take roughly 10x longer per job, and resume is per **finished** job,
so `--time` must cover the longest single job rather than the average.

---

## 6. GPU-less login nodes: what is expected, and what is a real failure

An Alliance login node has **no GPU and no GPU driver**. All three of the following are
**normal on a login node and are not setup failures**:

```python
torch.cuda.is_available()      # -> False
jax.default_backend()          # -> 'cpu', possibly after a CUDA_ERROR_NO_DEVICE message
pynvml.nvmlInit()              # -> NVMLError_DriverNotLoaded
```

The setup verification therefore distinguishes two different things:

| Question | Where it can be answered |
|---|---|
| Is the package/plugin **installed**? | login node — `import torch`, `import jax`, `import pynvml` |
| Is a GPU **visible**? | GPU compute node only |

In particular, `pynvml` **importing** on a login node is a useful dependency check and
nothing more. Whether NVML process-memory sampling actually *works* can only be verified
inside an allocated GPU job.

### Optional compute-node verification

```bash
salloc --account=<your-account> --gpus-per-node=h100:1 \
       --cpus-per-task=8 --mem=48G --time=0:20:00      # a100:1 on Narval
cd "$SCRATCH/controllable-image-generation/repo"
source activate_cluster.sh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch:', torch.cuda.is_available())"       # expect True
python -c "import jax; print('jax:', jax.default_backend(), jax.devices())" # expect gpu
python -c "import pynvml; pynvml.nvmlInit(); print('nvml devices:', pynvml.nvmlDeviceGetCount())"
ptxas --version | tail -2      # expect V12.9.x on Rorqual
exit
```

`jax: cpu` **inside a GPU job** is the dangerous one: JAX models still produce correct
numbers, 50–100× slower, so the job hits its wall clock instead of failing. Fix it from a
login node with `pip install --no-index jax_cuda12_plugin jax_cuda12_pjrt`.

---

## 7. Memory metrics: three different numbers

These are **not interchangeable** and must never be compared as if they were.

| Field | What it measures | Comparable across frameworks? |
|---|---|---|
| `gpu_process_peak_gib` | peak **PID-resident** GPU memory of this process during the measured reconstruction, sampled through NVML with a fresh sampler per atomic job | **Yes — this is the JiT-vs-pMF comparison metric** |
| `torch.cuda.max_memory_allocated` | PyTorch's own allocator high-water mark | No — PyTorch only |
| JAX `peak_bytes_in_use` | a **lifetime** high-water mark with no reset API | No — JAX only, and not per job |

`gpu_process_peak_gib` is a **job** peak at the configured batch size. It is never divided
by the batch, so it is not per-image memory.

When PID-level NVML memory cannot be obtained, `gpu_process_peak_gib` stays **null** and
`gpu_process_memory_source` reports why. It is *never* filled in from total device memory
(`nvmlDeviceGetMemoryInfo(...).used`) and never from a framework counter — a test asserts,
by parsing `src/memory.py`'s AST, that the device-total call is not present in the code.

If you see `gpu_process_memory_source = unavailable` for every job, the usual cause is a
missing `pynvml`. `setup_cluster.sh` now installs `nvidia-ml-py` automatically (Alliance
wheelhouse first, PyPI as a fallback).

---

## 8. Job arrays: shard outputs and aggregation

Array tasks share one run directory. Every **mutable top-level** file an array task writes
carries a `_shardNN` suffix, so no task can overwrite another's:

```text
results_shardNN.csv            results_per_image_shardNN.csv
results_shardNN.jsonl          experiment_log_shardNN.jsonl
checks_shardNN.json            run_metadata_shardNN.json
```

Per-job artefacts (`metadata.json`, `results.npz`, `images/`) live in per-spec directories
and are disjoint by construction. `config.yaml` / `resolved_config.yaml` are identical for
every task and are written once. Figures are skipped in array tasks — they have fixed
filenames — and are built once by `--aggregate`.

**Aggregation is required, not optional:**

```bash
python run.py --config <config> --run-id run_<ARRAY_JOB_ID> --aggregate
```

It produces the canonical, merged:

| File | Merge rule |
|---|---|
| `results.csv` | one row per finished job, ordered by plan position |
| `results_per_image.csv` | one row per (job, image); no duplicates, no dropped rows |
| `checks.json` | all shards' checks; **both** model families present; de-duplicated on (scope, name), and where shards disagree the **failing** result is kept |
| `run_metadata.json` | per-model dictionaries unioned; each shard's complete payload preserved under `shard_metadata` |
| `aggregate_metadata.json` | what was found, what is missing, which shards contributed |

plus per-job NPZ/metadata (untouched) and figures when not disabled.

Aggregation is **deterministic** (re-running it produces byte-identical CSVs) and tolerates
**partial completion**: missing jobs are listed explicitly and the merge proceeds as a
clearly-labelled partial result rather than failing or silently dropping data.

---

## 9. Updating an existing checkout after a `git pull`

**You do not need to delete or rebuild the virtualenv.**

```bash
# LOGIN node
cd "$SCRATCH/controllable-image-generation/repo"
git pull

# Install anything newly required and regenerate activate_cluster.sh,
# keeping the existing venv and without re-downloading assets:
bash setup_cluster.sh --refresh --venv "$HOME/controllable-image-generation-env"

source activate_cluster.sh
```

`--refresh` reuses the existing environment, installs only what is missing (currently
`nvidia-ml-py`), re-runs the import verification, and rewrites `activate_cluster.sh` with
the correct cluster-specific module lines. On Rorqual this is what switches the generated
activation from the old unversioned CUDA to `cuda/12.9` + `cudnn/9.13.1.26`.

`cluster.env` is **not** overwritten if it already exists, so your account and defaults
survive. Delete it first if you want it regenerated.

Only stage assets again if the configuration asks for something new:

```bash
python run.py --config <config> --prefetch
```
