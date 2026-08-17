# Troubleshooting

Every entry below is a failure that actually occurred while bringing this repository up,
with its cause and fix. Most are packaging or environment quirks rather than bugs in the
experiment code.

Jump to: [clusters](#alliance--compute-canada-clusters) ·
[Colab](#google-colab) · [transfers](#file-transfer-and-ssh) ·
[experiments](#experiment-level)

---

## Alliance / Compute Canada clusters

### `Failed to connect to github.com port 443` / `pip` hangs

You are on a **compute node**. Check the hostname in your prompt: `narval1`/`narval2` is a
login node, `ng10202` is compute. A shell opened by `salloc` is on a compute node, and those
have no internet.

Keep two terminals: one on a login node for `git`/`pip`/`--prefetch`, one on a compute node
for running. The filesystem is shared, so a pull on the login node is immediately visible
to the compute node.

### `ConnectionError: Couldn't reach 'ILSVRC/imagenet-1k' (OfflineModeIsEnabled)`

Offline mode is doing its job: the run needs images that are not staged. Prefetch a pool at
least as large as your `num_images`, from a **login node**:

```bash
python run.py --config configs/experiments.yaml --cache-root $SCRATCH/mpcflow/cache --prefetch
```

A pool of N satisfies any run of N or fewer, so stage the largest you plan to use.

### `Failed to build 'pyarrow-noinstall'` when installing `datasets`

The Alliance wheelhouse ships a **deliberately broken dummy** `pyarrow` whose only purpose
is to print instructions. `pyarrow` comes from a module, and **it must be loaded before the
virtualenv is activated**:

```bash
module --force purge
module load StdEnv/2023 python/3.11 gcc arrow/25.0.0
source ~/mpcflow-env/bin/activate
pip install --no-index datasets
python -c "import pyarrow, datasets; print(pyarrow.__version__, datasets.__version__)"
```

If `arrow/25.0.0` is absent, `module spider arrow` lists the versions.

A common variant: loading the modules only inside `setup_cluster.sh`. That happens in a
subshell and does not affect your interactive shell — load them yourself too.

**Alternative that avoids `datasets` entirely:** the image cache is one portable `.npz` of a
few hundred KB. Build it on Colab with `--prefetch`, download
`cache/data/imagenet_val_<N>_256.npz`, and copy it to the cluster's `cache/data/`. The
loader checks that file before touching `datasets`.

### `FileNotFoundError: .../lpips/weights/v0.1/alex.pth`

Not a download failure. The `lpips` package bundles a ~6 KB calibration file, and the
`lpips+computecanada` wheel omits it — the package imports fine, then dies on the missing
path. `--prefetch` repairs this automatically. Manually:

```python
python -c "from src.metrics import repair_lpips_weights; repair_lpips_weights()"
```

Note that `pip install --force-reinstall --no-deps lpips` reinstalls the *same* broken wheel,
because Alliance prepends its wheelhouse to pip's index. Use
`--index-url https://pypi.org/simple` if you want the real PyPI build.

### `jax: cpu` on a GPU node

The most costly failure mode, because it is silent: results stay correct but run 50–100×
slower, so the job hits its wall clock rather than failing. Modern JAX keeps CUDA support in
separate plugin packages that `pip install --no-index jax` does not pull in. From a **login
node**:

```bash
avail_wheels jax jaxlib jax_cuda12_plugin jax_cuda12_pjrt
pip install --no-index jax_cuda12_plugin jax_cuda12_pjrt
```

Verify on a compute node: `python -c "import jax; print(jax.default_backend())"` → `gpu`.
`run.py` prints a loud banner when it detects this.

### `ModuleNotFoundError: No module named 'wandb'`

The pMF and iMF repositories import `wandb` at module scope from their logging utilities,
even though this repository never logs. Recent versions install an inert stub automatically.
On older checkouts, `pip install wandb` on a login node also works.

### `JIT cannot run on tpu` when there is no TPU

Fixed. The accelerator probe used to treat `/dev/vfio/*` as a TPU indicator, but
`/dev/vfio/vfio` is the standard VFIO container device present on any host with IOMMU
enabled. `git pull` for the fix, or force it: `runtime.accelerator: gpu`.

### `Fatal Python error: PyGILState_Release ... state: finalizing`

Harmless if it appears **after** a completion message. A hundred independently built native
extensions (pyarrow, torch, jax, aiohttp, PIL) tear down in an order none of them agreed on.
It occurs strictly after the work is done and written. `run.py` now exits before interpreter
finalization so it no longer turns a successful run into a non-zero exit code.

### `Your local changes would be overwritten by merge`

You edited a tracked file, usually `slurm/*.sh`, and the upstream version changed the same
lines. Those files no longer need editing — `submit.sh` supplies the account and GPU:

```bash
git checkout -- slurm/run_array.sh slurm/run_single.sh
git pull
```

Put your settings in `cluster.env` (gitignored) instead.

### `ERROR: no allocation account`

`submit.sh` auto-detects from `~/projects` only when you have exactly one. Otherwise:

```bash
echo 'MPCFLOW_ACCOUNT=def-yourpi' >> cluster.env
```

### Job seems stuck after `staging cache -> …`

Normal. That copies ~2 GB from the shared filesystem to node-local NVMe with a single silent
`cp`; it takes 1–4 minutes. Confirm with `du -sh $SLURM_TMPDIR/cache` from another shell —
the number should grow.

Staging speeds up repeated checkpoint reads but keeps the JAX compilation cache inside the
copy, so each job recompiles (~40 s). For short jobs it is a net loss; comment out the
staging block in `slurm/run_single.sh` if you prefer.

### `hwloc_set_cpubind() failed: Invalid argument`

Cosmetic. JAX tries to pin threads to NUMA nodes inside a cgroup that only grants a slice of
the node. Ignore it.

### `Couldn't access the Hub … Defaulting to existing file`

Offline mode working as designed — it found the cached checkpoint and used it.

---

## Google Colab

### `jaxlib` errors immediately after setup

You skipped the restart. Installing JAX replaces shared libraries the running process has
already imported. `Runtime → Restart session`, then continue after the setup cell. Needed
once; re-running the setup is a no-op afterwards.

### Gated-dataset error / missing token

Add a Colab secret named `HF_TOKEN` (🔑 in the sidebar) with notebook access enabled, and
accept the licence at <https://huggingface.co/datasets/ILSVRC/imagenet-1k>. The code checks
Colab secrets, then the environment, then `.env`.

### Broken `PIL` / `ImageDraw` errors

A partially removed Pillow leaves `PIL` importable while `ImageDraw` and `ImageFilter` are
broken, which breaks stroke-geometry extraction. `setup_colab.sh` repairs this; if it
recurs, restart the runtime and re-run it.

### Session disconnects mid-run

Nothing is lost. Every atomic job is written the moment it finishes. Re-run with
`--run-id <existing>` and finished jobs are reused rather than recomputed.

### Out of memory on a T4

T4s have 16 GB and no BF16. Reduce `num_images`, set `models.<name>.batch_size: 1`, enable
fewer experiments, or avoid large planning horizons (`K`, `num_mpc_steps`), which hold more
of the control graph in memory.

---

## File transfer and SSH

### `scp: mkdir ~/Desktop/…: No such file or directory` (Windows)

PowerShell passes `~` through literally for **local** paths. Use `$HOME` or `.`, and create
the directory first:

```powershell
mkdir $HOME\Desktop\figures -Force
scp -r <user>@narval.alliancecan.ca:/scratch/<user>/mpcflow/outputs/run_<ID>/figures .
```

`~` on the **remote** side is fine — the remote shell expands it.

### `ssh-keygen` put the key somewhere unexpected

Typing a bare name at the "Enter file in which to save the key" prompt creates a *relative*
path, so the key lands in your current directory. Always pass `-f`:

```powershell
ssh-keygen -t ed25519 -C "alliance" -f "$HOME\.ssh\id_ed25519"
Get-Content $HOME\.ssh\id_ed25519.pub
```

Delete any misplaced pair, especially from a cloud-synced folder. Only the `.pub` file is
ever shared.

---

## Experiment level

### Every reconstruction is flagged `!` (worse than the degraded input)

Not a bug — a finding. The `!` marks a reconstruction whose PSNR is below the observation
itself. At `t0: 0.8` you destroy 80% of the signal, and most strategies cannot rebuild it in
a handful of steps; the generative prior invents plausible content that is not the ground
truth, which PSNR punishes.

Sweep `t0: [0.3, 0.5, 0.6, 0.8]` to find where each strategy crosses that line. That is the
more informative experiment.

### LPIPS column is empty

`lpips` is missing or its bundled weights are (see above). It is a headline metric — fix it
rather than proceeding. `metrics.lpips: false` disables it deliberately, and the planner
warns when you do.

### Results differ slightly between identical runs

PyTorch models are bit-identical. JAX models drift ~0.005 LPIPS from nondeterministic
reduction ordering. If a comparison you care about is smaller than that, run
`runtime.replicate: 1`, `2`, … to bound it.

### `padded N` in the job output

Fixed-batch models (pMF, iMF) compile for one shape, so a short final chunk is repeat-padded.
Padded rows carry their own measurement and control and are dropped before metrics, so they
cannot influence results — but they do mean a 1-image job costs a full batch of compute.

### The plan has more jobs than expected

Sweep lists expand as a Cartesian product. `--dry-run` shows the resolved count;
`runtime.max_atomic_jobs` (default 256) stops a runaway. Raise it deliberately with
`--max-jobs` if you mean it.

### A model failed to load and its jobs are `skipped`

Other models still run — `continue_on_experiment_error` defaults to true. The failure and
its error are recorded in `results.csv`, never silently dropped. Fix the cause and re-run
with `--run-id <existing>`; completed jobs are reused.
