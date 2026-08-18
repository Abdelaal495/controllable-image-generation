# Running Controllable Image Generation for Inverse Problems on Alliance clusters

Tested on **Narval**; the same procedure applies to **Nibi** and **Rorqual**, and to any
other Digital Research Alliance of Canada cluster with the same software stack.

This repository benchmarks several reconstruction/control strategies — **SDEdit, MPC-RHC,
MPC-Δt, PnP-Flow, and D-Flow** — across standard Flow Matching and MeanFlow models. The
cluster workflow is therefore described here in method-neutral terms rather than as an
"MPC-Flow" workflow.

Two parts: [first-time setup](#part-1--first-time-setup) (once per cluster, ~30 minutes)
and [everyday use](#part-2--everyday-use) (once set up).

Replace `<user>` with your Alliance username and `<account>` with your allocation
(`def-…`, `rrg-…` or `ctb-…`) throughout.

> **Compatibility note.** Some shell scripts still use legacy internal environment-variable
> names beginning with `MPCFLOW_`. Those names are part of the current script interface, so this
> guide leaves them intact where they are technically required. They should be read as legacy
> implementation identifiers, not as a description of the repository's present scope.
>
> This guide uses the neutral storage root
> `$SCRATCH/controllable-image-generation` and the virtualenv
> `~/controllable-image-generation-env`. The setup commands below explicitly pass those paths,
> so they remain consistent with the existing scripts.

---

## The one fact that explains everything

**Compute nodes have no internet access.** Login nodes do.

```text
  login node                        compute node
  narval1, narval2, …               ng10202, ng30604, …
  internet   YES                    internet   NO
  GPU        NO                     GPU        YES
  use for    git, pip, downloads    use for    running experiments

            └──── /home, /scratch, /project: THE SAME FILES ────┘
```

Every download — checkpoints, ImageNet images, LPIPS weights, git repositories — happens
once on a login node via `run.py --prefetch`. Jobs then run fully offline. The filesystem
is shared, so a `git pull` on the login node is instantly visible to a running job.

A shell inside `salloc` is on a **compute node**, so `git pull` and `pip install` will hang
there. That is expected, not a fault.

---

# Part 1 — First-time setup

## 1.1 Find your account and check your quota

```bash
ssh <user>@narval.alliancecan.ca        # or nibi. / rorqual.

ls ~/projects                            # your allocation(s), e.g. def-smith
echo $SCRATCH                            # e.g. /scratch/<user>
diskusage_report
```

## 1.2 Clone into `$SCRATCH`

`/home` is ~50 GB, backed up, and slow for heavy I/O. Caches and outputs are gigabytes and
regenerable, so they belong on `$SCRATCH`.

```bash
mkdir -p "$SCRATCH/controllable-image-generation"
cd "$SCRATCH/controllable-image-generation"
git clone https://github.com/Abdelaal495/controllable-image-generation repo
cd repo
pwd        # $SCRATCH/controllable-image-generation/repo
```

> `$SCRATCH` is **purged** after ~60 days without access. Copy finished runs to
> `~/projects/<account>/$USER/` when you are done (§2.6).

## 1.3 Hugging Face token

ImageNet-1k is gated. Accept the licence at
<https://huggingface.co/datasets/ILSVRC/imagenet-1k>, create a token at
<https://huggingface.co/settings/tokens>, then:

```bash
cp .env.example .env
nano .env          # HF_TOKEN=hf_...
```

`.env` is gitignored. Skip this entirely if you use `data.source: local_folder`.

## 1.4 Load modules — order matters

```bash
module --force purge
module load StdEnv/2023 python/3.11 gcc arrow/25.0.0 cuda cudnn
```

Two details that cost real time if you get them wrong:

* **`arrow` must be loaded before the virtualenv is activated.** The Alliance wheelhouse
  ships a *dummy* `pyarrow` wheel that fails on purpose and tells you to load this module.
  `datasets` depends on `pyarrow`, and `datasets` is what downloads ImageNet.
* **Load the modules in your own shell, not only inside a script.** `setup_cluster.sh` loads
  them in a subshell; those changes do not survive back to your interactive shell.

If `arrow/25.0.0` does not exist, run `module spider arrow` and pick an available version.

## 1.5 Build the environment and stage every asset

Set neutral cache/output paths, then run the repository's setup script:

```bash
export MPCFLOW_CACHE_ROOT="$SCRATCH/controllable-image-generation/cache"
export MPCFLOW_OUTPUT_ROOT="$SCRATCH/controllable-image-generation/outputs"

bash setup_cluster.sh --venv "$HOME/controllable-image-generation-env"
```

`MPCFLOW_CACHE_ROOT` and `MPCFLOW_OUTPUT_ROOT` are the current legacy environment-variable
names expected by the scripts; the actual paths above are method-neutral.

The setup script builds a virtualenv from Alliance wheels (`pip install --no-index`, so
PyTorch and JAX link against the cluster's own CUDA), installs the remaining dependencies
from PyPI, then downloads everything required by the current experiment configuration.
Expect **15–25 minutes**, mostly checkpoint transfer.

It also writes two files for you:

* `activate_cluster.sh` — modules + venv + environment variables. Job scripts source it.
* `cluster.env` — allocation/GPU/submission defaults. It is gitignored, so `git pull` does
  not conflict with local cluster settings.

**Verify before submitting anything.** A failure here is fixable; the same failure inside a
job wastes an allocation:

```bash
python -c "import json; r=json.load(open('$SCRATCH/controllable-image-generation/cache/prefetch_report.json')); \
print('models:', {k: v['status'] for k, v in r['models'].items()}); \
print('data:  ', r['data']); print('lpips: ', r['lpips'])"
```

You want every requested model `ok`, a `data` entry with a `count`, and `lpips: ok`.

## 1.6 Confirm the GPU stack

Two things commonly go wrong and both are silent, so check explicitly. Grab an interactive
GPU for a few minutes:

```bash
salloc --account=<account> --gpus-per-node=a100:1 --cpus-per-task=8 --mem=48G --time=0:30:00
```

(`a100:1` on Narval; `h100:1` on Nibi and Rorqual.) Then:

```bash
cd "$SCRATCH/controllable-image-generation/repo"
source activate_cluster.sh

python -c "import torch; print('torch cuda:', torch.cuda.is_available())"
python -c "import jax; print('jax:', jax.default_backend(), jax.devices())"
```

Both must report a GPU. **`jax: cpu` is the dangerous one** — JAX-based models still give
correct numbers, just 50–100× slower, so a job hits its wall clock instead of failing.
Modern JAX keeps CUDA support in separate plugin packages. Fix from a **login node**:

```bash
avail_wheels jax jaxlib jax_cuda12_plugin jax_cuda12_pjrt
pip install --no-index jax_cuda12_plugin jax_cuda12_pjrt
```

`run.py` prints a loud banner if it detects this at run time.

## 1.7 Smoke test

Still inside `salloc`:

```bash
python run.py --config configs/experiments.yaml --dry-run
python run.py --config configs/experiments.yaml --num-images 1 --experiments denoising
exit
```

The header should report a GPU accelerator and a SLURM compute-node environment with
`offline=True`. Check that the structural/model checks complete successfully and that LPIPS
is reported as a number rather than `n/a`.

Because the repository now contains five reconstruction strategies, the exact number of
individual checks/jobs may differ from older MPC-only documentation. Treat the resolved
`--dry-run` plan as the source of truth.

Setup is now complete and normally does not need repeating.

---

# Part 2 — Everyday use

Everything here runs on a **login node**. `sbatch` submits and returns immediately; you do
not need an interactive session, and you can close your laptop afterwards.

## 2.1 Open a session

```bash
ssh <user>@narval.alliancecan.ca
cd "$SCRATCH/controllable-image-generation/repo"
source activate_cluster.sh
```

## 2.2 Edit the experiment

`configs/experiments.yaml` is the main file you normally edit. It controls the participating
models, inverse problems, methods, and method-specific hyperparameter sweeps.

Check the resolved workload before spending GPU time:

```bash
python run.py --config configs/experiments.yaml --dry-run
```

This is especially important now that lists may sweep parameters for SDEdit, MPC, PnP-Flow,
and D-Flow and expand as a Cartesian product.

## 2.3 Stage anything new

Only needed when the current configuration requests an asset that has not already been
staged — for example, more ImageNet examples or an additional model:

```bash
python run.py \
  --config configs/experiments.yaml \
  --cache-root "$SCRATCH/controllable-image-generation/cache" \
  --prefetch
```

An image pool of N covers every run of N or fewer, so stage the largest you intend to use.

## 2.4 Submit

```bash
bash submit.sh --dry-run        # print the sbatch command, submit nothing
bash submit.sh                  # one GPU, whole config
bash submit.sh --array 8        # 8 shards in parallel
```

`submit.sh` reads `cluster.env` and passes the account, GPU type and shard count as `sbatch`
flags, which override the `#SBATCH` defaults — so **no tracked file is ever edited** and
`git pull` never conflicts.

Override resources for one submission when needed:

```bash
bash submit.sh --time 6:00:00 --mem 96G
bash submit.sh --array 16 --time 1:00:00
```

GPU type is auto-detected (`narval` → `a100:1`, `nibi`/`rorqual` → `h100:1`), and the
allocation is auto-detected from `~/projects` when you have exactly one.

If auto-detection cannot identify your allocation, either pass it explicitly:

```bash
bash submit.sh --account <account>
```

or set the legacy variable expected by the current submission script in `cluster.env`:

```bash
echo 'MPCFLOW_ACCOUNT=<account>' >> cluster.env
```

## 2.5 Watch it

```bash
squeue -u "$USER"                         # PD = pending, R = running, empty = done
tail -f logs/*<JOBID>*.out               # Ctrl+C stops watching, not the job
tail -f logs/*<JOBID>*.err
grep "\->" logs/*<JOBID>*.out            # concise per-job result lines, when present
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
scancel <JOBID>
```

The wildcard form deliberately avoids depending on the current legacy SLURM log prefix.

The log file does not exist until the job starts. The first thing a job may do is copy the
cache to node-local NVMe (`staging cache -> …`); that is a single silent `cp` and can take a
few minutes. Confirm progress with:

```bash
du -sh "$SLURM_TMPDIR/cache"
```

from another shell on the allocated node.

Typical milestones are: plan → structural checks → first model loads → its reconstruction
jobs → model release → next model → summary tables → figures.

Depending on the configuration, the job list can now include SDEdit, MPC-RHC, MPC-Δt,
PnP-Flow, and D-Flow.

## 2.6 Collect results

```bash
ls "$SCRATCH/controllable-image-generation/outputs/run_<JOBID>/"
```

If you used `--array`, merge the shards first on a login node (seconds, no GPU):

```bash
python run.py \
  --config configs/experiments.yaml \
  --run-id run_<ARRAY_JOB_ID> \
  --aggregate
```

`<ARRAY_JOB_ID>` is the `%A` number shared by all tasks. You can find the matching files with:

```bash
ls logs/*<ARRAY_JOB_ID>*
```

Preserve important runs before `$SCRATCH` is purged:

```bash
mkdir -p ~/projects/<account>/$USER/results
cp -r "$SCRATCH/controllable-image-generation/outputs/run_<JOBID>" \
      ~/projects/<account>/$USER/results/
```

## 2.7 View figures

**Browser (no copying).** Alliance runs JupyterHub, e.g.
<https://jupyterhub.narval.alliancecan.ca>. Log in, request a small CPU-only session, then
browse to:

```text
scratch/controllable-image-generation/outputs/
```

and open the generated PNGs.

The repository may produce comparison grids, method summaries, quality-vs-cost plots, and
quality-vs-memory plots depending on the completed jobs.

**Copy to your machine.** Tar first — one transfer beats hundreds of small files:

```bash
# on the cluster
cd "$SCRATCH/controllable-image-generation/outputs"
tar czf ~/run_<JOBID>.tar.gz run_<JOBID>
```

```bash
# on your machine
scp <user>@narval.alliancecan.ca:~/run_<JOBID>.tar.gz .
tar xzf run_<JOBID>.tar.gz
```

On **Windows PowerShell**, `~` is not expanded for local paths — use `$HOME` or `.`:

```powershell
scp <user>@narval.alliancecan.ca:~/run_<JOBID>.tar.gz .
```

**Globus** (<https://www.globus.org>, collection `computecanada#narval`) is better for large
transfers because it resumes interrupted copies.

---

## Optional: skip repeated Duo prompts with an SSH key

Every `scp` otherwise costs a password plus a Duo passcode. A CCDB-registered key can satisfy
the MFA requirement on its own.

```bash
ssh-keygen -t ed25519 -C "alliance" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

On Windows PowerShell:

```powershell
ssh-keygen -t ed25519 -C "alliance" -f "$HOME\.ssh\id_ed25519"
Get-Content $HOME\.ssh\id_ed25519.pub
```

Paste the public key at <https://ccdb.alliancecan.ca/ssh_authorized_keys>. It propagates in
a few minutes.

The public/private key pair is mathematically linked. CCDB copies the **public** key to
`~/.ssh/authorized_keys` on the clusters. On connection, the server asks the client to sign
a fresh challenge; the signature is verified against the public key. The private key is
never transmitted.

Only ever share the `.pub` file. If you set a passphrase, load the key once per session:

```powershell
Start-Service ssh-agent
ssh-add $HOME\.ssh\id_ed25519
```

Verify with:

```bash
ssh -v <user>@narval.alliancecan.ca 2>&1 | grep -i "Authenticated"
```

and confirm that authentication mentions `publickey`.

---

## Reference

| Path | Contents |
|---|---|
| `$SCRATCH/controllable-image-generation/repo` | repository checkout |
| `$SCRATCH/controllable-image-generation/cache` | checkpoints, model repositories, images, JAX compilation cache |
| `$SCRATCH/controllable-image-generation/outputs` | run directories |
| `$SCRATCH/controllable-image-generation/repo/logs` | SLURM logs |
| `~/controllable-image-generation-env` | virtualenv |

| Task | Command |
|---|---|
| Activate | `source activate_cluster.sh` |
| Plan only | `python run.py --config … --dry-run` |
| Stage assets | `python run.py --config … --prefetch` (login node) |
| Submit | `bash submit.sh` / `bash submit.sh --array N` |
| Merge shards | `python run.py --config … --run-id … --aggregate` |
| Queue | `squeue -u $USER` |
| Cancel | `scancel <JOBID>` |

### Legacy script identifiers

The current shell layer still uses several `MPCFLOW_*` variables internally, including
`MPCFLOW_CACHE_ROOT`, `MPCFLOW_OUTPUT_ROOT`, `MPCFLOW_ACCOUNT`, and `MPCFLOW_CONFIG`.
They are retained for backward compatibility. Renaming those variables properly requires a
coordinated update to `setup_cluster.sh`, `submit.sh`, the SLURM scripts, `cluster.env.example`,
and the relevant environment-detection code — not a documentation-only change.

Anything unexpected: see [`troubleshooting.md`](troubleshooting.md), which records the
cluster/environment failures encountered while bringing this repository up and their fixes.
