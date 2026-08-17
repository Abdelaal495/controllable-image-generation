# Running on Alliance (Compute Canada) clusters

Tested on **Narval**; the same procedure applies to **Nibi** and **Rorqual**, and to any
other Alliance cluster with the same software stack.

Two parts: [first-time setup](#part-1--first-time-setup) (once per cluster, ~30 minutes)
and [everyday use](#part-2--everyday-use) (once set up).

Replace `<user>` with your Alliance username and `<account>` with your allocation
(`def-…`, `rrg-…` or `ctb-…`) throughout.

---

## The one fact that explains everything

**Compute nodes have no internet access.** Login nodes do.

```
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
mkdir -p $SCRATCH/mpcflow
cd $SCRATCH/mpcflow
git clone <your-repo-url> repo
cd repo
pwd        # $SCRATCH/mpcflow/repo
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

```bash
bash setup_cluster.sh
```

This builds a virtualenv from Alliance wheels (`pip install --no-index`, so torch and JAX
link against the cluster's own CUDA), installs the rest from PyPI, then downloads
everything the config needs. Expect **15–25 minutes**, mostly the ~2 GB of checkpoints.

It also writes two files for you:

* `activate_cluster.sh` — modules + venv + all environment variables. Job scripts source it.
* `cluster.env` — your account and GPU type. Gitignored, so `git pull` never conflicts.

**Verify before submitting anything.** A failure here is fixable; the same failure inside a
job wastes an allocation:

```bash
python -c "import json; r=json.load(open('$SCRATCH/mpcflow/cache/prefetch_report.json')); \
print('models:', {k: v['status'] for k, v in r['models'].items()}); \
print('data:  ', r['data']); print('lpips: ', r['lpips'])"
```

You want every model `ok`, a `data` entry with a `count`, and `lpips: ok`.

## 1.6 Confirm the GPU stack

Two things commonly go wrong and both are silent, so check explicitly. Grab an interactive
GPU for a few minutes:

```bash
salloc --account=<account> --gpus-per-node=a100:1 --cpus-per-task=8 --mem=48G --time=0:30:00
```

(`a100:1` on Narval; `h100:1` on Nibi and Rorqual.) Then:

```bash
cd $SCRATCH/mpcflow/repo
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

The header should read `Accelerator : gpu (…)` and `Environment : slurm_compute [narval] |
offline=True`. You want `ok=8 failed=0 skipped=0` and LPIPS as a number, not `n/a`.

Setup is now complete and never needs repeating.

---

# Part 2 — Everyday use

Everything here runs on a **login node**. `sbatch` submits and returns immediately; you do
not need an interactive session, and you can close your laptop afterwards.

## 2.1 Open a session

```bash
ssh <user>@narval.alliancecan.ca
cd $SCRATCH/mpcflow/repo
source activate_cluster.sh
```

## 2.2 Edit the experiment

`configs/experiments.yaml` is the only file you normally touch. Check the size of any change
before spending GPU time:

```bash
python run.py --config configs/experiments.yaml --dry-run
```

## 2.3 Stage anything new

Only needed when you increase `num_images` or add a model. It is a no-op otherwise:

```bash
python run.py --config configs/experiments.yaml --cache-root $SCRATCH/mpcflow/cache --prefetch
```

An image pool of N covers every run of N or fewer, so stage the largest you intend to use.

## 2.4 Submit

```bash
bash submit.sh --dry-run        # print the sbatch command, submit nothing
bash submit.sh                  # one GPU, whole config
bash submit.sh --array 8        # 8 GPUs in parallel
```

`submit.sh` reads `cluster.env` and passes the account, GPU type and shard count as sbatch
flags, which override the `#SBATCH` defaults — so **no tracked file is ever edited** and
`git pull` never conflicts. Override anything for one submission:

```bash
bash submit.sh --time 6:00:00 --mem 96G
bash submit.sh --array 16 --time 1:00:00
```

GPU type is auto-detected (`narval` → `a100:1`, `nibi`/`rorqual` → `h100:1`), and the
account from `~/projects` when you have exactly one.

## 2.5 Watch it

```bash
squeue -u $USER                          # PD = pending, R = running, empty = done
tail -f logs/mpcflow-<JOBID>.out         # Ctrl+C stops watching, not the job
tail -f logs/mpcflow-<JOBID>.err
grep "\->" logs/mpcflow-<JOBID>.out      # just the per-job results
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
scancel <JOBID>
```

The log file does not exist until the job starts. The first thing a job does is copy the
~2 GB cache to node-local NVMe (`staging cache -> …`); that is a single silent `cp` and
takes 1–4 minutes. Confirm progress with
`du -sh $SLURM_TMPDIR/cache` from another shell.

Milestones, in order: plan → structural checks → first model loads → its jobs → released →
next model → summary tables → figures.

## 2.6 Collect results

```bash
ls $SCRATCH/mpcflow/outputs/run_<JOBID>/
```

If you used `--array`, merge the shards first (login node, seconds, no GPU):

```bash
python run.py --config configs/experiments.yaml --run-id run_<ARRAY_JOB_ID> --aggregate
```

`<ARRAY_JOB_ID>` is the `%A` number shared by all tasks, visible in the log filenames
`logs/mpcflow-<A>_<a>.out`.

Preserve before `$SCRATCH` is purged:

```bash
mkdir -p ~/projects/<account>/$USER/results
cp -r $SCRATCH/mpcflow/outputs/run_<JOBID> ~/projects/<account>/$USER/results/
```

## 2.7 View figures

**Browser (no copying).** Alliance runs JupyterHub, e.g.
<https://jupyterhub.narval.alliancecan.ca>. Log in, request a small CPU-only session, then
browse to `scratch/mpcflow/outputs/` and click any PNG.

**Copy to your machine.** Tar first — one transfer beats hundreds of small files:

```bash
# on the cluster
cd $SCRATCH/mpcflow/outputs && tar czf ~/run_<JOBID>.tar.gz run_<JOBID>
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
transfers; it resumes on failure.

---

## Optional: skip the Duo prompt with an SSH key

Every `scp` otherwise costs a password plus a Duo passcode. A CCDB-registered key satisfies
the MFA requirement on its own.

```bash
ssh-keygen -t ed25519 -C "alliance" -f ~/.ssh/id_ed25519    # -f avoids a misplaced key
cat ~/.ssh/id_ed25519.pub
```

On Windows PowerShell:

```powershell
ssh-keygen -t ed25519 -C "alliance" -f "$HOME\.ssh\id_ed25519"
Get-Content $HOME\.ssh\id_ed25519.pub
```

Paste the public key at <https://ccdb.alliancecan.ca/ssh_authorized_keys>. It propagates in
a few minutes.

How it works: the pair is mathematically linked. CCDB copies the **public** key to
`~/.ssh/authorized_keys` on every cluster. On connect, the server sends a fresh random
challenge; your client signs it with the **private** key; the server verifies the signature
against the public key. The private key is never transmitted — the server only ever sees
signatures over random challenges, which only the matching private key could produce. That
is strictly stronger than a password, which is why it is accepted in place of Duo.

Only ever share the `.pub` file. If you set a passphrase, load it once per session:

```powershell
Start-Service ssh-agent; ssh-add $HOME\.ssh\id_ed25519
```

Verify: `ssh -v <user>@narval.alliancecan.ca 2>&1 | grep -i "Authenticated"` should mention
`publickey`.

---

## Reference

| Path | Contents |
|---|---|
| `$SCRATCH/mpcflow/repo` | the repository |
| `$SCRATCH/mpcflow/cache` | checkpoints, model repos, images, JAX compile cache |
| `$SCRATCH/mpcflow/outputs` | run directories |
| `$SCRATCH/mpcflow/repo/logs` | SLURM logs |
| `~/mpcflow-env` | virtualenv |

| Task | Command |
|---|---|
| Activate | `source activate_cluster.sh` |
| Plan only | `python run.py --config … --dry-run` |
| Stage assets | `python run.py --config … --prefetch` (login node) |
| Submit | `bash submit.sh` / `bash submit.sh --array N` |
| Merge shards | `python run.py --config … --run-id … --aggregate` |
| Queue | `squeue -u $USER` |
| Cancel | `scancel <JOBID>` |

Anything unexpected: see [`troubleshooting.md`](troubleshooting.md), which lists every
failure encountered while bringing this up on Narval, with its cause and fix.
