#!/bin/bash
# =====================================================================================
# One GPU, one job: the whole configuration end to end.
#
#     sbatch slurm/run_single.sh
#
# EDIT --account. Everything else has a working default.
# Run `bash setup_cluster.sh` on a LOGIN NODE first: this job has no internet.
# =====================================================================================
#SBATCH --job-name=mpcflow
#SBATCH --account=def-CHANGEME            # <-- your allocation, e.g. def-yoursupervisor
#SBATCH --gpus-per-node=1                 # Nibi/Rorqual: h100:1   Narval: a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mpcflow-%j.out
#SBATCH --error=logs/mpcflow-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source activate_cluster.sh                # written by setup_cluster.sh

echo "=========================================================================="
echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "SLURM_TMPDIR=$SLURM_TMPDIR"
echo "=========================================================================="

# Stage the read-only cache onto node-local NVMe. The checkpoints are read repeatedly and
# $SCRATCH is a shared parallel filesystem; $SLURM_TMPDIR is local and fast. Skip this (and
# drop the --cache-root override) if the copy costs more than it saves.
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "$MPCFLOW_CACHE_ROOT" ]; then
  echo "staging cache -> $SLURM_TMPDIR/cache"
  cp -r "$MPCFLOW_CACHE_ROOT" "$SLURM_TMPDIR/cache"
  RUN_CACHE="$SLURM_TMPDIR/cache"
  export HF_HOME="$RUN_CACHE/huggingface"
  export TORCH_HOME="$RUN_CACHE/torch"
else
  RUN_CACHE="$MPCFLOW_CACHE_ROOT"
fi

python run.py \
  --config configs/experiments.yaml \
  --cache-root "$RUN_CACHE" \
  --output-root "$MPCFLOW_OUTPUT_ROOT" \
  --run-id "run_${SLURM_JOB_ID}"

echo "results: $MPCFLOW_OUTPUT_ROOT/run_${SLURM_JOB_ID}"
