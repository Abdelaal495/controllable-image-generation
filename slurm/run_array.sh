#!/bin/bash
# =====================================================================================
# Job ARRAY: split the atomic jobs across independent GPUs, then merge.
#
#     sbatch slurm/run_array.sh                     # 4 shards (see --array below)
#
# Each task runs `--shard K/N` over the SAME resolved plan. Shards are contiguous over a
# model-major ordering, so a task normally loads ONE checkpoint. Tasks never write the same
# file: each owns results_shardKK.csv plus its own per-job directories.
#
# Merge afterwards (login node, seconds, no GPU):
#     source activate_cluster.sh
#     python run.py --config configs/experiments.yaml --run-id run_<ARRAYJOBID> --aggregate
#
# EDIT --account and, if you like, --array.
# =====================================================================================
#SBATCH --job-name=mpcflow-array
#SBATCH --account=def-CHANGEME            # <-- your allocation
#SBATCH --array=0-3                       # N shards; must match NUM_SHARDS below
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mpcflow-%A_%a.out
#SBATCH --error=logs/mpcflow-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source activate_cluster.sh

NUM_SHARDS=4                              # keep in sync with --array=0-(N-1)
SHARD="${SLURM_ARRAY_TASK_ID}"
# Every task shares ONE run id so the shards land in the same directory and can be merged.
RUN_ID="run_${SLURM_ARRAY_JOB_ID}"

echo "=========================================================================="
echo "array $SLURM_ARRAY_JOB_ID task $SHARD/$NUM_SHARDS on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "run id: $RUN_ID"
echo "=========================================================================="

if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "$MPCFLOW_CACHE_ROOT" ]; then
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
  --run-id "$RUN_ID" \
  --shard "${SHARD}/${NUM_SHARDS}" \
  --no-figures                            # figures are built once, by --aggregate

echo "shard $SHARD done. Merge with:"
echo "  python run.py --config configs/experiments.yaml --run-id $RUN_ID --aggregate"
