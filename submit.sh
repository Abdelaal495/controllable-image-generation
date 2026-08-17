#!/usr/bin/env bash
# =====================================================================================
# Submit a job without editing any tracked file.
#
#     bash submit.sh                  # one GPU, the whole configuration
#     bash submit.sh --array          # job array, MPCFLOW_SHARDS shards (default 4)
#     bash submit.sh --array 8        # job array, 8 shards
#     bash submit.sh --dry-run        # print the sbatch command and exit
#     bash submit.sh --time 6:00:00 --mem 96G      # override anything, once
#
# Cluster-specific settings live in `cluster.env`, which is GITIGNORED and written for you
# by setup_cluster.sh.  #SBATCH lines are plain comments and cannot read variables, but
# sbatch command-line flags OVERRIDE them -- which is what this wrapper uses, so the job
# scripts themselves never need editing and `git pull` never conflicts.
# =====================================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

[ -f cluster.env ] && . ./cluster.env

# ------------------------------------------------------------------ defaults
ACCOUNT="${MPCFLOW_ACCOUNT:-}"
GPU="${MPCFLOW_GPU:-}"
TIME="${MPCFLOW_TIME:-03:00:00}"
MEM="${MPCFLOW_MEM:-64G}"
CPUS="${MPCFLOW_CPUS:-8}"
SHARDS="${MPCFLOW_SHARDS:-4}"
CONFIG="${MPCFLOW_CONFIG:-configs/experiments.yaml}"
ARRAY=0
DRY=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --array)
      ARRAY=1
      if [ $# -ge 2 ] && [[ "${2:-}" =~ ^[0-9]+$ ]]; then SHARDS="$2"; shift; fi
      shift ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --gpu)     GPU="$2";     shift 2 ;;
    --time)    TIME="$2";    shift 2 ;;
    --mem)     MEM="$2";     shift 2 ;;
    --cpus)    CPUS="$2";    shift 2 ;;
    --config)  CONFIG="$2";  shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# ------------------------------------------------------------------ auto-detect
if [ -z "$GPU" ]; then
  case "${CC_CLUSTER:-}" in
    narval)        GPU="a100:1" ;;
    nibi|rorqual)  GPU="h100:1" ;;
    *)             GPU="1" ;;          # any GPU the scheduler offers
  esac
fi
if [ -z "$ACCOUNT" ] && [ -d "$HOME/projects" ]; then
  # Exactly one allocation is the common case; more than one is ambiguous, so ask.
  mapfile -t found < <(ls -1 "$HOME/projects" 2>/dev/null | grep -E '^(def|rrg|ctb)-')
  [ "${#found[@]}" -eq 1 ] && ACCOUNT="${found[0]}"
fi
if [ -z "$ACCOUNT" ]; then
  echo "ERROR: no allocation account. Set it once:"
  echo "    echo 'MPCFLOW_ACCOUNT=def-yourpi' >> cluster.env"
  echo "  or pass it:  bash submit.sh --account def-yourpi"
  [ -d "$HOME/projects" ] && echo "  Candidates: $(ls -1 "$HOME/projects" | tr '\n' ' ')"
  exit 2
fi

# ------------------------------------------------------------------ build the command
if [ $ARRAY -eq 1 ]; then
  SCRIPT="slurm/run_array.sh"
  # The job script reads SLURM_ARRAY_TASK_COUNT, so the shard count can never drift out of
  # sync with --array the way a hard-coded NUM_SHARDS could.
  CMD=(sbatch --account="$ACCOUNT" --gpus-per-node="$GPU" --cpus-per-task="$CPUS"
       --mem="$MEM" --time="$TIME" --array="0-$((SHARDS - 1))"
       --export=ALL,MPCFLOW_CONFIG="$CONFIG" "${EXTRA[@]}" "$SCRIPT")
else
  SCRIPT="slurm/run_single.sh"
  CMD=(sbatch --account="$ACCOUNT" --gpus-per-node="$GPU" --cpus-per-task="$CPUS"
       --mem="$MEM" --time="$TIME"
       --export=ALL,MPCFLOW_CONFIG="$CONFIG" "${EXTRA[@]}" "$SCRIPT")
fi

echo "cluster : ${CC_CLUSTER:-unknown}"
echo "account : $ACCOUNT"
echo "gpu     : $GPU        cpus: $CPUS   mem: $MEM   time: $TIME"
echo "config  : $CONFIG"
[ $ARRAY -eq 1 ] && echo "shards  : $SHARDS  (array 0-$((SHARDS - 1)))"
echo "command : ${CMD[*]}"
echo

if [ $DRY -eq 1 ]; then
  echo "--dry-run: nothing submitted."
  exit 0
fi
mkdir -p logs
"${CMD[@]}"
