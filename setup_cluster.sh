#!/usr/bin/env bash
# =====================================================================================
# Environment setup for Digital Research Alliance of Canada (Compute Canada) clusters:
# Nibi, Narval, Rorqual (and any other Alliance cluster with the same software stack).
#
#     bash setup_cluster.sh                    # build the venv, then prefetch assets
#     bash setup_cluster.sh --venv-only        # build the venv, skip downloads
#     bash setup_cluster.sh --prefetch-only    # assume the venv exists, just download
#     bash setup_cluster.sh --python 3.12      # pick a different Python module
#     bash setup_cluster.sh --venv $SCRATCH/env  # put the venv somewhere else
#
# RUN THIS ON A LOGIN NODE. It needs the internet, and compute nodes have none.
# It is light enough for the login-node policy (a few CPU-minutes; the heavy part is I/O).
#
# Three cluster facts drive everything here:
#   1. Compute nodes have NO internet. Every checkpoint, dataset and git repository must be
#      staged first -- that is what the prefetch step does.
#   2. Alliance ships its own optimised wheels. `pip install --no-index` uses them and
#      avoids the CUDA/cuDNN dependency mess; PyPI is only a fallback for what is missing.
#   3. Anaconda is not permitted. This uses `virtualenv` on top of a `python` module.
# =====================================================================================
set -uo pipefail

PY_VERSION="3.11"
VENV=""
DO_VENV=1
DO_PREFETCH=1
CONFIG="configs/experiments.yaml"

while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY_VERSION="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --venv-only) DO_PREFETCH=0; shift ;;
    --prefetch-only) DO_VENV=0; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ where are we?
CLUSTER="${CC_CLUSTER:-unknown}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: this script is running inside a SLURM job (job ${SLURM_JOB_ID})."
  echo "       Compute nodes have no internet. Run it on a LOGIN node instead."
  exit 1
fi
if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: no 'module' command found. This script is for Alliance clusters."
  echo "       On Colab or a personal machine use setup_colab.sh instead."
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SCRATCH:=$HOME/scratch}"
VENV="${VENV:-$HOME/mpcflow-env}"
# /home is small and backed up; caches and outputs are large and regenerable -> $SCRATCH.
CACHE_ROOT="${MPCFLOW_CACHE_ROOT:-$SCRATCH/mpcflow/cache}"
OUTPUT_ROOT="${MPCFLOW_OUTPUT_ROOT:-$SCRATCH/mpcflow/outputs}"

echo "=============================================================================="
echo " SDEdit vs. MPC-Flow -- Alliance cluster setup"
echo "   cluster      : ${CLUSTER}"
echo "   repository   : ${REPO_DIR}"
echo "   virtualenv   : ${VENV}"
echo "   cache root   : ${CACHE_ROOT}     (checkpoints, repos, images)"
echo "   output root  : ${OUTPUT_ROOT}"
echo "   python module: python/${PY_VERSION}"
echo "=============================================================================="

mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"

# ------------------------------------------------------------------ virtual environment
if [ $DO_VENV -eq 1 ]; then
  echo
  echo "--- [1/2] Virtual environment ---------------------------------------------"
  module --force purge >/dev/null 2>&1
  module load StdEnv/2023 >/dev/null 2>&1 || true
  module load "python/${PY_VERSION}" || { echo "ERROR: no python/${PY_VERSION} module."; \
      echo "Available:"; module spider python 2>&1 | head -20; exit 1; }
  # cuda/cudnn are what the Alliance JAX and PyTorch wheels link against.
  module load cuda cudnn >/dev/null 2>&1 || module load cuda >/dev/null 2>&1 || \
      echo "  (no cuda module loaded; CPU-only wheels will still work)"
  # The 'arrow' module provides pyarrow, which `datasets` requires. The Alliance wheelhouse ships a
  # DUMMY pyarrow wheel that fails on purpose and tells you to load this module instead --
  # and it MUST be loaded BEFORE the virtualenv is activated, or `pip install datasets`
  # dies with "Failed to build 'pyarrow-noinstall'".
  if module load gcc arrow >/dev/null 2>&1 || module load arrow >/dev/null 2>&1; then
      echo '  arrow module loaded (provides pyarrow, which datasets needs)'
  else
      echo '  ! no arrow module found: datasets will not install, so the gated ImageNet'
      echo "    download will fail. Use data.source: local_folder, or run"
      echo "    'module spider arrow' and pass the version you find."
  fi

  if [ ! -d "$VENV" ]; then
    echo "  creating $VENV"
    virtualenv --no-download "$VENV" || exit 1
  else
    echo "  reusing existing $VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --no-index --upgrade pip

  echo
  echo "  Alliance wheels (--no-index): optimised builds, no CUDA dependency hell"
  # Order matters: torch and jax first so nothing else drags in a PyPI CUDA stack.
  pip install --no-index torch torchvision || \
      echo "  ! torch wheel unavailable -- JiT and SiT will not run"
  pip install --no-index jax flax optax orbax-checkpoint || \
      echo "  ! jax stack unavailable -- pMF and iMF will not run"
  pip install --no-index numpy scipy matplotlib pillow scikit-image pyyaml tqdm \
      pandas h5py || true
  # Available as Alliance wheels on most clusters; harmless if they fall through to PyPI.
  pip install --no-index huggingface_hub safetensors transformers datasets \
      ml_collections absl-py || true

  echo
  echo "  Remaining packages from PyPI (login nodes have internet)"
  for pkg in "diffusers>=0.36,<0.40" "timm==0.9.16" "einops>=0.8.0" "lpips>=0.1.4" \
             "python-dotenv>=1.0" "dm-tree>=0.1.8" "cached_property>=1.5"; do
    pip install "$pkg" || echo "  ! failed: $pkg"
  done
  # These four are needed only if a wheel was missing above.
  python -c "import datasets" 2>/dev/null || pip install "datasets>=2.19" || true
  python -c "import ml_collections" 2>/dev/null || pip install ml_collections || true

  echo
  echo "  Verifying:"
  python - <<'PYEOF'
import importlib
for name, label in [("numpy", "numpy"), ("yaml", "PyYAML"), ("PIL", "Pillow"),
                    ("skimage", "scikit-image"), ("matplotlib", "matplotlib"),
                    ("torch", "torch"), ("jax", "jax"), ("optax", "optax"),
                    ("flax", "flax"), ("diffusers", "diffusers"), ("timm", "timm"),
                    ("lpips", "lpips"), ("datasets", "datasets"),
                    ("huggingface_hub", "huggingface_hub"),
                    ("ml_collections", "ml_collections"), ("pyarrow", "pyarrow (arrow module)")]:
    try:
        module = importlib.import_module(name)
        print("    %-18s %s" % (label, getattr(module, "__version__", "ok")))
    except Exception as exc:
        print("    %-18s MISSING (%s)" % (label, type(exc).__name__))
PYEOF
  echo
  echo "  NOTE: torch/jax report no GPU on a login node -- login nodes have none."
  echo "        Device visibility is checked inside the job, not here."
  if ! python -c "import datasets" >/dev/null 2>&1; then
    echo
    echo "  !! 'datasets' is MISSING. It needs pyarrow, which comes from the arrow module."
    echo "     Fix with:"
    echo "         deactivate"
    echo "         module load gcc arrow"
    echo "         source ${VENV}/bin/activate"
    echo "         pip install --no-index datasets"
    echo "     Without it the gated ImageNet download cannot run."
  fi
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || { echo "ERROR: no venv at $VENV"; exit 1; }
fi

# ------------------------------------------------------------------ prefetch
if [ $DO_PREFETCH -eq 1 ]; then
  echo
  echo "--- [2/2] Staging assets for offline compute nodes ------------------------"
  cd "$REPO_DIR" || exit 1
  if [ ! -f .env ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "  ! No .env and no HF_TOKEN. ImageNet-1k is GATED and the download will fail."
    echo "    cp .env.example .env   then set HF_TOKEN=hf_...   and re-run with"
    echo "    bash setup_cluster.sh --prefetch-only"
  fi
  # Keep the Hugging Face cache with everything else on $SCRATCH: /home is small and the
  # checkpoints are gigabytes.
  export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
  mkdir -p "$HF_HOME" "$TORCH_HOME"
  python run.py --config "$CONFIG" --cache-root "$CACHE_ROOT" --prefetch
fi

# ------------------------------------------------------------------ activation snippet
ACTIVATE="$REPO_DIR/activate_cluster.sh"
cat > "$ACTIVATE" <<EOF
# Generated by setup_cluster.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ) for cluster ${CLUSTER}.
# Source this from a job script or an interactive session:
#     source $(basename "$ACTIVATE")
module --force purge >/dev/null 2>&1
module load StdEnv/2023 >/dev/null 2>&1 || true
module load python/${PY_VERSION} >/dev/null 2>&1
module load cuda cudnn >/dev/null 2>&1 || module load cuda >/dev/null 2>&1 || true
# arrow supplies pyarrow for datasets; it must be loaded BEFORE activating the venv.
module load gcc arrow >/dev/null 2>&1 || module load arrow >/dev/null 2>&1 || true
source ${VENV}/bin/activate
export MPCFLOW_CACHE_ROOT="${CACHE_ROOT}"
export MPCFLOW_OUTPUT_ROOT="${OUTPUT_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
# Compute nodes have no internet: fail fast instead of blocking on a socket.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# JAX should not preallocate the whole device: PyTorch may share the process.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
EOF
echo
echo "=============================================================================="
echo " Wrote ${ACTIVATE}"
echo
echo " Next -- interactive smoke test (one GPU, 30 minutes):"
echo "   salloc --account=def-YOURPI --gpus-per-node=1 --cpus-per-task=8 \\"
echo "          --mem=48G --time=0:30:00"
echo "   source activate_cluster.sh"
echo "   python run.py --config ${CONFIG} --dry-run"
echo "   python run.py --config ${CONFIG} --num-images 1 --experiments denoising"
echo
echo " Then -- batch:        sbatch slurm/run_single.sh"
echo "         job array:    sbatch slurm/run_array.sh"
echo "=============================================================================="
