#!/usr/bin/env bash
# =====================================================================================
# Environment setup for the SDEdit vs. MPC-Flow repository.
#
#     bash setup_colab.sh                 # detect the accelerator, install what is needed
#     bash setup_colab.sh --cpu           # force the CPU wheels
#     bash setup_colab.sh --no-jax        # JiT/SiT only (skip the JAX stack)
#     bash setup_colab.sh --no-torch      # pMF/iMF only
#     bash setup_colab.sh --skip-repos    # dependencies only
#     bash setup_colab.sh --force         # ignore the completion marker and reinstall
#
# This script is the notebooks' setup cells, extracted and made idempotent.  It writes a
# marker keyed on (accelerator, stack) so re-running it is cheap.
#
# IMPORTANT -- runtime restart.  Installing or upgrading JAX replaces shared libraries that
# a running Python process has already imported.  If the JAX stack is (re)installed, this
# script says so at the end and you must restart the Colab runtime ONCE before running
# anything.  Nothing else needs a restart, and re-running the script after a restart is a
# no-op because the marker is already there.
#
# Version policy: the pins below come from the notebooks that are known to work on Colab.
# Do not casually "modernise" them.  `diffusers` is installed once at the INTERSECTION of
# the two stacks' constraints (>=0.36,<0.40) rather than twice with different bounds.
# =====================================================================================
set -uo pipefail

CACHE_ROOT="${CACHE_ROOT:-cache}"
REPO_ROOT="${CACHE_ROOT}/repos"
WANT_JAX=1
WANT_TORCH=1
SKIP_REPOS=0
FORCE=0
ACCEL=""

for arg in "$@"; do
  case "$arg" in
    --cpu) ACCEL="cpu" ;;
    --gpu) ACCEL="gpu" ;;
    --tpu) ACCEL="tpu" ;;
    --no-jax) WANT_JAX=0 ;;
    --no-torch) WANT_TORCH=0 ;;
    --skip-repos) SKIP_REPOS=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

PY="${PYTHON:-python3}"
PIP="$PY -m pip"

# ------------------------------------------------------------------ accelerator detection
if [ -z "$ACCEL" ]; then
  if [ -n "${COLAB_TPU_ADDR:-}" ] || [ -n "${TPU_WORKER_ID:-}" ] || ls /dev/accel* >/dev/null 2>&1; then
    ACCEL="tpu"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    ACCEL="gpu"
  else
    ACCEL="cpu"
  fi
fi

STACK="jax${WANT_JAX}_torch${WANT_TORCH}"
MARKER="${CACHE_ROOT}/.setup_${ACCEL}_${STACK}"
mkdir -p "$CACHE_ROOT" "$REPO_ROOT"

echo "=============================================================================="
echo " SDEdit vs. MPC-Flow -- environment setup"
echo "   accelerator : ${ACCEL}"
echo "   JAX stack   : $([ $WANT_JAX -eq 1 ] && echo yes || echo no)   (pMF, iMF)"
echo "   Torch stack : $([ $WANT_TORCH -eq 1 ] && echo yes || echo no)   (JiT, SiT)"
echo "   cache root  : ${CACHE_ROOT}"
echo "=============================================================================="

NEEDS_RESTART=0

if [ -f "$MARKER" ] && [ $FORCE -eq 0 ]; then
  echo "Setup already completed for this configuration (${MARKER})."
  echo "Delete the marker or pass --force to reinstall."
else
  # ---------------------------------------------------------------- JAX ecosystem
  if [ $WANT_JAX -eq 1 ]; then
    echo
    echo "--- JAX ecosystem (pMF, iMF) ---------------------------------------------"
    case "$ACCEL" in
      tpu) $PIP install -q --upgrade "jax[tpu]" \
             -f https://storage.googleapis.com/jax-releases/libtpu_releases.html ;;
      gpu) $PIP install -q --upgrade "jax[cuda12]" ;;
      *)   $PIP install -q --upgrade "jax" ;;
    esac
    $PIP install -q --upgrade flax optax orbax-checkpoint \
        "tensorstore>=0.1.67" "ml-dtypes>=0.4.0"
    $PIP install -q --upgrade ml_collections pyyaml dm-tree cached_property absl-py
    NEEDS_RESTART=1
  fi

  # ---------------------------------------------------------------- PyTorch ecosystem
  if [ $WANT_TORCH -eq 1 ]; then
    echo
    echo "--- PyTorch ecosystem (JiT, SiT) -----------------------------------------"
    # Deliberately does NOT install or replace torch itself: Colab already ships a CUDA
    # build, and overwriting it with a CPU wheel would silently disable the GPU.
    $PIP install -q "transformers>=4.46" "accelerate>=1.1" \
        "huggingface_hub>=0.34" "safetensors>=0.4.5" "timm==0.9.16" "einops>=0.8.0"
    if ! $PY -c "import torch" >/dev/null 2>&1; then
      echo "  ! PyTorch is not importable in this runtime."
      echo "    JiT and SiT need it. On Colab, select a GPU runtime; otherwise install a"
      echo "    build matching your accelerator, e.g."
      echo "      pip install torch --index-url https://download.pytorch.org/whl/cu121"
    else
      echo "  torch present: $($PY -c 'import torch; print(torch.__version__)')"
    fi
  fi

  # ---------------------------------------------------------------- diffusers, once
  # Installed at the intersection of both stacks' constraints so a mixed run cannot end up
  # with a version only one of them tolerates.
  $PIP install -q "diffusers>=0.36,<0.40"

  # ---------------------------------------------------------------- shared utilities
  echo
  echo "--- Shared utilities ------------------------------------------------------"
  $PIP install -q --upgrade matplotlib scikit-image scipy tqdm python-dotenv datasets
  $PIP install -q lpips || echo "  ! lpips failed to install; the LPIPS columns will be empty."

  # In a JAX-only run torch is still imported by the iMF repository's VAE utilities; a CPU
  # wheel is fine there because no tensor ever reaches the accelerator.
  if [ $WANT_JAX -eq 1 ] && [ $WANT_TORCH -eq 0 ]; then
    if ! $PY -c "import torch" >/dev/null 2>&1; then
      $PIP install -q torch torchvision \
          --index-url https://download.pytorch.org/whl/cpu || true
    fi
  fi

  # ---------------------------------------------------------------- Pillow repair
  # A partially-removed Pillow leaves PIL importable while ImageDraw/ImageFilter are broken,
  # which silently breaks stroke-geometry extraction.  Force one clean, validated version.
  echo
  echo "--- Pillow repair ---------------------------------------------------------"
  $PIP uninstall -y Pillow >/dev/null 2>&1 || true
  $PY - <<'PYEOF' || true
import glob, os, shutil, site
roots = []
try: roots.extend(site.getsitepackages())
except Exception: pass
try: roots.append(site.getusersitepackages())
except Exception: pass
for root in roots:
    if not root or not os.path.isdir(root):
        continue
    for pattern in ("PIL", "Pillow-*.dist-info", "pillow-*.dist-info",
                    "Pillow-*.egg-info", "pillow-*.egg-info"):
        for path in glob.glob(os.path.join(root, pattern)):
            print("Removing residual:", path)
            shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
PYEOF
  $PIP install -q --no-cache-dir --force-reinstall "Pillow==12.3.0"
  if $PY -c "import PIL; from PIL import Image, ImageDraw, ImageFilter; print('  Pillow OK:', PIL.__version__)"; then
    :
  else
    echo "  ! Pillow is still inconsistent. Restart the runtime and re-run this script."
  fi

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MARKER"
  echo "accelerator=${ACCEL} stack=${STACK}" >> "$MARKER"
fi

# ------------------------------------------------------------------ model repositories
if [ $SKIP_REPOS -eq 0 ]; then
  echo
  echo "--- Model repositories (pinned revisions) --------------------------------"
  clone_at() {   # url dirname rev
    local url="$1" dir="${REPO_ROOT}/$2" rev="${3:-}"
    if [ ! -d "${dir}/.git" ]; then
      echo "  cloning $2"
      git clone --quiet "$url" "$dir" || { echo "  ! clone failed: $url"; return 1; }
    fi
    if [ -n "$rev" ]; then
      (cd "$dir" && git fetch --quiet --depth 1 origin "$rev" >/dev/null 2>&1 \
        && git checkout --quiet "$rev" >/dev/null 2>&1) || true
    fi
    echo "  $2 @ $(cd "$dir" && git rev-parse HEAD | cut -c1-12)"
  }
  if [ $WANT_JAX -eq 1 ]; then
    clone_at https://github.com/Lyy-iiis/pMF.git       pMF       75f6073042c21f7104686261a0c4784db4ede9d1
    clone_at https://github.com/Lyy-iiis/imeanflow.git imeanflow bf60cd7cb653f6628e59d48034b333c5eba445e2
  fi
  if [ $WANT_TORCH -eq 1 ]; then
    clone_at https://github.com/LTH14/JiT.git   JiT
    clone_at https://github.com/willisma/SiT.git SiT
  fi
  # run.py clones anything still missing at start-up, so this step is a convenience.
fi

# ------------------------------------------------------------------ Hugging Face token
echo
if [ -f .env ]; then
  echo "Found .env (HF_TOKEN is read from it, from the environment, or from Colab secrets)."
else
  echo "No .env found. ImageNet-1k is a GATED dataset:"
  echo "    cp .env.example .env     then put your token in HF_TOKEN="
  echo "  or set a Colab secret named HF_TOKEN, or export HF_TOKEN=..."
  echo "  You must also accept the licence at"
  echo "    https://huggingface.co/datasets/ILSVRC/imagenet-1k"
fi

echo
echo "=============================================================================="
if [ $NEEDS_RESTART -eq 1 ]; then
  echo " JAX was (re)installed: RESTART THE RUNTIME ONCE before running anything."
  echo "   Colab: Runtime -> Restart session."
  echo " Re-running this script afterwards is a no-op (the marker is already written)."
  echo
fi
echo " Next:"
echo "   python run.py --config configs/experiments.yaml --dry-run   # plan only"
echo "   python run.py --config configs/experiments.yaml             # full run"
echo "=============================================================================="
