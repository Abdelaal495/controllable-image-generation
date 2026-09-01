#!/usr/bin/env bash
# =====================================================================================
# Cluster-conditional module selection.  ONE generic file, no per-cluster copies.
#
# Sourced by setup_cluster.sh for BOTH of the places that must agree:
#   1. the module stack loaded while the virtualenv is built and verified;
#   2. the module stack written into the generated activate_cluster.sh.
#
# They agree because they call the SAME function here.  When those two drifted apart, a
# venv built against one CUDA and jobs run against another is the result -- and it is
# invisible until JAX prints a ptxas warning inside a paid GPU job.
#
# ------------------------------------------------------------------------------------
# NARVAL IS DELIBERATELY NOT LISTED BELOW.
#
# Narval already works with the Alliance DEFAULT (unversioned) CUDA/cuDNN modules, and
# "works" is worth more than "tidy". Any cluster that is not named here keeps exactly the
# previous behaviour, byte for byte:
#
#     module load cuda cudnn  ||  module load cuda  ||  (carry on without)
#
# ------------------------------------------------------------------------------------
# RORQUAL pins cuda/12.9 + cudnn/9.13.1.26 because:
#
#   * the unversioned default there resolves to CUDA 12.6.2 (ptxas V12.6.77), and JAX
#     warns at runtime that compilers up to 12.6.2 can miscompile some clamping edge
#     cases, recommending 12.6.3 or newer;
#   * `module spider cudnn/9.13.1.26` requires StdEnv/2023 + cudacore/.12.9.1, so it is
#     the cuDNN that pairs with cuda/12.9. The newer cudnn/9.21.1.3 requires CUDA 13.2
#     and is therefore NOT interchangeable here.
#
# To pin a different stack on some other cluster, either add a case below or export
# MPCFLOW_CUDA_MODULE / MPCFLOW_CUDNN_MODULE before running setup_cluster.sh.
# =====================================================================================

# The CUDA module to pin for a cluster, or "" to keep the Alliance default.
mpcflow_cuda_module() {
  if [ -n "${MPCFLOW_CUDA_MODULE:-}" ]; then printf '%s' "$MPCFLOW_CUDA_MODULE"; return; fi
  case "${1:-}" in
    rorqual) printf '%s' "cuda/12.9" ;;
    *)       printf '%s' "" ;;
  esac
}

# The cuDNN module to pin for a cluster, or "" to keep the Alliance default.
mpcflow_cudnn_module() {
  if [ -n "${MPCFLOW_CUDNN_MODULE:-}" ]; then printf '%s' "$MPCFLOW_CUDNN_MODULE"; return; fi
  case "${1:-}" in
    rorqual) printf '%s' "cudnn/9.13.1.26" ;;
    *)       printf '%s' "" ;;
  esac
}

# The exact shell line that loads the CUDA stack, for cluster $1.
#
# Pinned clusters still fall back to the unversioned modules if the pinned versions ever
# disappear from the module tree, so a module-tree change degrades to the old behaviour
# instead of failing the setup outright.
mpcflow_cuda_load_line() {
  local cuda cudnn
  cuda="$(mpcflow_cuda_module "${1:-}")"
  cudnn="$(mpcflow_cudnn_module "${1:-}")"
  if [ -n "$cuda" ]; then
    printf '%s' "module load ${cuda} ${cudnn} >/dev/null 2>&1 || module load ${cuda} >/dev/null 2>&1 || module load cuda cudnn >/dev/null 2>&1 || module load cuda >/dev/null 2>&1 || true"
  else
    printf '%s' "module load cuda cudnn >/dev/null 2>&1 || module load cuda >/dev/null 2>&1 || true"
  fi
}

# A one-line human description of what was selected, for the setup banner and the docs.
mpcflow_cuda_description() {
  local cuda cudnn
  cuda="$(mpcflow_cuda_module "${1:-}")"
  cudnn="$(mpcflow_cudnn_module "${1:-}")"
  if [ -n "$cuda" ]; then
    printf '%s' "${cuda} + ${cudnn}  (pinned for ${1:-unknown})"
  else
    printf '%s' "cuda + cudnn  (Alliance default for ${1:-unknown})"
  fi
}

# The full module preamble that activate_cluster.sh uses.  $1 = cluster, $2 = python version.
mpcflow_module_preamble() {
  cat <<PREAMBLE_EOF
module --force purge >/dev/null 2>&1
module load StdEnv/2023 >/dev/null 2>&1 || true
module load python/${2} >/dev/null 2>&1
$(mpcflow_cuda_load_line "${1:-}")
# arrow supplies pyarrow for datasets; it must be loaded BEFORE activating the venv.
module load gcc arrow >/dev/null 2>&1 || module load arrow >/dev/null 2>&1 || true
PREAMBLE_EOF
}
