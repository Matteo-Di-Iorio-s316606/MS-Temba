#!/usr/bin/env bash
# =========================================================
#  IMPORTANT: this file must be SOURCED, not executed.
#  Usage:  source activate_mstemba.sh
# =========================================================
#
# To be SOURCED (not executed):  source activate_mstemba.sh
#
# Activates the mstemba_v2 conda env, loads CUDA, and exports REPO.
# Idempotent: safe to source multiple times in the same shell.

# Refuse to be executed (vs sourced): if executed, $0 is the script path;
# if sourced, $0 is the parent shell name (e.g. "-bash" or "bash").
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: this script must be SOURCED, not executed." >&2
    echo "  Use:  source $0" >&2
    exit 1
fi

# NB: no `set -e` because conda's own scripts can return non-zero in
# corner cases without it being fatal. We rely on the failure messages
# below if something is wrong.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mstemba_v2
module load cuda/11.8.0_gcc-10.4.0

export REPO="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"

# Confirm
echo "[activate] conda env : ${CONDA_DEFAULT_ENV:-<none>}"
echo "[activate] python    : $(command -v python)"
echo "[activate] CUDA_HOME : ${CUDA_HOME:-<unset>}"
echo "[activate] REPO      : $REPO"