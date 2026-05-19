#!/usr/bin/env bash
# Sanity check: 1 epoch on Charades I3D. Self-contained for OAR batch.

# #OAR -n sanity_charades_i3d
# #OAR -q besteffort
# #OAR -p host='esterel-31.sophia.grid5000.fr'
# #OAR -l host=1/gpu=1,walltime=1
# #OAR --stdout experiments/oar_logs/%jobname%.%jobid%.stdout
# #OAR --stderr experiments/oar_logs/%jobname%.%jobid%.stderr

set -euo pipefail

REPO_ROOT="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOBID="${OAR_JOB_ID:-local}"
OUTDIR="$REPO_ROOT/experiments/sanity_charades_i3d_${TIMESTAMP}_oar${JOBID}"
mkdir -p "$OUTDIR" "$REPO_ROOT/experiments/oar_logs"

# ---- Self-contained env setup (never assume an outer shell did this) ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mstemba_v2
module load cuda/11.8.0_gcc-10.4.0

bash "$REPO_ROOT/scripts/check_env.sh" || {
    echo "[FATAL] env check failed, aborting before training"
    exit 1
}

# ---- Env check: tee, not redirect, so failures are visible immediately ----
{
  echo "=== Sanity check Charades I3D ==="
  echo "date     : $(date -Is)"
  echo "host     : $(hostname)"
  echo "oar jobid: ${OAR_JOB_ID:-<interactive>}"
  echo "conda env: ${CONDA_DEFAULT_ENV:-<none>}"
  echo "which py : $(which python)"
  echo ""
  python - <<'PY'
import torch, mamba_ssm, causal_conv1d
print("torch         :", torch.__version__, "| cuda:", torch.version.cuda)
print("device        :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("mamba_ssm     :", mamba_ssm.__file__)
print("causal_conv1d :", causal_conv1d.__file__)
PY
  echo ""
  # `|| true` to neutralize SIGPIPE under pipefail; ignore failures only here
  nvidia-smi | head -n 12 || true
} 2>&1 | tee "$OUTDIR/env_check.txt"

cd "$REPO_ROOT/vim"

echo ""
echo "==> Training (1 ep, Charades I3D, bs=5)"
echo "==> Output: $OUTDIR"

python MSTemba_main.py \
  -dataset charades \
  -mode rgb \
  -backbone i3d \
  -model mstemba \
  -train True \
  -rgb_root "$REPO_ROOT/data/features/charades_i3d" \
  -num_clips 256 \
  -skip 0 \
  -comp_info False \
  -epochs 1 \
  -unisize True \
  -alpha_l 1 \
  -beta_l 0.05 \
  -batch_size 5 \
  -resume "" \
  -output_dir "$OUTDIR" \
  2>&1 | tee "$OUTDIR/training.log"

echo ""
echo "==> Sanity completato. Output: $OUTDIR"