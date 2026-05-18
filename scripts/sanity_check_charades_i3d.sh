#!/usr/bin/env bash
# Sanity check: 1 epoca su Charades I3D.

set -euo pipefail

REPO_ROOT="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$REPO_ROOT/experiments/sanity_charades_i3d_${TIMESTAMP}"
mkdir -p "$OUTDIR"

{
  echo "=== Sanity check Charades I3D ==="
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "env : ${CONDA_DEFAULT_ENV:-<none>}"
  echo ""
  python -c "
import torch, mamba_ssm, causal_conv1d
print('torch:', torch.__version__, '| cuda:', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('mamba_ssm   :', mamba_ssm.__file__)
print('causal_conv1d:', causal_conv1d.__file__)
"
  echo ""
  nvidia-smi | head -12
} > "$OUTDIR/env_check.txt"

cat "$OUTDIR/env_check.txt"

cd "$REPO_ROOT/vim"

echo ""
echo "==> Avvio training (1 epoca Charades I3D, batch=5)"
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
  -output_dir "$OUTDIR" \
  2>&1 | tee "$OUTDIR/training.log"

echo ""
echo "==> Sanity completato. Output: $OUTDIR"
