#!/usr/bin/env bash
# =========================================================
# E2 — Skeleton SCD-Net w=16, Linear projection, bidirectional Mamba
#
# Replicates Exp 5 (old repo: 9.46 mAP unidir) under v2 bidirectional setup.
# Bridge baseline: w=16 aligned to CLIP's ~1.5 FPS, T_clipped=256.
# Reference: Context Doc, C.3 Phase 2B.1
# Expected outcome: 10.5-12.0 mAP val_full. Gate: ≥10.5 → proceed E1/E3/E4.
# =========================================================

#OAR -n run_charades_scdnet_w16_E2
#OAR -p "host='esterel-33.sophia.grid5000.fr' OR host='esterel-34.sophia.grid5000.fr' OR host='esterel-35.sophia.grid5000.fr' OR host='esterel-39.sophia.grid5000.fr' OR host='esterel-40.sophia.grid5000.fr'"
#OAR -q besteffort
#OAR -O experiments/oar_logs/charades_scdnet_w16_E2.%jobid%.stdout
#OAR -E experiments/oar_logs/charades_scdnet_w16_E2.%jobid%.stderr
#OAR --checkpoint 600
#OAR --signal SIGUSR2

set -uo pipefail

# ---- Env setup (self-contained, see Context Sect. B for rationale) ----
export REPO="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"
cd "$REPO"
source activate_mstemba.sh
bash scripts/check_env.sh || { echo "ENV CHECK FAILED"; exit 1; }

# ---- Experiment paths ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOBID="${OAR_JOB_ID:-manual}"
EXPNAME="charades_scdnet_w16_E2_${TIMESTAMP}_oar${JOBID}"
EXPDIR="$REPO/experiments/${EXPNAME}"
mkdir -p "$EXPDIR"

# ---- Feature dir (symlink set up in feat/skeleton, see SESSION_README) ----
FEAT_DIR="$REPO/data/features/charades_scdnet_w16"
[[ -d "$FEAT_DIR" ]] || { echo "FEAT_DIR missing: $FEAT_DIR"; exit 1; }

# ---- Run metadata (for reproducibility) ----
{
  echo "EXPNAME=${EXPNAME}"
  echo "JOBID=${JOBID}"
  echo "HOST=$(hostname)"
  echo "GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "GIT_SHA=$(git rev-parse HEAD)"
  echo "GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)"
  echo "FEAT_DIR=${FEAT_DIR}"
  echo "TIMESTAMP=${TIMESTAMP}"
} > "$EXPDIR/run_meta.txt"

# ---- Auto-resume from checkpoint_last if present ----
RESUME_FLAG=""
if [[ -f "$EXPDIR/checkpoint_last.pth" ]]; then
    RESUME_FLAG="-resume $EXPDIR/checkpoint_last.pth"
    echo "[resume] checkpoint_last.pth found, will continue"
fi

# ---- Launch training ----
# Hparams from Exp 5 (old repo): drop=0.1, drop_path=0.1, wd=0.05, patience=15
# num_clips=256 (default): w=16 → T~47, fits comfortably with no truncation
python -u vim/MSTemba_main.py \
    -dataset charades \
    -mode rgb \
    -backbone scdnet \
    -model mstemba \
    -train True \
    -rgb_root "$FEAT_DIR" \
    -num_clips 256 \
    -skip 0 \
    -comp_info False \
    -epochs 50 \
    -unisize True \
    -alpha_l 1 \
    -beta_l 0.05 \
    -batch_size 5 \
    --drop 0.1 \
    --drop-path 0.1 \
    --weight-decay 0.05 \
    -early_stop_patience 15 \
    -early_stop_min_delta 0.0 \
    -save_every_epoch True \
    -output_dir "$EXPDIR" \
    $RESUME_FLAG \
    2>&1 | tee -a "$EXPDIR/training.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "[exit] training python exited with code $EXIT_CODE"
exit $EXIT_CODE
