#!/usr/bin/env bash
# =========================================================
# E1 — Skeleton SCD-Net w=1 native, Linear projection, bidirectional Mamba
#
# Replicates Exp 10 (old repo: 9.11 mAP unidir, num_clips=2400) under v2 bidir.
# Reference: Context Doc, C.3 Phase 2B.2 + consolidated timeline Exp 10
# Expected outcome: 9.5-11 mAP val_full. Sensible to label-noise overfit.
#
# IMPORTANT:
#   - Requires Ampere GPU (sm_86): A40 or A6000. NOT Hopper (H100 = sm_90 = incompatible kernel).
#   - T=2400 + B=5 + bidir → estimated ~5 GB VRAM peak.
#   - Epoch time ~25-35 min → 50 ep ≈ 25h. Walltime 24h, expect besteffort resumes.
#
# Two ways to launch:
#   A) Batch: oarsub -S vim/scripts/run_charades_scdnet_w1_E1.sh
#      (uses #OAR directives below; SIGUSR2 + checkpoint works)
#   B) Interactive: oarsub -I -p host='esterel-XX...' -l host=1/gpu=1,walltime=24
#      then bash vim/scripts/run_charades_scdnet_w1_E1.sh
#      (#OAR directives are ignored; no SIGUSR2; checkpoints only at epoch end)
# =========================================================

#OAR -n charades_scdnet_w1_E1
#OAR -p "host='esterel-33.sophia.grid5000.fr' OR host='esterel-34.sophia.grid5000.fr' OR host='esterel-35.sophia.grid5000.fr' OR host='esterel-39.sophia.grid5000.fr' OR host='esterel-40.sophia.grid5000.fr'"
#OAR -q besteffort
#OAR -l host=1/gpu=1,walltime=12
#OAR -O experiments/oar_logs/charades_scdnet_w1_E1.%jobid%.stdout
#OAR -E experiments/oar_logs/charades_scdnet_w1_E1.%jobid%.stderr
#OAR --checkpoint 600
#OAR --signal SIGUSR2

set -uo pipefail

# ---- Env setup ----
export REPO="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"
cd "$REPO"
source activate_mstemba.sh
bash scripts/check_env.sh || { echo "ENV CHECK FAILED"; exit 1; }

# ---- Hard fail if not on Ampere (sm_86) ----
# Defense in depth: even if OAR placed us wrong, abort before training corrupts a 24h slot.
CC=$(python3 -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability(0))))" 2>/dev/null)
if [[ "$CC" != "8.6" ]]; then
    echo "[FATAL] GPU compute capability is $CC, expected 8.6 (Ampere). Abort." >&2
    echo "Mamba kernel was compiled for sm_86 only; will crash with 'no kernel image'." >&2
    exit 2
fi
echo "[ok] GPU compute capability: $CC"

# ---- Experiment paths ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOBID="${OAR_JOB_ID:-manual}"
EXPNAME="charades_scdnet_w1_E1_${TIMESTAMP}_oar${JOBID}"
EXPDIR="$REPO/experiments/${EXPNAME}"
mkdir -p "$EXPDIR"

# ---- Feature dir ----
FEAT_DIR="$REPO/data/features/charades_scdnet_full"
[[ -d "$FEAT_DIR" ]] || { echo "FEAT_DIR missing: $FEAT_DIR"; exit 1; }

# ---- Run metadata ----
{
    echo "EXPNAME=${EXPNAME}"
    echo "JOBID=${JOBID}"
    echo "HOST=$(hostname)"
    echo "GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "GPU_CC=${CC}"
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
python -u vim/MSTemba_main.py \
    -dataset charades \
    -mode rgb \
    -backbone scdnet \
    -model mstemba \
    -train True \
    -rgb_root "$FEAT_DIR" \
    -num_clips 2400 \
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