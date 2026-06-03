#!/usr/bin/env bash
# =========================================================
# E3 — Skeleton SCD-Net w=16, MLP projection (4096→1024→256), bidirectional Mamba
#
# Methodological control vs E2 (Linear projection): tests whether a single
# Linear is too brutal a projection from D=4096 SCD-Net features and whether
# a 2-layer MLP exposes useful signal that Linear collapses.
#
# Pure ablation vs E2: identical hparams, identical features (w=16),
# identical bidir setup, identical seed, identical batch size. The ONLY
# changes are `-model mstemba_mlp_proj` and the two MLP hparams
# (proj_hidden_dim, proj_dropout).
#
# Expected outcome: 10.5-11.5 mAP Full (modest gain from extra params +
# non-linearity).
#
# Decision criterion (pre-registered, do not redefine post-hoc):
#   PRIMARY — count of classes with AP_skel > AP_clip per-class at best epoch:
#     ≤ 4/157  → pivot to T1-T6 ablation confirmed definitively
#     5-9/157  → ambiguous, check if ≥2 of top-5 expected classes recover
#                (c112 closing closet, c97 walking, c135 sitting bed,
#                 c106 drinking, c151 std-sit); if no → pivot anyway
#     ≥ 10/157 → reopen fusion, design cross-modal module
#   SECONDARY — mAP Full sanity:
#     <10.0 = overfit signal (MLP worse than Linear, conclusive pivot)
#     10.0-11.5 = expected band
#     >11.5 = unexpected, reconsider
#
# Reference: chat session "MSTemba v2 skeleton bidir timeline E1 E2"
#            (2026-05-26), Context Doc Sect. C.3 + pivot rationale.
# =========================================================
#OAR -n run_charades_scdnet_w16_E3
#OAR -p "host='esterel-33.sophia.grid5000.fr' OR host='esterel-34.sophia.grid5000.fr' OR host='esterel-35.sophia.grid5000.fr' OR host='esterel-39.sophia.grid5000.fr' OR host='esterel-40.sophia.grid5000.fr'"
#OAR -q besteffort
#OAR -O experiments/oar_logs/charades_scdnet_w16_E3.%jobid%.stdout
#OAR -E experiments/oar_logs/charades_scdnet_w16_E3.%jobid%.stderr
#OAR --checkpoint 600
#OAR --signal SIGUSR2

set -uo pipefail

# ---- Env setup ----
export REPO="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"
cd "$REPO"
source activate_mstemba.sh
bash scripts/check_env.sh || { echo "ENV CHECK FAILED"; exit 1; }

# ---- Experiment paths ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOBID="${OAR_JOB_ID:-manual}"
EXPNAME="charades_scdnet_w16_E3_${TIMESTAMP}_oar${JOBID}"
EXPDIR="$REPO/experiments/${EXPNAME}"
mkdir -p "$EXPDIR"

# ---- Feature dir (same as E2 — NO new pre-computation needed) ----
FEAT_DIR="$REPO/data/features/charades_scdnet_w16"
[[ -d "$FEAT_DIR" ]] || { echo "FEAT_DIR missing: $FEAT_DIR"; exit 1; }

# ---- Run metadata ----
{
  echo "EXPNAME=${EXPNAME}"
  echo "JOBID=${JOBID}"
  echo "HOST=$(hostname)"
  echo "GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "GIT_SHA=$(git rev-parse HEAD)"
  echo "GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)"
  echo "FEAT_DIR=${FEAT_DIR}"
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "MODEL=mstemba_mlp_proj"
  echo "PROJ_HIDDEN_DIM=1024"
  echo "PROJ_DROPOUT=0.1"
} > "$EXPDIR/run_meta.txt"

# ---- Auto-resume from checkpoint_last if present ----
RESUME_FLAG=""
if [[ -f "$EXPDIR/checkpoint_last.pth" ]]; then
    RESUME_FLAG="-resume $EXPDIR/checkpoint_last.pth"
    echo "[resume] checkpoint_last.pth found, will continue"
fi

# ---- Launch training ----
# Hparams IDENTICAL to E2: drop=0.1, drop_path=0.1, wd=0.05, patience=15,
#                          batch=5, num_clips=256, alpha_l=1, beta_l=0.05.
# Only ablation variables vs E2: -model + MLP hparams.
python -u vim/MSTemba_main.py \
    -dataset charades \
    -mode rgb \
    -backbone scdnet \
    -model mstemba_mlp_proj \
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
    -proj_hidden_dim 1024 \
    -proj_dropout 0.1 \
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