#!/usr/bin/env bash
set -euo pipefail

OUTDIR="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba/runs/tsu/clip/seed0"
mkdir -p "$OUTDIR"

RESUME="False"
if [ -f "$OUTDIR/checkpoint_last.pth" ]; then RESUME="$OUTDIR/checkpoint_last.pth"; fi

{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "pwd: $(pwd)"
  echo "git_commit: $(git -C . rev-parse HEAD 2>/dev/null || echo NA)"
  echo "git_status:"; git -C . status -sb 2>/dev/null || true
  echo "module_list:"; module list 2>&1 || true
  python -c "import torch; print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"
  echo "nvcc:"; nvcc -V || true
  echo "script: $0"
  echo "resume: $RESUME"
} > "$OUTDIR/run_meta.txt"

cd vim

python MSTemba_main.py \
  -dataset tsu \
  -mode rgb \
  -backbone clip \
  -model mstemba \
  -train True \
  -seed 0 \
  -resume "$RESUME" \
  -save_every 1 \
  -rgb_root "/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba/data/hf_features/Temporal_Action_Detection/tsu_features_clip_l14" \
  -num_clips 2500 \
  -skip 0 \
  --lr 4.5e-4 \
  -comp_info False \
  -epochs 140 \
  -unisize True \
  -alpha_l 1 \
  -beta_l 0.05 \
  -batch_size 1 \
  -output_dir "$OUTDIR" \
  2>&1 | tee -a "$OUTDIR/training.log"
