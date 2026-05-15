#!/usr/bin/env bash
# Setup env conda mstemba_v2 da zero, secondo specifiche README upstream.
# Lanciare SUL NODO GPU, non sul frontend (serve nvcc + GPU per testare).
#
# Tempo stimato: 25-45 min (incluse compilazioni mamba e causal_conv1d).

set -euo pipefail

REPO_ROOT="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"
ENV_NAME="mstemba_v2"

cd "$REPO_ROOT"

echo "=========================================="
echo "Setup env $ENV_NAME"
echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "=========================================="

# === STEP 1: CUDA module ===
echo ""
echo "[1/7] Caricamento CUDA module"
# Su Grid5000 Sophia il module init è in posti diversi: provo i path noti
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
elif [ -f /usr/share/modules/init/bash ]; then
  source /usr/share/modules/init/bash
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
  source /usr/share/lmod/lmod/init/bash
elif command -v module >/dev/null 2>&1; then
  echo "  -> module già disponibile in PATH"
else
  echo "  ERRORE: impossibile inizializzare 'module'"
  exit 1
fi

if module load cuda/11.8.0_gcc-10.4.0 2>/dev/null; then
  echo "  -> cuda/11.8.0 caricato (matcha README upstream torch 2.1.1+cu118)"
elif module load cuda/12.1.1_gcc-10.4.0 2>/dev/null; then
  echo "  -> cuda/12.1.1 caricato (fallback; userò torch cu121)"
  CUDA_FALLBACK_121=1
else
  echo "  ERRORE: nessun module CUDA disponibile, abort"
  module avail cuda 2>&1 | head -20
  exit 1
fi
nvcc --version

# === STEP 2: Conda init ===
echo ""
echo "[2/7] Init conda"
source ~/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -q "^${ENV_NAME}\s"; then
  echo "  ERRORE: env $ENV_NAME esiste già. Cancellalo prima con:"
  echo "    conda env remove -n $ENV_NAME"
  exit 1
fi

# === STEP 3: Crea env Python 3.10.13 ===
echo ""
echo "[3/7] Creazione env Python 3.10.13"
conda create -n "$ENV_NAME" python=3.10.13 -y
conda activate "$ENV_NAME"
python -V

# === STEP 4: Install PyTorch ===
echo ""
echo "[4/7] Install PyTorch 2.1.1"
if [ "${CUDA_FALLBACK_121:-0}" = "1" ]; then
  echo "  -> uso wheel cu121"
  pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
else
  echo "  -> uso wheel cu118 (come da README upstream)"
  pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
fi

python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
print('compute capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')
"

# === STEP 5: vim_requirements.txt ===
echo ""
echo "[5/7] Install vim_requirements.txt"
pip install -r vim/vim_requirements.txt

# === STEP 6: causal_conv1d ===
echo ""
echo "[6/7] Build causal_conv1d (~5 min)"
# Forza arch list a quella della GPU detected (RTX 8000 = 7.5; H100 = 9.0; A100 = 8.0)
export TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch
m, n = torch.cuda.get_device_capability(0)
print(f"{m}.{n}")
PY
)"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
export MAX_JOBS=4

cd "$REPO_ROOT/causal-conv1d"
pip install -e . --no-build-isolation
python -c "import causal_conv1d; print('causal_conv1d OK:', causal_conv1d.__file__)"

# === STEP 7: mamba-1p1p1 ===
echo ""
echo "[7/7] Build mamba-1p1p1 (~15-25 min)"
cd "$REPO_ROOT/mamba-1p1p1"
pip install -e . --no-build-isolation
python -c "import mamba_ssm; print('mamba_ssm OK:', mamba_ssm.__file__)"

cd "$REPO_ROOT"

echo ""
echo "=========================================="
echo "SETUP COMPLETATO"
echo "  env: $ENV_NAME"
echo "  attivare con: conda activate $ENV_NAME"
echo "=========================================="