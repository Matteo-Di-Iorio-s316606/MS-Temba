#!/usr/bin/env bash
# Environment health check for mstemba_v2.
#
# Verifies that conda env, CUDA module, Python dependencies, and project
# layout are all in a runnable state. Idempotent: safe to call from any
# script before launching training. Exits with code 0 on success, 1 on
# any failure (so you can chain it with `&&`).
#
# Usage:
#   bash scripts/check_env.sh                  # standalone
#   bash scripts/check_env.sh && python ...    # gate other commands

set -uo pipefail   # NOT -e: we handle errors ourselves to print all failures

REPO_ROOT="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"
EXPECTED_ENV="mstemba_v2"
EXPECTED_PYTHON="3.10"
EXPECTED_TORCH_MAJOR="2."
EXPECTED_CUDA="11.8"

FAIL=0
pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; FAIL=1; }
info() { printf "  · %s\n" "$1"; }

# ---- 1. Conda env active ----
echo "[1/5] Conda environment"
if ! command -v conda >/dev/null 2>&1; then
    # Try sourcing conda if not on PATH (typical in OAR batch shell).
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    fail "conda not found on PATH and ~/miniconda3 not sourceable"
elif [[ "${CONDA_DEFAULT_ENV:-<none>}" != "$EXPECTED_ENV" ]]; then
    info "Current env: ${CONDA_DEFAULT_ENV:-<none>}, expected $EXPECTED_ENV. Activating."
    if conda activate "$EXPECTED_ENV" 2>/dev/null; then
        pass "Activated $EXPECTED_ENV"
    else
        fail "Cannot activate $EXPECTED_ENV (does it exist? run conda env list)"
    fi
else
    pass "$EXPECTED_ENV already active"
fi

# ---- 2. CUDA module ----
echo "[2/5] CUDA module"
if [[ -z "${CUDA_HOME:-}" ]] && ! command -v nvcc >/dev/null 2>&1; then
    info "CUDA not in env, attempting module load"
    if command -v module >/dev/null 2>&1; then
        module load cuda/11.8.0_gcc-10.4.0 2>/dev/null \
            && pass "module load cuda/11.8 OK" \
            || fail "module load cuda failed"
    else
        fail "module command unavailable and CUDA not in env"
    fi
else
    pass "CUDA visible (CUDA_HOME=${CUDA_HOME:-<unset>}, nvcc=$(command -v nvcc || echo none))"
fi

# ---- 3. Python deps ----
echo "[3/5] Python dependencies"
PYCHECK=$(python - <<'PY' 2>&1
import sys
try:
    import torch, mamba_ssm, causal_conv1d
    print(f"PY_VERSION={sys.version.split()[0]}")
    print(f"TORCH={torch.__version__}")
    print(f"TORCH_CUDA={torch.version.cuda}")
    print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
    print(f"MAMBA={mamba_ssm.__version__ if hasattr(mamba_ssm, '__version__') else 'unknown'}")
    print(f"CCONV={causal_conv1d.__version__ if hasattr(causal_conv1d, '__version__') else 'unknown'}")
    if torch.cuda.is_available():
        print(f"GPU={torch.cuda.get_device_name(0)}")
        print(f"CC={'.'.join(map(str, torch.cuda.get_device_capability(0)))}")
except Exception as e:
    print(f"IMPORT_ERROR={e}")
    sys.exit(1)
PY
)
PY_EXIT=$?
if [[ $PY_EXIT -ne 0 ]]; then
    fail "Python import failed:"
    echo "$PYCHECK" | sed 's/^/      /'
else
    while IFS='=' read -r key val; do
        case "$key" in
            PY_VERSION)     [[ "$val" == ${EXPECTED_PYTHON}* ]] && pass "Python $val" || fail "Python $val, expected ${EXPECTED_PYTHON}.x" ;;
            TORCH)          [[ "$val" == ${EXPECTED_TORCH_MAJOR}* ]] && pass "torch $val" || fail "torch $val, expected ${EXPECTED_TORCH_MAJOR}x" ;;
            TORCH_CUDA)     [[ "$val" == "$EXPECTED_CUDA" ]] && pass "torch CUDA $val" || fail "torch CUDA $val, expected $EXPECTED_CUDA" ;;
            CUDA_AVAILABLE) [[ "$val" == "True" ]] && pass "CUDA available in torch" || fail "torch.cuda.is_available()=False (no GPU? wrong driver?)" ;;
            MAMBA)          pass "mamba_ssm $val" ;;
            CCONV)          pass "causal_conv1d $val" ;;
            GPU)            info "GPU: $val" ;;
            CC)             info "Compute capability: $val (mamba needs >= 7.0)" ;;
        esac
    done <<< "$PYCHECK"
fi

# ---- 4. nvidia-smi sanity ----
echo "[4/5] nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [[ -n "$GPU_LINE" ]]; then
        pass "nvidia-smi sees: $GPU_LINE"
    else
        fail "nvidia-smi runs but returns no GPU"
    fi
else
    fail "nvidia-smi not on PATH"
fi

# ---- 5. Project layout ----
echo "[5/5] Project layout"
[[ -d "$REPO_ROOT" ]]                              && pass "repo root exists" || fail "repo root missing: $REPO_ROOT"
[[ -f "$REPO_ROOT/vim/MSTemba_main.py" ]]          && pass "vim/MSTemba_main.py present" || fail "vim/MSTemba_main.py missing"
[[ -d "$REPO_ROOT/data/features/charades_i3d" ]]   && pass "charades_i3d features dir present" || fail "data/features/charades_i3d missing"
[[ -f "$REPO_ROOT/data/charades.json" ]]           && pass "charades.json annotation present" || fail "data/charades.json missing"

echo
if [[ $FAIL -eq 0 ]]; then
    printf "\033[32m== env check PASSED ==\033[0m\n"
    exit 0
else
    printf "\033[31m== env check FAILED ==\033[0m\n"
    exit 1
fi