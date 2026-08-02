#!/usr/bin/env bash
# ===========================================================================
# CLASP — environment setup.
#
# Requirements:
#   - Python 3.12
#   - CUDA 12.x toolkit on PATH (nvcc): needed to build fast-hadamard-transform
#     and the fused inference kernels. The nvcc MAJOR version must match the
#     torch build (torch 2.6.0+cu124 -> CUDA 12.x). Check: nvcc --version
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3.12}

echo "[install] Creating venv (.venv) with $PYTHON ..."
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "[install] Installing torch 2.6.0 (CUDA 12.4 wheels) ..."
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "[install] Installing python dependencies ..."
pip install -r requirements.txt

echo "[install] Building fused CUDA inference kernels (optional for accuracy"
echo "          experiments; required for the speed benchmarks) ..."
if command -v nvcc >/dev/null 2>&1; then
    bash scripts/build_kernels.sh
else
    echo "[install] WARNING: nvcc not found — skipping kernel build."
    echo "          Accuracy experiments (quantize/PPL/zero-shot) still work;"
    echo "          run 'bash scripts/build_kernels.sh' later for speed tests."
fi

echo "[install] Done. Activate with: source .venv/bin/activate"
