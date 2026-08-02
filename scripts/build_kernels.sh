#!/usr/bin/env bash
# Build the fused CUDA inference kernels (VQ4 / GPTQ / TurboQuant dequant-matmul).
# Requires nvcc with a major CUDA version matching the installed torch build
# (torch 2.6.0+cu124 -> CUDA 12.x nvcc).
#
# Usage:
#   bash scripts/build_kernels.sh          # build in-place
#   bash scripts/build_kernels.sh --clean  # clean + rebuild
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNEL_DIR="$ROOT/inference/kernels"
PY=${PY:-python}

if [[ "${1:-}" == "--clean" ]]; then
    echo "[build_kernels] Cleaning previous build artifacts ..."
    rm -rf "$KERNEL_DIR/build" "$KERNEL_DIR"/*.so "$KERNEL_DIR"/*.egg-info
fi

echo "[build_kernels] Building CUDA kernels in $KERNEL_DIR ..."
cd "$KERNEL_DIR"
$PY setup.py build_ext --inplace

echo "[build_kernels] Done. Built extensions:"
ls -lh "$KERNEL_DIR"/*.so 2>/dev/null || echo "  (no .so files found — build may have failed)"
