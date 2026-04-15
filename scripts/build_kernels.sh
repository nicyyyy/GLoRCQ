#!/usr/bin/env bash
# Build fused CUDA kernels for GLoRCQ inference.
#
# Usage:
#   bash scripts/build_kernels.sh          # build with uv
#   bash scripts/build_kernels.sh --clean  # clean + rebuild
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLORCQ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNEL_DIR="$GLORCQ_ROOT/inference/kernels"

if [[ "${1:-}" == "--clean" ]]; then
    echo "[build_kernels] Cleaning previous build artifacts ..."
    rm -rf "$KERNEL_DIR/build" "$KERNEL_DIR"/*.so "$KERNEL_DIR"/*.egg-info
fi

echo "[build_kernels] Building CUDA kernels in $KERNEL_DIR ..."
cd "$KERNEL_DIR"
uv run python setup.py build_ext --inplace

echo "[build_kernels] Done. Built extensions:"
ls -lh "$KERNEL_DIR"/*.so 2>/dev/null || echo "  (no .so files found — build may have failed)"
