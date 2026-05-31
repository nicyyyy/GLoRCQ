#!/usr/bin/env bash
# First-time environment setup for GLoRCQ.
#
# Usage:
#   cd glorcq/
#   bash scripts/setup_env.sh
#
# What it does:
#   1. Creates a uv virtualenv (.venv)
#   2. Installs Python dependencies from pyproject.toml
#   3. Installs TurboQuant (thirdpart dependency)
#   4. Builds CUDA inference kernels
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLORCQ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$GLORCQ_ROOT"

echo "============================================================"
echo "  GLoRCQ Environment Setup"
echo "  Root: $GLORCQ_ROOT"
echo "============================================================"

# 1. Create virtualenv
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtualenv ..."
    uv venv .venv
else
    echo "[1/4] Virtualenv already exists, skipping."
fi

# 2. Install Python dependencies
echo "[2/4] Installing Python dependencies ..."
uv pip install -e ".[turboquant]"

# 3. Install TurboQuant if thirdpart/ exists
TURBOQUANT_DIR="$GLORCQ_ROOT/thirdpart/turboquant"
if [ -d "$TURBOQUANT_DIR" ]; then
    echo "[3/5] Installing TurboQuant from thirdpart/ ..."
    uv pip install -e "$TURBOQUANT_DIR"
else
    echo "[3/5] thirdpart/turboquant not found, skipping."
    echo "       (TurboQuant is optional; only needed for --use_turboquant mode)"
fi

# 4. Install fast-hadamard-transform if thirdpart/ exists
HADAMARD_DIR="$GLORCQ_ROOT/thirdpart/fast-hadamard-transform"
if [ -d "$HADAMARD_DIR" ]; then
    echo "[4/5] Installing fast-hadamard-transform from thirdpart/ ..."
    uv pip install --no-build-isolation "$HADAMARD_DIR"
else
    echo "[4/5] thirdpart/fast-hadamard-transform not found, skipping."
    echo "       (Optional; only needed for --rotation_type hadamard)"
    echo "       Clone with: git clone https://github.com/Dao-AILab/fast-hadamard-transform.git thirdpart/fast-hadamard-transform"
fi

# 5. Build CUDA kernels
echo "[5/5] Building CUDA inference kernels ..."
bash "$SCRIPT_DIR/build_kernels.sh"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Activate with: source .venv/bin/activate"
echo "  Or use:        uv run python ..."
echo "============================================================"
