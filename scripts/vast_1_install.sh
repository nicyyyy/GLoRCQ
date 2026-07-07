#!/bin/bash
# Step 1/3: install env + build kernels on a fresh vast.ai H200 container.
#
# Usage:
#   cd /workspace
#   git clone -b exp/e11-tileq-cross-layer https://github.com/nicyyyy/GLoRCQ.git
#   bash GLoRCQ/scripts/vast_1_install.sh
#
# Verify success at end: torch importable + 3 .so files built.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLORCQ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK=${WORK:-/workspace/glorcq_speed}
mkdir -p "$WORK"
cd "$WORK"

echo "===== [$(date)] Machine check ====="
nvidia-smi --query-gpu=index,name,memory.total --format=csv | head -5
nvcc --version 2>&1 | tail -1
g++ --version | head -1
df -h /workspace | tail -1
free -h | head -2

echo ""
echo "===== [$(date)] Install uv + venv ====="
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv --version
uv venv .venv --python 3.12 --clear
export VIRTUAL_ENV=$WORK/.venv
PY=$VIRTUAL_ENV/bin/python
$PY --version

echo ""
echo "===== [$(date)] Install torch matching nvcc ====="
NVCC_VER=$(nvcc --version 2>&1 | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
case "$NVCC_VER" in
    12.6*) TORCH_IDX=cu126; TORCH_VER=2.7.1 ;;
    12.8*|12.9*) TORCH_IDX=cu128; TORCH_VER=2.8.0 ;;
    13.*)  TORCH_IDX=cu130; TORCH_VER=2.9.1 ;;
    *)     TORCH_IDX=cu124; TORCH_VER=2.6.0 ;;
esac
echo "nvcc $NVCC_VER -> torch $TORCH_VER $TORCH_IDX"

# Torch is big — increase timeout, retry once if fails
export UV_HTTP_TIMEOUT=300
uv pip install "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$TORCH_IDX" \
    || uv pip install "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$TORCH_IDX"

echo ""
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ok', torch.cuda.is_available())" \
    || { echo "ERROR: torch install failed"; exit 1; }

echo ""
echo "===== [$(date)] Install rest of deps ====="
uv pip install "transformers==4.51.3" datasets accelerate peft huggingface_hub wheel packaging ninja sentencepiece protobuf

# fast-hadamard-transform stub (VQ4 doesn't need it)
$PY -c "import fast_hadamard_transform" 2>/dev/null || {
    echo "  installing fast-hadamard-transform stub"
    cat > $VIRTUAL_ENV/lib/python3.12/site-packages/fast_hadamard_transform.py <<'STUB'
def hadamard_transform(*a, **k):
    raise NotImplementedError("stub — VQ4 doesn't use this")
STUB
}

echo ""
echo "===== [$(date)] Install glorcq (editable, no-deps) ====="
cd "$GLORCQ_ROOT"
uv pip install -e . --no-build-isolation --no-deps
cd "$WORK"

echo ""
echo "===== [$(date)] Build CUDA kernels (H200 sm_90) ====="
cd "$GLORCQ_ROOT/inference/kernels"
export TORCH_CUDA_ARCH_LIST=9.0
$PY setup.py build_ext --inplace 2>&1 | tee /tmp/kernel_build.log | tail -8
echo "---"
ls -lh *.so 2>&1
n_so=$(ls *.so 2>/dev/null | wc -l)
if [ "$n_so" -lt 3 ]; then
    echo "ERROR: expected 3 .so files, got $n_so"
    echo "See /tmp/kernel_build.log for details:"
    tail -40 /tmp/kernel_build.log
    exit 1
fi
cd "$WORK"

echo ""
# Verify kernel import (need to cd into repo root; glorcq package has packages=[] in pyproject)
cd "$GLORCQ_ROOT"
$PY -c "from inference.kernels import is_vq4_cuda_available; print('vq4 kernel:', is_vq4_cuda_available())"
cd "$WORK"

echo ""
echo "===== [$(date)] STEP 1 DONE ====="
echo "Next: bash $GLORCQ_ROOT/scripts/vast_2_download.sh"
