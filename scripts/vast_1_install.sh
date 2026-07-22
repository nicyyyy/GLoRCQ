#!/bin/bash
# Step 1/3: install env + build CUDA kernels on a fresh GPU machine
# (vast.ai container, lab server, cloud VM — anything with nvcc + a GPU).
#
# Usage:
#   git clone -b exp/e11-tileq-cross-layer https://github.com/nicyyyy/GLoRCQ.git
#   bash GLoRCQ/scripts/vast_1_install.sh
#
# Optional: WORK=/path/to/workdir bash ... (default: /workspace/glorcq_speed
# on vast.ai, else $HOME/glorcq_speed). Needs ~50 GB free disk at $WORK.
# Verify success at end: torch importable + 3 .so kernel files built.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLORCQ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -z "${WORK:-}" ]; then
    if [ -d /workspace ] && [ -w /workspace ]; then WORK=/workspace/glorcq_speed
    else WORK=$HOME/glorcq_speed; fi
fi
mkdir -p "$WORK"
cd "$WORK"

echo "===== [$(date)] Machine check ====="
nvidia-smi --query-gpu=index,name,memory.total --format=csv | head -5
nvcc --version 2>&1 | tail -1
g++ --version | head -1
df -h "$WORK" | tail -1
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
# Explicit pin: to keep ALL speed numbers on ONE CUDA version (fair speedup
# ratios — a version-wide speed factor cancels in the ratio, but MIXING
# versions within one table does not), set both, e.g. for CUDA 12.8:
#   TORCH_IDX=cu128 TORCH_VER=2.8.0 bash vast_1_install.sh
# A CUDA-13-driver machine runs a cu128 torch fine (driver is backward-compat);
# to also BUILD the kernels against 12.8, install a 12.8 toolkit and put its
# nvcc first (CUDA_HOME=/usr/local/cuda-12.8 or conda cuda-toolkit=12.8).
if [ -n "${TORCH_IDX:-}" ] && [ -n "${TORCH_VER:-}" ]; then
    echo "using pinned torch $TORCH_VER $TORCH_IDX (nvcc reports $NVCC_VER)"
else
    case "$NVCC_VER" in
        12.6*) TORCH_IDX=cu126; TORCH_VER=2.7.1 ;;
        12.8*|12.9*) TORCH_IDX=cu128; TORCH_VER=2.8.0 ;;
        13.*)  TORCH_IDX=cu130; TORCH_VER=2.9.1 ;;
        *)     TORCH_IDX=cu124; TORCH_VER=2.6.0 ;;
    esac
    echo "nvcc $NVCC_VER -> torch $TORCH_VER $TORCH_IDX"
fi

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

# fast-hadamard-transform stub (VQ4 doesn't need it; real wheel is slow to build)
$PY -c "import fast_hadamard_transform" 2>/dev/null || {
    echo "  installing fast-hadamard-transform stub"
    SITE_PKGS=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    cat > "$SITE_PKGS/fast_hadamard_transform.py" <<'STUB'
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
echo "===== [$(date)] Build CUDA kernels ====="
# setup.py bakes -gencode for sm_80/86/89/90 (A100/A6000/4090/H100/H200).
# NOTE: TORCH_CUDA_ARCH_LIST is ignored (explicit gencodes win). For Blackwell
# (sm_100/120) add the arch to the gencode list in inference/kernels/setup.py.
#
# torch's cpp_extension REFUSES to build if the nvcc major version mismatches
# torch's CUDA (e.g. system nvcc 13.0 vs torch cu128). Auto-detect a matching
# nvcc in /usr/local AND conda envs (install one with:
#   conda create -n cuda128 -y -c nvidia cuda-toolkit=12.8 ).
TCUDA=$($PY -c "import torch; print(torch.version.cuda or '')" 2>/dev/null)
NVCC_MATCH=""
for n in $(which -a nvcc 2>/dev/null) "/usr/local/cuda-$TCUDA/bin/nvcc" \
         /opt/conda/bin/nvcc /opt/conda/envs/*/bin/nvcc /venv/*/bin/nvcc \
         "$HOME"/miniconda3/bin/nvcc "$HOME"/miniconda3/envs/*/bin/nvcc \
         "$HOME"/anaconda3/bin/nvcc "$HOME"/anaconda3/envs/*/bin/nvcc; do
    [ -x "$n" ] || continue
    v=$("$n" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')
    if [ -n "$TCUDA" ] && [ "$v" = "$TCUDA" ]; then NVCC_MATCH="$n"; break; fi
done
if [ -n "$NVCC_MATCH" ]; then
    echo "  using matching CUDA $TCUDA nvcc: $NVCC_MATCH"
    export CUDA_HOME="$(dirname "$(dirname "$NVCC_MATCH")")"
    export PATH="$(dirname "$NVCC_MATCH"):$PATH"
else
    echo "  WARN: no nvcc matching torch CUDA $TCUDA found; build may fail on a"
    echo "        version-mismatch. Install one: conda create -n cuda$( echo $TCUDA | tr -d . ) -y -c nvidia cuda-toolkit=$TCUDA"
fi
cd "$GLORCQ_ROOT/inference/kernels"
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
