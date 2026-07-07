#!/bin/bash
# One-shot: setup env + download 3 real-quant models + run GLoRCQ inference speed bench.
#
# Usage on a fresh vast.ai H200 container (repo is now public, no PAT needed):
#   cd /workspace
#   git clone -b exp/e11-tileq-cross-layer https://github.com/nicyyyy/GLoRCQ.git
#   cd GLoRCQ
#   bash scripts/run_vast_h200_speed.sh
#
# Idempotent — safe to re-run after failures.
set +e  # continue on errors (better for resume)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLORCQ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK=${WORK:-/workspace/glorcq_speed}
mkdir -p "$WORK"
cd "$WORK"
# Use a real dir with symlink to repo (avoids the "symlink over existing dir" gotcha)
[ -e GLoRCQ ] || ln -sfn "$GLORCQ_ROOT" GLoRCQ

echo ""
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
$PY --version || { echo "ERROR: venv broken"; exit 1; }

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
uv pip install "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/$TORCH_IDX 2>&1 | tail -3
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ok', torch.cuda.is_available())" 2>&1 | tail -3

echo ""
echo "===== [$(date)] Install rest of deps ====="
uv pip install "transformers==4.51.3" datasets accelerate peft huggingface_hub wheel packaging ninja sentencepiece protobuf 2>&1 | tail -3
# fast-hadamard-transform is not needed by VQ4 path — install stub if wheel fails
uv pip install fast-hadamard-transform --no-build-isolation 2>&1 | tail -3
$PY -c "import fast_hadamard_transform" 2>/dev/null || {
    echo "  fast-hadamard-transform install failed; installing stub (VQ4 doesn't need it)"
    cat > $VIRTUAL_ENV/lib/python3.12/site-packages/fast_hadamard_transform.py <<'STUB'
def hadamard_transform(*a, **k):
    raise NotImplementedError("stub — VQ4 doesn't use this")
STUB
}

echo ""
echo "===== [$(date)] Install glorcq package (editable) ====="
cd "$GLORCQ_ROOT"
uv pip install -e . --no-build-isolation --no-deps 2>&1 | tail -3

echo ""
echo "===== [$(date)] Build CUDA kernels (H200 sm_90) ====="
cd "$GLORCQ_ROOT/inference/kernels"
export TORCH_CUDA_ARCH_LIST=9.0
$PY setup.py build_ext --inplace 2>&1 | tee /tmp/kernel_build.log | tail -8
echo "---"
ls -lh *.so 2>&1
cd "$WORK"

# Verify vq4 kernel available (needed by real-quant graph path)
$PY -c "from inference.kernels import is_vq4_cuda_available; print('vq4 kernel:', is_vq4_cuda_available())" 2>&1 | tail -3

echo ""
echo "===== [$(date)] Download 3 real-quant checkpoints from HF ====="
export HF_HOME=$WORK/hf_cache
mkdir -p "$HF_HOME" ckpts
for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
    if [ -f "ckpts/$m/config.json" ] && ls ckpts/$m/model-*.safetensors >/dev/null 2>&1; then
        echo "  already have $m ($(du -sh ckpts/$m | cut -f1))"
        continue
    fi
    mkdir -p "ckpts/$m"
    echo "  downloading $m ..."
    $VIRTUAL_ENV/bin/huggingface-cli download "Tsingyow/GLoRCQ-${m}-real" \
        --local-dir "ckpts/$m" --max-workers 8 > "/tmp/dl_${m}.log" 2>&1 &
done
wait
echo "---"
for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
    printf "  %-22s " "$m"
    du -sh "ckpts/$m" 2>/dev/null | cut -f1
    n=$(ls "ckpts/$m/model-"*.safetensors 2>/dev/null | wc -l)
    for f in config.json cross_layer_info.pt model.safetensors.index.json; do
        [ -f "ckpts/$m/$f" ] && printf "%s ✓ " "$f" || printf "%s ✗ " "$f"
    done
    printf "shards=%d\n" $n
done

echo ""
echo "===== [$(date)] Run inference speed tests ====="
mkdir -p speed_results
# expandable_segments helps if GPU has residual fragmentation from prior runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
    echo ""
    echo "========== $m =========="
    CUDA_VISIBLE_DEVICES=0 $PY "$GLORCQ_ROOT/inference/eval_speed.py" \
        --model_path "ckpts/$m" --batch_size 1 --prompt_len 128 --gen_len 128 --max_seq_len 512 \
        2>&1 | tee "speed_results/${m}.log"
done

echo ""
echo "===== [$(date)] DONE. Summary ====="
grep -H -E "Standard:|Graph:|Speedup:" speed_results/*.log
