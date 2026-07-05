#!/bin/bash
# One-shot: setup env + download 3 real-quant models from HF + run inference speed bench.
#
# Assumes you are in the GLoRCQ repo root (i.e. this script is at scripts/run_vast_h200_speed.sh).
# Idempotent: safe to re-run after failures.
#
# Usage on a fresh vast.ai H200 container:
#   cd /workspace
#   git clone -b exp/e11-tileq-cross-layer https://<PAT>@github.com/nicyyyy/GLoRCQ.git
#   cd GLoRCQ
#   bash scripts/run_vast_h200_speed.sh
set +e  # continue on errors (better for resume)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLORCQ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK=${WORK:-/workspace/glorcq_speed}
mkdir -p $WORK
cd $WORK
ln -sfn $GLORCQ_ROOT GLoRCQ

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
# Use --clear to overwrite any pre-existing .venv (vast.ai templates often ship one)
uv venv .venv --python 3.12 --clear
export VIRTUAL_ENV=$WORK/.venv
PY=$VIRTUAL_ENV/bin/python
$PY --version || { echo "ERROR: venv broken"; exit 1; }

echo ""
echo "===== [$(date)] Install torch matching nvcc ====="
NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
case "$NVCC_VER" in
    12.6*) TORCH_IDX=cu126; TORCH_VER=2.7.1 ;;
    12.8*) TORCH_IDX=cu128; TORCH_VER=2.8.0 ;;
    12.9*) TORCH_IDX=cu128; TORCH_VER=2.8.0 ;;
    13.*)  TORCH_IDX=cu130; TORCH_VER=2.9.1 ;;
    *)     TORCH_IDX=cu124; TORCH_VER=2.6.0 ;;
esac
echo "nvcc $NVCC_VER -> torch $TORCH_VER $TORCH_IDX"
uv pip install "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/$TORCH_IDX 2>&1 | tail -3

echo ""
echo "===== [$(date)] Install rest of deps ====="
uv pip install "transformers==4.51.3" datasets accelerate peft huggingface_hub wheel packaging ninja sentencepiece protobuf 2>&1 | tail -3
# fast-hadamard-transform: try wheel first, fall back to source, fall back to stub
uv pip install fast-hadamard-transform --no-build-isolation 2>&1 | tail -3
$PY -c "import fast_hadamard_transform" 2>/dev/null || {
    echo "  installing stub (VQ4 path doesn't call it)"
    cat > $VIRTUAL_ENV/lib/python3.12/site-packages/fast_hadamard_transform.py <<'STUB'
def hadamard_transform(*a, **k):
    raise NotImplementedError("stub — VQ4 doesn't need this")
STUB
}

echo ""
echo "===== [$(date)] Install glorcq package (editable) ====="
cd GLoRCQ
uv pip install -e . --no-build-isolation --no-deps 2>&1 | tail -3
cd $WORK

echo ""
echo "===== [$(date)] Build CUDA kernels ====="
cd GLoRCQ/inference/kernels
export TORCH_CUDA_ARCH_LIST=9.0
$PY setup.py build_ext --inplace 2>&1 | tail -8
ls -lh *.so 2>&1
cd $WORK

echo ""
echo "===== [$(date)] Download 3 real-quant checkpoints from HF ====="
export HF_HOME=$WORK/hf_cache
mkdir -p $HF_HOME ckpts
for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
    if [ -f ckpts/$m/config.json ]; then
        echo "  already have $m"
        continue
    fi
    mkdir -p ckpts/$m
    $VIRTUAL_ENV/bin/huggingface-cli download "Tsingyow/GLoRCQ-${m}-real" \
        --local-dir ckpts/$m 2>&1 | tail -2 &
done
wait
du -sh ckpts/* 2>&1

echo ""
echo "===== [$(date)] Run inference speed tests ====="
mkdir -p speed_results
for m in qwen1.5-moe-a2.7b mixtral-8x7b qwen3-30b-a3b; do
    echo "--- $m ---"
    CUDA_VISIBLE_DEVICES=0 $PY GLoRCQ/inference/eval_speed.py \
        --model_path ckpts/$m --batch_size 1 --prompt_len 128 --gen_len 128 --max_seq_len 512 \
        2>&1 | tee speed_results/${m}.log | grep -E "Standard|Graph|Speedup|tok/s|Error|error" || true
done

echo ""
echo "===== [$(date)] DONE. Summary ====="
grep -H "tok/s" speed_results/*.log
