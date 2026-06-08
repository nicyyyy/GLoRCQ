#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    uv venv --python /nvme3/miniconda3/bin/python .venv
    uv pip install -r requirements.txt
fi

export HF_HOME=/home/qyyang/resource_dir/hf_cache/
export TRANSFORMERS_CACHE=/home/qyyang/resource_dir/hf_cache/
export HF_DATASETS_CACHE=/home/qyyang/resource_dir/hf_cache/
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_DISABLE_CACHING=0

model_path="Qwen/Qwen1.5-MoE-A2.7B"
# model_path="mistralai/Mixtral-8x7B-v0.1"
save_dir="/home/qyyang/resource_dir/GLoRCQ_out"
qbit=2
rank=128       # INT4-packed SV: 128 rank uses same storage as rank=64 int8
n_iter=5
G_moe=128
G_attn=4
sv_bits=4      # real 4-bit packing → 50% storage, rank doubled vs sv_bits=8
u_fp16=1  # store shared U in fp16 (no int8 quantization)

mkdir -p ./logs

logname="glorcq_${qbit}bit_rank${rank}_ufp16_sv${sv_bits}"

CUDA_VISIBLE_DEVICES=1 \
uv run python run_quantize.py \
    --model_path "$model_path" \
    --output_path "$save_dir/${model_path}-glorcq-${qbit}bit-rank${rank}-ufp16-sv${sv_bits}" \
    --qbit $qbit \
    --w_clip \
    --rank $rank \
    --G_moe $G_moe \
    --G_attn $G_attn \
    --nsamples 128 \
    --n_iter $n_iter \
    --hessian_svd \
    --use_turboquant \
    --search_act_alpha \
    --u_fp16 \
    --sv_bits $sv_bits \
    2>&1 | tee ./logs/${logname}.log
# Note: no --real_quant, so weights are saved as fp16 W_approx for PPL eval

CUDA_VISIBLE_DEVICES=1 \
uv run python evaluate/eval_ppl.py \
    --model_path "$save_dir/${model_path}-glorcq-${qbit}bit-rank${rank}-ufp16-sv${sv_bits}" \
    2>&1 | tee -a ./logs/${logname}.log
