#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ ! -d ".venv" ]; then
    bash scripts/setup_env.sh
fi

export HF_HOME=/nvme1/yqy/huggingface
export TRANSFORMERS_CACHE=/nvme1/yqy/huggingface/transformers
export HF_DATASETS_CACHE=/nvme1/yqy/huggingface/datasets
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_DISABLE_CACHING=0

# model_path="Qwen/Qwen1.5-MoE-A2.7B"
model_path="mistralai/Mixtral-8x7B-v0.1"
save_dir="./output"
qbit=2
rank=64
n_iter=10
G_moe=128
G_attn=4

mkdir -p ./logs

CUDA_VISIBLE_DEVICES=1 \
uv run python run_quantize.py \
    --model_path "$model_path" \
    --output_path "$save_dir/${model_path}-glorcq-${qbit}bit-rank${rank}-hybrid" \
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
    2>&1 | tee ./logs/glorcq_${qbit}bit_rank${rank}_hybrid.log

CUDA_VISIBLE_DEVICES=1 \
uv run python evaluate/eval_ppl.py \
    --model_path "$save_dir/${model_path}-glorcq-${qbit}bit-rank${rank}-hybrid" \
    2>&1 | tee -a ./logs/glorcq_${qbit}bit_rank${rank}_hybrid.log
