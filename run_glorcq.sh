#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ ! -d ".venv" ]; then
    uv venv --python /nvme3/miniconda3/bin/python .venv
    uv pip install -r requirements.txt
fi

export HF_HOME=/nvme1/yqy/huggingface
export TRANSFORMERS_CACHE=/nvme1/yqy/huggingface/transformers
export HF_DATASETS_CACHE=/nvme1/yqy/huggingface/datasets
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_DISABLE_CACHING=0

model_path="Qwen/Qwen1.5-MoE-A2.7B"
# model_path="mistralai/Mixtral-8x7B-v0.1"
save_dir="./output"
qbit=2
rank=64
n_iter=5
G_moe=128
G_attn=4

mkdir -p ./log

CUDA_VISIBLE_DEVICES=3 \
uv run python glorcq/run_quantize.py \
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
    2>&1 | tee ./log/glorcq_${qbit}bit_rank${rank}_hybrid.log

CUDA_VISIBLE_DEVICES=1 \
uv run python eval/evaluation.py \
    "$save_dir/${model_path}-glorcq-${qbit}bit-rank${rank}-hybrid" \
    "wikitext2" \
    2>&1 | tee -a ./log/glorcq_${qbit}bit_rank${rank}_hybrid.log
