#!/usr/bin/env bash
# End-to-end test: quantize → save → evaluate PPL → benchmark speed
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# export HF_HOME=/nvme1/yqy/huggingface
export HF_HOME=/nvme1/yqy/huggingface
export TRANSFORMERS_CACHE=/nvme1/yqy/huggingface/transformers
export HF_DATASETS_CACHE=/nvme1/yqy/huggingface/datasets
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_DISABLE_CACHING=0

export CUDA_VISIBLE_DEVICES=1

MODEL=Qwen/Qwen1.5-MoE-A2.7B
QBIT=2
RANK=64
G_moe=128
NSAMPLES=128
N_ITER=1

save_dir="./output"
tag="${MODEL}-glorcq-${QBIT}bit-rank${RANK}"

mkdir -p ./glorcq/logs

echo "=== Step 1: Fake-quant quantization ==="
uv run python glorcq/run_quantize.py \
    --model_path $MODEL \
    --output_path "${save_dir}/${tag}-fakequant" \
    --qbit $QBIT --rank $RANK --G_moe $G_moe \
    --nsamples $NSAMPLES --n_iter $N_ITER --w_clip \
    --hessian_svd --use_turboquant --search_act_alpha \
     2>&1 | tee "./glorcq/logs/e2e_${QBIT}bit_rank${RANK}.log"

echo "=== Step 2: Real-quant quantization ==="
uv run python glorcq/run_quantize.py \
    --model_path $MODEL \
    --output_path "${save_dir}/${tag}-realquant" \
    --qbit $QBIT --rank $RANK --G_moe $G_moe \
    --nsamples $NSAMPLES --n_iter $N_ITER --w_clip --real_quant \
    --hessian_svd --use_turboquant --search_act_alpha

echo "=== Step 3: PPL evaluation (fake-quant) ==="
uv run python evaluate/eval_ppl.py \
    --model_path "${save_dir}/${tag}-fakequant" \
    --output_json "${save_dir}/${tag}-fakequant/ppl.json"

echo "=== Step 4: Speed benchmark (real-quant) ==="
uv run python evaluate/eval_speed.py \
    --model_path "${save_dir}/${tag}-realquant" \
    --prompt_len 64 --gen_len 64 --num_runs 3

echo "=== Step 5: Results ==="
echo "--- Fake-quant PPL ---"
cat "${save_dir}/${tag}-fakequant/ppl.json"
