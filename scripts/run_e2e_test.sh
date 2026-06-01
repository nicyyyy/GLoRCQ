#!/usr/bin/env bash
export HF_HOME=/home/qyyang/resource_dir/hf_cache/
export TRANSFORMERS_CACHE=/home/qyyang/resource_dir/hf_cache/
export HF_DATASETS_CACHE=/home/qyyang/resource_dir/hf_cache/
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_DISABLE_CACHING=0

export CUDA_VISIBLE_DEVICES=3

MODEL=Qwen/Qwen1.5-MoE-A2.7B
QBIT=2
RANK=64
G_moe=128
# NSAMPLES=128
NSAMPLES=2
N_ITER=1

save_dir="/home/qyyang/resource_dir/GLoRCQ_out/"
tag="${MODEL}-glorcq-${QBIT}bit-rank${RANK}"

mkdir -p ./logs

# echo "=== Step 1: Fake-quant quantization ==="
# uv run python run_quantize.py \
#     --model_path $MODEL \
#     --output_path "${save_dir}/${tag}-fakequant" \
#     --qbit $QBIT --rank $RANK --G_moe $G_moe \
#     --nsamples $NSAMPLES --n_iter $N_ITER --w_clip \
#     --hessian_svd --use_turboquant --search_act_alpha \
#      2>&1 | tee "./logs/e2e_${QBIT}bit_rank${RANK}.log"

# echo "=== Step 2: Real-quant quantization ==="
# uv run python run_quantize.py \
#     --model_path $MODEL \
#     --output_path "${save_dir}/${tag}-realquant" \
#     --qbit $QBIT --rank $RANK --G_moe $G_moe \
#     --nsamples $NSAMPLES --n_iter $N_ITER --w_clip --real_quant \
#     --hessian_svd --use_turboquant --search_act_alpha --rotation_type hadamard \
#     2>&1 | tee "./logs/e2e_real_quant_${QBIT}bit_rank${RANK}.log"

# echo "=== Step 3: PPL evaluation (fake-quant) ==="
# uv run python evaluate/eval_ppl.py \
#     --model_path "${save_dir}/${tag}-fakequant" \
#     --output_json "${save_dir}/${tag}-fakequant/ppl.json"

echo "=== Step 4: Speed benchmark (real-quant) ==="
uv run python evaluate/eval_speed.py \
    --model_path "${save_dir}/${tag}-realquant" \
    --hf_model_path Qwen/Qwen1.5-MoE-A2.7B \
    --prompt_len 64 --gen_len 64 --num_runs 3 \
    2>&1 | tee "./logs/e2e_speed_${QBIT}bit_rank${RANK}.log"

# echo "=== Step 5: Results ==="
# echo "--- Fake-quant PPL ---"
# cat "${save_dir}/${tag}-fakequant/ppl.json"

# 导出 Chrome trace 可视化（可选）
# uv run python evaluate/profile_speed.py \
#       --model_path ${save_dir}/${tag}-realquant \
#       --trace_path ./profile_realquant.json --gen_len 64
