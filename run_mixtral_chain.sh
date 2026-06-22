#!/bin/bash
# Mixtral-8x7B-v0.1 experiment chain
# Run from inside test:0 (srun session in SLURM step cgroup)
# This avoids the 5-minute GPU kill from the cluster monitor
#
# Usage: from tmux test:0 pane:
#   bash run_mixtral_chain.sh 2>&1 | tee logs/mixtral_chain.log
#
# Model: mistralai/Mixtral-8x7B-v0.1
# Architecture: 32 layers, 8 experts/layer, no shared expert
# FP16 size: ~47 GB → fits on GPU4 (80 GB A100)
# Total MoE experts: 256 (32 layers × 8)
cd /home/qyyang/repo/GLoRCQ

wait_for_gpu4() {
    echo "[$(date)] Waiting for GPU4 to be free..."
    while true; do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=4 2>/dev/null)
        if [ "$USED" -lt 2000 ]; then
            echo "[$(date)] GPU4 is free (${USED} MB used)"
            break
        fi
        echo "[$(date)] GPU4 has ${USED} MB used, waiting 60s..."
        sleep 60
    done
}

echo "=== Mixtral-8x7B chain started at $(date) ==="
echo "Running from inside SLURM step cgroup (no 5-min kill)"

# Step M1: GLoRCQ quantization
# G_moe=32: 256 experts / 8 groups = 32 experts per group
# n_lora_iter=2: balance quality vs compute (Mixtral is 4x larger than Qwen)
# n_iter=5: same as Qwen SOTA
wait_for_gpu4
echo "[$(date)] === Step M1: GLoRCQ quantization ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_quantize.py \
    --model_path mistralai/Mixtral-8x7B-v0.1 \
    --output_path /home/qyyang/resource_dir/GLoRCQ_out/mixtral_8x7b_sota \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --n_iter 5 \
    --G_moe 32 --G_attn 32 \
    --uv_bits 8 --sv_bits 8 \
    --n_lora_iter 2 \
    --use_turboquant --hessian_svd --search_act_alpha \
    > logs/mixtral_quant.log 2>&1
echo "[$(date)] Step M1 done"

# Step M2: PPL eval (WikiText-2)
wait_for_gpu4
echo "[$(date)] === Step M2: Mixtral PPL eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/mixtral_8x7b_sota \
    --device cuda:0 \
    > logs/mixtral_ppl.log 2>&1
echo "[$(date)] Step M2 done: $(grep 'WikiText-2 PPL' logs/mixtral_ppl.log)"

# Step M3: Zero-shot downstream eval (5 tasks)
wait_for_gpu4
echo "[$(date)] === Step M3: Mixtral zero-shot eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_zeroshot.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/mixtral_8x7b_sota \
    --device cuda:0 \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --batch_size 1 \
    --output_json logs/mixtral_zeroshot_results.json \
    > logs/mixtral_zeroshot.log 2>&1
echo "[$(date)] Step M3 done"
grep -E "acc_norm|acc\b" logs/mixtral_zeroshot.log | tail -8 || true

# Step M4: FP16 baseline PPL (for comparison)
wait_for_gpu4
echo "[$(date)] === Step M4: Mixtral FP16 PPL eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path mistralai/Mixtral-8x7B-v0.1 \
    --device cuda:0 \
    > logs/mixtral_fp16_ppl.log 2>&1
echo "[$(date)] Step M4 done: $(grep 'WikiText-2 PPL' logs/mixtral_fp16_ppl.log)"

# Step M5: FP16 zero-shot eval (for comparison)
wait_for_gpu4
echo "[$(date)] === Step M5: Mixtral FP16 zero-shot eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_zeroshot.py \
    --model_path mistralai/Mixtral-8x7B-v0.1 \
    --device cuda:0 \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --batch_size 1 \
    --output_json logs/mixtral_fp16_zeroshot_results.json \
    > logs/mixtral_fp16_zeroshot.log 2>&1
echo "[$(date)] Step M5 done"
grep -E "acc_norm|acc\b" logs/mixtral_fp16_zeroshot.log | tail -8 || true

echo "=== Mixtral-8x7B chain completed at $(date) ==="
