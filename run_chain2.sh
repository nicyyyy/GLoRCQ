#!/bin/bash
# Experiment chain v2 - run from inside test:0 (srun session in SLURM step cgroup)
# This avoids the 5-minute GPU kill from the cluster monitor
cd /home/qyyang/repo/GLoRCQ

wait_for_gpu4() {
    echo "[$(date)] Waiting for GPU4 to be free..."
    while true; do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=4 2>/dev/null)
        if [ "$USED" -lt 2000 ]; then
            echo "[$(date)] GPU4 is free (${USED} MB used)"
            break
        fi
        echo "[$(date)] GPU4 has ${USED} MB used, waiting 30s..."
        sleep 30
    done
}

echo "=== Chain v2 started at $(date) ==="
echo "Running from inside SLURM step cgroup (no 5-min kill)"

# Step A: GPTQ-only baseline quantization (no LoRA)
wait_for_gpu4
echo "[$(date)] === Step A: GPTQ-only baseline quantization ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path /home/qyyang/resource_dir/GLoRCQ_out/gptq_only_baseline \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --n_iter 5 --uv_bits 8 \
    --use_turboquant --hessian_svd --search_act_alpha --no_lora \
    > logs/gptq_only_baseline.log 2>&1
echo "[$(date)] Step A done"

# Step B: GPTQ-only PPL eval
wait_for_gpu4
echo "[$(date)] === Step B: GPTQ-only PPL eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/gptq_only_baseline \
    --device cuda:0 \
    > logs/gptq_only_ppl.log 2>&1
echo "[$(date)] Step B done: $(grep 'WikiText-2 PPL' logs/gptq_only_ppl.log)"

# Step C: real_quant speed test (SOTA rank config: rank_attn=512, rank_down=128)
wait_for_gpu4
echo "[$(date)] === Step C: real_quant speed test (rank_attn=512, rank_down=128) ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path /home/qyyang/resource_dir/GLoRCQ_out/speed_test_r128 \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --n_iter 1 --rank_attn 512 --rank_down 128 \
    --G_moe 128 --G_attn 24 --uv_bits 8 --sv_bits 8 \
    --n_lora_iter 1 --use_turboquant --hessian_svd --search_act_alpha --real_quant \
    > logs/speed_test_realquant_r128.log 2>&1
echo "[$(date)] Step C done"

# Step D: Speed evaluation
wait_for_gpu4
echo "[$(date)] === Step D: Speed eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_speed.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/speed_test_r128 \
    --hf_model_path Qwen/Qwen1.5-MoE-A2.7B \
    > logs/e2e_speed_sota_r128.log 2>&1
echo "[$(date)] Step D done"
grep -E "Standard|Graph|FP16|Speedup|Peak GPU|tok/s" logs/e2e_speed_sota_r128.log | tail -10 || true

# Step E: No-sharing ablation (G_moe=1440, each expert independent)
wait_for_gpu4
echo "[$(date)] === Step E: No-sharing ablation (G_moe=1440) ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path /home/qyyang/resource_dir/GLoRCQ_out/no_sharing_G1440 \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --n_iter 5 --G_moe 1440 --G_attn 24 \
    --uv_bits 8 --sv_bits 8 \
    --n_lora_iter 2 --use_turboquant --hessian_svd --search_act_alpha \
    > logs/no_sharing_G1440.log 2>&1
echo "[$(date)] Step E done"

# Step F: No-sharing PPL eval
wait_for_gpu4
echo "[$(date)] === Step F: No-sharing PPL eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/no_sharing_G1440 \
    --device cuda:0 \
    > logs/no_sharing_G1440_ppl.log 2>&1
echo "[$(date)] Step F done: $(grep 'WikiText-2 PPL' logs/no_sharing_G1440_ppl.log)"

# Step G: FP16 zero-shot eval (batch_size=1 for memory safety)
wait_for_gpu4
echo "[$(date)] === Step G: FP16 zero-shot eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_zeroshot.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --device cuda:0 \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --batch_size 1 \
    --output_json logs/fp16_zeroshot_results.json \
    > logs/zeroshot_fp16_bs1.log 2>&1
echo "[$(date)] Step G done"
grep -E "acc_norm|acc\b" logs/zeroshot_fp16_bs1.log | tail -8 || true

echo "=== Chain v2 completed at $(date) ==="
