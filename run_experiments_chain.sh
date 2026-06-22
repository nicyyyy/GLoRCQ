#!/bin/bash
# Chain of experiments to run sequentially on GPU4
# Run from /home/qyyang/repo/GLoRCQ/
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

echo "=== Experiment pipeline started at $(date) ==="

# 1. FP16 PPL eval
wait_for_gpu4
echo "[$(date)] === Step 1: FP16 PPL eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --device cuda:0 \
    > logs/fp16_ppl.log 2>&1
echo "[$(date)] FP16 PPL done: $(grep 'WikiText-2 PPL' logs/fp16_ppl.log)"

# 1b. FP16 zero-shot eval (via lm_eval CLI for stability)
wait_for_gpu4
echo "[$(date)] === Step 1b: FP16 zero-shot eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m lm_eval \
    --model hf \
    --model_args "pretrained=Qwen/Qwen1.5-MoE-A2.7B,dtype=float16,trust_remote_code=True" \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --batch_size 4 \
    --device cuda:0 \
    --output_path logs/lm_eval_fp16_results \
    > logs/zeroshot_fp16_baseline.log 2>&1 || echo "[WARN] FP16 zero-shot eval failed or was skipped"

# 2. GPTQ-only baseline quantization
wait_for_gpu4
echo "[$(date)] === Step 2: GPTQ-only baseline quantization ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path /home/qyyang/resource_dir/GLoRCQ_out/gptq_only_baseline \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --n_iter 5 --uv_bits 8 \
    --use_turboquant --hessian_svd --search_act_alpha --no_lora \
    > logs/gptq_only_baseline.log 2>&1
echo "[$(date)] GPTQ-only quant done"

# 3. GPTQ-only PPL eval
wait_for_gpu4
echo "[$(date)] === Step 3: GPTQ-only PPL eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/gptq_only_baseline \
    --device cuda:0 \
    > logs/gptq_only_ppl.log 2>&1
echo "[$(date)] GPTQ-only PPL done: $(grep 'WikiText-2 PPL' logs/gptq_only_ppl.log)"

# 4. real_quant speed test (SOTA rank config, n_iter=1 for speed)
wait_for_gpu4
echo "[$(date)] === Step 4: real_quant speed test ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path /home/qyyang/resource_dir/GLoRCQ_out/speed_test_r128 \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --n_iter 1 --rank_attn 512 --rank_down 128 \
    --G_moe 128 --G_attn 24 --uv_bits 8 --sv_bits 8 \
    --n_lora_iter 1 --use_turboquant --hessian_svd --search_act_alpha --real_quant \
    > logs/speed_test_realquant_r128.log 2>&1
echo "[$(date)] real_quant done"

# 5. Speed evaluation
wait_for_gpu4
echo "[$(date)] === Step 5: Speed eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_speed.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/speed_test_r128 \
    --hf_model_path Qwen/Qwen1.5-MoE-A2.7B \
    > logs/e2e_speed_sota_r128.log 2>&1
echo "[$(date)] Speed eval done"
grep -E "Standard|Graph|FP16|Speedup|Peak GPU" logs/e2e_speed_sota_r128.log || true

# 6. No-sharing ablation (G_moe=1440, each expert independent)
wait_for_gpu4
echo "[$(date)] === Step 6: No-sharing ablation (G_moe=1440) ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path /home/qyyang/resource_dir/GLoRCQ_out/no_sharing_G1440 \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --n_iter 5 --G_moe 1440 --G_attn 24 \
    --uv_bits 8 --sv_bits 8 \
    --n_lora_iter 2 --use_turboquant --hessian_svd --search_act_alpha \
    > logs/no_sharing_G1440.log 2>&1
echo "[$(date)] No-sharing quant done"

# 7. No-sharing PPL eval
wait_for_gpu4
echo "[$(date)] === Step 7: No-sharing PPL eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/no_sharing_G1440 \
    --device cuda:0 \
    > logs/no_sharing_G1440_ppl.log 2>&1
echo "[$(date)] No-sharing PPL done: $(grep 'WikiText-2 PPL' logs/no_sharing_G1440_ppl.log)"

echo "=== All experiments completed at $(date) ==="
