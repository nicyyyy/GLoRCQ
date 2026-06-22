#!/bin/bash
# Follow-up: retry jobs killed in exp_chain (Steps 1-3 were killed by transient SIGKILL)
# Also runs FP16 zero-shot. Start this after exp_chain finishes (or run in parallel).
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

echo "=== Follow-up pipeline started at $(date) ==="

# F1. GPTQ-only baseline (retry of killed Step 2)
wait_for_gpu4
echo "[$(date)] === F1: GPTQ-only baseline quantization (retry) ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_quantize.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path /home/qyyang/resource_dir/GLoRCQ_out/gptq_only_baseline \
    --qbit 2 --groupsize 128 --nsamples 128 \
    --rank 32 --n_iter 5 --uv_bits 8 \
    --use_turboquant --hessian_svd --search_act_alpha --no_lora \
    > logs/gptq_only_baseline_retry.log 2>&1
echo "[$(date)] F1 done"

# F2. GPTQ-only PPL eval
wait_for_gpu4
echo "[$(date)] === F2: GPTQ-only PPL eval ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_ppl.py \
    --model_path /home/qyyang/resource_dir/GLoRCQ_out/gptq_only_baseline \
    --device cuda:0 \
    > logs/gptq_only_ppl_retry.log 2>&1
echo "[$(date)] F2 done: $(grep 'WikiText-2 PPL' logs/gptq_only_ppl_retry.log)"

# F3. FP16 zero-shot eval (batch_size=1 for memory safety)
wait_for_gpu4
echo "[$(date)] === F3: FP16 zero-shot eval (batch_size=1) ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_zeroshot.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --device cuda:0 \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --batch_size 1 \
    --output_json logs/fp16_zeroshot_results.json \
    > logs/zeroshot_fp16_bs1.log 2>&1
echo "[$(date)] F3 done"
grep -E "arc_challenge|arc_easy|winogrande|hellaswag|piqa" logs/zeroshot_fp16_bs1.log | tail -10 || true

echo "=== Follow-up pipeline completed at $(date) ==="
