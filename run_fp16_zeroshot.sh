#!/bin/bash
# Follow-up: FP16 baseline zero-shot eval (after exp_chain completes)
# Uses eval_zeroshot.py with batch_size=1 to minimize memory pressure
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

wait_for_gpu4
echo "[$(date)] === FP16 zero-shot eval (batch_size=1) ==="
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluate/eval_zeroshot.py \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --device cuda:0 \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --batch_size 1 \
    --output_json logs/fp16_zeroshot_results.json \
    > logs/zeroshot_fp16_bs1.log 2>&1
echo "[$(date)] FP16 zero-shot done"
grep -E "acc_norm|acc\b|results" logs/zeroshot_fp16_bs1.log | tail -20 || true

echo "=== Done at $(date) ==="
